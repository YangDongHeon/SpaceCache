"""Convert a ScanNet .sens RGB-D video into a per-frame point-cloud sequence
in the on-disk format of preprocess_scannet.py (read by ScanNetVideoDataset).

Each selected depth frame is back-projected to 3D and written as its own
ScanNet-style scene directory:

    <out>/frame_{i:06d}/
        coord.npy      (M,3) float32   camera-frame xyz (world: --frame world)
        color.npy      (M,3) uint8
        normal.npy     (M,3) float32   estimated from the depth map
        segment20.npy  (M,)  int16     nyu40->20-class index, -1 ignore
        segment200.npy (M,)  int16     scannet200 index,      -1 ignore
        pose.npy       (4,4) float32   camera to world (axisAlignment folded
                                       in with --axis-align)

Labels are remapped from the 2D label-filt images as in preprocess_scannet.py:
raw scannet id -> nyu40id -> VALID_CLASS_IDS_20 index, and raw id ->
VALID_CLASS_IDS_200 index. A first pass records the byte offset of each
selected frame, then a process pool decodes and writes the frames
(--num-workers). Requires numpy, pandas and imageio (no open3d).

The evaluation sequences (scripts/reproduce/prepare_scannet_video.sh) use
3 fps (--frame-skip 10 of the 30 Hz capture), camera-frame coords (ApplyPose
maps them to the world frame at load time) and no --axis-align, like
data/scannet:

    python scripts/preprocess_scannet_video.py \\
        --sens <scans>/scene0426_00/scene0426_00.sens \\
        --label-zip <scans>/scene0426_00/scene0426_00_2d-label-filt.zip \\
        --out data/scannetvideo/val/scene0426_00 --frame-skip 10
"""
import argparse
import os
import struct
import sys
import zlib
import zipfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import imageio.v2 as imageio
import numpy as np
import pandas as pd

COLOR_COMPRESSION = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
DEPTH_COMPRESSION = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_DEFAULT = os.path.join(
    REPO, "pointcept/datasets/preprocessing/scannet/meta_data")


# ----------------------------- .sens header ------------------------------
def read_header(path):
    """Read the .sens header; return (header dict, file pos of first frame)."""
    f = open(path, "rb")
    struct.unpack("I", f.read(4))[0]                          # version
    nlen = struct.unpack("Q", f.read(8))[0]
    f.read(nlen)                                              # sensor name
    f.read(64)                                               # intrinsic color
    f.read(64)                                               # extrinsic color
    K_depth = np.frombuffer(f.read(64), np.float32).reshape(4, 4).copy()
    f.read(64)                                               # extrinsic depth
    h = dict(
        K_depth=K_depth,
        color_comp=COLOR_COMPRESSION[struct.unpack("i", f.read(4))[0]],
        depth_comp=DEPTH_COMPRESSION[struct.unpack("i", f.read(4))[0]],
        color_w=struct.unpack("I", f.read(4))[0],
        color_h=struct.unpack("I", f.read(4))[0],
        depth_w=struct.unpack("I", f.read(4))[0],
        depth_h=struct.unpack("I", f.read(4))[0],
        depth_shift=struct.unpack("f", f.read(4))[0],
        num_frames=struct.unpack("Q", f.read(8))[0],
    )
    pos = f.tell()
    f.close()
    return h, pos


def scan_offsets(path, first_pos, num_frames, skip):
    """Fast pass: byte offset (pose start) of every Nth frame. No decode."""
    f = open(path, "rb")
    f.seek(first_pos)
    offsets = []
    for i in range(num_frames):
        off = f.tell()
        f.read(64 + 16)                                      # pose + 2 timestamps
        csz = struct.unpack("Q", f.read(8))[0]
        dsz = struct.unpack("Q", f.read(8))[0]
        if i % skip == 0:
            offsets.append((i, off))
        f.seek(csz + dsz, 1)
    f.close()
    return offsets


def decode_color(b, h):
    if h["color_comp"] in ("jpeg", "png"):
        return imageio.imread(b)[..., :3]
    return np.frombuffer(b, np.uint8).reshape(h["color_h"], h["color_w"], 3)


def decode_depth(b, h):
    if h["depth_comp"] == "zlib_ushort":
        b = zlib.decompress(b)
    return np.frombuffer(b, np.uint16).reshape(h["depth_h"], h["depth_w"]).copy()


# ----------------------------- label remap -------------------------------
def build_label_luts(meta_dir):
    """id->segment20 index and id->segment200 index, matching preprocess_scannet."""
    sys.path.insert(0, meta_dir)
    from scannet200_constants import VALID_CLASS_IDS_20, VALID_CLASS_IDS_200
    tsv = os.path.join(meta_dir, "scannetv2-labels.combined.tsv")
    df = pd.read_csv(tsv, sep="\t", header=0)
    df = df[df["id"].notna()]
    df["nyu40id"] = df["nyu40id"].fillna(0)                   # some rows lack nyu40id
    max_id = int(df["id"].max())
    nyu_of_id = np.zeros(max_id + 1, np.int64)
    for rid, nyu in zip(df["id"].to_numpy(), df["nyu40id"].to_numpy()):
        nyu_of_id[int(rid)] = int(nyu)
    v20 = {int(c): i for i, c in enumerate(VALID_CLASS_IDS_20)}
    v200 = {int(c): i for i, c in enumerate(VALID_CLASS_IDS_200)}
    seg20_lut = np.full(max_id + 1, -1, np.int16)
    seg200_lut = np.full(max_id + 1, -1, np.int16)
    for i in range(max_id + 1):
        seg20_lut[i] = v20.get(int(nyu_of_id[i]), -1)
        seg200_lut[i] = v200.get(i, -1)
    return seg20_lut, seg200_lut, max_id


# ----------------------------- geometry ----------------------------------
def grid_points(depth, K, depth_shift):
    """Full (H,W,3) point grid in the camera frame; invalid depth -> z=0."""
    H, W = depth.shape
    z = depth.astype(np.float32) / depth_shift
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.stack([x, y, z], -1), z


def estimate_normals(P, valid):
    """Per-pixel normals from neighbor cross-products, oriented toward camera.
    Pixels whose neighbors are invalid fall back to the view direction."""
    H, W = valid.shape
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1, :] = P[2:, :] - P[:-2, :]
    oku = np.zeros((H, W), bool); okv = np.zeros((H, W), bool)
    oku[:, 1:-1] = valid[:, 2:] & valid[:, :-2]
    okv[1:-1, :] = valid[2:, :] & valid[:-2, :]
    n = np.cross(du, dv)
    bad = ~(oku & okv)
    view = -P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-8)
    n[bad] = view[bad]
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    flip = np.sum(n * P, axis=-1) > 0
    n[flip] *= -1
    return n.astype(np.float32)


def read_axis_align(info_path):
    if not info_path or not os.path.exists(info_path):
        return None
    with open(info_path) as f:
        for line in f:
            if line.startswith("axisAlignment"):
                vals = [float(x) for x in line.split("=")[1].split()]
                return np.array(vals, np.float32).reshape(4, 4)
    return None


# ----------------------------- worker ------------------------------------
_G = {}


def _init_worker(path, header, cfg, seg20_lut, seg200_lut, max_id, label_zip):
    _G["f"] = open(path, "rb")
    _G["h"] = header
    _G["cfg"] = cfg
    _G["seg20"] = seg20_lut
    _G["seg200"] = seg200_lut
    _G["max_id"] = max_id
    _G["lz"] = zipfile.ZipFile(label_zip) if label_zip else None
    _G["lnames"] = set(_G["lz"].namelist()) if label_zip else None


def _process(task):
    i, off = task
    f, h, cfg = _G["f"], _G["h"], _G["cfg"]
    f.seek(off)
    pose = np.frombuffer(f.read(64), np.float32).reshape(4, 4).copy()
    if not np.all(np.isfinite(pose)):
        return (i, 0)
    f.read(16)                                               # timestamps
    csz = struct.unpack("Q", f.read(8))[0]
    dsz = struct.unpack("Q", f.read(8))[0]
    color = decode_color(f.read(csz), h)
    depth = decode_depth(f.read(dsz), h)

    P, z = grid_points(depth, h["K_depth"], h["depth_shift"])
    valid = (z > 0) & (z < cfg["trunc"])
    normals = estimate_normals(P, valid)

    st = cfg["stride"]
    sel = np.zeros_like(valid)
    sel[::st, ::st] = True
    m = valid & sel
    vs, us = np.nonzero(m)
    cam = P[vs, us]
    nrm = normals[vs, us]

    cu = np.clip((us * h["color_w"] / h["depth_w"]).astype(np.int64), 0, h["color_w"] - 1)
    cv = np.clip((vs * h["color_h"] / h["depth_h"]).astype(np.int64), 0, h["color_h"] - 1)
    rgb = color[cv, cu].astype(np.uint8)

    align = cfg["align"]
    T = align @ pose if align is not None else pose
    if cfg["frame"] == "world":
        coord = (cam @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        nrm = (nrm @ T[:3, :3].T).astype(np.float32)
    else:
        coord = cam.astype(np.float32)

    fdir = os.path.join(cfg["out"], f"frame_{i:06d}")
    os.makedirs(fdir, exist_ok=True)
    np.save(os.path.join(fdir, "coord.npy"), coord)
    np.save(os.path.join(fdir, "color.npy"), rgb)
    np.save(os.path.join(fdir, "normal.npy"), nrm)
    np.save(os.path.join(fdir, "pose.npy"), T.astype(np.float32))
    if _G["lz"] is not None:
        for cand in (f"label-filt/{i}.png", f"{i}.png"):
            if cand in _G["lnames"]:
                lbl = imageio.imread(_G["lz"].read(cand))
                lh, lw = lbl.shape[:2]
                lu = np.clip((us * lw / h["depth_w"]).astype(np.int64), 0, lw - 1)
                lv = np.clip((vs * lh / h["depth_h"]).astype(np.int64), 0, lh - 1)
                ids = np.clip(lbl[lv, lu].astype(np.int64), 0, _G["max_id"])
                np.save(os.path.join(fdir, "segment20.npy"), _G["seg20"][ids])
                np.save(os.path.join(fdir, "segment200.npy"), _G["seg200"][ids])
                break
    return (i, len(coord))


# ----------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", required=True)
    ap.add_argument("--info", default=None, help="scene*.txt (for axisAlignment)")
    ap.add_argument("--label-zip", default=None, help="2d-label-filt.zip (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default=META_DEFAULT, help="scannet meta_data dir")
    ap.add_argument("--frame-skip", type=int, default=20,
                    help="keep every Nth frame (ignored if --fps given)")
    ap.add_argument("--fps", type=float, default=None,
                    help="target fps; derives frame-skip from --src-fps")
    ap.add_argument("--src-fps", type=float, default=30.0)
    ap.add_argument("--pixel-stride", type=int, default=1)
    ap.add_argument("--depth-trunc", type=float, default=8.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--frame", choices=["local", "world"], default="local")
    ap.add_argument("--axis-align", action="store_true")
    ap.add_argument("--num-workers", type=int, default=mp.cpu_count(),
                    help="parallel worker processes (default: all cores)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    h, first_pos = read_header(args.sens)
    skip = args.frame_skip
    if args.fps is not None:
        skip = max(1, round(args.src_fps / args.fps))
    print(f"frames={h['num_frames']} depth={h['depth_w']}x{h['depth_h']} "
          f"color={h['color_w']}x{h['color_h']} skip={skip} frame={args.frame} "
          f"workers={args.num_workers}")

    offsets = scan_offsets(args.sens, first_pos, h["num_frames"], skip)
    if args.max_frames:
        offsets = offsets[:args.max_frames]
    print(f"selected {len(offsets)} frames")

    align = read_axis_align(args.info) if args.axis_align else None
    seg20_lut = seg200_lut = None
    max_id = 0
    if args.label_zip:
        seg20_lut, seg200_lut, max_id = build_label_luts(args.meta)

    cfg = dict(out=args.out, trunc=args.depth_trunc, stride=args.pixel_stride,
               frame=args.frame, align=align)
    init_args = (args.sens, h, cfg, seg20_lut, seg200_lut, max_id, args.label_zip)

    n = 0
    with ProcessPoolExecutor(max_workers=args.num_workers,
                             initializer=_init_worker, initargs=init_args) as ex:
        for i, pts in ex.map(_process, offsets, chunksize=4):
            n += 1
            if n % 100 == 0:
                print(f"  {n}/{len(offsets)} (last frame={i}, pts={pts})")
    print(f"done: {n} frame scene-dirs -> {args.out}")


if __name__ == "__main__":
    main()
