"""Render the streaming demo (GT | Baseline | ToMe | SpaceCache).

Simulates a LiDAR stream in wall-clock time (2 Hz labelled Waymo sweeps,
or 5 Hz with the waymo_litept_5hz preset). Each frame arrives at its own
timestamp (only segments whose frames are evenly spaced at the preset's
period are accepted); each method processes every frame in order (FIFO),
starting when the frame has arrived and the previous one is done, and takes
the latency measured for that frame (mean over eval passes 2..N of the
latency runs). A panel shows the latest frame its method has finished, so a
method slower than the stream falls further behind the sensor.

Reads `<root>/dump/<method>/` (dump_frames.py) and
`<root>/latency/<run name>/per_frame_timing.jsonl` (run.py with timing),
both produced with spacecache/demo/waymo_utonia_demo.py.

    PYTHONPATH=. python spacecache/demo/render_demo.py \\
        --segment 9231652062943496183 --start_frame 3 \\
        --out figures/demo.webp --gif_width 1400 --quality 80
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_DRAW = None


def _draw_one(now):
    return _DRAW(now)


def run_frames(draw, t_frames, jobs):
    """Render draw(t) for every t, on `jobs` forked processes (they inherit
    the loaded state, so only the times and the images are sent)."""
    global _DRAW
    if jobs <= 1 or len(t_frames) < 2:
        out = []
        for i, t in enumerate(t_frames):
            out.append(draw(t))
            if i % 20 == 0:
                print(f"{i}/{len(t_frames)}", flush=True)
        return out
    import multiprocessing as mp
    _DRAW = draw
    out = []
    with mp.get_context("fork").Pool(jobs) as pool:
        # contiguous chunks keep each worker's frame cache warm
        chunk = max(1, len(t_frames) // (jobs * 4))
        for i, im in enumerate(pool.imap(_draw_one, t_frames, chunk)):
            out.append(im)
            if i % 40 == 0:
                print(f"{i}/{len(t_frames)}", flush=True)
    return out

# One preset per demo. methods: (dump dir, latency run, full-val Waymo
# mIoU with flash attention, eval at 2 Hz). gt: the stream is labelled, so
# a ground-truth panel and the error rate are shown.
PRESETS = {
    "waymo_utonia": dict(
        root="output/demo/waymo_utonia_demo", period=0.5, gt=True,
        backbone="Utonia", data="Waymo", panel_w=480, panel_h=440,
        methods={
            "backbone": ("backbone", "backbone_flash", 67.17),
            "tome": ("tome_r004", "tome_flash_r0.04", 63.85),
            "ours": ("spacecache_s34_c095",
                     "spacecache_s34_flash_L8_conf0.95_triton", 67.21),
        }),
    # every other 10 Hz sweep, mostly unlabelled (preprocess_waymo_stream.py);
    # frames keep their own timestamps (uniform=False), so a sweep missing
    # from the log just arrives late
    "waymo_litept_5hz": dict(
        root="output/demo/waymo_litept_5hz", period=0.2, gt=False,
        uniform=False,
        backbone="LitePT", data="Waymo", panel_w=640, panel_h=480,
        methods={
            "backbone": ("backbone", "backbone_flash", 72.82),
            # the merge ratio whose latency on the 8888 clip matches
            # SpaceCache's (about 184 vs 182 ms), under 0.2 s
            "tome": ("tome_r012", "tome_flash_r0.12", 70.97),
            "ours": ("spacecache_s34_c095",
                     "spacecache_s34_flash_L8_conf0.95_triton", 72.75),
        }),
    # faster operating points: SpaceCache theta=0.8 at about ToMe r=0.18's
    # latency on the 8888 clip (157 vs 165 ms)
    "waymo_litept_5hz_fast": dict(
        root="output/demo/waymo_litept_5hz", period=0.2, gt=False,
        uniform=False,
        backbone="LitePT", data="Waymo", panel_w=640, panel_h=480,
        methods={
            "backbone": ("backbone", "backbone_flash", 72.82),
            "tome": ("tome_r018", "tome_flash_r0.18", 66.81),
            "ours": ("spacecache_s34_c080",
                     "spacecache_s34_flash_L8_conf0.8_triton", 72.35),
        }),
    # dense attention as in Table 1; ToMe at the merge ratio whose latency
    # matches SpaceCache's (212 vs 210 ms on the latency slice, 2 Hz)
    "waymo_litept_5hz_dense": dict(
        root="output/demo/waymo_litept_5hz", period=0.2, gt=False,
        uniform=False,
        backbone="LitePT", data="Waymo", panel_w=640, panel_h=480,
        methods={
            "backbone": ("backbone_dense", "backbone_dense", 72.82),
            "tome": ("tome_dense_r016", "tome_dense_r0.16", 69.11),
            "ours": ("spacecache_s34_dense_c095",
                     "spacecache_s34_dense_L8_conf0.95_triton", 72.02),
        }),
}
PRESET = PRESETS["waymo_utonia"]
ROOT = PRESET["root"]
METHODS = PRESET["methods"]
PERIOD = PRESET["period"]  # s between sweeps, also the latency budget

# Waymo semseg classes (22) in the SemanticKITTI colours (semantic-kitti.yaml
# color_map, converted from BGR); classes KITTI lacks take the nearest one.
PALETTE = np.array([
    (100, 150, 245),  # Car                 <- car
    (80, 30, 180),    # Truck               <- truck
    (100, 80, 250),   # Bus                 <- bus
    (0, 0, 255),      # Other Vehicle       <- other-vehicle
    (150, 30, 90),    # Motorcyclist        <- motorcyclist
    (255, 40, 200),   # Bicyclist           <- bicyclist
    (255, 30, 30),    # Pedestrian          <- person
    (255, 0, 0),      # Sign                <- traffic-sign
    (255, 0, 0),      # Traffic Light       <- traffic-sign
    (255, 240, 150),  # Pole                <- pole
    (50, 255, 255),   # Construction Cone   <- other-object
    (100, 230, 245),  # Bicycle             <- bicycle
    (30, 60, 150),    # Motorcycle          <- motorcycle
    (255, 200, 0),    # Building            <- building
    (0, 175, 0),      # Vegetation          <- vegetation
    (135, 60, 0),     # Tree Trunk          <- trunk
    (175, 0, 75),     # Curb                <- other-ground
    (255, 0, 255),    # Road                <- road
    (150, 255, 170),  # Lane Marker         <- lane-marking
    (150, 240, 80),   # Other Ground        <- terrain
    (255, 150, 255),  # Walkable            <- parking
    (75, 0, 75),      # Sidewalk            <- sidewalk
], dtype=np.float32)
IGNORE_RGB = np.array((160, 160, 165), dtype=np.float32)

# Colours by theme. GRAY_SCENE: correct points with --error_style gray.
THEMES = {
    "light": dict(
        BG=(255, 255, 255), FG=(22, 24, 30), DIM=(120, 124, 132),
        LABEL=(22, 24, 30), OK=(30, 150, 70), BAD=(215, 45, 45),
        ACCENT=(0, 92, 200), GRAY_SCENE=(200, 203, 209),
        LANE_BG=(234, 236, 240), SLOT_A=(205, 209, 217),
        SLOT_B=(178, 183, 193), SLOT_FUTURE=(238, 240, 244),
        DIM_SCENE=0.38, DIFF=(225, 30, 30)),
    "dark": dict(
        BG=(16, 18, 22), FG=(235, 236, 240), DIM=(150, 154, 162),
        LABEL=(235, 236, 240), OK=(90, 200, 120), BAD=(235, 80, 80),
        ACCENT=(255, 215, 0), GRAY_SCENE=(80, 84, 92),
        LANE_BG=(30, 33, 39), SLOT_A=(72, 77, 88), SLOT_B=(52, 56, 65),
        SLOT_FUTURE=(36, 39, 46), DIM_SCENE=0.42, DIFF=(255, 80, 80)),
}
# DIFF: on an unlabelled stream, points a method predicts differently from
# the baseline (red, on grey)

# How misclassified points are drawn in the prediction panels; each colour
# stays clear of the UI's green / red.
#   gray:  correct points in plain grey (no class colours), errors hot pink;
#          the only colour in the panel is an error
#   cyan:  correct points in faded class colours, errors cyan
#   black: correct points in faded class colours, errors black (white on
#          the dark theme)
ERROR_STYLES = {
    "gray": dict(scene="gray", light=(235, 0, 140), dark=(255, 60, 170)),
    "cyan": dict(scene="faded", light=(0, 185, 230), dark=(0, 220, 255)),
    "black": dict(scene="faded", light=(0, 0, 0), dark=(255, 255, 255)),
}
ERROR_SCENE = "gray"


def apply_theme(name, error_style="gray"):
    """Set the colour globals read at draw time."""
    t = THEMES[name]
    g = globals()
    for k in ("BG", "FG", "DIM", "LABEL", "OK", "BAD", "ACCENT", "LANE_BG",
              "SLOT_A", "SLOT_B", "SLOT_FUTURE", "DIM_SCENE", "GRAY_SCENE"):
        g[k] = t[k]
    es = ERROR_STYLES[error_style]
    g["ERROR_RGB"] = np.array(es[name], np.float32)
    g["ERROR_SCENE"] = es["scene"]
    g["DIFF_RGB"] = np.array(t["DIFF"], np.float32)


apply_theme("light")

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


_FONTS = {}


def font(size, bold=False):
    path = FONT_B if bold else FONT
    if (path, size) not in _FONTS:
        _FONTS[path, size] = ImageFont.truetype(path, size)
    return _FONTS[path, size]


# --------------------------------------------------------------------------
# data


def load_latency(root, run, n):
    """Per-frame latency (s): mean over eval passes 2..N, as in the paper's
    latency tables (pass 1 warms up the allocator and kernels)."""
    with open(os.path.join(root, "latency", run,
                           "per_frame_timing.jsonl")) as f:
        t = np.array([json.loads(line)["total_ms"] for line in f])
    if len(t) < n or len(t) % n:
        raise SystemExit(f"{run}: {len(t)} timed frames, expected a "
                         f"multiple of {n}")
    passes = t.reshape(-1, n)
    return (passes[1:] if len(passes) > 1 else passes).mean(0) / 1000.0


def segment_frames(frames, key):
    idx = [i for i, f in enumerate(frames) if key in f]
    if not idx:
        raise SystemExit(f"no frame matches segment {key!r}")
    return idx


def arrival_times(names):
    """Arrival (s) of each frame from its timestamp directory (us).

    Waymo's labelled frames are 0.5 s apart in about half of the validation
    segments; the others have 5.5 s gaps or 0.1-0.4 s spacing, which a 2 Hz
    stream cannot represent, so they are rejected."""
    ts = np.array([int(os.path.basename(n)) for n in names]) / 1e6
    arr = ts - ts[0]
    dt = np.diff(arr)
    if not PRESET.get("uniform", True):
        # a sweep missing from the 10 Hz log only lengthens one slot
        return arr
    if len(dt) and np.abs(dt - PERIOD).max() > 0.05:
        bad = int(np.argmax(np.abs(dt - PERIOD)))
        raise SystemExit(
            f"frames are not {PERIOD} s apart (frame {bad}->{bad + 1}: "
            f"{dt[bad]:.2f} s); pick a segment with uniform spacing")
    return arr


def simulate(lat, arr):
    """FIFO on the stream: (arrival, start, done) per frame (s)."""
    n = len(lat)
    start, done = np.zeros(n), np.zeros(n)
    t = 0.0
    for k in range(n):
        start[k] = max(arr[k], t)
        done[k] = t = start[k] + lat[k]
    return arr, start, done


class Frames:
    """Lazy per-frame loader of one dump, in a shared world frame."""

    def __init__(self, dump_dir, idx, origin):
        self.dir, self.idx, self.origin = dump_dir, idx, origin
        self._cache = {}

    def __getitem__(self, k):
        if k not in self._cache:
            d = np.load(os.path.join(self.dir, f"{self.idx[k]:05d}.npz"))
            pose = d["pose"]
            xyz = (d["coord"].astype(np.float64) @ pose[:3, :3].T
                   + (pose[:3, 3] - self.origin)).astype(np.float32)
            self._cache[k] = dict(xyz=xyz, pred=d["pred"], gt=d["gt"],
                                  hit3=d["hit3"] if "hit3" in d else None)
            if len(self._cache) > 6:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[k]


# --------------------------------------------------------------------------
# camera and rasterizer


def ego_track(poses, origin):
    pos = np.stack([p[:3, 3] - origin for p in poses])
    yaw = np.unwrap([math.atan2(p[1, 0], p[0, 0]) for p in poses])
    return pos, yaw


def interp_track(track, x):
    """Ego position and yaw at fractional frame index x."""
    pos, yaw = track
    x = float(np.clip(x, 0, len(yaw) - 1))
    i = min(int(x), len(yaw) - 2) if len(yaw) > 1 else 0
    a = x - i if len(yaw) > 1 else 0.0
    j = min(i + 1, len(yaw) - 1)
    return pos[i] * (1 - a) + pos[j] * a, yaw[i] * (1 - a) + yaw[j] * a


class Camera:
    def __init__(self, w, h, fov=58.0, back=24.0, up=15.0, ahead=14.0):
        self.w, self.h = w, h
        self.f = (w / 2) / math.tan(math.radians(fov) / 2)
        self.back, self.up, self.ahead = back, up, ahead

    def look(self, pos, yaw):
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        eye = pos + np.array([0, 0, self.up]) - self.back * fwd
        tgt = pos + self.ahead * fwd + np.array([0, 0, -2.0])
        f = tgt - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.eye, self.R = eye, np.stack([r, u, f]).astype(np.float32)
        return self

    def project(self, xyz):
        c = (xyz - self.eye.astype(np.float32)) @ self.R.T
        z = c[:, 2]
        keep = z > 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            px = self.w / 2 + self.f * c[:, 0] / z
            py = self.h / 2 - self.f * c[:, 1] / z
        return px, py, z, keep


def rasterize(cam, xyz, rgb, size=2, img=None):
    """Splat points (far to near) into an HxWx3 float image (new or `img`)."""
    w, h = cam.w, cam.h
    if img is None:
        img = np.empty((h, w, 3), np.float32)
        img[:] = BG
    px, py, z, keep = cam.project(xyz)
    keep &= (px >= 0) & (px < w - size) & (py >= 0) & (py < h - size)
    px, py, z, rgb = (px[keep].astype(np.int32), py[keep].astype(np.int32),
                      z[keep], rgb[keep])
    # depth cue: fade far points toward the background
    fade = np.clip((z - 20.0) / 90.0, 0, 0.75)[:, None]
    rgb = rgb * (1 - fade) + np.asarray(BG, np.float32) * fade
    order = np.argsort(-z, kind="stable")
    px, py, rgb = px[order], py[order], rgb[order]
    for dy in range(size):
        for dx in range(size):
            img[py + dy, px + dx] = rgb
    return img


def draw_ego(img, cam, pos, yaw, sensor_h=2.1):
    """Ego vehicle as a white chevron on the ground, pointing forward."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    # the vehicle frame origin is near the rear axle; centre on the body
    for lx, ly in ((3.6, 0.0), (0.2, 1.3), (1.0, 0.0), (0.2, -1.3)):
        pts.append(pos + np.array([c * lx - s * ly, s * lx + c * ly,
                                   -sensor_h + 0.3]))
    px, py, _, keep = cam.project(np.asarray(pts, np.float32))
    if keep.all():
        poly = list(zip(px.tolist(), py.tolist()))
        ImageDraw.Draw(img).polygon(poly, fill=FG, outline=BG)


def colors(labels):
    lab = labels.astype(np.int64)
    out = np.empty((len(lab), 3), np.float32)
    ok = (lab >= 0) & (lab < len(PALETTE))
    out[ok] = PALETTE[lab[ok]]
    out[~ok] = IGNORE_RGB
    return out


# --------------------------------------------------------------------------
# panels


def panel_image(cam, fr, which, pos, yaw, ss, ref=None):
    """Render one panel at supersampling factor ss.

    which: gt / plain (class colours of the GT / the prediction); pred
    (labelled stream: correct points in grey or faded class colours, wrong
    ones larger and on top in ERROR_RGB); diff (unlabelled stream: grey,
    with the points predicted differently from `ref`, the baseline's frame,
    in DIFF_RGB). Returns (image, number of highlighted points)."""
    if which == "diff":
        pred = fr["pred"].astype(np.int64)
        rp = ref["pred"].astype(np.int64)
        m = min(len(pred), len(rp))
        diff = np.zeros(len(pred), bool)
        # ignored points (the short-range LiDARs on the unlabelled stream)
        # are left out, as the error rate leaves out the ignore label
        diff[:m] = (pred[:m] != rp[:m]) & (fr["gt"][:m] >= 0)
        rgb = np.repeat(np.asarray(GRAY_SCENE, np.float32)[None],
                        len(pred), 0)
        img = rasterize(cam, fr["xyz"], rgb, size=ss + 1)
        n = int(diff.sum())
        if n:
            rasterize(cam, fr["xyz"][diff],
                      np.repeat(DIFF_RGB[None], n, 0), size=ss + 2, img=img)
        im = Image.fromarray(img.astype(np.uint8))
        draw_ego(im, cam, pos, yaw)
        return im.resize((cam.w // ss, cam.h // ss), Image.LANCZOS), n
    if which in ("gt", "plain"):
        if which == "gt":
            lab = fr["gt"]
        else:
            # ignored points in grey, as in the GT and the diff panels
            lab = np.where(fr["gt"] >= 0, fr["pred"].astype(np.int64), -1)
        img = rasterize(cam, fr["xyz"], colors(lab), size=ss + 1)
        n_wrong = None
    else:
        gt = fr["gt"].astype(np.int64)
        wrong = (gt >= 0) & (fr["pred"].astype(np.int64) != gt)
        if ERROR_SCENE == "gray":
            rgb = np.repeat(np.asarray(GRAY_SCENE, np.float32)[None],
                            len(gt), 0)
        else:
            rgb = (colors(fr["pred"]) * DIM_SCENE
                   + np.asarray(BG) * (1 - DIM_SCENE))
        img = rasterize(cam, fr["xyz"], rgb, size=ss + 1)
        n_wrong = int(wrong.sum())
        if n_wrong:
            rasterize(cam, fr["xyz"][wrong],
                      np.repeat(ERROR_RGB[None], n_wrong, 0),
                      size=ss + 2, img=img)
    im = Image.fromarray(img.astype(np.uint8))
    draw_ego(im, cam, pos, yaw)
    return im.resize((cam.w // ss, cam.h // ss), Image.LANCZOS), n_wrong


NAMES = {"backbone": "Baseline", "tome": "ToMe", "ours": "SpaceCache"}


def lanes(d, x, y, w, now, arr, sim=None, lat=None, span=4.0, past=0.75):
    """Scrolling timeline over `span` seconds with `now` at `past` of the
    width, marked by a vertical line.

    Top lane: one slot per input frame, from its arrival to the next one
    (shades alternate, frame number inside; frames still to come are pale).
    Bottom lane (methods only): one block per inference, green when it ends
    inside its frame's slot (result within one period of the arrival, queue
    wait included) and red once it runs past the slot."""
    t0 = now - span * past
    t1 = t0 + span

    def X(t):
        return x + (min(max(t, t0), t1) - t0) / span * w

    tick_h, lane_h = 24, 30
    bottom = y + tick_h + (6 + lane_h if sim is not None else 0)
    d.rectangle((x, y, x + w, y + tick_h), fill=LANE_BG)
    for k, ak in enumerate(arr):
        end = arr[k + 1] if k + 1 < len(arr) else ak + PERIOD
        if end < t0 or ak > t1:
            continue
        col = (SLOT_FUTURE if ak > now
               else (SLOT_A if k % 2 == 0 else SLOT_B))
        sx0, sx1 = X(ak), X(end)
        if sx1 - sx0 < 3:  # clipped to a sliver at the window edge
            continue
        d.rectangle((sx0 + 1, y, sx1 - 1, y + tick_h), fill=col)
        num = str(k + 1)
        tw = d.textlength(num, font=font(15, True))
        if sx1 - sx0 > tw + 6:
            d.text(((sx0 + sx1 - tw) / 2, y + 3), num,
                   fill=DIM if ak > now else FG, font=font(15, True))
    if sim is not None:
        ly = y + tick_h + 6
        d.rectangle((x, ly, x + w, ly + lane_h), fill=LANE_BG)
        a, st, dn = sim
        for k in range(len(a)):
            if st[k] > now or dn[k] < t0:
                continue
            # a block in flight turns red the moment its frame is one
            # period old
            col = OK if min(dn[k], now) - a[k] <= PERIOD else BAD
            bx0, bx1 = X(st[k]), X(min(dn[k], now))
            d.rectangle((bx0 + 1, ly + 3, max(bx1 - 1, bx0 + 2),
                         ly + lane_h - 3), fill=col)
    xn = X(now)
    d.line((xn, y - 6, xn, bottom + 4), fill=FG, width=3)
    d.polygon(((xn - 7, y - 14), (xn + 7, y - 14), (xn, y - 5)), fill=FG)


def cum_acc(counts, k):
    """Point accuracy (%) over frames 0..k (labelled points only).

    A running mIoU swings by several points over the first frames of a
    segment as rare classes come and go; the point error rate does not."""
    return float(counts["intersection"][:k + 1].sum()
                 / max(counts["target"][:k + 1].sum(), 1)) * 100


def load_counts(root, run, n_all, idx):
    """Per-frame class intersection / union / target of the first pass."""
    c = np.load(os.path.join(root, "latency", run, "per_frame_counts.npz"))
    return {k: c[k][:n_all][idx] for k in ("intersection", "union", "target")}


def load_segment(args):
    """Dump indices, arrival times and ego track of the chosen segment."""
    ref = os.path.join(args.root, "dump", METHODS["ours"][0])
    meta = json.load(open(os.path.join(ref, "frames.json")))
    idx = segment_frames(meta["frames"], args.segment)[args.start_frame:]
    n = len(idx) if args.max_frames is None else min(args.max_frames,
                                                     len(idx))
    idx = idx[:n]
    arr = arrival_times([meta["frames"][i] for i in idx])
    origin = np.load(os.path.join(ref, f"{idx[0]:05d}.npz"))["pose"][:3, 3]
    poses = [np.load(os.path.join(ref, f"{i:05d}.npz"))["pose"] for i in idx]
    return meta, idx, arr, origin, ego_track(poses, origin)


def compose(args):
    """Columns GT | Baseline | ToMe | SpaceCache (no GT on an unlabelled
    stream), one backbone.

    Under each prediction panel: its delay (finish time minus arrival time
    of the frame on screen) with a timeline around the current time, and on
    a labelled stream its error rate over the frames it has finished."""
    meta, idx, arr, origin, track = load_segment(args)
    n = len(idx)

    frames = {k: Frames(os.path.join(args.root, "dump", v[0]), idx, origin)
              for k, v in METHODS.items()}
    # dump order == eval order of the latency runs (same config)
    n_all = len(meta["frames"])
    lat = {k: load_latency(args.root, v[1], n_all)[idx]
           for k, v in METHODS.items()}
    sim = {k: simulate(v, arr) for k, v in lat.items()}
    gt = PRESET["gt"]
    counts = ({k: load_counts(args.root, v[1], n_all, idx)
               for k, v in METHODS.items()} if gt else None)
    changed = None
    if not gt:
        # per frame: points predicted differently from the baseline, total
        def _load(key, i):
            return np.load(os.path.join(args.root, "dump", METHODS[key][0],
                                        f"{i:05d}.npz"))
        changed = {}
        for key in ("tome", "ours"):
            nd, nt = np.zeros(n), np.zeros(n)
            for j, i in enumerate(idx):
                b = _load("backbone", i)
                p = _load(key, i)["pred"]
                m = min(len(p), len(b["pred"]))
                # ignored points left out (see panel_image)
                valid = b["gt"][:m] >= 0
                nd[j] = ((p[:m] != b["pred"][:m]) & valid).sum()
                nt[j] = valid.sum()
            changed[key] = (nd, nt)

    cols = (("gt",) if gt else ()) + ("backbone", "tome", "ours")
    pw, ph, gap = args.panel_w, args.panel_h, 14
    # --sync: every panel shows the current frame (no queue), for comparing
    # methods at matched latency; the timelines are dropped
    sync = args.sync
    # no legend row when it would be empty (synced unlabelled stream: the
    # disagreement dot sits next to its number)
    head, block_h = 74, (140 if sync else 210)
    foot = 16 if (sync and not gt) else 72
    W = len(cols) * pw + (len(cols) + 1) * gap
    H = head + ph + block_h + foot
    ss = args.ss
    cam = Camera(pw * ss, ph * ss, fov=62, back=22, up=16, ahead=12)
    title_txt = ("SpaceCache vs. ToMe in real time" if args.sync
                 else f"SpaceCache on a {1 / PERIOD:g} Hz LiDAR stream")
    header = [("Backbone", PRESET["backbone"]), ("Data", PRESET["data"])]
    span = 8 * PERIOD  # seconds shown by the timelines

    end_t = (arr[-1] + PERIOD if sync
             else max(s_[2][-1] for s_ in sim.values())) + args.hold
    t_frames = (np.array([args.preview_time])
                if args.preview_time is not None
                else np.arange(0.0, end_t, 1.0 / args.fps))
    def draw(now):
        canvas = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(canvas)
        d.text((gap, 16), title_txt, fill=FG, font=font(36, True))
        bx = gap + d.textlength(title_txt, font=font(36, True)) + 28
        for label, value in header:
            d.line((bx, 20, bx, 58), fill=DIM, width=2)
            d.text((bx + 22, 18), label, fill=FG, font=font(32))
            vx = bx + 22 + d.textlength(label + " ", font=font(32))
            d.text((vx, 16), value, fill=FG, font=font(34, True))
            bx = vx + d.textlength(value, font=font(34, True)) + 28
        if not gt and args.diff_legend == "header":
            # legend of the disagreement points in the title row
            d.line((bx, 20, bx, 58), fill=DIM, width=2)
            d.ellipse((bx + 22, 28, bx + 44, 50),
                      fill=tuple(DIFF_RGB.astype(int)))
            d.text((bx + 56, 18), "disagrees with baseline", fill=FG,
                   font=font(32))
        d.text((W - 190, 20), f"{min(now, arr[-1] + PERIOD):4.1f} s",
               fill=DIM, font=font(32, True))
        for c, key in enumerate(cols):
            x0, y0 = gap + c * (pw + gap), head
            accent = ACCENT if key == "ours" else FG
            if key == "gt" or sync:
                k = int(np.searchsorted(arr, now, side="right")) - 1
                x = float(np.interp(now, arr, np.arange(n)))
                fr = frames["backbone" if key == "gt" else key][k]
            else:
                a, st, dn = sim[key]
                k = int(np.searchsorted(dn, now, side="right")) - 1
                if k >= 0:
                    nxt = dn[k + 1] if k + 1 < n else dn[k] + PERIOD
                    x = k + np.clip((now - dn[k]) / (nxt - dn[k]), 0, 1)
                    fr = frames[key][k]
            if k >= 0:
                pos, yaw = interp_track(track, x)
                cam.look(pos, yaw)
                if key == "gt":
                    mode = "gt"
                elif gt:
                    mode = "pred"
                else:
                    mode = "plain" if key == "backbone" else "diff"
                im, _ = panel_image(
                    cam, fr, mode, pos, yaw, ss,
                    ref=frames["backbone"][k] if mode == "diff" else None)
            else:
                im = Image.new("RGB", (pw, ph), BG)
            canvas.paste(im, (x0, y0))
            if key == "ours":
                d.rectangle((x0 - 6, y0 - 6, x0 + pw + 5, H - foot + 2),
                            outline=ACCENT, width=6)
            # panel header: name left, val mIoU right
            title = "Ground truth" if key == "gt" else NAMES[key]
            d.text((x0 + 14, y0 + 10), title, fill=accent,
                   font=font(30, True))
            if key != "gt":
                txt = f"{METHODS[key][2]:.1f}"
                tw = d.textlength(txt, font=font(30, True))
                # green when within 1 point of the backbone, red below
                low = METHODS[key][2] < METHODS["backbone"][2] - 1.0
                d.text((x0 + pw - 14 - tw, y0 + 10), txt,
                       fill=BAD if low else OK, font=font(30, True))
                lw = d.textlength("mIoU", font=font(24))
                d.text((x0 + pw - 24 - tw - lw, y0 + 15), "mIoU", fill=LABEL,
                       font=font(24))

            # bottom block: "Latency 0.69 s" with the timeline under it, then
            # "Error rate 5.2%" (labelled streams)
            by = y0 + ph + 16
            nx, lw_ = x0 + 14, pw - 28

            def stat(y, label, value, col, dot=None):
                lx_ = nx
                if dot is not None:
                    # the points this number counts, as a legend in place
                    d.ellipse((lx_, y + 17, lx_ + 20, y + 37), fill=dot)
                    lx_ += 32
                d.text((lx_, y + 9), label, fill=LABEL, font=font(30))
                vx = lx_ + d.textlength(label + "  ", font=font(30))
                d.text((vx, y), value, fill=col, font=font(40, True))
                return vx + d.textlength(value, font=font(40, True))

            if key == "gt":
                stat(by, "LiDAR", f"{1 / PERIOD:g} Hz", FG)
                lanes(d, nx, by + 64, lw_, now, arr, span=span)
                continue
            if k < 0:
                continue
            if sync:
                # latency of the frame on screen (no queue), red when it
                # misses the frame period
                ex = stat(by, "Latency", f"{lat[key][k] * 1000:.0f} ms",
                          OK if lat[key][k] <= PERIOD else BAD)
                # the budget: one frame period of the stream
                d.text((ex + 14, by + 9),
                       f"/ {PERIOD * 1000:.0f} ms ({1 / PERIOD:g} Hz)",
                       fill=LABEL, font=font(30))
                ry = by + 64
            else:
                a, st, dn = sim[key]
                delay = dn[k] - a[k]
                stat(by, "Latency", f"{delay:.2f} s",
                     OK if delay <= PERIOD else BAD)
                lanes(d, nx, by + 64, lw_, now, arr, sim[key], lat[key],
                      span=span)
                ry = by + 138
            if not gt:
                if key != "backbone":
                    # share of points predicted differently from the
                    # baseline, over the frames shown so far
                    def _rate(m):
                        nd, nt = changed[m]
                        return nd[:k + 1].sum() / max(nt[:k + 1].sum(), 1) * 100
                    rate = _rate(key)
                    other = _rate("ours" if key == "tome" else "tome")
                    # green for the method that keeps more of the baseline
                    stat(ry, "Prediction disagreement", f"{rate:.1f}%",
                         OK if rate <= other else BAD,
                         dot=(tuple(DIFF_RGB.astype(int))
                              if args.diff_legend == "inline" else None))
                continue
            err = 100 - cum_acc(counts[key], k)
            base_err = 100 - cum_acc(counts["backbone"], k)
            # red when more than 10% more points are wrong than with the
            # backbone on the same frames, green otherwise (as for the mIoU)
            worse = key != "backbone" and err > base_err * 1.10
            col = BAD if worse else OK
            ex = stat(ry, "Error rate", f"{err:.1f}%", col)
            if key != "backbone":
                # change from the backbone in percentage points
                diff = round(err - base_err, 1) + 0.0  # no "-0.0"
                sign = "+" if diff >= 0 else "−"
                d.text((ex + 12, ry + 9), f"({sign}{abs(diff):.1f}%p)",
                       fill=col, font=font(30, True))
        # legend
        ly, lx = H - foot + 14, gap + 4
        f = font(34)
        items = ([] if sync else
                 [("now", FG, 18, "now"),
                  ("slot", None, 64, f"new frame every {PERIOD:g} s"),
                  ("box", OK, 40, "real-time"),
                  ("box", BAD, 40, "lagging")])
        if gt:
            items.append(("dot", tuple(ERROR_RGB.astype(int)), 18,
                          "misclassified point"))
        for kind, col, wdt, txt in items:
            if kind == "now":
                cx = lx + wdt // 2
                d.polygon(((lx, ly + 3), (lx + wdt, ly + 3), (cx, ly + 15)),
                          fill=col)
                d.line((cx, ly + 13, cx, ly + 40), fill=col, width=4)
            elif kind == "dot":
                d.ellipse((lx, ly + 13, lx + wdt, ly + 13 + wdt), fill=col)
            elif kind == "slot":
                half = wdt // 2
                d.rectangle((lx, ly + 6, lx + half - 1, ly + 38), fill=SLOT_A)
                d.rectangle((lx + half + 1, ly + 6, lx + wdt, ly + 38),
                            fill=SLOT_B)
            else:
                d.rectangle((lx, ly + 6, lx + wdt, ly + 38), fill=col)
            lx += wdt + 14
            d.text((lx, ly), txt, fill=FG, font=f)
            lx += d.textlength(txt, font=f) + 64
        return canvas

    return run_frames(draw, t_frames, args.jobs)


CACHED = np.array((245, 160, 80), np.float32)    # served from the cache
COMPUTED = np.array((150, 120, 230), np.float32)  # recomputed this frame


def compose_cache(args):
    """Single panel: which voxels SpaceCache served from its stage-3 cache."""
    _, idx, arr, origin, track = load_segment(args)
    n = len(idx)
    frames = Frames(os.path.join(args.root, "dump", METHODS["ours"][0]), idx,
                    origin)

    pw, ph, head = 1280, 720, 64
    W, H = pw, head + ph
    ss = args.ss
    cam = Camera(pw * ss, ph * ss, fov=55, back=30, up=22, ahead=10)
    out = []
    t_frames = np.arange(0.0, arr[-1] + PERIOD + args.hold, 1.0 / args.fps)
    for ti, now in enumerate(t_frames):
        k = int(np.searchsorted(arr, now, side="right")) - 1
        fr = frames[k]
        hit = fr["hit3"]
        rgb = np.where(hit[:, None], CACHED, COMPUTED)
        pos, yaw = interp_track(track, float(np.interp(now, arr,
                                                       np.arange(n))))
        cam.look(pos, yaw)
        img = rasterize(cam, fr["xyz"], rgb, size=ss + 1)
        im = Image.fromarray(img.astype(np.uint8))
        draw_ego(im, cam, pos, yaw)
        im = im.resize((pw, ph), Image.LANCZOS)
        canvas = Image.new("RGB", (W, H), BG)
        canvas.paste(im, (0, head))
        d = ImageDraw.Draw(canvas)
        d.text((16, 10), "SpaceCache: reuse what was already seen",
               fill=FG, font=font(24, True))
        d.text((16, 42), "Stage-3 tokens (0.4 m regions) of each frame, "
               "Waymo val, Utonia", fill=DIM, font=font(13))
        lx = W - 470
        d.rectangle((lx, 18, lx + 16, 34), fill=tuple(CACHED.astype(int)))
        d.text((lx + 22, 16), "cached (blocks skipped)", fill=FG,
               font=font(15))
        d.rectangle((lx + 230, 18, lx + 246, 34),
                    fill=tuple(COMPUTED.astype(int)))
        d.text((lx + 252, 16), "computed", fill=FG, font=font(15))
        rate = float(hit.mean()) * 100
        d.text((lx, 40), f"frame {k + 1}/{n} · cached {rate:4.1f}% of "
               "input voxels", fill=DIM, font=font(14))
        out.append(canvas)
        if ti % 20 == 0:
            print(f"{ti}/{len(t_frames)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True,
                    help="substring of the segment directory name")
    ap.add_argument("--out", default=None,
                    help="animated .gif / .webp (optional with --png_dir)")
    ap.add_argument("--preset", default="waymo_utonia", choices=PRESETS)
    ap.add_argument("--root", default=None,
                    help="directory holding dump/ and latency/ "
                         "(default: the preset's)")
    ap.add_argument("--mode", default="stream", choices=("stream", "cache"),
                    help="stream = side-by-side wall-clock comparison; cache = "
                         "single panel of cache hits / misses")
    ap.add_argument("--sync", action="store_true",
                    help="stream: all panels show the current frame (no "
                         "queueing), with each frame's latency in ms")
    ap.add_argument("--start_frame", type=int, default=0,
                    help="skip the first N frames of the segment; the clip, "
                         "the queue and the running error rate start at N "
                         "(predictions keep the cache state of the full run)")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--hold", type=float, default=1.5)
    ap.add_argument("--panel_w", type=int, default=None,
                    help="default: the preset's")
    ap.add_argument("--panel_h", type=int, default=None)
    ap.add_argument("--ss", type=int, default=2, help="supersampling")
    ap.add_argument("--jobs", type=int,
                    default=max(1, (os.cpu_count() or 2) // 2),
                    help="processes rendering frames in parallel (stream)")
    ap.add_argument("--theme", default="light", choices=THEMES,
                    help="light: white background, dark text; dark: the "
                         "earlier dark background")
    ap.add_argument("--error_style", default="gray", choices=ERROR_STYLES,
                    help="how misclassified points are drawn (see "
                         "ERROR_STYLES)")
    ap.add_argument("--diff_legend", default="inline",
                    choices=("inline", "header"),
                    help="unlabelled stream: legend of the disagreement "
                         "points next to each rate (inline) or in the "
                         "title row (header)")
    ap.add_argument("--preview_time", type=float, default=None,
                    help="stream: render only the frame at this time (s), "
                         "to try styles quickly")
    ap.add_argument("--font", default=None,
                    help="regular .ttf/.otf (e.g. Aptos Narrow); "
                         "default DejaVu Sans")
    ap.add_argument("--font_bold", default=None,
                    help="bold .ttf/.otf (default: --font)")
    ap.add_argument("--quality", type=int, default=75,
                    help="WebP quality when --out ends with .webp")
    ap.add_argument("--gif_width", type=int, default=0,
                    help="downscale the GIF to this width (0: keep)")
    ap.add_argument("--png_dir", default=None,
                    help="write every frame as PNG here (for mp4)")
    args = ap.parse_args()
    global PRESET, ROOT, METHODS, PERIOD
    PRESET = PRESETS[args.preset]
    ROOT, METHODS, PERIOD = PRESET["root"], PRESET["methods"], PRESET["period"]
    args.root = args.root or ROOT
    args.panel_w = args.panel_w or PRESET["panel_w"]
    args.panel_h = args.panel_h or PRESET["panel_h"]
    global FONT, FONT_B
    FONT = args.font or FONT
    FONT_B = args.font_bold or args.font or FONT_B
    apply_theme(args.theme, args.error_style)

    if not (args.out or args.png_dir):
        raise SystemExit("give --out and/or --png_dir")
    frames = compose(args) if args.mode == "stream" else compose_cache(args)
    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)
        for i, im in enumerate(frames):
            im.save(os.path.join(args.png_dir, f"{i:05d}.png"),
                    compress_level=1)
        print("wrote", args.png_dir, len(frames), "frames")
    if not args.out:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.gif_width:
        h = round(frames[0].height * args.gif_width / frames[0].width)
        frames = [im.resize((args.gif_width, h), Image.LANCZOS)
                  for im in frames]
    save_kw = dict(save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0)
    if args.out.lower().endswith(".webp"):
        # animated WebP: a fraction of the GIF size, shown by GitHub
        frames[0].save(args.out, quality=args.quality, method=4, **save_kw)
    else:
        frames[0].save(args.out, optimize=False, **save_kw)
    print("wrote", args.out, len(frames), "frames")


if __name__ == "__main__":
    main()
