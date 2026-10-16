"""Stage-wise temporal overlap of encoder tokens (Fig. 2b).

For consecutive frames (A, B) of a sequence and each encoder stage s, the
fraction of B's stage-s tokens whose world cell (base_voxel * 2^s) is also
occupied by A's stage-s tokens. Token sets follow the GridPooling rule
(cluster by grid_coord // stride, mean of member coordinates), so no model
is loaded.

    PYTHONPATH=. python \\
        spacecache/figure/motivation/temporal_overlap/stage_overlap.py \\
        --config spacecache/configs/waymo/utonia_dec/stage_overlap.py \\
        --name waymo --num_workers 8

Writes `<save_path>/<name>.json` (per-stage aggregate and per-pair records)
for `plot_stage_overlap.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _ROOT)

from pointcept.datasets import build_dataset  # noqa: E402
from pointcept.utils.config import Config  # noqa: E402

from spacecache.common.helpers import (  # noqa: E402
    extract_grid_size, extract_random_scale, match_b_to_cells, set_seed,
    to_global, voxel_hash,
)


log = logging.getLogger("stage_overlap")

# grid_coord is packed into one int64 key (20 bits per axis) instead of
# torch.unique(dim=0), which sorts rows and is much slower. `_pack` checks
# the range, since a wraparound would silently merge distant cells.
_PACK_BITS = 20


def _pack(gc):
    assert int(gc.min()) >= 0 and int(gc.max()) < (1 << _PACK_BITS), (
        f"grid_coord out of the {_PACK_BITS}-bit packing range "
        f"[{int(gc.min())}, {int(gc.max())}]"
    )
    return (gc[:, 0] << (2 * _PACK_BITS)) | (gc[:, 1] << _PACK_BITS) | gc[:, 2]


def _mean_by_cluster(x, inverse, n_clusters):
    """Per-cluster mean of `x`, as GridPooling's `segment_csr(..., "mean")`."""
    out = torch.zeros(n_clusters, x.shape[1], dtype=x.dtype, device=x.device)
    out.index_add_(0, inverse, x)
    cnt = torch.zeros(n_clusters, dtype=x.dtype, device=x.device)
    cnt.index_add_(0, inverse, torch.ones_like(inverse, dtype=x.dtype))
    return out / cnt.clamp(min=1).unsqueeze(1)


def pooling_pyramid(coord, grid_coord, strides):
    """Token coordinates at every encoder stage, without the model.

    Returns `len(strides) + 1` levels: level 0 is the input voxel set and
    level d is what `enc{d}`'s GridPooling produces (cluster by
    `grid_coord // stride` in the frame-local lattice, coordinate = mean of
    the members, applied level by level). `coord` (N, 3) float64 and
    `grid_coord` (N, 3) long must be on the same device.
    """
    levels = [coord]
    gc, cur = grid_coord, coord
    for stride in strides:
        gc = torch.div(gc, stride, rounding_mode="trunc")
        uniq, inverse = torch.unique(_pack(gc), return_inverse=True)
        cur = _mean_by_cluster(cur, inverse, int(uniq.numel()))
        # all rows of a cluster share the divided grid_coord, so a scatter
        # yields the cluster's grid_coord without unpacking the key
        gc_next = torch.zeros(int(uniq.numel()), 3, dtype=gc.dtype,
                              device=gc.device)
        gc_next[inverse] = gc
        gc = gc_next
        levels.append(cur)
    return levels


# ---------------------------------------------------------------------------
#  Sequences and pairs
# ---------------------------------------------------------------------------


def segment_ids(dataset):
    """Per-frame id of the continuous sequence the frame belongs to.

    Pairs are only formed within one sequence. Uses, in order: the
    dataset's `sequence_index` (Waymo; nuScenes with scene ids),
    timestamp-contiguous runs for info dicts with a `timestamp` (a gap
    above 4x the median gap starts a new run), and the frame's directory
    for path-list datasets (e.g. ScanNet video).
    """
    seq_idx = getattr(dataset, "sequence_index", None)
    if seq_idx is not None:
        return [int(s) for s in seq_idx]
    data_list = getattr(dataset, "data_list", None)
    if not data_list:
        return [0] * len(dataset)
    if isinstance(data_list[0], dict) and "timestamp" in data_list[0]:
        ts = [float(d["timestamp"]) for d in data_list]
        diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        pos = sorted(d for d in diffs if d > 0)
        med = pos[len(pos) // 2] if pos else 1.0
        thr = 4.0 * med                     # nuScenes: 4 * 0.5 s = 2.0 s
        seg, cur = [0], 0
        for d in diffs:
            if d <= 0 or d > thr:
                cur += 1
            seg.append(cur)
        return seg
    return [os.path.dirname(str(d)) for d in data_list]


def frame_times(dataset):
    """Per-frame capture time in seconds, or None if unavailable.

    Used only to report the measured frame gap. Sources: `timestamp` of the
    nuScenes info dicts (s), the Waymo frame directory name (us) and the
    SemanticKITTI `times.txt` (s).
    """
    data_list = getattr(dataset, "data_list", None)
    if not data_list:
        return None
    if isinstance(data_list[0], dict):
        if "timestamp" not in data_list[0]:
            return None
        return [float(d["timestamp"]) for d in data_list]

    out, times_cache = [], {}
    for path in data_list:
        path = str(path)
        stem = os.path.basename(path.rstrip("/")).split(".")[0]
        if stem.isdigit() and len(stem) >= 13:        # Waymo: microseconds
            out.append(int(stem) * 1e-6)
            continue
        seq_dir = os.path.dirname(os.path.dirname(path))  # .../<seq>/velodyne
        times_path = os.path.join(seq_dir, "times.txt")
        if times_path not in times_cache:
            if not os.path.exists(times_path):
                return None
            with open(times_path) as f:
                times_cache[times_path] = [float(x) for x in f if x.strip()]
        times = times_cache[times_path]
        if not stem.isdigit() or int(stem) >= len(times):
            return None
        out.append(times[int(stem)])
    return out


def build_pairs(dataset, stride, max_pairs_per_scene=None, max_pairs=None):
    """(idx_a, idx_b, segment) for every sampled pair.

    B is `stride` frames after A in the same sequence. Under a cap, anchors
    are spread evenly over each sequence (and pairs over the split) rather
    than taken from the start.
    """
    segs = segment_ids(dataset)
    by_seg = defaultdict(list)
    for idx, seg in enumerate(segs):
        by_seg[seg].append(idx)

    pairs = []
    for seg in sorted(by_seg, key=str):
        frames = by_seg[seg]
        pos = {idx: p for p, idx in enumerate(frames)}
        anchors = [i for i in frames if pos[i] + stride < len(frames)]
        if not anchors:
            continue
        if (max_pairs_per_scene is not None
                and len(anchors) > max_pairs_per_scene):
            take = np.linspace(0, len(anchors) - 1, max_pairs_per_scene)
            anchors = [anchors[int(round(t))] for t in take]
        for a in anchors:
            pairs.append((a, frames[pos[a] + stride], seg))
    pairs.sort(key=lambda p: p[0])
    if max_pairs is not None and len(pairs) > max_pairs:
        take = np.linspace(0, len(pairs) - 1, max_pairs)
        pairs = [pairs[int(round(t))] for t in take]
    return pairs


# ---------------------------------------------------------------------------
#  Pose
# ---------------------------------------------------------------------------


class PoseSource:
    """lidar -> world pose (4x4, metres) for a frame index.

    Uses the `pose` collected by the pipeline when present. SemanticKITTI
    poses are read from `<sequence>/poses.txt` and moved to the lidar frame
    with the calibration `Tr` (`SemanticKITTIImagePointDataset.get_pose`).
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._seq_cache = {}

    def __call__(self, idx, data):
        pose = data.get("pose")
        if pose is not None:
            if not torch.is_tensor(pose):
                pose = torch.as_tensor(np.asarray(pose))
            return pose.to(torch.float64).reshape(4, 4)
        return self._kitti_pose(idx)

    def _kitti_pose(self, idx):
        from pointcept.datasets.semantic_kitti import (
            SemanticKITTIImagePointDataset as K,
        )
        path = str(self.dataset.data_list[idx])
        seq_dir = os.path.dirname(os.path.dirname(path))
        frame = int(os.path.basename(path).split(".")[0])
        if seq_dir not in self._seq_cache:
            calib = K.parse_calib_file(os.path.join(seq_dir, "calib.txt"))
            self._seq_cache[seq_dir] = (
                os.path.join(seq_dir, "poses.txt"), calib["Tr"],
            )
        poses_path, tr = self._seq_cache[seq_dir]
        return torch.as_tensor(K.get_pose(poses_path, frame, tr),
                               dtype=torch.float64)


# ---------------------------------------------------------------------------
#  Per-frame / per-pair work
# ---------------------------------------------------------------------------


class _FrameSet(Dataset):
    """The frames a run needs, in index order, one per __getitem__.

    Each frame is seeded by its index, so the points drawn by
    `GridSample(mode="train")` do not depend on `--num_workers`.
    """

    def __init__(self, base, indices, seed):
        self.base, self.indices, self.seed = base, indices, seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        set_seed(self.seed + idx)
        return idx, self.base[idx]


def frame_stream(dataset, indices, seed, num_workers):
    frames = _FrameSet(dataset, indices, seed)
    if num_workers <= 0:
        for i in range(len(frames)):
            yield frames[i]
        return
    loader = DataLoader(
        frames, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
    )
    for item in loader:
        yield item


def lattice_grid(coord, data, pose, coord_scale, base_voxel, strides,
                 lattice):
    """Integer lattice the pooling clusters on, per `lattice` mode.

    All modes share the cell size and differ only in where the cell walls
    are anchored:

      ego          `grid_coord` as built by the val pipeline.
      world        coordinates mapped to the world frame (R c + t).
      world_trans  ego axes, shifted by the world translation expressed on
                   the ego axes (c + R^T t).
      world_yaw    rotated by the yaw of R only (ego up axis), plus t.

    Returns (grid_coord, recluster); `recluster` is False only for `ego`.
    """
    if lattice == "ego":
        return (data["grid_coord"].to(device=coord.device, dtype=torch.long),
                False)

    rot = pose[:3, :3]
    trans = pose[:3, 3] * coord_scale
    if lattice == "world":
        ref = coord @ rot.T + trans
    elif lattice == "world_yaw":
        yaw = torch.atan2(rot[1, 0], rot[0, 0])
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        rot_z = torch.eye(3, dtype=coord.dtype, device=coord.device)
        rot_z[0, 0], rot_z[0, 1] = cos, -sin
        rot_z[1, 0], rot_z[1, 1] = sin, cos
        ref = coord @ rot_z.T + trans
    elif lattice == "world_trans":
        # R^T t: the world offset on this frame's axes (shift, no rotation)
        ref = coord + trans @ rot
    else:
        raise ValueError(f"unknown lattice {lattice}")

    gc = torch.floor(ref / base_voxel).to(torch.long)
    snap = 1
    for st in strides:
        snap *= int(st)
    # shift by whole deepest-level cells: the partition is unchanged and the
    # key stays within _pack's range
    gc = gc - torch.div(gc.min(dim=0).values, snap,
                        rounding_mode="floor") * snap
    return gc, True


def frame_stages(data, pose, coord_scale, strides, base_voxel, device,
                 lattice="ego"):
    """Per-stage token coordinates of one frame, in the world frame.

    Pooling averages ego coordinates and maps the result to the world at
    the end, which is exact because `to_global` is affine; `lattice` only
    selects the partition. For a non-ego lattice, level 0 re-clusters the
    points GridSample already drew on the ego lattice.
    """
    coord = data["coord"].to(device=device, dtype=torch.float64)
    pose = pose.to(device)
    gc, recluster = lattice_grid(coord, data, pose, coord_scale, base_voxel,
                                 strides, lattice)
    if recluster:
        # a leading stride of 1 re-clusters level 0 on the new lattice;
        # entry 0 (the GridSample points) is not a stage
        levels = pooling_pyramid(coord, gc, [1] + list(strides))[1:]
    else:
        levels = pooling_pyramid(coord, gc, strides)
    return [to_global(c, pose, scale=coord_scale) for c in levels]


def pair_overlap(stages_a, stages_b, cell_sizes):
    """Per-stage overlap of B against A, in the world frame."""
    out = []
    for depth, (ca, cb, cell) in enumerate(
            zip(stages_a, stages_b, cell_sizes)):
        cells_a = torch.unique(voxel_hash(ca, cell))
        matched, _ = match_b_to_cells(cb, cells_a, cell)
        n_b = int(matched.numel())
        n_matched = int(matched.sum().item())
        out.append(dict(
            depth=depth,
            n_a=int(ca.shape[0]),
            n_b=n_b,
            n_matched=n_matched,
            overlap=n_matched / max(n_b, 1),
        ))
    return out


# ---------------------------------------------------------------------------
#  Aggregation + IO
# ---------------------------------------------------------------------------


def _is_identity(pose, tol=1e-6):
    """Whether `pose` is the identity (already applied upstream)."""
    return bool(torch.allclose(pose.to(torch.float64),
                               torch.eye(4, dtype=torch.float64,
                                         device=pose.device), atol=tol))


def _quantiles(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return None
    return dict(
        mean=float(a.mean()), std=float(a.std()),
        p25=float(np.percentile(a, 25)), p50=float(np.percentile(a, 50)),
        p75=float(np.percentile(a, 75)),
        min=float(a.min()), max=float(a.max()),
    )


def aggregate(records, cell_sizes_m, match_cell_sizes_m=None):
    """Per-stage aggregate over the pair records.

    `overlap` holds quantiles of the per-pair ratio (one vote per pair);
    `overlap_weighted` pools tokens over all pairs (sum matched / sum
    tokens).
    """
    stages = []
    match_cell_sizes_m = match_cell_sizes_m or cell_sizes_m
    for depth, cell_m in enumerate(cell_sizes_m):
        per_pair = [r["per_depth"][depth] for r in records]
        n_matched = sum(d["n_matched"] for d in per_pair)
        n_b = sum(d["n_b"] for d in per_pair)
        stages.append(dict(
            depth=depth,
            cell_m=cell_m,
            match_cell_m=match_cell_sizes_m[depth],
            n_pairs=len(per_pair),
            overlap=_quantiles([d["overlap"] for d in per_pair]),
            overlap_weighted=n_matched / max(n_b, 1),
            tokens_a_mean=float(np.mean([d["n_a"] for d in per_pair]))
            if per_pair else 0.0,
            tokens_b_mean=float(np.mean([d["n_b"] for d in per_pair]))
            if per_pair else 0.0,
        ))
    return stages


def stage_geometry(cfg, base_voxel, n_stages=None):
    """(strides, cumulative cell size per stage) from the backbone config.

    `enc0` runs at the input resolution and stages 1..S-1 start with a
    GridPooling, so stage s has cell `base_voxel * prod(stride[:s])`.
    """
    backbone = cfg.get("model", {}).get("backbone", {})
    strides = [int(s) for s in backbone.get("stride", (2, 2, 2, 2))]
    depths = backbone.get("enc_depths", None)
    if depths is not None:
        n_stages = n_stages or len(depths)
    n_stages = n_stages or len(strides) + 1
    strides = strides[: n_stages - 1]
    cells, cur = [base_voxel], base_voxel
    for s in strides:
        cur *= s
        cells.append(cur)
    return strides, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Dataset config, e.g. "
                         "spacecache/configs/waymo/utonia_dec/stage_overlap.py.")
    ap.add_argument("--stride", type=int, default=None,
                    help="Frames between A and B (default: cfg.stride).")
    ap.add_argument("--max_pairs_per_scene", type=int, default=None,
                    help="Pairs per sequence (default: "
                         "cfg.max_pairs_per_scene, None = all).")
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="Total pairs (default: cfg.max_pairs, None = all).")
    ap.add_argument("--n_stages", type=int, default=None,
                    help="Number of encoder stages (default: "
                         "len(backbone.enc_depths)).")
    ap.add_argument("--out", default=None, help="Override cfg.save_path.")
    ap.add_argument("--name", default=None,
                    help="Output basename (default: the config's stem).")
    ap.add_argument("--mf", type=float, default=1.0,
                    help="Match factor: the matching cell is "
                         "base_voxel * 2^s * mf.")
    ap.add_argument("--lattice", default="ego",
                    choices=("ego", "world", "world_trans", "world_yaw"),
                    help="Lattice the pooling clusters on (see "
                         "lattice_grid); ego uses the pipeline's "
                         "grid_coord.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num_workers", type=int, default=0,
                    help="Data loader workers; results do not depend on "
                         "the count.")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    cfg = Config.fromfile(args.config)
    requested = str(args.device or cfg.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.info("cuda unavailable — falling back to cpu")
        requested = "cpu"
    device = torch.device(requested)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 37))
    stride = (args.stride if args.stride is not None
              else int(cfg.get("stride", 1)))
    max_per_scene = (args.max_pairs_per_scene
                     if args.max_pairs_per_scene is not None
                     else cfg.get("max_pairs_per_scene", None))
    max_pairs = (args.max_pairs if args.max_pairs is not None
                 else cfg.get("max_pairs", None))
    out_dir = args.out or cfg.get("save_path",
                                  "output/motivation/stage_overlap")
    name = args.name or os.path.splitext(os.path.basename(args.config))[0]
    if abs(args.mf - 1.0) > 1e-9:
        name = f"{name}_mf{args.mf:g}".replace(".", "p")
    if args.lattice != "ego":
        # separate name and subdirectory, so the plot's default glob over
        # save_path only picks up ego runs
        name = f"{name}_{args.lattice}"
        if args.out is None:
            out_dir = os.path.join(out_dir, f"{args.lattice}_lattice")
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    dataset = build_dataset(cfg.data.val)
    base_voxel = extract_grid_size(cfg.data.val.transform)
    coord_scale = extract_random_scale(cfg.data.val.transform)
    strides, cells = stage_geometry(cfg, base_voxel, args.n_stages)
    cells_m = [c / coord_scale for c in cells]
    # pooling uses the stage's own cell, matching uses that cell times mf;
    # the two lattices coincide only at mf = 1
    match_cells = [c * float(args.mf) for c in cells]
    match_cells_m = [c / coord_scale for c in match_cells]

    log.info("dataset %s: %d frames | base_voxel=%g (%.3f m physical), "
             "coord_scale=%g | lattice=%s", cfg.data.val.type, len(dataset),
             base_voxel, base_voxel / coord_scale, coord_scale, args.lattice)
    log.info("stages: %s", ", ".join(
        f"S{d} cell={m:.3f} m" for d, m in enumerate(cells_m)))
    if abs(args.mf - 1.0) > 1e-9:
        log.info("match_factor=%g -> matching cells: %s", args.mf, ", ".join(
            f"S{d} {m:.3f} m" for d, m in enumerate(match_cells_m)))

    pairs = build_pairs(dataset, stride, max_per_scene, max_pairs)
    n_segments = len({p[2] for p in pairs})
    times = frame_times(dataset)
    log.info("pairs: %d over %d drives (stride=%d frames)",
             len(pairs), n_segments, stride)
    if not pairs:
        raise SystemExit("no pairs — check stride / split / scene_list")

    pose_of = PoseSource(dataset)
    pairs_of = defaultdict(list)
    for p, (ia, ib, _) in enumerate(pairs):
        pairs_of[ia].append(p)
        pairs_of[ib].append(p)
    remaining = Counter()
    for ia, ib, _ in pairs:
        remaining[ia] += 1
        remaining[ib] += 1
    needed = sorted(remaining)

    # each frame is loaded once and freed after its last pair
    store, records, done = {}, [], set()
    for idx, data in frame_stream(dataset, needed, seed, args.num_workers):
        pose = pose_of(idx, data)
        store[idx] = dict(
            stages=frame_stages(data, pose, coord_scale, strides,
                                base_voxel, device, args.lattice),
            # ApplyPose (indoor video) moves coord to the world frame and
            # sets pose to the identity, so ego motion is unknown there
            translation=(None if _is_identity(pose)
                         else pose[:3, 3].to(torch.float64).cpu().numpy()),
        )
        for p in pairs_of[idx]:
            ia, ib, seg = pairs[p]
            if p in done or ia not in store or ib not in store:
                continue
            per_depth = pair_overlap(store[ia]["stages"],
                                     store[ib]["stages"], match_cells)
            t_a, t_b = store[ia]["translation"], store[ib]["translation"]
            ego = (None if t_a is None or t_b is None
                   else float(np.linalg.norm(t_b - t_a)))
            records.append(dict(
                idx_a=int(ia), idx_b=int(ib), segment=str(seg),
                ego_m=ego,
                dt_s=(float(times[ib] - times[ia]) if times else None),
                per_depth=per_depth,
            ))
            done.add(p)
            for j in (ia, ib):
                remaining[j] -= 1
                if remaining[j] <= 0:
                    store.pop(j, None)
            if len(records) % 25 == 0 or len(records) == len(pairs):
                log.info("pairs %d/%d | S0 %.1f%% -> S%d %.1f%%",
                         len(records), len(pairs),
                         100 * per_depth[0]["overlap"], len(cells) - 1,
                         100 * per_depth[-1]["overlap"])

    stages = aggregate(records, cells_m, match_cells_m)
    gaps = [r["dt_s"] for r in records if r["dt_s"] is not None]
    result = dict(
        config=args.config,
        dataset=dict(
            type=cfg.data.val.type,
            label=(cfg.get("figure_label", name) if args.lattice == "ego"
                   else f"{cfg.get('figure_label', name)} "
                        f"({args.lattice} lattice)"),
            split=str(cfg.data.val.get("split", "val")),
            n_frames=len(dataset),
            n_drives_sampled=n_segments,
        ),
        sampling=dict(
            stride=stride,
            frame_gap_s=cfg.get("frame_gap_s", None),
            measured_gap_s_median=(float(np.median(gaps)) if gaps else None),
            n_pairs=len(records),
            max_pairs_per_scene=max_per_scene,
            max_pairs=max_pairs,
            seed=seed,
        ),
        grid=dict(
            base_voxel=base_voxel, coord_scale=coord_scale,
            base_voxel_m=base_voxel / coord_scale,
            strides=strides, cell_m=cells_m, lattice=args.lattice,
            match_factor=float(args.mf), match_cell_m=match_cells_m,
        ),
        ego=_quantiles([r["ego_m"] for r in records
                        if r["ego_m"] is not None]) or None,
        stages=stages,
        pairs=records,
    )
    out_path = os.path.join(out_dir, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    log.info("=" * 62)
    log.info("%-6s %10s %10s %10s %12s", "stage", "cell (m)", "overlap",
             "weighted", "tokens/frame")
    for s in stages:
        log.info("S%-5d %10.3f %9.1f%% %9.1f%% %12.0f", s["depth"],
                 s["cell_m"], 100 * s["overlap"]["mean"],
                 100 * s["overlap_weighted"], s["tokens_b_mean"])
    if result["ego"] is not None:
        log.info("sensor displacement per pair: median %.2f m "
                 "(p25 %.2f, p75 %.2f)", result["ego"]["p50"],
                 result["ego"]["p25"], result["ego"]["p75"])
    else:
        log.info("sensor displacement: n/a (the pose was consumed upstream, "
                 "e.g. ApplyPose on an indoor video pipeline)")
    if gaps:
        log.info("measured frame gap: median %.3f s", np.median(gaps))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
