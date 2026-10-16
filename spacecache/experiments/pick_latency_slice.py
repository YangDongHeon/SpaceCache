"""Pick the fixed cost-measurement slice (latency / FLOPs / memory runs).

The slice is made of whole sequences (SpaceCache resets its cache at every
sequence boundary), with their number K pinned so the cold-start fraction
K / n_frames matches the full split's within `--seed_tol`. The sequences are
chosen so the per-frame voxel-count distribution after the val transform
(computed on CPU, without the model) matches the full split in 1-Wasserstein
distance: greedy selection, swap repair, swap refinement. `--subset_log`
also matches the SpaceCache recompute ratio. The per-frame counts are cached,
so re-selecting with another `--target_frames` is fast.

    # Waymo
    PYTHONPATH=. python spacecache/experiments/pick_latency_slice.py \\
        --config spacecache/configs/waymo/utonia_dec.py \\
        --target_frames 300 --num_workers 8

    # nuScenes: sequences are scenes if the info pkl has scene ids
    # (pointcept/datasets/preprocessing/nuscenes/add_scene_id.py), else
    # timestamp-contiguous runs
    PYTHONPATH=. python spacecache/experiments/pick_latency_slice.py \\
        --config spacecache/configs/nuscenes/utonia_dec.py \\
        --target_frames 300 --num_workers 16

Writes `<out_dir>/<name>_cost_slice.json` with the chosen sequence keys, the
global frame indices and, when the dataset supports it, the `scene_list` to
paste into the latency-slice config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
))

from pointcept.datasets import build_dataset  # noqa: E402
import pointcept.datasets.waymo  # noqa: F401,E402
from pointcept.utils.config import Config  # noqa: E402

# sequence ids come from the SpaceCache policy itself, so the slice uses
# exactly its cache-reset unit
from spacecache.policies.spacecache import SpaceCachePolicy  # noqa: E402


# --- per-frame voxel counts -----------------------------------------------


class _CountOnly(torch.utils.data.Dataset):
    """Val dataset wrapper whose workers return only (idx, n_voxels, class
    histogram), so the transformed point clouds never cross the worker
    boundary."""

    def __init__(self, base, num_classes):
        self.base = base
        self.num_classes = int(num_classes)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = self.base[i]
        n = int(np.asarray(d["coord"]).shape[0])
        hist = np.zeros(self.num_classes, dtype=np.int64)
        seg = d.get("segment")
        if seg is not None:
            s = np.asarray(seg).reshape(-1)
            s = s[(s >= 0) & (s < self.num_classes)]
            if s.size:
                hist = np.bincount(s, minlength=self.num_classes)[
                    : self.num_classes]
        return int(i), n, hist


def collect_counts(dataset, num_classes, num_workers):
    """Per-frame voxel count + class histogram over the whole split."""
    loader = torch.utils.data.DataLoader(
        _CountOnly(dataset, num_classes), batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=lambda b: b[0])
    n_total = len(dataset)
    counts = np.zeros(n_total, dtype=np.int64)
    hists = np.zeros((n_total, int(num_classes)), dtype=np.int64)
    for done, (i, n, hist) in enumerate(loader, 1):
        counts[i] = n
        hists[i] = hist
        if done % 250 == 0 or done == n_total:
            print(f"  counted {done}/{n_total} frames", flush=True)
    return counts, hists


# --- distribution matching -------------------------------------------------


def w1(a, b):
    """1-Wasserstein distance between two 1-D empirical samples.

    W1 = integral |F_a(x) - F_b(x)| dx, evaluated exactly on the merged
    support. Latency depends on the whole frame-size distribution (the tail
    costs the most), not only on its mean and std."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float("inf")
    grid = np.concatenate([a, b])
    grid.sort()
    step = np.diff(grid)
    ca = np.searchsorted(a, grid[:-1], side="right") / a.size
    cb = np.searchsorted(b, grid[:-1], side="right") / b.size
    return float(np.sum(np.abs(ca - cb) * step))


def greedy_select(counts, seg_of_frame, target_frames, n_segments=None,
                  window=None):
    """Greedy forward selection of whole segments minimising W1 against the
    full-split frame-size distribution.

    `n_segments` pins the segment count K and `window` is the (lo, hi) frame
    range the slice must land in. The cache resets once per segment, so the
    slice's seed fraction is K / n_frames; main derives the window from the
    target seed fraction. Segment lengths vary, so without both constraints
    the greedy drifts towards short or long segments and biases the seed
    fraction, which only affects the streaming policies.

    Returns (chosen segment keys, {key: per-frame counts}); chosen is None
    if the constrained search dead-ends. `n_segments=None` adds segments
    until the frame target is reached, without seed-fraction control."""
    seg_keys = sorted(set(seg_of_frame))
    seg_arr = np.asarray(seg_of_frame)
    per_seg = {k: counts[seg_arr == k] for k in seg_keys}
    lens = {k: per_seg[k].size for k in seg_keys}

    if n_segments is None:
        chosen, cur = [], np.array([], dtype=np.float64)
        while cur.size < target_frames:
            best_k, best_s, best_cur = None, None, None
            for k in seg_keys:
                if k in chosen:
                    continue
                cand = np.concatenate([cur, per_seg[k]])
                s = w1(cand, counts)
                if best_s is None or s < best_s:
                    best_k, best_s, best_cur = k, s, cand
            if best_k is None:
                break
            chosen.append(best_k)
            cur = best_cur
        return chosen, per_seg

    lo, hi = window
    k_seg = int(n_segments)
    chosen, cur = [], np.array([], dtype=np.float64)
    for step in range(k_seg):
        left = k_seg - step - 1                # picks remaining after this one
        avail = [k for k in seg_keys if k not in chosen]
        best_k, best_s, best_cur = None, None, None
        for k in avail:
            n_after = cur.size + lens[k]
            # look-ahead over the segments still available: `left` more
            # picks add between the sum of the `left` shortest and the sum of
            # the `left` longest lengths
            if left:
                pool = sorted(lens[j] for j in avail if j != k)
                if len(pool) < left:
                    continue
                if n_after + sum(pool[-left:]) < lo:
                    continue
                if n_after + sum(pool[:left]) > hi:
                    continue
            elif not (lo <= n_after <= hi):
                continue
            cand = np.concatenate([cur, per_seg[k]])
            s = w1(cand, counts)
            if best_s is None or s < best_s:
                best_k, best_s, best_cur = k, s, cand
        if best_k is None:
            # the look-ahead is necessary but not sufficient (achievable
            # sums are sparse); the caller falls back to greedy_fixed_k and
            # swap repair
            return None, per_seg
        chosen.append(best_k)
        cur = best_cur
    return chosen, per_seg


def greedy_fixed_k(counts, per_seg, k_seg):
    """Pick exactly `k_seg` segments by W1 alone, ignoring the frame window.
    Feeds `repair` when the constrained greedy dead-ends."""
    chosen, cur = [], np.array([], dtype=np.float64)
    for _ in range(int(k_seg)):
        best_k, best_s, best_cur = None, None, None
        for k in per_seg:
            if k in chosen:
                continue
            cand = np.concatenate([cur, per_seg[k]])
            s = w1(cand, counts)
            if best_s is None or s < best_s:
                best_k, best_s, best_cur = k, s, cand
        chosen.append(best_k)
        cur = best_cur
    return chosen


def load_subset_ratios(path):
    """Per-segment recompute statistics of a full-val SpaceCache run.

    Reads its `spacecache_per_frame.jsonl` and returns ({segment: (sum
    subset_size, sum n_d3)}, overall subset ratio) over non-seed frames. The
    recompute ratio depends on ego motion rather than frame size, so it is
    matched separately. Seed frames (subset 1.0 by construction) are
    excluded, as the seed fraction is controlled by the segment count."""
    per, tot_s, tot_n = {}, 0.0, 0.0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("is_seed"):
                continue
            k = r["segment"]
            s, n = float(r["subset_size"]), float(r["n_d3"])
            a, b = per.get(k, (0.0, 0.0))
            per[k] = (a + s, b + n)
            tot_s += s
            tot_n += n
    return per, (tot_s / tot_n if tot_n else None)


def make_checks(per_seg, window, sub_per_seg=None, sub_target=None,
                sub_tol=0.02):
    """Bundle the constraints into (violation, n_frames, subset_ratio).

    `violation` is 0 when the selection satisfies both the frame window
    (which pins the seed fraction) and the subset-ratio band, and otherwise
    grows with the relative distance outside them, so repair can descend on
    it even when both are violated."""
    lo, hi = window
    mid = 0.5 * (lo + hi)

    def n_frames(sel):
        return sum(per_seg[k].size for k in sel)

    def subset_ratio(sel):
        if not sub_per_seg:
            return None
        s = sum(sub_per_seg.get(k, (0.0, 0.0))[0] for k in sel)
        n = sum(sub_per_seg.get(k, (0.0, 0.0))[1] for k in sel)
        return (s / n) if n else None

    def violation(sel):
        v = 0.0
        n = n_frames(sel)
        if n < lo:
            v += (lo - n) / mid
        elif n > hi:
            v += (n - hi) / mid
        sr = subset_ratio(sel)
        if sr is not None and sub_target:
            rel = abs(sr / sub_target - 1.0)
            if rel > sub_tol:
                v += rel - sub_tol
        return v

    return violation, n_frames, subset_ratio


def repair(chosen, per_seg, counts, violation):
    """Walk an infeasible selection back inside the constraints with 1-for-1
    swaps, each step taking the swap that reduces the violation most and
    breaking ties on W1. Segment count is preserved throughout."""
    chosen = list(chosen)
    for _ in range(6 * len(chosen)):
        if violation(chosen) <= 0:
            return chosen, True
        best = None
        for i in range(len(chosen)):
            for k in per_seg:
                if k in chosen:
                    continue
                trial = list(chosen)
                trial[i] = k
                key = (violation(trial),
                       w1(np.concatenate([per_seg[j] for j in trial]), counts))
                if best is None or key < best[0]:
                    best = (key, trial)
        if best is None or best[0][0] >= violation(chosen):
            return chosen, violation(chosen) <= 0
        chosen = best[1]
    return chosen, violation(chosen) <= 0


def reachable_sums(lengths, k_seg, cap):
    """Bitmask of every frame total reachable with exactly `k_seg` segments
    (bit s set <=> sum s reachable). Used to check the requested window
    before searching and to report nearby reachable totals if it has none."""
    masks = [0] * (int(k_seg) + 1)
    masks[0] = 1
    full = (1 << (int(cap) + 1)) - 1
    for length in lengths:
        for j in range(int(k_seg) - 1, -1, -1):
            if masks[j]:
                masks[j + 1] |= (masks[j] << int(length)) & full
    return masks[int(k_seg)]


def sums_in_range(mask, lo, hi):
    return [s for s in range(max(int(lo), 0), int(hi) + 1) if (mask >> s) & 1]


def refine_swaps(chosen, per_seg, counts, violation, passes=3):
    """Local search: try replacing each chosen segment with an unchosen one
    (greedy forward selection cannot undo an early pick).

    Swaps are 1-for-1, so K is preserved, and swaps that would break a
    constraint (frame window or subset-ratio band) are rejected, so
    refinement only lowers W1 from a feasible selection."""
    all_keys = list(per_seg)

    def score(sel):
        return w1(np.concatenate([per_seg[k] for k in sel]), counts)

    best = list(chosen)
    best_s = score(best)
    for _ in range(passes):
        improved = False
        for i in range(len(best)):
            for k in all_keys:
                if k in best:
                    continue
                trial = list(best)
                trial[i] = k
                if violation(trial) > 0:
                    continue
                s = score(trial)
                if s < best_s - 1e-12:
                    best, best_s, improved = trial, s, True
        if not improved:
            break
    return best, best_s


# --- reporting -------------------------------------------------------------


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(
        n=int(x.size), mean=float(x.mean()), std=float(x.std()),
        p10=float(np.percentile(x, 10)), p50=float(np.percentile(x, 50)),
        p90=float(np.percentile(x, 90)), max=float(x.max()),
    )


def seg_display_name(dataset, seg_of_frame, key):
    """Human-readable segment key and, where the dataset supports it, the
    exact string `data.val.scene_list` matches on:

      * Waymo          the segment directory name (path-list data_list)
      * nuScenes       `scene_name` ("scene-0003") if the info pkl has it
                       (preprocessing/nuscenes/add_scene_id.py); otherwise
                       the positional run index, not a valid scene_list entry
    """
    data_list = getattr(dataset, "data_list", None)
    idx = int(np.argmax(np.asarray(seg_of_frame) == key))
    if data_list is None or len(data_list) == 0:
        return str(key)
    d = data_list[idx]
    if isinstance(d, str):
        return os.path.basename(os.path.dirname(d))
    if isinstance(d, dict) and d.get("scene_name"):
        return str(d["scene_name"])
    return str(key)


def supports_scene_list(dataset):
    """True when `seg_display_name` returns strings the dataset can be
    filtered by, i.e. a pasteable `data.val.scene_list`."""
    data_list = getattr(dataset, "data_list", None)
    if not data_list:
        return False
    d = data_list[0]
    return isinstance(d, str) or (isinstance(d, dict)
                                  and bool(d.get("scene_name")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target_frames", type=int, default=300,
                    help="approximate slice size in frames; sets the "
                         "segment count")
    ap.add_argument("--seed_tol", type=float, default=0.02,
                    help="relative tolerance of the slice's seed fraction "
                         "(n_seg/n_frames) vs the full split's; sets the "
                         "frame window")
    ap.add_argument("--tol", type=float, default=0.1,
                    help="frame-count tolerance used only with "
                         "--free_segments (no seed-fraction control)")
    ap.add_argument("--subset_log", default=None,
                    help="spacecache_per_frame.jsonl of a full-val "
                         "SpaceCache run; also match its recompute ratio")
    ap.add_argument("--subset_tol", type=float, default=0.02,
                    help="relative tolerance of the slice's recompute "
                         "ratio vs the full split's (with --subset_log)")
    ap.add_argument("--n_segments", type=int, default=None,
                    help="pin the number of segments (default: "
                         "round(target_frames * n_seg_total / n_frames_total), "
                         "which matches the full split's seed fraction)")
    ap.add_argument("--free_segments", action="store_true",
                    help="do not pin the segment count: pick by W1 alone "
                         "until the frame target (no seed-fraction control)")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--name", default=None,
                    help="output basename (default: config stem)")
    ap.add_argument("--out_dir", default="spacecache/experiments/slices")
    ap.add_argument("--cache", default=None,
                    help="npz of the per-frame counts, reused when present "
                         "(default: <out_dir>/<name>_counts.npz)")
    ap.add_argument("--recount", action="store_true",
                    help="ignore an existing count cache and recount")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    name = args.name or os.path.splitext(os.path.basename(args.config))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    cache = args.cache or os.path.join(args.out_dir, f"{name}_counts.npz")

    print(f"=== latency-slice selection | {name} ===")
    dataset = build_dataset(cfg.data.val)
    n_total = len(dataset)
    num_classes = int(cfg.data.num_classes)
    print(f"    val frames    : {n_total}")

    seg_of_frame = SpaceCachePolicy._build_seg_ids(dataset)
    assert seg_of_frame is not None and len(seg_of_frame) == n_total, (
        "could not derive segment ids for this dataset — the slice must be "
        "whole-segment, so selection cannot proceed")
    seg_arr = np.asarray(seg_of_frame)
    n_segments = len(set(seg_of_frame))
    print(f"    segments      : {n_segments} "
          f"(mean {n_total / n_segments:.1f} frames each)")

    if os.path.exists(cache) and not args.recount:
        z = np.load(cache, allow_pickle=True)
        counts, hists = z["counts"], z["hists"]
        assert counts.size == n_total, (
            f"count cache {cache} has {counts.size} frames but the dataset "
            f"has {n_total} — pass --recount")
        print(f"    counts        : loaded from {cache}")
    else:
        print(f"    counting voxels with {args.num_workers} workers "
              f"(val transform, no model) ...")
        counts, hists = collect_counts(dataset, num_classes, args.num_workers)
        np.savez_compressed(cache, counts=counts, hists=hists)
        print(f"    counts        : saved {cache}")

    # --- select ---
    # pin the segment count so the slice's seed fraction (n_seg/n_frames)
    # matches the full split's
    sf_target = n_segments / n_total
    if args.free_segments:
        k_seg = None
        window = (args.target_frames * (1.0 - args.tol),
                  args.target_frames * (1.0 + args.tol))
    else:
        k_seg = args.n_segments or max(
            1, int(round(args.target_frames * sf_target)))
        # frame window from the seed fraction: sf = k_seg / n_frames, so
        # |sf/sf_target - 1| <= seed_tol maps to n_frames in
        # [k/(sf_target*(1+tol)), k/(sf_target*(1-tol))]; the realised
        # frame count (~k/sf_target) may differ from --target_frames
        window = (k_seg / (sf_target * (1.0 + args.seed_tol)),
                  k_seg / (sf_target * (1.0 - args.seed_tol)))
        print(f"    segment count : pinned to {k_seg}")
        print(f"    seed fraction : target {sf_target:.4f} "
              f"(+/-{100 * args.seed_tol:.0f}%) -> frames in "
              f"[{window[0]:.0f}, {window[1]:.0f}]")
        # segment lengths are uneven, so a tight window may contain no
        # achievable total
        seg_lens = [int((seg_arr == k).sum()) for k in sorted(set(seg_of_frame))]
        mask = reachable_sums(seg_lens, k_seg, int(window[1]) + 1)
        ok_sums = sums_in_range(mask, window[0], window[1])
        if not ok_sums:
            near = sums_in_range(mask, window[0] - 40, window[1] + 40)
            raise SystemExit(
                f"no {k_seg}-segment slice can total "
                f"{window[0]:.0f}-{window[1]:.0f} frames.\n"
                f"  reachable totals nearby : {near}\n"
                f"  segment lengths present : "
                f"{sorted(set(seg_lens))}\n"
                f"  -> raise --seed_tol, or pick a --target_frames whose "
                f"k/sf_full lands on a reachable total.")
        print(f"    reachable     : {len(ok_sums)} totals in window "
              f"({ok_sums[0]}..{ok_sums[-1]})")

    sub_per_seg, sub_target = (None, None)
    if args.subset_log:
        sub_per_seg, sub_target = load_subset_ratios(args.subset_log)
        print(f"    cache hit     : full-split subset ratio {sub_target:.4f} "
              f"(+/-{100 * args.subset_tol:.0f}%) from "
              f"{os.path.basename(os.path.dirname(args.subset_log))}")

    chosen, per_seg = greedy_select(
        counts, seg_of_frame, args.target_frames, n_segments=k_seg,
        window=window)
    violation, n_frames_of, subset_of = make_checks(
        per_seg, window, sub_per_seg, sub_target, args.subset_tol)
    if chosen is None:
        print("    note          : constrained greedy dead-ended; "
              "rebuilding by W1 alone + swap repair")
        chosen = greedy_fixed_k(counts, per_seg, k_seg)
    # the greedy only knows the frame window; the subset-ratio band (if
    # any) is imposed here, so repair runs even when the greedy succeeded
    chosen, fixed = repair(chosen, per_seg, counts, violation)
    if not fixed:
        raise SystemExit(
            f"swap repair could not satisfy the constraints "
            f"(frames in [{window[0]:.0f}, {window[1]:.0f}]"
            + (f", subset ratio within {100 * args.subset_tol:.0f}% of "
               f"{sub_target:.4f}" if sub_target else "")
            + ") — raise --seed_tol / --subset_tol or change --target_frames")
    chosen, score = refine_swaps(chosen, per_seg, counts, violation)
    chosen = sorted(chosen, key=lambda k: int(np.argmax(seg_arr == k)))

    mask = np.isin(seg_arr, chosen)
    frame_idx = np.nonzero(mask)[0]
    slice_counts = counts[frame_idx]

    # reference: the first N frames of the split (`--max_frames N`)
    prefix = counts[: int(frame_idx.size)]
    prefix_w1 = w1(prefix, counts)

    print("")
    print(f"[ selected {len(chosen)} segments / {frame_idx.size} frames ]")
    print(f"  W1 (selected vs full) : {score:12.2f} voxels")
    print(f"  W1 (naive prefix)     : {prefix_w1:12.2f} voxels"
          f"   -> {prefix_w1 / max(score, 1e-9):.1f}x worse")
    print("")
    print(f"  {'segment':64s} {'frames':>7s} {'mean N':>10s}")
    print("  " + "-" * 84)
    for k in chosen:
        nm = seg_display_name(dataset, seg_of_frame, k)
        c = counts[seg_arr == k]
        print(f"  {nm[:64]:64s} {c.size:7d} {c.mean():10.0f}")

    st_slice, st_full = describe(slice_counts), describe(counts)
    print("")
    print("[ voxel-count distribution ]")
    print(f"  {'':8s} {'n':>7s} {'mean':>10s} {'std':>10s} {'p10':>10s} "
          f"{'p50':>10s} {'p90':>10s} {'max':>10s}")
    for lab, st in (("slice", st_slice), ("full", st_full)):
        print(f"  {lab:8s} {st['n']:7d} {st['mean']:10.0f} {st['std']:10.0f} "
              f"{st['p10']:10.0f} {st['p50']:10.0f} {st['p90']:10.0f} "
              f"{st['max']:10.0f}")
    print(f"  mean delta: {100 * (st_slice['mean'] / st_full['mean'] - 1):+.2f}%"
          f"   p90 delta: {100 * (st_slice['p90'] / st_full['p90'] - 1):+.2f}%")

    # class mix: not used for mIoU, but a sanity check that the scenes are
    # representative
    h_slice = hists[frame_idx].sum(0).astype(np.float64)
    h_full = hists.sum(0).astype(np.float64)
    p, q = h_slice / max(h_slice.sum(), 1), h_full / max(h_full.sum(), 1)
    print(f"  class-mix L1 distance: {float(np.abs(p - q).sum()):.4f} "
          f"(0 = identical, 2 = disjoint)")

    # seed fraction: a ratio above 1 means more cache resets per frame than
    # the full split, which slows down only the streaming policies
    sf_slice = len(chosen) / frame_idx.size
    sf_full = n_segments / n_total
    ratio = sf_slice / sf_full
    if abs(ratio - 1) <= max(args.seed_tol, 1e-9):
        note = "   ok"
    elif ratio > 1:
        note = "   <-- WARNING: extra cold seeds, penalises `ours` on Speed"
    else:
        note = "   <-- WARNING: too few cold seeds, flatters `ours` on Speed"
    print(f"  seed fraction: slice {sf_slice:.4f}  full {sf_full:.4f}  "
          f"ratio {ratio:.3f}{note}")
    print(f"  segment lengths: mean {frame_idx.size / len(chosen):.1f} "
          f"(full {n_total / n_segments:.1f})")

    # recompute ratio: sets the stage-3 cost of SpaceCache
    sub_slice = subset_of(chosen) if sub_target else None
    if sub_slice is not None:
        rel = sub_slice / sub_target - 1.0
        print(f"  subset ratio  : slice {sub_slice:.4f}  full {sub_target:.4f}"
              f"  ratio {1 + rel:.3f}"
              + ("   ok" if abs(rel) <= args.subset_tol else
                 "   <-- WARNING: `ours` mis-stated by this much on FLOPs/Speed"))

    # --- write ---
    seg_names = [seg_display_name(dataset, seg_of_frame, k) for k in chosen]
    is_path_ds = supports_scene_list(dataset)
    out = dict(
        config=args.config,
        dataset=name,
        n_frames_total=int(n_total),
        n_segments_total=int(n_segments),
        target_frames=int(args.target_frames),
        n_frames=int(frame_idx.size),
        n_segments=len(chosen),
        w1_voxels=float(score),
        w1_naive_prefix=float(prefix_w1),
        seed_fraction=float(sf_slice),
        seed_fraction_full=float(sf_full),
        subset_ratio=(None if sub_slice is None else float(sub_slice)),
        subset_ratio_full=(None if sub_target is None else float(sub_target)),
        seg_keys=[int(k) if isinstance(k, (int, np.integer)) else str(k)
                  for k in chosen],
        seg_names=seg_names,
        # for `data.val.scene_list`: WaymoDataset matches these by substring
        # on its path list, NuScenesSceneMixin by exact scene_name /
        # scene_token; None if the dataset cannot be filtered by name
        scene_list=(seg_names if is_path_ds else None),
        # indices into the unfiltered val dataset, a record of the chosen
        # frames for datasets without a scene_list
        frame_indices=[int(i) for i in frame_idx],
        stats=dict(slice=st_slice, full=st_full),
        class_mix_l1=float(np.abs(p - q).sum()),
    )
    out_path = os.path.join(args.out_dir, f"{name}_cost_slice.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    if is_path_ds:
        print("\nPaste into the latency-slice config:")
        print("data = dict(val=dict(scene_list=[")
        for nm in seg_names:
            print(f'    "{nm}",')
        print("]))")
    else:
        print(f"\nNo scene_list for this dataset — its segments have no "
              f"names to filter on. For nuScenes, run "
              f"preprocessing/nuscenes/add_scene_id.py and re-run this.")


if __name__ == "__main__":
    main()
