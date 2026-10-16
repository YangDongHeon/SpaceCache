"""Merge `--scene_shard K/N` runs into the unsharded run's results.json.

Shards cover disjoint sets of whole scenes and store the raw per-class
`intersection` / `union` / `target` counts, so summing them reproduces an
unsharded run exactly. For every `<run>_shardKofN` group with all N shards
present, writes `<cell>/<run>/results.json` with the metrics and
`per_class_iou` recomputed from the summed counts, the SpaceCache aggregates
recomputed from the pooled `spacecache_per_frame.jsonl` logs, the
`--region_stats` counters summed and the CostMeter means frame-weighted.

    PYTHONPATH=. python spacecache/experiments/merge_shards.py [--dry-run] \\
        --cell output/outdoor/accuracy/waymo/utonia_dec
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

_SHARD = re.compile(r"^(.*)_shard(\d+)of(\d+)$")


def _load_frame_log(shard_dir, res):
    """The shard's SpaceCache frame log (last eval pass, as in run.py), or
    None if absent."""
    path = os.path.join(shard_dir, "spacecache_per_frame.jsonl")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    fpp = res.get("frames_per_pass")
    if res.get("passes", 1) > 1 and fpp:
        rows = rows[-int(fpp):]
    return rows


def _stream_aggregate(rows):
    """run.py's SpaceCache aggregates, recomputed over the pooled rows."""
    out = {}
    ns = [f for f in rows if not f.get("is_seed", False)] or rows
    for src, dst in (("match_ratio", "spacecache_hit_rate"),
                     ("subset_size", "spacecache_miss_tokens"),
                     ("match_ratio_d4", "spacecache_hit_rate_s4"),
                     ("subset_size_d4", "spacecache_miss_tokens_s4")):
        xs = [f[src] for f in ns if f.get(src) is not None]
        if xs:
            out[dst] = float(sum(xs) / len(xs))
    for sfx, osfx in (("", ""), ("_d4", "_s4")):
        hk = sum(int(f.get("n_hit_ok" + sfx) or 0) for f in ns)
        hn = sum(int(f.get("n_hit_l0" + sfx) or 0) for f in ns)
        mk = sum(int(f.get("n_miss_ok" + sfx) or 0) for f in ns)
        mn = sum(int(f.get("n_miss_l0" + sfx) or 0) for f in ns)
        if hn + mn > 0:
            out["spacecache_acc_hit" + osfx] = hk / hn if hn else None
            out["spacecache_acc_miss" + osfx] = mk / mn if mn else None
            if not sfx:
                out["spacecache_hit_fraction_input"] = hn / (hn + mn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True,
                    help="directory holding the <run>_shardKofN runs")
    ap.add_argument("--allow-partial", action="store_true",
                    help="merge even if some shards are missing")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    groups = {}
    for d in sorted(os.listdir(args.cell)):
        m = _SHARD.match(d)
        if m and os.path.isfile(
                os.path.join(args.cell, d, "results.json")):
            base, k, n = m.group(1), int(m.group(2)), int(m.group(3))
            groups.setdefault((base, n), {})[k] = d

    for (base, n), shards in sorted(groups.items()):
        if len(shards) < n and not args.allow_partial:
            print(f"SKIP {base}: {len(shards)}/{n} shards present")
            continue
        shards_data = {}
        for k, d in shards.items():
            with open(os.path.join(args.cell, d, "results.json")) as f:
                shards_data[k] = json.load(f)
        # round-robin and --shard_balance cut the scenes differently; a set
        # mixing the two overlaps some scenes and misses others
        modes = {r.get("scene_shard_mode") or "round_robin"
                 for r in shards_data.values()}
        if len(modes) > 1:
            print(f"SKIP {base}: shards mix scene_shard modes {sorted(modes)}"
                  f" — delete one mode's {base}_shard*of{n} dirs and rerun")
            continue
        base = dict(shards_data[min(shards_data)])
        inter = np.zeros(len(base["intersection"]))
        union = np.zeros(len(base["union"]))
        target = np.zeros(len(base["target"]))
        n_frames = n_counted = 0
        for k, r in sorted(shards_data.items()):
            inter += np.asarray(r["intersection"], dtype=float)
            union += np.asarray(r["union"], dtype=float)
            target += np.asarray(r["target"], dtype=float)
            n_frames += int(r.get("n_frames", 0))
            n_counted += int(r.get("n_frames_counted", 0))
        iou = inter / (union + 1e-10)
        acc = inter / (target + 1e-10)
        base["intersection"] = inter.tolist()
        base["union"] = union.tolist()
        base["target"] = target.tolist()
        base["mIoU"] = float(np.mean(inter / (union + 1e-10)))
        base["mAcc"] = float(np.mean(inter / (target + 1e-10)))
        base["allAcc"] = float(inter.sum() / (target.sum() + 1e-10))
        base["n_frames"] = n_frames
        base["n_frames_counted"] = n_counted
        names = list(base.get("per_class_iou") or {})
        if len(names) == len(iou):
            base["per_class_iou"] = {nm: float(v)
                                     for nm, v in zip(names, iou)}
        # SpaceCache aggregates: exact from the pooled frame logs; without
        # a log for every shard, shard 0's values are kept and flagged
        logs = [_load_frame_log(os.path.join(args.cell, shards[k]), r)
                for k, r in sorted(shards_data.items())]
        if any(r.get("spacecache_hit_rate") is not None
               for r in shards_data.values()):
            if all(lg is not None for lg in logs):
                base.update(_stream_aggregate(
                    [row for lg in logs for row in lg]))
                base["spacecache_merged"] = "frame_log"
            else:
                base["spacecache_merged"] = "shard0_only"
        rd = [r.get("spacecache_region_stats") for r in shards_data.values()]
        if all(isinstance(d, dict) for d in rd) and rd:
            base["spacecache_region_stats"] = {
                k: int(sum(int(d.get(k, 0)) for d in rd))
                for k in sorted({k for d in rd for k in d})}
        ag = [r.get("spacecache_region_age") for r in shards_data.values()]
        if ag and all(isinstance(d, dict) for d in ag):
            base["spacecache_region_age"] = {
                k: [int(sum(v)) for v in zip(*(d[k] for d in ag))]
                for k in ag[0] if all(k in d for d in ag)}
        # CostMeter summaries are per-frame means: pool them weighted by
        # each shard's metered frame count; peak_* / cache_* fields keep
        # the first shard's value
        mets = [(r.get("metrics"), int((r.get("metrics") or {})
                                       .get("n_frames") or 0))
                for r in shards_data.values()]
        mets = [(m, w) for m, w in mets if isinstance(m, dict) and w > 0]
        if mets:
            wsum = sum(w for _, w in mets)
            merged = dict(mets[0][0])
            for key, v0 in mets[0][0].items():
                if key == "n_frames":
                    merged[key] = wsum
                elif isinstance(v0, (int, float)) and not isinstance(
                        v0, bool) and not key.startswith(("peak_", "cache_")):
                    merged[key] = sum(float(m.get(key) or 0) * w
                                      for m, w in mets) / wsum
                elif isinstance(v0, dict) and key != "peak_by_stage" and all(
                        isinstance(x, (int, float)) for x in v0.values()):
                    ks = sorted({k for m, _ in mets
                                 for k in (m.get(key) or {})})
                    merged[key] = {
                        k: sum(float((m.get(key) or {}).get(k, 0)) * w
                               for m, w in mets) / wsum for k in ks}
            merged["merged_shards"] = "frame-weighted means; peaks shard0"
            base["metrics"] = merged
        base["run_name"] = base["run_name"].split("_shard")[0]
        base["scene_shard"] = None
        base["merged_from_shards"] = sorted(shards_data.keys())
        dest = os.path.join(args.cell, base["run_name"])
        print(f"MERGE {base['run_name']}: mIoU {base['mIoU']:.4f} "
              f"({len(shards_data)}/{n} shards, {n_frames} frames)")
        if not args.dry_run:
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(dest, "results.json"), "w") as f:
                json.dump(base, f, indent=2)


if __name__ == "__main__":
    main()
