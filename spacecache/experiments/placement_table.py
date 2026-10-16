"""Placement ablation table: SpaceCache at a single encoder stage or at the
third and fourth stages together (native region size, fixed lifetime).

Accuracy and cache-hit fractions come from the merged full-validation runs,
FLOPs, latency, cache-operation overhead and cache size from the
latency-slice runs (clean pass of a multi-pass run).

    PYTHONPATH=. python spacecache/experiments/placement_table.py --thetas 0.85
"""

from __future__ import annotations

import argparse
import json
import os

from spacecache.common import naming
from spacecache.common.latency import frame_ms

ABL = "output/ablation"
CELL = "waymo/utonia_dec"
STAGES = ("s0", "s1", "s2", "s3", "s4", "s34")
LABEL = {"s0": "0", "s1": "1", "s2": "2", "s3": "3", "s4": "4",
         "s34": "3+4"}
OPS = ("match", "subset", "restore", "stage", "commit")


def _load(path):
    p = os.path.join(path, "results.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def _ops_median(bd, key, names):
    d = bd.get(key) or {}
    return sum(float(d.get(k, 0.0)) for k in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc_root", default=f"{ABL}/accuracy/{CELL}")
    ap.add_argument("--lat_root", default=f"{ABL}/latency/{CELL}")
    ap.add_argument("--baseline_acc", default=None,
                    help="run dir of the backbone accuracy (full val); "
                         "default <acc_root>/backbone_<acc_mode>")
    ap.add_argument("--acc_mode", default="dense",
                    choices=("dense", "flash"),
                    help="attention of the accuracy runs (dense, as in "
                         "scripts/reproduce)")
    ap.add_argument("--thetas", default="0.85",
                    help="comma-separated confidence thresholds")
    ap.add_argument("--life", type=int, default=8)
    ap.add_argument("--blocks", default="3,3,3,12,3",
                    help="transformer blocks per encoder stage")
    ap.add_argument("--no_triton", action="store_true",
                    help="the latency runs ran without --triton_cache")
    ap.add_argument("--expiry", default="staggered",
                    choices=("staggered", "fixed"),
                    help="lifetime rule of the runs to read")
    ap.add_argument("--out", default=f"{ABL}/placement_table.md")
    args = ap.parse_args()

    blocks = [int(x) for x in args.blocks.split(",")]
    base_lat = _load(os.path.join(args.lat_root,
                                  naming.run_name(naming.BACKBONE, "dense")))
    if base_lat is None:
        raise SystemExit(f"missing backbone latency run in {args.lat_root}")
    base_ms = frame_ms(base_lat)
    base_fl = (base_lat.get("metrics") or {}).get("flops_encoder")
    base_acc = _load(args.baseline_acc
                     or os.path.join(args.acc_root, naming.run_name(
                         naming.BACKBONE, args.acc_mode)))
    base_miou = base_acc["mIoU"] if base_acc else None

    lines = [
        f"# Placement ablation ({CELL}, L={args.life}, native regions)", "",
        f"backbone: mIoU {base_miou * 100:.2f} (full val), "
        f"{base_ms:.1f} ms (latency slice)" if base_miou is not None
        else f"backbone: {base_ms:.1f} ms (latency slice)", "",
        "Saved = backbone latency - frame latency (Commit Gate included); "
        "Overhead = median lookup + subset + restore + staging + commit.", "",
        "| Stage | theta | Blocks | Hit (%) | FLOPs down (%) | Saved (ms) | "
        "Overhead (ms) | Cache (MiB) | mIoU |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for theta in (float(x) for x in args.thetas.split(",")):
        for s in STAGES:
            kw = dict(lifetime=args.life, confidence=theta,
                      expiry=args.expiry)
            policy = f"spacecache_{s}"
            acc = _load(os.path.join(args.acc_root, naming.run_name(
                policy, args.acc_mode, **kw)))
            lat = _load(os.path.join(args.lat_root, naming.run_name(
                policy, "dense", triton=not args.no_triton, **kw)))
            if acc is None or lat is None:
                lines.append(f"| {LABEL[s]} | {theta:g} | missing run | | | "
                             f"| | | |")
                continue
            n_blk = (blocks[3] + blocks[4] if s == "s34"
                     else blocks[int(s[1])])
            hit = f"{acc['spacecache_hit_rate'] * 100:.0f}"
            if acc.get("spacecache_hit_rate_s4") is not None:
                hit += f" / {acc['spacecache_hit_rate_s4'] * 100:.0f}"
            met = lat.get("metrics") or {}
            fl_down = (100.0 * (1.0 - met["flops_encoder"] / base_fl)
                       if base_fl and met.get("flops_encoder") else float("nan"))
            bd = lat.get("spacecache_ms") or {}
            overhead = (_ops_median(bd, "median", OPS)
                        + _ops_median(bd, "s4_median", OPS))
            saved = base_ms - frame_ms(lat, stat="median")
            cache = ((met.get("cache_bytes_mean") or 0)
                     + (met.get("cache_bytes_d4_mean") or 0)) / 2 ** 20
            lines.append(
                f"| {LABEL[s]} | {theta:g} | {n_blk} | {hit} | {fl_down:.1f} "
                f"| {saved:.1f} | {overhead:.1f} | {cache:.0f} "
                f"| {acc['mIoU'] * 100:.2f} |")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
