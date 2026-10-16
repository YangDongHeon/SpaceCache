"""Table 2: indoor comparison (ScanNet video, LitePT-B).

The operating points are those of the paper, the swept values of Appendix
Table 5 (ScanNet column), listed in scripts/reproduce/table2_points.txt.
mIoU and total GFLOPs (as in Table 1) come from the accuracy runs of
indoor_accuracy.sh (10 scenes, 2,221 frames); kernel time and speedup from
the Orin runs of
indoor_latency_orin.sh (`graph_fair.csv` of graph_fair_table.py: the mean
GPU kernel time per frame over the 50 captured frames, Commit Gate
included, and the paired ratio of means to the backbone).

    PYTHONPATH=. python spacecache/experiments/indoor_table.py [--orin <graph_fair.csv>]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

from spacecache.common import naming

# method -> Table 2 row, in table order
ROWS = [
    (naming.TOME, "+ ToMe*"),
    (naming.ALGM, "+ ALGM*"),
    (naming.GITMERGE3D, "+ GitMerge3D†"),
    (naming.SPACECACHE_S3, "+ SpaceCache Stage-3"),
    (naming.SPACECACHE_S34, "+ SpaceCache Stage-3,4"),
]
ORIN_GLOB = "output/indoor/kernel_time/*/piecewise/graph_fair.csv"


def row_of(variant):
    return dict(ROWS).get(naming.policy_of(variant))


def load_points(path):
    """[(budget, run name)] from table2_points.txt."""
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            budget, _cell, variant, _args = line.split("|", 3)
            pts.append((int(budget), variant))
    return pts


def load_run(acc_root, variant):
    rp = os.path.join(acc_root, variant, "results.json")
    if not os.path.isfile(rp):
        return None, None
    with open(rp) as f:
        r = json.load(f)
    miou = r.get("mIoU")
    fl = (r.get("metrics") or {}).get("flops_total")
    return (None if miou is None else 100.0 * miou,
            None if not fl else fl / 1e9)


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="scripts/reproduce/table2_points.txt")
    ap.add_argument("--acc_root",
                    default="output/indoor/accuracy/"
                            "scannet_video/litept_base_meshnormal")
    ap.add_argument("--orin", default=None,
                    help="graph_fair.csv of the Orin runs (default: the "
                         f"first match of {ORIN_GLOB})")
    ap.add_argument("--out", default=None,
                    help="markdown output (default: output/indoor/table2.md)")
    args = ap.parse_args()

    orin = args.orin or next(iter(sorted(glob.glob(ORIN_GLOB))), None)
    kern = {}
    if orin and os.path.isfile(orin):
        with open(orin, newline="") as f:
            for r in csv.DictReader(f):
                kern[r.get("run")] = (_float(r.get("k_frame_ms")),
                                          _float(r.get("k_frame_x")))

    def cells(variant):
        miou, gfl = load_run(args.acc_root, variant)
        ms, sp = kern.get(variant, (None, None))
        return miou, ms, gfl, sp

    def fmt(x, nd=2, suf=""):
        return "--" if x is None else f"{x:.{nd}f}{suf}"

    def group(c):
        miou, ms, gfl, sp = c
        return (f" {fmt(miou)} | {fmt(ms, 1)} | {fmt(gfl, 1)} |"
                f" {fmt(sp, 2, '×')} |")

    by_row = {}
    for budget, variant in load_points(args.points):
        name = row_of(variant)
        if name is None:
            raise SystemExit(f"unknown method of {variant} in {args.points}")
        by_row.setdefault(name, {})[budget] = cells(variant)

    head = ("| Method | mIoU ≤1pt | Kernel ms | GFLOPs | Speedup |"
            " mIoU ≤2pt | Kernel ms | GFLOPs | Speedup |")
    base = cells(naming.run_name(naming.BACKBONE, "dense"))
    lines = [
        "# Table 2: indoor comparison (ScanNet video, LitePT-B)", "",
        f"accuracy: {args.acc_root} · kernel time: "
        f"{orin or 'no graph_fair.csv found'}", "",
        head, "|---|" + "---|" * 8,
        "| **LitePT-B** |" + group(base) + group(base)]
    for _, name in ROWS:
        pts = by_row.get(name, {})
        empty = (None, None, None, None)
        lines.append(f"| {name} |" + group(pts.get(1, empty))
                     + group(pts.get(2, empty)))
    text = "\n".join(lines) + "\n"
    print(text)
    out = args.out or "output/indoor/table2.md"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
