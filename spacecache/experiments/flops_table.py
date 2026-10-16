"""FLOPs by component (the GFLOPs rows of Table 6) at the Fig. 7 points.

Each method is taken at its point within the budget, chosen as in
efficiency_breakdown.py (lowest encoder FLOPs among the outdoor sweep points
of Waymo / Utonia), and its per-role FLOPs (encoder + decoder, GFLOPs per
frame) are read from the latency run (`metrics.flops_by_role`, pass 1).

    PYTHONPATH=. python spacecache/experiments/flops_table.py
    PYTHONPATH=. python spacecache/experiments/flops_table.py --budget 0.01

Writes output/efficiency/flops_by_component_m<budget>pt.md with the
same rows as the paper table (and a LaTeX body under it).
"""

from __future__ import annotations

import argparse
import os

from spacecache.common import naming
from spacecache.experiments.efficiency_breakdown import (
    CELL, MAIN, _load, auto_picks, family,
)

# paper row -> cost_meter roles
ROWS = [
    ("CPE (sparse conv. + linear)", ("cpe_conv", "cpe_linear")),
    ("QKV projection", ("qkv",)),
    ("Attention (QK^T, AV)", ("qk_av",)),
    ("Output projection", ("proj",)),
    ("MLP", ("mlp",)),
    ("Other linear layers", ("other_linear", "other_conv")),
]
OVERHEAD = ("matching",)


def gflops(run, roles):
    fr = ((run or {}).get("metrics") or {}).get("flops_by_role") or {}
    return sum(float(fr.get(k, 0)) for k in roles) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.02,
                    help="mIoU budget below the backbone (0.02 = -2 pt)")
    ap.add_argument("--acc_dir", default=f"{MAIN}/accuracy/{CELL}")
    ap.add_argument("--acc_mode", default="dense",
                    choices=("dense", "flash"))
    ap.add_argument("--lat_dir", default=f"{MAIN}/latency/{CELL}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    picks, _, _ = auto_picks(a.acc_dir, a.lat_dir, a.budget,
                             acc_mode=a.acc_mode, by="flops")
    runs = [(label, v, _load([a.lat_dir], v)) for label, v, *_ in picks]
    base_total = None
    header = ["Component"] + [label for label, _, _ in runs]
    lines = [f"# FLOPs by component (GFLOPs / frame), {CELL}, "
             f"-{a.budget * 100:g} pt", "",
             "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header),
             "| setting | " + " | ".join(v for _, v, _ in runs) + " |"]
    tex = []
    for name, roles in ROWS:
        vals = [gflops(r, roles) for _, _, r in runs]
        lines.append(f"| {name} | " + " | ".join(f"{x:,.1f}" for x in vals)
                     + " |")
        tex.append(f"{name} & " + " & ".join(f"{x:,.1f}" for x in vals)
                   + r" \\")
    # SpaceCache's selection is the key transform (~9 multiply-adds per
    # token, not recorded by CostMeter), shown as +<0.001 as in the paper
    over = []
    for _, v, r in runs:
        x = gflops(r, OVERHEAD)
        over.append("--" if family(v)[0] == naming.BACKBONE
                    else f"+{x:,.1f}" if x >= 0.05 else "+<0.001")
    lines.append("| Overhead (scoring) | " + " | ".join(over) + " |")
    totals = [((r or {}).get("metrics") or {}).get("flops_total", 0) / 1e9
              for _, _, r in runs]
    for (label, v, _), t in zip(runs, totals):
        if family(v)[0] == naming.BACKBONE:
            base_total = t
    lines.append("| Total | " + " | ".join(f"{t:,.1f}" for t in totals)
                 + " |")
    if base_total:
        lines.append("| Reduction | " + " | ".join(
            "--" if abs(t - base_total) < 1e-9
            else f"{100 * (1 - t / base_total):.1f}%" for t in totals) + " |")
    text = "\n".join(lines + ["", "```", *tex, "```"])
    print(text)
    out = a.out or (f"output/efficiency/"
                    f"flops_by_component_m{a.budget * 100:g}pt.md")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
