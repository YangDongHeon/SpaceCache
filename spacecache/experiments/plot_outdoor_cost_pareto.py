"""Outdoor accuracy-vs-cost Pareto plot.

Joins full-val mIoU (`--acc_root`) with latency-slice latency and FLOPs
(`--lat_root`) on (method, swept value) and draws a latency row and a FLOPs
row for {waymo, nuscenes} x {utonia_dec, litept_base}. The paper panel,
spacecache/figure/experiment/outdoor/plot_pareto.py, imports METHODS and the
loaders from here.

    PYTHONPATH=. python spacecache/experiments/plot_outdoor_cost_pareto.py \\
        [--acc_root output/outdoor/accuracy] \\
        [--lat_root output/outdoor/latency] \\
        [--out_prefix output/outdoor/pareto]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacecache.common import naming
from spacecache.common.latency import frame_ms

# (method, label, colour, marker); colours from the paper palette, with
# both SpaceCache variants in the red family
METHODS = [
    (naming.GITMERGE3D,     "GitMerge3D†",                 "#a1ff0a", "s"),
    (naming.TOME,           "ToMe*",                       "#147df5", "v"),
    (naming.ALGM,           "ALGM*",                       "#ffd300", "D"),
    (naming.SPACECACHE_S3,  "SpaceCache Stage-3 (ours)",   "#ff8a8a", "^"),
    (naming.SPACECACHE_S34, "SpaceCache Stage-3,4 (ours)", "#ff0000", "P"),
]


def key_of(run):
    """-> (method, swept value) of a run directory name; SpaceCache with the
    confidence rule off has swept value 0.0. (None, None) for shards and
    foreign directories."""
    d = naming.parse_run_name(run)
    if d is None or d["shard"] is not None:
        return None, None
    if d["policy"] == naming.BACKBONE:
        return naming.BACKBONE, None
    if d["policy"] not in {k for k, *_ in METHODS}:
        return None, None
    return d["policy"], d["swept"]


def load_acc(cell_dir):
    """-> {method: {swept: mIoU}}, backbone mIoU (merged runs only)."""
    pts, base = {}, None
    for variant in sorted(os.listdir(cell_dir)):
        if "shard" in variant:
            continue
        rp = os.path.join(cell_dir, variant, "results.json")
        if not os.path.isfile(rp):
            continue
        try:
            r = json.load(open(rp))
        except Exception:
            continue
        k, sw = key_of(variant)
        if k is None or r.get("mIoU") is None:
            continue
        if k == naming.BACKBONE:
            base = float(r["mIoU"])
        else:
            pts.setdefault(k, {})[sw] = float(r["mIoU"])
    return pts, base


def load_lat(cell_dir):
    """-> {method: {swept: (ms, flops_enc_G, flops_total_G)}}, backbone.

    The tuple index is the x axis: 0 latency, 1 encoder FLOPs, 2 total
    FLOPs. Variants containing any name in `drop` are skipped.
    """
    pts, base = {}, None
    for variant in sorted(os.listdir(cell_dir)):
        rp = os.path.join(cell_dir, variant, "results.json")
        if not os.path.isfile(rp):
            continue
        try:
            r = json.load(open(rp))
        except Exception:
            continue
        k, sw = key_of(variant)
        if k is None:
            continue
        # mean over passes 2..N, Commit Gate included
        ms = frame_ms(r)
        met = r.get("metrics") or {}
        fl, ft = met.get("flops_encoder"), met.get("flops_total")
        if ms is None:
            continue
        val = (float(ms),
               float(fl) / 1e9 if fl else None,
               float(ft) / 1e9 if ft else None)
        if k == naming.BACKBONE:
            base = val
        else:
            pts.setdefault(k, {})[sw] = val
    return pts, base


def draw(ax, acc, acc_base, lat, lat_base, xidx, xlabel):
    ax.axhline(acc_base, color="k", ls="-", lw=1.2,
               label=f"backbone ({acc_base:.3f})")
    if lat_base:
        ax.plot(lat_base[xidx], acc_base, "k*", ms=14, zorder=5)
    # budget line, plus a faint -2 pt reference
    thr = acc_base - BUDGET
    ax.axhline(thr, color="k", ls="--", lw=1.1, alpha=0.8,
               label=f"-{BUDGET*100:.0f}pt budget")
    if BUDGET != 0.02:
        ax.axhline(acc_base - 0.02, color="k", ls=":", lw=0.8, alpha=0.35)
    for key, label, color, marker in METHODS:
        a, l = acc.get(key, {}), lat.get(key, {})
        # connect the points in sweep order, not along the cost axis
        xy = [(l[s][xidx], a[s], s) for s in sorted(a) if s in l
              and l[s][xidx] is not None]
        if not xy:
            continue
        ax.plot([x for x, _, _ in xy], [y for _, y, _ in xy],
                color=color, marker=marker, ms=5, lw=1.4,
                markeredgecolor="white", markeredgewidth=0.7,
                label=label)
        # ring the cheapest point on this axis that meets the budget
        ok = [(x, y) for x, y, _ in xy if y >= thr]
        if ok:
            px, py = min(ok)
            ax.scatter([px], [py], s=170, facecolors="none",
                       edgecolors=color, linewidths=2.2, zorder=6)
            ax.scatter([px], [py], s=170, facecolors="none",
                       edgecolors="white", linewidths=0.8, zorder=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mIoU")
    ax.grid(alpha=0.25)
    # x range: data span (backbone star included) plus a small pad
    xs = [p[xidx] for p in [lat_base] if p]
    for key, *_ in METHODS:
        a, l = acc.get(key, {}), lat.get(key, {})
        xs += [l[s][xidx] for s in a if s in l and l[s][xidx] is not None]
    if xs:
        lo, hi = min(xs), max(xs)
        pad = 0.04 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)


BUDGET = 0.01  # set from --budget in main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc_root",
                    default="output/outdoor/accuracy")
    ap.add_argument(
        "--lat_root",
        default="output/outdoor/latency")
    ap.add_argument("--budget", type=float, default=0.01)
    ap.add_argument("--out_prefix",
                    default="output/outdoor/pareto")
    args = ap.parse_args()
    global BUDGET
    BUDGET = args.budget

    cells = [(ds, bb) for ds in ("waymo", "nuscenes")
             for bb in ("utonia_dec", "litept_base")]

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    for j, (ds, bb) in enumerate(cells):
        acc, acc_base = load_acc(os.path.join(args.acc_root, ds, bb))
        lat, lat_base = load_lat(os.path.join(args.lat_root, ds, bb))
        if acc_base is None:
            continue
        for i, (xidx, xl) in enumerate(((0, "latency (ms/frame)"),
                                        (1, "encoder FLOPs (G/frame)"))):
            ax = axes[i][j]
            draw(ax, acc, acc_base, lat, lat_base, xidx, xl)
            ax.set_title(f"{ds} / {bb}", fontsize=9)
            if j == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Outdoor pareto — accuracy (full val) vs cost "
                 "(latency slice, dense attention)", fontsize=12)
    fig.tight_layout()
    out = f"{args.out_prefix}.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
