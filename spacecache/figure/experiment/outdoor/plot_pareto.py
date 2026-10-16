"""Outdoor Pareto figure: full-val mIoU against latency and FLOPs.

Two rows x four panels, FLOPs per frame (top) and latency (bottom), for
{Waymo, nuScenes} x {Utonia, LitePT}. Each curve rings its fewest-FLOPs
point within the 1-point budget (the Table 1 pick) in both rows. The
loaders and method styles are imported from
spacecache/experiments/plot_outdoor_cost_pareto.py.

    PYTHONPATH=. python spacecache/figure/experiment/outdoor/plot_pareto.py

Writes figure/experiment/outdoor/pareto.{png,pdf,csv}; `--plane` adds
pareto_plane.svg (the plot plane without text) and `--split` one svg per
panel.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
for _ in range(6):
    _ROOT = os.path.dirname(_ROOT)
    if os.path.isdir(os.path.join(_ROOT, "pointcept")):
        break
sys.path.insert(0, _ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from matplotlib.transforms import Bbox, offset_copy  # noqa: E402

from spacecache.common import naming  # noqa: E402
from spacecache.figure.style import (  # noqa: E402
    INK, INK2, PAPER_RC, add_font_args, box_axes, font_rc, ink_bbox,
    plane_only, save, scaled_rc,
)
from spacecache.experiments.plot_outdoor_cost_pareto import (  # noqa: E402
    METHODS, load_acc, load_lat,
)

CELLS = [(ds, bb) for ds in ("waymo", "nuscenes")
         for bb in ("utonia_dec", "litept_base")]

# top-row FLOPs: (index into load_lat's tuples, axis label). 'total'
# matches the latency row, which times the whole forward.
FLOPS = {"total": (2, "GFLOPs"),
         "encoder": (1, "encoder GFLOPs")}
DS_LABEL = {"waymo": "Waymo", "nuscenes": "nuScenes"}
BB_LABEL = {"utonia_dec": "Utonia", "litept_base": "LitePT"}

# line width and marker size in points
LW, MS = 1.6, 4.2

# long dash for the -2pt reference, distinct from the -1pt budget's "--"
DASH_2PT = (0, (6, 4))

# ring around each method's pick (fewest-FLOPs point within the budget):
# diameter and edge width in points, just outside the marker's white edge
RING_MS, RING_MEW = 5.0, 0.9

# matplotlib centres a marker's bounding box, not its ink, on the point; a
# triangle's centroid is off by a third of its half-height, so the ring is
# shifted by that much (in marker half-heights)
MARK_DY = {"^": -1 / 3, "v": 1 / 3}

# z-order overrides: the lighter SpaceCache (Stage-3) curve is drawn above
# Stage-3,4 where they overlap; the legend keeps the METHODS order
ZORDER = {naming.SPACECACHE_S3: 3.9, naming.SPACECACHE_S34: 3.8}

# fixed y window in mIoU for every panel, so equal drops look equal: YSPAN
# high with YHEAD of it above the baseline, one tick every YTICK
YSPAN, YHEAD, YTICK = 0.045, 0.006, 0.01


def split_svgs(fig, axes, out_dir, name, pad=0.0, suffix=""):
    """Save each panel as its own svg, cropped from the assembled figure.

    Cropping instead of redrawing keeps the axes size, line widths and
    text sizes identical across the files. Each crop is the panel's tight
    bbox (plus `pad` inches); the shared legend is in none of them.
    Returns [(path, bbox in inches)].
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()
    written = []
    for i, row in enumerate(("flops", "lat")):
        for j, (ds, bb) in enumerate(CELLS):
            ax = axes[i][j]
            bb_in = ax.get_tightbbox(rend).transformed(inv)
            box = Bbox.from_extents(bb_in.x0 - pad, bb_in.y0 - pad,
                                    bb_in.x1 + pad, bb_in.y1 + pad)
            path = os.path.join(
                out_dir, f"{name}_{ds}_{bb}_{row}{suffix}.svg")
            fig.savefig(path, bbox_inches=box, pad_inches=0.0,
                        transparent=True)
            written.append((path, box))
    return written


def draw(ax, acc, acc_base, lat, lat_base, xidx, budget, show_2pt,
         pick_idx=2):
    """One panel: reference lines, one curve per method, a ring per pick.

    The pick is the point with the fewest FLOPs (`pick_idx` into the
    load_lat tuples) within the budget, so both rows ring the same point.
    """
    # recessive reference lines: baseline, -budget and -2pt
    ax.axhline(acc_base, color=INK, ls="-", lw=0.9, alpha=0.75,
               label="baseline")
    ax.axhline(acc_base - budget, color=INK, ls="--", lw=0.8, alpha=0.55,
               label=f"-{budget * 100:.0f}pt")
    ax.axhline(acc_base - 0.02, color=INK, ls=DASH_2PT, lw=0.8, alpha=0.4,
               label="-2pt" if show_2pt else "_nolegend_")
    if lat_base:
        ax.plot(lat_base[xidx], acc_base, marker="*", ms=9,
                color=INK, markeredgecolor="white", markeredgewidth=0.5,
                zorder=5, label="_nolegend_")
    for zi, (key, label, color, marker) in enumerate(METHODS):
        a, l = acc.get(key, {}), lat.get(key, {})
        xy = [(l[s][xidx], a[s], s) for s in sorted(a) if s in l
              and l[s][xidx] is not None]
        if not xy:
            continue
        z = ZORDER.get(key, 3.0 + 0.1 * zi)
        ax.plot([x for x, _, _ in xy], [y for _, y, _ in xy],
                color=color, marker=marker, ms=MS, lw=LW,
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=z, label=label)
        ok = [s for _, y, s in xy
              if y >= acc_base - budget and l[s][pick_idx] is not None]
        if ok:
            s_pick = min(ok, key=lambda s: l[s][pick_idx])
            px, py = l[s_pick][xidx], a[s_pick]
            # plot (not scatter) so that ms is the ring's diameter in
            # points; the nudge is a point offset on the data transform,
            # independent of the axis limits
            tr = ax.transData
            dy = MARK_DY.get(marker, 0.0) * MS / 2.0
            if dy:
                tr = offset_copy(tr, fig=ax.figure, y=dy, units="points")
            ax.plot([px], [py], marker="o", ms=RING_MS, ls="none",
                    markerfacecolor="none", markeredgecolor=color,
                    markeredgewidth=RING_MEW, zorder=6, transform=tr,
                    label="_nolegend_")
    ax.set_xlim(None)
    # x range from the baseline and the points inside the y window only;
    # points below the window would only widen the axis with empty space
    y_lo = acc_base + YHEAD - YSPAN
    xs = [p[xidx] for p in [lat_base] if p]
    for key, *_ in METHODS:
        a, l = acc.get(key, {}), lat.get(key, {})
        xs += [l[s][xidx] for s in a if s in l and l[s][xidx] is not None
               and a[s] >= y_lo]
    if xs:
        lo, hi = min(xs), max(xs)
        pad = 0.05 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
    # y: fixed YSPAN window anchored at the baseline
    ax.set_ylim(y_lo, acc_base + YHEAD)
    ax.yaxis.set_major_locator(MultipleLocator(YTICK))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc_root",
                    default="output/outdoor/accuracy")
    ap.add_argument("--lat_root",
                    default="output/outdoor/latency")
    ap.add_argument("--budget", type=float, default=0.01)
    ap.add_argument("--flops", default="total", choices=list(FLOPS),
                    help="FLOPs on the top row and of the ringed pick: "
                         "total (whole forward, like the latency row) or "
                         "encoder only")
    ap.add_argument("--out", default="figure/experiment/outdoor")
    ap.add_argument("--name", default="pareto")
    ap.add_argument("--width", type=float, default=8.4,
                    help="authored width in inches")
    ap.add_argument("--row_h", type=float, default=0.72,
                    help="height of one panel row in inches")
    ap.add_argument("--font_scale", type=float, default=1.0)
    ap.add_argument("--plane", action="store_true",
                    help="also write <name>_plane.svg: the plot plane "
                         "without labels, titles or legend")
    ap.add_argument("--plane_text", default="all",
                    choices=["outside", "all"],
                    help="text hidden in the plane: all, or only the text "
                         "outside the axes")
    ap.add_argument("--split", action="store_true",
                    help="also write each panel as "
                         "<name>_<ds>_<bb>_{lat,flops}.svg (planes with "
                         "--plane)")
    ap.add_argument("--plane_pad", type=float, default=0.0,
                    help="margin around the plane and split crops, in "
                         "inches")
    add_font_args(ap)
    args = ap.parse_args()

    plt.rcParams.update(font_rc(scaled_rc(PAPER_RC, args.font_scale),
                                args.font_family, args.font_file))

    # margins in inches: legend row on top, x-label row at the bottom
    TOP, BOT, HGAP, WGAP = 0.62, 0.30, 0.40, 0.30
    W = args.width
    H = TOP + BOT + 2 * args.row_h + HGAP
    fig, axes = plt.subplots(2, 4, figsize=(W, H))
    fig.patch.set_facecolor("#fcfcfb")
    fig.subplots_adjust(top=1 - TOP / H, bottom=BOT / H,
                        left=0.055, right=0.995,
                        hspace=HGAP / args.row_h, wspace=WGAP)

    for j, (ds, bb) in enumerate(CELLS):
        acc, acc_base = load_acc(os.path.join(args.acc_root, ds, bb))
        lat, lat_base = load_lat(os.path.join(args.lat_root, ds, bb))
        if acc_base is None:
            continue
        for i, (xidx, xl) in enumerate((FLOPS[args.flops],
                                        (0, "latency (ms)"))):
            ax = axes[i][j]
            box_axes(ax, grid_axis="both")
            draw(ax, acc, acc_base, lat, lat_base, xidx, args.budget,
                 show_2pt=(i == 0 and j == 0),
                 pick_idx=FLOPS[args.flops][0])
            if i == 0:
                ax.set_title(f"{DS_LABEL[ds]} · {BB_LABEL[bb]}",
                             fontsize=plt.rcParams["axes.titlesize"],
                             fontweight="bold", color=INK, pad=3)
            # both rows carry their x label, as in the paper figure
            ax.set_xlabel(xl, color=INK)
            if j == 0:
                ax.set_ylabel("mIoU", color=INK)

    # one shared legend in the top margin
    handles, labels = [], []
    for j in range(4):  # collect from whichever panel drew everything
        h, l = axes[0][j].get_legend_handles_labels()
        if len(h) > len(handles):
            handles, labels = h, l
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.53, 1.0), ncol=len(labels),
               frameon=False, fontsize=6.5,
               columnspacing=1.0, handlelength=1.6)

    rows = []
    for ds, bb in CELLS:
        acc, acc_base = load_acc(os.path.join(args.acc_root, ds, bb))
        lat, _ = load_lat(os.path.join(args.lat_root, ds, bb))
        for key, label, *_ in METHODS:
            for s in sorted(acc.get(key, {})):
                if s in lat.get(key, {}):
                    ms, fl, ft = lat[key][s]
                    rows.append((f"{ds}/{bb}", label, s, acc[key][s],
                                 f"{ms:.2f}",
                                 f"{fl:.1f}" if fl else "",
                                 f"{ft:.1f}" if ft else ""))

    save(fig, args.out, args.name, formats=("png", "pdf", "csv"),
         rows=rows,
         header=("cell", "method", "sweep", "mIoU", "ms",
                 "gflops_enc", "gflops_total"),
         close=not (args.plane or args.split))
    print(f"wrote {args.out}/{args.name}.{{png,pdf,csv}}")
    # the plane is cut from the same figure, so it stays registered with
    # the full panel
    if args.plane:
        plane_only(fig, args.plane_text, log=print)
        pbox = ink_bbox(fig, args.plane_pad)
        ppath = os.path.join(args.out, f"{args.name}_plane.svg")
        fig.savefig(ppath, bbox_inches=pbox, pad_inches=0.0,
                    transparent=True)
        print(f"wrote {ppath}  "
              f"({pbox.width:.3f} x {pbox.height:.3f} in)")
    # after plane_only, so --plane --split writes bare panels
    if args.split:
        for path, box in split_svgs(fig, axes, args.out, args.name,
                                    pad=args.plane_pad,
                                    suffix="_plane" if args.plane else ""):
            print(f"wrote {path}  "
                  f"({box.width:.3f} x {box.height:.3f} in)")
    if args.plane or args.split:
        plt.close(fig)


if __name__ == "__main__":
    main()
