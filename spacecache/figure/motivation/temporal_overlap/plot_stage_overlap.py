"""Plot the stage-wise temporal overlap of `stage_overlap.py` (Fig. 2b).

One line per dataset over the encoder stages: the share of a frame's
stage-s tokens whose world cell the previous frame already occupied
(per-pair median, or mean with `--stat mean`), with the interquartile
range across pairs as a band.

    PYTHONPATH=. python \\
        spacecache/figure/motivation/temporal_overlap/plot_stage_overlap.py \\
        --in_dir output/motivation/stage_overlap

Reads every `*.json` under `--in_dir` (or the `--json` paths) and writes
`<out>/<name>.{png,pdf,csv,caption.txt}` plus `<name>.axes.json` (default
`figure/motivation/temporal_overlap/a_stage_overlap`).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from spacecache.figure.style import (  # noqa: E402
    GRID, INK, PAPER_RC, SURFACE, box_axes, dataset_style, pt, save,
    scaled_rc, short_label, add_font_args, add_type_args, font_rc,
    type_rc, ink_bbox, fit_axes_width,
    nudge_legend, snap_legend_right, y_ticks_bounded, drop_corner_tick,
    plane_only,
)


# Draw order, matched against the stem of each run's json.
ORDER = ("scannet", "waymo", "nuscenes", "kitti", "semantic_kitti")

# Line colour per dataset, keyed by the lowercased short label.
DS_CLR = {"scannet": "#147df5", "waymo": "#ff0000", "nuscenes": "#ffd300"}


def ds_color(label, fallback):
    return DS_CLR.get(short_label(label).lower(), fallback)


def load_runs(paths):
    runs = []
    for path in paths:
        with open(path) as f:
            run = json.load(f)
        run["_stem"] = os.path.splitext(os.path.basename(path))[0]
        runs.append(run)

    def rank(run):
        stem = run["_stem"].lower()
        for i, key in enumerate(ORDER):
            if key in stem:
                return (i, stem)
        return (len(ORDER), stem)

    return sorted(runs, key=rank)


def _spread(values, min_gap):
    """Push end-of-line labels at least `min_gap` apart, keeping their
    order."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for k in range(1, len(order)):
        lo, hi = order[k - 1], order[k]
        if out[hi] - out[lo] < min_gap:
            out[hi] = out[lo] + min_gap
    return out


STAT_KEY = {"median": "p50", "mean": "mean"}


def draw_overlap(ax, runs, n_stages, stat="median", label_gap=6.0,
                 mark_stage=None, legend=True, end_labels=True,
                 ylabel="Temporal overlap  (%)", label_size=None,
                 tick_step=20, legend_size=None, legend_loc="upper left",
                 leg_kw=None, legend_borderpad=0.20, ylim=(0.0, 100.0),
                 ymargin=0.0, x_end_pad=0.62):
    """Draw the overlap curves on `ax`; returns the last-stage values."""
    x = np.arange(n_stages)
    label_size = label_size or pt(1.0)
    box_axes(ax)
    # open frame: no top/right spines
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if mark_stage is not None and 0 <= mark_stage < n_stages:
        ax.axvspan(mark_stage - 0.42, mark_stage + 0.42,
                   color=GRID, alpha=0.32, lw=0, zorder=0)

    ends = []
    for i, run in enumerate(runs):
        # colour follows the dataset name, not the position in `runs`
        color, marker = dataset_style(run["dataset"]["label"],
                                      fallback_index=i)
        color = ds_color(run["dataset"]["label"], color)
        stages = run["stages"][:n_stages]
        key = STAT_KEY[stat]
        center = np.array([100 * s["overlap"][key] for s in stages])
        p25 = np.array([100 * s["overlap"]["p25"] for s in stages])
        p75 = np.array([100 * s["overlap"]["p75"] for s in stages])
        ax.fill_between(x, p25, p75, color=color, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, center, color=color, lw=2.6, marker=marker,
                markersize=7.0, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3,
                solid_capstyle="round",
                label=short_label(run["dataset"]["label"]), clip_on=False)
        ends.append(center[-1])

    if end_labels:
        for i, (run, y_text) in enumerate(zip(runs, _spread(ends, label_gap))):
            # with a legend the end label is the value only; without one it
            # also names the series
            text = (f"{ends[i]:.0f}%" if legend_loc != "none" and len(runs) > 1
                    else f"{short_label(run['dataset']['label'])}  "
                         f"{ends[i]:.0f}%")
            color, _ = dataset_style(run["dataset"]["label"],
                                     fallback_index=i)
            color = ds_color(run["dataset"]["label"], color)
            ax.annotate(text, xy=(x[-1], ends[i]),
                        xytext=(x[-1] + 0.16, y_text), color=color,
                        fontsize=label_size, fontweight="bold", va="center",
                        ha="left", annotation_clip=False,
                        path_effects=[pe.withStroke(linewidth=1.8,
                                                    foreground=SURFACE)])

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{d}" for d in range(n_stages)])
    ax.set_xlabel("Stage")
    # the end labels live inside the axes: room for them comes from the x
    # range past the last stage, not from a shorter axes box
    ax.set_xlim(-0.32, n_stages - 1 + (x_end_pad if end_labels else 0.32))

    # `ylim` is the tick span; the drawn limits extend `ymargin` beyond it
    lo, hi = ylim
    ticks = [v for v in range(0, 101, tick_step)
             if lo - 1e-9 <= v <= hi + 1e-9]
    y_ticks_bounded(ax, ticks, ymargin)
    ax.set_yticklabels([f"{v}" for v in ticks])
    ax.set_ylabel(ylabel)

    if legend and len(runs) > 1 and legend_loc != "none":
        if legend_loc == "outside":
            # one row above the axes
            ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01),
                      ncol=len(runs), frameon=False,
                      labelcolor=INK, fontsize=legend_size,
                      **(leg_kw or {}))
        else:
            ax.legend(loc=legend_loc, borderaxespad=0.35, labelcolor=INK,
                      framealpha=1.0, edgecolor=GRID, fancybox=False,
                      borderpad=legend_borderpad, fontsize=legend_size,
                      **(leg_kw or {}))
    return ends


def build_panel(runs, args):
    n_stages = min(len(r["stages"]) for r in runs)
    if any(len(r["stages"]) != n_stages for r in runs):
        print(f"[warn] runs disagree on stage count; drawing the first "
              f"{n_stages}")

    cells = [[s["cell_m"] for s in r["stages"][:n_stages]] for r in runs]
    shared_cells = all(
        np.allclose(c, cells[0], rtol=1e-6, atol=1e-9) for c in cells
    )

    # the y title runs along the axes height: raise the canvas height until
    # it fits (unless the axes box is pinned)
    ylabel = args.ylabel or "Temporal overlap  (%)"
    axis_pt = PAPER_RC["axes.labelsize"] * args.font_scale
    tick_pt = PAPER_RC["xtick.labelsize"] * args.font_scale
    label_in = len(ylabel) * 0.62 * axis_pt / 72.0 + 0.25
    margins_in = (axis_pt + tick_pt) * 1.9 / 72.0 + 0.30
    height = args.height if args.axes_rect else max(args.height,
                                                    label_in + margins_in)
    if height > args.height + 1e-6:
        print(f"  [fit] y title {ylabel!r} needs {label_in:.2f}in along the "
              f"axes at {axis_pt:.1f}pt; height "
              f"{args.height:.2f} -> {height:.2f}in")

    fig, ax = plt.subplots(figsize=(args.width, height))
    fig.patch.set_facecolor(SURFACE)
    # legend spacings are in font-size units, so they follow --font_scale
    draw_overlap(ax, runs, n_stages, stat=args.stat, label_gap=args.label_gap,
                 mark_stage=args.mark_stage,
                 end_labels=not args.no_end_labels,
                 legend_loc=args.legend_loc,
                 legend_borderpad=args.legend_borderpad,
                 ylim=args.ylim_pair, ymargin=args.ymargin,
                 x_end_pad=args.x_end_pad,
                 leg_kw=dict(handlelength=args.legend_handlelen,
                             handletextpad=args.legend_handlepad,
                             columnspacing=args.legend_colspace,
                             labelspacing=args.legend_labelspace),
                 **({"ylabel": args.ylabel} if args.ylabel else {}))

    if args.title:
        ax.set_title(args.title, fontsize=pt(1.33), fontweight="bold",
                     color=INK, loc="left", pad=6)
    if args.axes_rect:
        ax.set_position(list(args.axes_rect))
        # centre-anchor the axis titles so their length cannot move them
        for axis, xy in zip((ax.yaxis, ax.xaxis),
                            getattr(args, "axes_labels", (None, None))):
            if xy:
                lab = axis.get_label()
                lab.set_ha("center")
                lab.set_va("center")
                axis.set_label_coords(*xy)
    else:
        fig.tight_layout(pad=0.4)
    return fig, n_stages, shared_cells


def build_rows(runs, n_stages):
    header = ["dataset", "stage", "cell_m", "overlap_mean", "overlap_p25",
              "overlap_p50", "overlap_p75", "overlap_token_weighted",
              "tokens_per_frame", "n_pairs"]
    rows = []
    for run in runs:
        for s in run["stages"][:n_stages]:
            rows.append([
                run["dataset"]["label"], s["depth"], f"{s['cell_m']:.4f}",
                f"{s['overlap']['mean']:.4f}", f"{s['overlap']['p25']:.4f}",
                f"{s['overlap']['p50']:.4f}", f"{s['overlap']['p75']:.4f}",
                f"{s['overlap_weighted']:.4f}", f"{s['tokens_b_mean']:.0f}",
                s["n_pairs"],
            ])
    return header, rows


def motion_split(run, depth, thr):
    """(stationary share, stationary mean, moving mean) of the overlap at
    one stage, with pairs split by ego displacement at `thr` metres."""
    pairs = [p for p in (run.get("pairs") or [])
             if p.get("ego_m") is not None]
    if not pairs:
        return None
    still = [p["per_depth"][depth]["overlap"] for p in pairs
             if p["ego_m"] < thr]
    moving = [p["per_depth"][depth]["overlap"] for p in pairs
              if p["ego_m"] >= thr]
    if not still or not moving:
        return None
    return (len(still) / len(pairs), float(np.mean(still)),
            float(np.mean(moving)))


def build_caption(runs, n_stages, args):
    """Caption draft filled in with the numbers of these runs."""
    shared_cells = all(
        abs(r["stages"][d]["cell_m"] - runs[0]["stages"][d]["cell_m"]) < 1e-9
        for r in runs for d in range(n_stages)
    )
    gaps = [r["sampling"].get("measured_gap_s_median")
            or r["sampling"].get("frame_gap_s") for r in runs]
    gaps = [g for g in gaps if g]
    one_gap = bool(gaps) and max(gaps) - min(gaps) < 0.05
    if one_gap:
        gap_txt = f"{gaps[0]:.1f} s apart"
    else:
        gap_txt = ", ".join(
            f"{r['dataset']['label']} {g:.2f} s"
            for r, g in zip(runs, gaps)) + " apart"
    gap = gaps[0] if gaps else 0.5
    last = n_stages - 1
    cell0 = runs[0]["stages"][0]["cell_m"]
    cell_last = runs[0]["stages"][last]["cell_m"]

    key = STAT_KEY[args.stat]
    per_dataset = []
    for run in runs:
        s0 = 100 * run["stages"][0]["overlap"][key]
        sl = 100 * run["stages"][last]["overlap"][key]
        per_dataset.append(
            f"{run['dataset']['label']} {s0:.0f}% -> {sl:.0f}%")
    n_pairs = sum(r["sampling"]["n_pairs"] for r in runs)
    n_drives = sum(r["dataset"]["n_drives_sampled"] for r in runs)

    cell_list = ", ".join(f"{s_['cell_m']:.2f}"
                          for s_ in runs[0]["stages"][:n_stages])
    by_range = {}
    for r in runs:
        span = (r["stages"][0]["cell_m"], r["stages"][last]["cell_m"])
        by_range.setdefault(span, []).append(r["dataset"]["label"])
    range_list = "; ".join(
        f"{'/'.join(labels)} {lo:.2f}-{hi:.2f} m"
        for (lo, hi), labels in by_range.items())
    where = (f"({cell0:.2f} m cells)" if shared_cells
             else "(the input voxel)")
    grows = (f"pooling coarsens the cell to {cell_last:.2f} m by stage "
             f"{last}" if shared_cells
             else f"pooling coarsens the cell stage by stage")
    caption = (
        f"(a) Stage-wise world overlap between consecutive frames of a "
        f"stream ({gap_txt}). Both frames of a pair are placed in one world "
        f"frame by their ego poses; at each encoder stage we measure the "
        f"fraction of the current frame's tokens that fall in a world cell "
        f"the previous frame already occupied. Stage s matches on a cell of "
        + (f"base_voxel x 2^s ({cell_list} m for S0..S{last}). "
           if shared_cells else f"base_voxel x 2^s ({range_list}). ")
        + f"At the input resolution "
        f"{where} that fraction is small on the outdoor streams — a lidar "
        f"sweep is sparse, and two sweeps rarely sample the same cell — but "
        f"{grows}, and the same "
        f"two frames are then found to share much of their token set "
        f"({'; '.join(per_dataset)}). The redundancy is present at the "
        f"input; it only becomes addressable with depth. Lines are per-pair "
        f"{args.stat}s over {n_pairs} frame pairs from {n_drives} "
        f"sequences, "
        f"bands the interquartile range across pairs."
    )

    numbers = ["", f"per-stage {args.stat} / mean / token-weighted (%):"]
    for run in runs:
        cells = "  ".join(
            f"S{s['depth']}={100 * s['overlap'][key]:.1f}"
            f"/{100 * s['overlap']['mean']:.1f}"
            f"/{100 * s['overlap_weighted']:.1f}"
            for s in run["stages"][:n_stages])
        numbers.append(
            f"  {run['dataset']['label']:<14} {cells}")
        numbers.append(
            f"  {'':<14} [{run['sampling']['n_pairs']} pairs, "
            f"{run['dataset']['n_drives_sampled']} sequences"
            + (f", sensor {run['ego']['p50']:.1f} m/pair median]"
               if run.get("ego") else ", sensor motion n/a]"))
        split = motion_split(run, 0, args.stationary_m)
        if split:
            share, still, moving = split
            numbers.append(
                f"  {'':<14} S0 by motion: {100 * share:.0f}% of pairs move "
                f"< {args.stationary_m} m (stopped) and overlap "
                f"{100 * still:.0f}%; the moving rest overlap "
                f"{100 * moving:.0f}%")
    return caption + "\n" + "\n".join(numbers)


def main():
    ap = add_type_args(add_font_args(argparse.ArgumentParser()))
    ap.add_argument("--json", action="append", default=None,
                    help="Run json; repeatable (default: every *.json "
                         "under --in_dir).")
    ap.add_argument("--in_dir", default="output/motivation/stage_overlap")
    ap.add_argument("--out", default="figure/motivation/temporal_overlap")
    ap.add_argument("--name", default="a_stage_overlap")
    ap.add_argument("--formats", default="png,pdf,csv,caption")
    ap.add_argument("--width", type=float, default=3.6)
    ap.add_argument("--height", type=float, default=2.6)
    ap.add_argument("--title", default="")
    ap.add_argument("--stat", default="median", choices=("median", "mean"),
                    help="Line statistic over pairs; both are written to "
                         "the CSV.")
    ap.add_argument("--stationary_m", type=float, default=0.1,
                    help="Ego displacement (m) below which a pair counts "
                         "as stopped in the caption's motion split.")
    ap.add_argument("--label_gap", type=float, default=6.0,
                    help="Minimum vertical gap (percentage points) between "
                         "the end labels.")
    ap.add_argument("--mark_stage", type=int, default=None,
                    help="Shade this stage.")
    ap.add_argument("--no_end_labels", action="store_true",
                    help="Do not print the value at the end of each line.")
    ap.add_argument("--font_scale", type=float, default=1.0,
                    help="Multiply every point size by this; use authored "
                         "width / displayed width to cancel LaTeX scaling.")
    ap.add_argument("--ylabel", default=None,
                    help="Override the y label (default: \"Temporal "
                         "overlap  (%%)\").")
    ap.add_argument("--ylim", default="0,100", metavar="LO,HI",
                    help="Tick span; the y spine ends at these ticks and "
                         "the limits extend --ymargin beyond them.")
    ap.add_argument("--ymargin", type=float, default=0.0,
                    help="Limits beyond the outermost ticks, as a fraction "
                         "of the tick span.")
    ap.add_argument("--legend_dx_px", type=float, default=0.0,
                    help="Shift the legend right by this many pixels of "
                         "the saved 300-dpi image.")
    ap.add_argument("--legend_dy_px", type=float, default=0.0,
                    help="Shift the legend up by this many saved pixels.")
    ap.add_argument("--x_end_pad", type=float, default=0.62,
                    help="Empty x range past the last stage, in stage "
                         "widths, for the end labels.")
    ap.add_argument("--plane", action="store_true",
                    help="Also write <name>_plane.svg: the plot area "
                         "without labels, title and legend.")
    ap.add_argument("--plane_text", default="outside",
                    choices=["outside", "all"],
                    help="Text removed from the plane: text outside the "
                         "axes, or all text.")
    ap.add_argument("--plane_pad", type=float, default=0.0,
                    help="Margin around the plane, in inches.")
    ap.add_argument("--fit_width", action="store_true",
                    help="With --match_axes, narrow the plot until its ink "
                         "fits the reference crop.")
    ap.add_argument("--legend_right", type=float, nargs="?", const=0.0,
                    default=None, metavar="MARGIN_IN",
                    help="Move the legend so its right edge sits MARGIN_IN "
                         "inches inside the crop's right edge (bare flag: "
                         "flush).")
    ap.add_argument("--legend_handlelen", type=float, default=0.9,
                    help="Length of the line sample in the legend, in "
                         "font-size units (matplotlib default 2.0).")
    ap.add_argument("--legend_handlepad", type=float, default=0.25,
                    help="Marker-to-label gap.")
    ap.add_argument("--legend_colspace", type=float, default=0.55,
                    help="Gap between legend columns.")
    ap.add_argument("--legend_labelspace", type=float, default=0.18,
                    help="Gap between legend rows.")
    ap.add_argument("--legend_borderpad", type=float, default=0.20,
                    help="Gap between the legend frame and its contents.")
    ap.add_argument("--legend_loc", default="upper left",
                    help="matplotlib legend loc, 'outside' for one row "
                         "above the axes, or 'none'.")
    ap.add_argument("--match_axes", default=None, metavar="PATH",
                    help="Take the canvas size, axes box, title positions "
                         "and crop from another panel's .axes.json "
                         "(overrides --axes_rect).")
    ap.add_argument("--axes_rect", type=float, nargs=4, default=None,
                    metavar=("L", "B", "W", "H"),
                    help="Pin the axes box to these figure fractions "
                         "instead of using tight_layout.")
    ap.add_argument("--bbox", default="tight", choices=("tight", "none"),
                    help="tight crops to the ink (or the --match_axes "
                         "crop); none saves at exactly --width x --height.")
    ap.add_argument("--pad_in", type=float, default=0.02,
                    help="Margin around the ink for the tight crop, in "
                         "inches.")
    ap.add_argument("--transparent", action="store_true",
                    help="Save with a transparent background.")
    args = ap.parse_args()
    args.ylim_pair = tuple(
        float(v) for v in args.ylim.replace(" ", "").split(","))
    if len(args.ylim_pair) != 2:
        raise SystemExit(f"--ylim wants \"lo,hi\", got {args.ylim!r}")
    if args.match_axes:
        with open(args.match_axes) as fh:
            ref = json.load(fh)
        args.axes_rect = ref["rect"]
        args.axes_labels = (ref.get("ylabel_xy"), ref.get("xlabel_xy"))
        args.axes_ref = ref
        # `rect` is in figure fractions, so the canvas size must match too
        if max(abs(a - b) for a, b in
               zip(ref["figsize"], (args.width, args.height))) > 1e-6:
            print(f"  [axes] canvas {args.width} x {args.height} -> "
                  f"{ref['figsize'][0]} x {ref['figsize'][1]} "
                  f"(adopted from {args.match_axes})")
            args.width, args.height = ref["figsize"]
        print(f"  [axes] pinned to {args.match_axes}: "
              + " ".join(f"{v:.4f}" for v in args.axes_rect))
    rc = type_rc(font_rc(PAPER_RC, args.font_family, args.font_file),
                 args.axis_pt, args.tick_pt)
    plt.rcParams.update(scaled_rc(rc, args.font_scale))
    # build_panel reads point sizes from PAPER_RC for its height fit
    PAPER_RC.update(rc)

    paths = args.json or sorted(glob.glob(os.path.join(args.in_dir, "*.json")))
    if not paths:
        raise SystemExit(
            f"no run json under {args.in_dir} — run "
            f"spacecache/figure/motivation/temporal_overlap/stage_overlap.py "
            f"first")
    runs = load_runs(paths)
    print(f"[plot] {len(runs)} run(s): "
          + ", ".join(r["dataset"]["label"] for r in runs))

    fig, n_stages, _ = build_panel(runs, args)
    header, rows = build_rows(runs, n_stages)
    caption = build_caption(runs, n_stages, args)
    # with --match_axes the reference crop is reused, so both panels are
    # saved at the same size
    ref_bbox = None
    _ref = getattr(args, "axes_ref", None)
    if _ref and _ref.get("bbox_in"):
        ref_bbox = Bbox.from_extents(*_ref["bbox_in"])
        if args.fit_width:
            fit_axes_width(fig, ref_bbox, log=print)
    # drop the corner tick before the crop is measured
    fig.canvas.draw()
    if drop_corner_tick(fig.axes[0]):
        print("  [ticks] dropped the tick mark sitting on the axes corner")
    crop = ref_bbox if ref_bbox is not None else ink_bbox(fig, args.pad_in)
    if args.legend_right is not None:
        snap_legend_right(fig, crop, args.legend_right, log=print)
    nudge_legend(fig, args.legend_dx_px, args.legend_dy_px)

    os.makedirs(args.out, exist_ok=True)
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        ref_ax = fig.axes[0]
        inv = ref_ax.transAxes.inverted()

        def _centre(lab):
            bb = lab.get_window_extent(rend)
            return [float(v) for v in
                    inv.transform((0.5 * (bb.x0 + bb.x1),
                                   0.5 * (bb.y0 + bb.y1)))]

        with open(os.path.join(args.out, f"{args.name}.axes.json"), "w") as fh:
            # layout sidecar for --match_axes: axes box, canvas, axis-title
            # centres (axes coords) and crop box (inches)
            json.dump({"rect": list(ref_ax.get_position().bounds),
                       "figsize": list(fig.get_size_inches()),
                       "ylabel_xy": _centre(ref_ax.yaxis.get_label()),
                       "xlabel_xy": _centre(ref_ax.xaxis.get_label()),
                       "bbox_in": list(crop.extents)}, fh)
    except OSError as e:
        print(f"  [axes] could not write the sidecar: {e}")
    written = save(fig, args.out, args.name,
                   formats=tuple(f.strip() for f in args.formats.split(",")),
                   rows=rows, header=header, caption=caption,
                   transparent=args.transparent, pad=0.0,
                   close=False,
                   bbox=(crop if args.bbox == "tight" else None))
    # the plane comes from the same figure, so it registers with the panel
    if args.plane:
        plane_only(fig, args.plane_text, log=print)
        pbox = ink_bbox(fig, args.plane_pad)
        ppath = os.path.join(args.out, f"{args.name}_plane.svg")
        fig.savefig(ppath, bbox_inches=pbox, pad_inches=0.0,
                    transparent=True)
        print(f"[plot] wrote {ppath}  "
              f"({pbox.width:.3f} x {pbox.height:.3f} in)")
    plt.close(fig)

    for path in written:
        print(f"[plot] wrote {path}")
    print()
    print(caption)


if __name__ == "__main__":
    main()
