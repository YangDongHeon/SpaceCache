"""Plot prototype agreement against encoder stage (Fig. 2c).

Reads the runs written by `cone_affinity.py` and draws one line per
(dataset, backbone): colour and marker follow the dataset, line style the
backbone. With `--by_model`, one line per backbone (mean over datasets),
with each dataset's value as a point. The default metric is soft
agreement over all reused tokens, token-weighted over pairs.

    PYTHONPATH=. python \\
        spacecache/figure/motivation/reuse_agreement/plot_cone_trend_multi.py \\
        --runs output/motivation/cone_affinity --by_model

Writes `<out>/<name>.{png,pdf,csv,caption.txt}` plus `<name>.axes.json`
(default `figure/motivation/reuse_agreement/b_cone_trend_multi`).
"""

from __future__ import annotations

import argparse
import glob
import json
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
import numpy as np  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from spacecache.figure.style import (  # noqa: E402
    DATA_SERIES, GRID, INK, INK2, SURFACE, PAPER_RC, box_axes,
    dataset_style, save, scaled_rc, short_label, add_font_args,
    add_type_args, font_rc, type_rc, ink_bbox, fit_axes_width,
    nudge_legend, snap_legend_right, y_ticks_bounded, drop_corner_tick,
    plane_only,
)

METRICS = {
    "soft": "Prototype agreement (%)",
    "top1": "Prototype agreement, Top-1 (%)",
    "top2": "Prototype agreement, Top-2 (%)",
    # 100 - soft
    "disagree": "Prototype disagreement (%)",
}

MODEL_LS = ("-", "--", "-.")

# Draw order; a missing run leaves a gap instead of shifting the others.
DEFAULT_DATASETS = "ScanNet-video,Waymo,nuScenes"
DEFAULT_MODELS = "Utonia,LitePT-B"


def load_runs(paths):
    runs = []
    for p in paths:
        for f in (sorted(glob.glob(os.path.join(p, "*.json")))
                  if os.path.isdir(p) else [p]):
            with open(f) as fh:
                r = json.load(fh)
            r["_path"] = f
            runs.append(r)
    return runs


def model_label(run):
    """(label, guessed): the run's `model.label`, or the last `_`-separated
    token of the file stem when the json carries no label."""
    lbl = (run.get("model") or {}).get("label")
    if lbl:
        return lbl, False
    stem = os.path.splitext(os.path.basename(run["_path"]))[0]
    return stem.rsplit("_", 1)[-1], True


def _agree_stages(run, subset):
    """Stages of `run` that carry an agreement block for `subset`."""
    return [s for s in run["stages"] if s.get("agreement", {}).get(subset)]


def _agree_pairs(run, depth, subset, key):
    """(per-pair values, per-pair token counts) of one agreement metric, or
    (None, None) if a pair record lacks it."""
    vs, ws = [], []
    for p in run.get("pairs") or []:
        a = ((p.get("depths") or {}).get(str(depth)) or {}).get("agreement")
        a = (a or {}).get(subset)
        if not a or key not in a:
            return None, None
        vs.append(a[key])
        ws.append(a["n"])
    if not vs:
        return None, None
    return np.asarray(vs, dtype=float), np.asarray(ws, dtype=float)


def _weighted_quantile(v, w, q):
    """Quantile `q` of `v` under weights `w` (mid-cumulative-weight
    interpolation)."""
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, v))


def agree_value(run, stage, subset, key, weight="token"):
    """Agreement at one stage: token-weighted mean over pairs (`token`) or
    plain mean over pairs (`pair`)."""
    if weight == "token":
        v, w = _agree_pairs(run, stage["depth"], subset, key)
        if v is not None and w.sum() > 0:
            return float((v * w).sum() / w.sum())
    return stage["agreement"][subset][key]["mean"]


def agree_spread(run, stage, subset, key, weight="token"):
    """(p25, p75) across pairs, weighted like `agree_value`."""
    if weight == "token":
        v, w = _agree_pairs(run, stage["depth"], subset, key)
        if v is not None and w.sum() > 0:
            return (_weighted_quantile(v, w, 0.25),
                    _weighted_quantile(v, w, 0.75))
    q = stage["agreement"][subset][key]
    return q["p25"], q["p75"]


def _spread(values, min_gap, lo, hi):
    """Push end-of-line labels at least `min_gap` apart, keeping their
    order, and keep them within [lo, hi]."""
    out = np.array(values, dtype=float)
    order = np.argsort(out)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    shift = max(0.0, out.max() - hi)
    if shift:
        out -= shift
    return np.clip(out, lo, hi)


def n_cones(run, stage, rule="pure", variant="reused"):
    """Mean number of cones per pair at `stage`."""
    v = stage.get("rules", {}).get(rule, {}).get(variant) or {}
    return float(v.get("n_cones_mean", float("nan")))


def main():
    ap = add_type_args(add_font_args(argparse.ArgumentParser()))
    ap.add_argument("--runs", nargs="+",
                    default=["output/motivation/cone_affinity"],
                    help="Run json files, or directories scanned for "
                         "*.json.")
    ap.add_argument("--datasets", default=DEFAULT_DATASETS,
                    help="Dataset labels (figure_label) in draw order.")
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help="Model labels (model_label) in draw order; they "
                         "set the line style.")
    ap.add_argument("--metric", default="soft", choices=sorted(METRICS),
                    help="Agreement metric; disagree = 100 - soft.")
    ap.add_argument("--subset", default="all",
                    choices=["all", "pure", "mixed"],
                    help="Reused tokens scored: all, or only single-class "
                         "(pure) or mixed ones.")
    ap.add_argument("--weight", default="token", choices=["token", "pair"],
                    help="token: one vote per token; pair: one vote per "
                         "frame pair.")
    ap.add_argument("--by_model", action="store_true",
                    help="One line per model (mean over datasets), with "
                         "each dataset's value as a point.")
    ap.add_argument("--band", action="store_true",
                    help="Draw each line's p25-p75 range across pairs "
                         "(not with --by_model).")
    ap.add_argument("--chance", action="store_true",
                    help="Dotted 1 / n_cones line per dataset, the "
                         "agreement of independent assignments (not with "
                         "--by_model).")
    ap.add_argument("--out", default="figure/motivation/reuse_agreement")
    ap.add_argument("--name", default="b_cone_trend_multi")
    ap.add_argument("--width", type=float, default=3.6,
                    help="Canvas width in inches.")
    ap.add_argument("--height", type=float, default=2.6,
                    help="Minimum canvas height in inches; raised to fit "
                         "the y title.")
    ap.add_argument("--ylabel", default=None,
                    help="Override the y axis title.")
    ap.add_argument("--font_scale", type=float, default=1.0,
                    help="Multiply every point size by this; use authored "
                         "width / displayed width to cancel LaTeX scaling.")
    ap.add_argument("--mark_scale", type=float, default=1.0,
                    help="Scale the markers in --by_model mode.")
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
    ap.add_argument("--transparent", action="store_true",
                    help="Save with a transparent background.")
    ap.add_argument("--pad_in", type=float, default=0.02,
                    help="Margin around the ink for the tight crop, in "
                         "inches.")
    ap.add_argument("--legend_dx_px", type=float, default=0.0,
                    help="Shift the legend right by this many pixels of "
                         "the saved 300-dpi image.")
    ap.add_argument("--legend_dy_px", type=float, default=0.0,
                    help="Shift the legend up by this many saved pixels.")
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
                    help="Gap between the two legend columns.")
    ap.add_argument("--legend_labelspace", type=float, default=0.18,
                    help="Gap between legend rows.")
    ap.add_argument("--legend_borderpad", type=float, default=0.20,
                    help="Gap between the legend frame and its contents.")
    ap.add_argument("--legend_loc", default="lower right",
                    help="matplotlib legend loc, or 'outside' for a legend "
                         "above the axes.")
    ap.add_argument("--yticks", type=int, default=4,
                    help="Approximate number of y ticks.")
    ap.add_argument("--ylim", default=None,
                    help="\"lo,hi\" y range (default: fitted to the data); "
                         "the tick span when --ymargin is set.")
    ap.add_argument("--ymargin", type=float, default=0.0,
                    help="If > 0, end the y spine at the outermost ticks "
                         "and put the limits this fraction of the tick span "
                         "beyond them.")
    ap.add_argument("--end_labels", action="store_true", default=False,
                    help="Print the value at the end of each line.")
    ap.add_argument("--label_gap", type=float, default=3.0,
                    help="Minimum gap between end labels, in percentage "
                         "points.")
    args = ap.parse_args()
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

    ds_order = [s.strip() for s in args.datasets.split(",") if s.strip()]
    m_order = [s.strip() for s in args.models.split(",") if s.strip()]
    if len(ds_order) > len(DATA_SERIES):
        raise SystemExit(
            f"{len(ds_order)} datasets exceeds the validated palette "
            f"({len(DATA_SERIES)}). A 5th dataset is not a generated hue — "
            f"facet the panel instead.")
    # colours are resolved by dataset name, not by position
    ds_style = {ds: dataset_style(ds, fallback_index=i)
                for i, ds in enumerate(ds_order)}
    if len(m_order) > len(MODEL_LS):
        raise SystemExit(f"{len(m_order)} models exceeds the linestyle set")

    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no runs under {args.runs}")

    want = {m.lower(): m for m in m_order}
    by_key = {}
    for r in runs:
        ds = (r.get("dataset") or {}).get("label", "?")
        ml, guessed = model_label(r)
        ml = want.get(ml.lower(), ml)
        if guessed:
            print(f"  [warn] {r['_path']}: no model.label in the json; read "
                  f"{ml!r} off the filename. Add `model_label` to its config "
                  f"and re-run to make it explicit.")
        if (ds, ml) in by_key:
            print(f"  [warn] duplicate ({ds}, {ml}); keeping "
                  f"{by_key[(ds, ml)]['_path']}, ignoring {r['_path']}")
            continue
        by_key[(ds, ml)] = r

    # PAPER_RC scaled by --font_scale, as in plot_stage_overlap.py
    base = type_rc(font_rc(PAPER_RC, args.font_family, args.font_file),
                   args.axis_pt, args.tick_pt)
    rc = scaled_rc(base, args.font_scale)
    plt.rcParams.update(rc)
    PAPER_RC.update(base)     # the height-fit below reads sizes from it
    axis_pt = PAPER_RC["axes.labelsize"] * args.font_scale
    tick_pt = PAPER_RC["xtick.labelsize"] * args.font_scale

    # legend spacings are in font-size units, so they follow --font_scale
    leg_kw = dict(handlelength=args.legend_handlelen,
                  handletextpad=args.legend_handlepad,
                  columnspacing=args.legend_colspace,
                  labelspacing=args.legend_labelspace)

    # the y title runs along the axes height: raise the canvas height until
    # it fits (unless the axes box is pinned)
    ylabel = args.ylabel or METRICS[args.metric]
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
    box_axes(ax)
    # open frame: no top/right spines
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    csv_rows, drawn, chances, band_extents = [], [], [], []
    n_stage = 0
    base_metric = "soft" if args.metric == "disagree" else args.metric

    if args.by_model:
        # one line per model (mean over datasets); colour = model
        m_clr = ("#ff0000", "#147df5")
        for mi, ml in enumerate(m_order):
            per_run = []
            runs_used = []
            for ds in ds_order:
                run = by_key.get((ds, ml))
                if run is None:
                    continue
                stages = _agree_stages(run, args.subset)
                if not stages:
                    continue
                per_run.append({s["depth"]:
                                100.0 * agree_value(run, s, args.subset,
                                                    base_metric,
                                                    args.weight)
                                for s in stages})
                runs_used.append((ds, run))
            if not per_run:
                print(f"  [miss] {ml}: no runs")
                continue
            depths = sorted({d for v in per_run for d in v})
            xs = np.arange(len(depths))
            # per-dataset values as dark points: marker = dataset, edge
            # colour = model
            jit = np.linspace(-0.03, 0.03, len(per_run))
            ds_mark = {"ScanNet": "X", "Waymo": "P", "nuScenes": "v"}
            for (ds, _), v, j in zip(runs_used, per_run, jit):
                ds_marker = ds_mark.get(short_label(ds), "X")
                pts = [v.get(d, np.nan) for d in depths]
                if args.metric == "disagree":
                    pts = [100.0 - p for p in pts]
                ax.scatter(xs + j, pts, s=(5.0 * args.mark_scale) ** 2,
                           facecolor="#1a1a1a",
                           edgecolor=m_clr[mi],
                           marker=ds_marker,
                           alpha=1.0, linewidths=0.6, zorder=3)
            M = np.array([[v.get(d, np.nan) for d in depths]
                          for v in per_run])
            mean = np.nanmean(M, axis=0)
            lo = np.nanmin(M, axis=0)
            hi = np.nanmax(M, axis=0)
            if args.metric == "disagree":
                mean, lo, hi = 100.0 - mean, 100.0 - hi, 100.0 - lo
            color = m_clr[mi]
            xs = np.arange(len(depths))
            n_stage = max(n_stage, len(depths))
            band_extents.extend([lo, hi])
            # mean line, drawn like the lines of plot_stage_overlap.py
            ax.plot(xs, mean, color=color, lw=2.6, ls=MODEL_LS[mi],
                    marker="o", ms=7.0 * args.mark_scale,
                    markerfacecolor=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.2,
                    alpha=0.85,
                    zorder=5, solid_capstyle="round", dash_capstyle="round")
            stage_like = [{"depth": d} for d in depths]
            drawn.append((f"{len(per_run)} datasets", ml, stage_like,
                          mean, runs_used[0][1]))
            for d, m, l, h in zip(depths, mean, lo, hi):
                csv_rows.append(("mean-of-datasets", ml, f"S{d}",
                                 round(m, 2), round(h - l, 2), "",
                                 sum(r["sampling"]["n_pairs"]
                                     for _, r in runs_used),
                                 sum(r["sampling"]["n_drives"]
                                     for _, r in runs_used)))
            print(f"  {ml:9s} mean over {len(per_run)} datasets "
                  + "  ".join(f"S{d}={m:.0f}" for d, m in zip(depths, mean)))
    else:
        for di, ds in enumerate(ds_order):
            color, marker = ds_style[ds]
            for mi, ml in enumerate(m_order):
                run = by_key.get((ds, ml))
                if run is None:
                    print(f"  [miss] {ds} / {ml}")
                    continue
                stages = _agree_stages(run, args.subset)
                if not stages:
                    print(f"  [miss] {ds} / {ml}: no agreement[{args.subset}] "
                          f"block in {run['_path']}")
                    continue
                n_stage = max(n_stage, len(stages))
                base_metric = ("soft" if args.metric == "disagree"
                               else args.metric)
                y = np.array([100.0 * agree_value(run, s, args.subset,
                                                  base_metric, args.weight)
                              for s in stages])
                if args.metric == "disagree":
                    y = 100.0 - y
                xs = np.arange(len(stages))
                if args.band:
                    lo, hi = zip(*(agree_spread(run, s, args.subset,
                                                base_metric, args.weight)
                                   for s in stages))
                    if args.metric == "disagree":
                        lo, hi = 1.0 - np.array(hi), 1.0 - np.array(lo)
                    ax.fill_between(xs, 100 * np.array(lo), 100 * np.array(hi),
                                    color=color, alpha=0.12, lw=0, zorder=2)
                ax.plot(xs, y, color=color, lw=2.6, ls=MODEL_LS[mi],
                        marker=marker, ms=7.0, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, zorder=3, solid_capstyle="round",
                        dash_capstyle="round")
                drawn.append((ds, ml, stages, y, run))
                nc = np.array([n_cones(run, s) for s in stages])
                chances.append((ds, color,
                                100.0 / np.where(nc > 0, nc, np.nan)))
                for s, v, c in zip(stages, y, nc):
                    csv_rows.append((ds, ml, f"S{s['depth']}", round(v, 2),
                                     round(c, 2),
                                     s["agreement"][args.subset]["n_tokens"],
                                     run["sampling"]["n_pairs"],
                                     run["sampling"]["n_drives"]))
                print(f"  {ds:14s} {ml:9s} "
                      + "  ".join(f"S{s['depth']}={v:.0f}"
                                  for s, v in zip(stages, y))
                      + f"   ({run['sampling']['n_pairs']} pairs / "
                        f"{run['sampling']['n_drives']} seqs)")

    if not drawn:
        raise SystemExit("nothing drawn — check --datasets / --models against "
                         "the runs' figure_label / model_label")

    if args.chance:
        seen = set()
        for ds, color, ch in chances:
            if ds in seen or not np.isfinite(ch).any():
                continue
            seen.add(ds)
            ax.plot(np.arange(len(ch)), ch, color=color, lw=0.9, ls=":",
                    alpha=0.85, zorder=1)

    x = np.arange(n_stage)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}" for i in x])
    ax.set_xlabel("Stage")
    ax.set_ylabel(ylabel, color=INK)
    ax.xaxis.label.set_color(INK)
    ax.set_xlim(-0.18, n_stage - 0.82)
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=args.yticks, integer=True))
    if args.ylim:
        lo, hi = (float(v) for v in args.ylim.split(","))
    else:
        allv = np.concatenate([y for *_, y, _ in drawn]
                              + band_extents)
        # round out to multiples of 5, with room for the markers
        lo = max(0.0, np.floor((np.nanmin(allv) - 4) / 5) * 5)
        hi = min(100.0, np.ceil((np.nanmax(allv) + 3) / 5) * 5)
    ax.set_ylim(lo, hi)
    # optional: end the spine at the outermost ticks and put the limits
    # `ymargin` of the tick span beyond them
    if args.ymargin:
        fig.canvas.draw()
        ticks = [t for t in ax.get_yticks() if lo - 1e-9 <= t <= hi + 1e-9]
        if len(ticks) >= 2:
            lo, hi = y_ticks_bounded(ax, ticks, args.ymargin)

    if args.end_labels:
        ends = [(float(y[-1]),
                 m_clr[m_order.index(ml)] if args.by_model
                 else ds_style[ds][0],
                 len(y) - 1)
                for ds, ml, _, y, _ in drawn]
        ys = _spread([e[0] for e in ends], args.label_gap, lo, hi)
        for (v, color, xi), yt in zip(ends, ys):
            ax.annotate(f"{v:.0f}", xy=(xi, v), xytext=(xi + 0.12, yt),
                        color=color, fontsize=tick_pt, fontweight="bold",
                        va="center", ha="left", zorder=6,
                        annotation_clip=False)

    if args.by_model:
        # legend: marker shape = dataset, coloured line = model
        ds_mark = {"ScanNet": "X", "Waymo": "P", "nuScenes": "v"}
        ds_h = [plt.Line2D([], [], color="none", lw=0,
                           marker=ds_mark.get(short_label(ds), "X"),
                           ms=5.0,
                           markerfacecolor="#1a1a1a",
                           markeredgecolor=INK, markeredgewidth=0.4)
                for ds in ds_order]
        m_h = [plt.Line2D([], [], color=m_clr[i], lw=2.6,
                          ls=MODEL_LS[i], marker="o", ms=6.0,
                          markerfacecolor=m_clr[i],
                          markeredgecolor=SURFACE, markeredgewidth=1.2,
                          alpha=0.85)
               for i, _ in enumerate(m_order)]
        handles = list(ds_h) + list(m_h)
        labels = [short_label(d) for d in ds_order] + list(m_order)
        leg_fs = PAPER_RC["legend.fontsize"] * args.font_scale
        if args.legend_loc == "outside":
            pad = plt.Line2D([], [], lw=0, marker="", alpha=0)
            inter, inter_labels = [], []
            for i in range(max(len(ds_h), len(m_h))):
                if i < len(ds_h):
                    inter.append(ds_h[i])
                    inter_labels.append(short_label(ds_order[i]))
                if i < len(m_h):
                    inter.append(m_h[i])
                    inter_labels.append(m_order[i])
            while len(inter) % len(ds_h):
                inter.append(pad)
                inter_labels.append("")
            leg = ax.legend(inter, inter_labels, loc="lower left",
                            bbox_to_anchor=(-0.02, 1.01),
                            ncol=len(ds_h), frameon=False,
                            labelcolor=INK, fontsize=leg_fs, **leg_kw)
        else:
            leg = ax.legend(handles, labels, loc=args.legend_loc, ncol=2,
                            borderaxespad=0.3, labelcolor=INK,
                            framealpha=1.0, edgecolor=GRID, fancybox=False,
                            borderpad=args.legend_borderpad,
                            fontsize=leg_fs, **leg_kw)
    else:
        ds_h = [plt.Line2D([], [], color=ds_style[ds][0], lw=2.6,
                           marker=ds_style[ds][1], ms=7.0,
                           markeredgecolor=SURFACE, markeredgewidth=1.2)
                for ds in ds_order]
        m_h = [plt.Line2D([], [], color=INK2, lw=2.6, ls=MODEL_LS[i])
               for i, _ in enumerate(m_order)]
        handles = list(ds_h) + list(m_h)
        labels = [short_label(d) for d in ds_order] + list(m_order)
    if not args.by_model and args.legend_loc == "outside":
        # legend above the axes: dataset and model handles are interleaved
        # (and padded) so the column-major fill puts datasets on the first
        # row and models on the second
        pad = plt.Line2D([], [], lw=0, marker="", alpha=0)
        inter, inter_labels = [], []
        for i in range(max(len(ds_h), len(m_h))):
            if i < len(ds_h):
                inter.append(ds_h[i])
                inter_labels.append(short_label(ds_order[i]))
            if i < len(m_h):
                inter.append(m_h[i])
                inter_labels.append(m_order[i])
        while len(inter) % len(ds_h):
            inter.append(pad)
            inter_labels.append("")
        ax.legend(inter, inter_labels, loc="lower left",
                  bbox_to_anchor=(-0.02, 1.01), ncol=len(ds_h), frameon=False,
                  labelcolor=INK, **leg_kw)
    elif not args.by_model:
        # two columns: datasets in the first, models in the second
        ax.legend(handles, labels, loc=args.legend_loc, ncol=2,
                  borderaxespad=0.35, labelcolor=INK,
                  framealpha=1.0, edgecolor=GRID, fancybox=False,
                  borderpad=args.legend_borderpad, **leg_kw)

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
        fig.tight_layout()

    s3 = []
    for ds, ml, stages, y, _ in drawn:
        i = next((j for j, s in enumerate(stages) if s["depth"] == 3), None)
        if i is not None:
            v = y[i] if args.metric != "disagree" else 100.0 - y[i]
            s3.append(f"{ds} {ml} {v:.0f}%")
    if args.by_model:
        cones = ""
        cones_sentence = ""
    else:
        cones = "; ".join(
            f"{ds} {n_cones(r, st[0]):.1f}->{n_cones(r, st[-1]):.1f}"
            for ds, st, r in
            {d: (d, st, r) for d, _, st, _, r in drawn}.values())
        cones_sentence = (
            f" Agreement is over the cones present in a pair, and that "
            f"count differs by dataset (S0->S4: {cones}), so a "
            f"cross-dataset reading of the level is weaker than the "
            f"reading of the trend.")
    caption = (
        f"Prototype agreement against encoder stage for {len(drawn)} "
        f"dataset-backbone combinations: the share of a token's class "
        f"affinity that survives being served from the previous frame's "
        f"cache instead of recomputed ({args.metric}"
        + (", i.e. 100 - soft agreement" if args.metric == "disagree" else "")
        + f", {args.subset} cells, "
        f"one token one vote). At Stage 3: " + "; ".join(s3) + ". Each run is "
        + ", ".join(sorted({f"{r['sampling']['n_pairs']} pairs over "
                            f"{r['sampling']['n_drives']} sequences"
                            for *_, r in drawn}))
        + ", sampled evenly across the full validation split."
        + cones_sentence)
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
    written = save(fig, args.out, args.name, rows=csv_rows,
                   header=("dataset", "model", "stage", args.metric,
                           "n_cones_mean", "n_tokens", "n_pairs",
                           "n_sequences"),
                   caption=caption, transparent=args.transparent,
                   pad=0.0, close=False,
                   bbox=(crop if args.bbox == "tight" else None))
    # the plane comes from the same figure, so it registers with the panel
    if args.plane:
        plane_only(fig, args.plane_text, log=print)
        pbox = ink_bbox(fig, args.plane_pad)
        ppath = os.path.join(args.out, f"{args.name}_plane.svg")
        fig.savefig(ppath, bbox_inches=pbox, pad_inches=0.0,
                    transparent=True)
        written.append(ppath)
        print(f"  [plane] {pbox.width:.3f} x {pbox.height:.3f} in")
    plt.close(fig)
    for p in written:
        print("wrote", p)
    print("\n" + caption)


if __name__ == "__main__":
    main()
