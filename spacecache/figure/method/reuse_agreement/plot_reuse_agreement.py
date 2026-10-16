"""Fig. 5 (Sec. 4.3): Reuse Agreement of SpaceCache entries.

  (a) confidence: RA by region confidence c(v) for the cached entries, the
      candidates (regions computed this frame on the miss subset, age 0)
      and all tokens, at the default lifetime L=8. Low confidence means low
      reusability for both populations the Commit Gate judges.
  (b) age: RA of the cached entries by entry age with an unbounded lifetime
      (L=inf), so every region stays cached. Reusability drops as the entry
      ages.

Both runs use theta=0, so no confidence eviction shapes the curves (d3+d4
SpaceCache, Waymo / Utonia, representative scenes, --reuse_agreement;
scripts/reproduce/reuse_agreement.sh). RA follows Appendix A.2. The panels share
the canvas and axes box of Fig. 2(b) when its .axes.json exists
(scripts/reproduce/motivation.sh).

    PYTHONPATH=. python spacecache/figure/method/reuse_agreement/plot_reuse_agreement.py
    PYTHONPATH=. python spacecache/figure/method/reuse_agreement/plot_reuse_agreement.py \\
        --depths 3,4 --max_age 20

Writes figure/method/reuse_agreement/{a_ra_conf,b_ra_age}.png (preview), .csv,
.caption.txt, and the vector data plane {a_ra_conf,b_ra_age}_plane.svg
(frame, grid, tick marks and curves, no text).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _ROOT)

from spacecache.figure.style import (                                # noqa: E402
    GRID, INK, INK2, PAPER_RC, SURFACE, add_font_args, box_axes, font_rc,
    ink_bbox, plane_only, save, scaled_rc, y_ticks_bounded,
)
from spacecache.experiments.reuse_agreement.analyze import (           # noqa: E402
    load_run, variant,
)

# colours and markers of the Fig. 2(b) panel (plot_stage_overlap.DS_CLR)
POP_STYLE = {
    "cached": ("#ff0000", "s", "Cached"),
    "candidate": ("#147df5", "^", "Candidate"),
    "all": ("#ffd300", "o", "All"),
}
# a second stage keeps its population's colour and switches to dashed
STAGE_LS = {3: "-", 4: (0, (4, 2))}
# canvas and axes box of Fig. 2(b), so the plot planes have the same size
MATCH_AXES = "figure/motivation/temporal_overlap/a_stage_overlap.axes.json"
MARKER_PT = 7.0 / 2 + 1.2         # marker radius + its edge, in points
FONT_FAMILY = "Aptos Narrow"


def _si(run):
    return {s: i for i, s in enumerate(run["stats"])}


def age_curve(run, depth, min_n, max_age):
    """RA of cache hits per entry age (1..max_age). -> (age, n, RA%)."""
    H = run["H"][f"H_t_d{depth}"]
    si = _si(run)
    h = H.sum(axis=(1, 2, 3))                          # (age, stat)
    ages = np.arange(1, min(max_age, h.shape[0] - 2) + 1)  # last = overflow
    n = h[ages, si["n"]]
    ra = 100 * h[ages, si["ra"]] / np.maximum(n, 1)
    keep = n >= min_n
    return ages[keep], n[keep], ra[keep]


def conf_curves(run, depth, bin_w, min_n):
    """RA per confidence bin for cached (age >= 1), candidate (age 0) and
    all tokens. -> {pop: (lo, hi, n, RA%)}."""
    H = run["H"][f"H_t_d{depth}"]
    si = _si(run)
    K = H.shape[1]
    step = max(int(round(bin_w * K)), 1)
    pops = dict(cached=H[1:].sum(axis=(0, 2, 3)),
                candidate=H[0].sum(axis=(1, 2)))
    pops["all"] = pops["cached"] + pops["candidate"]
    out = {}
    for name, h in pops.items():
        nb = K // step
        hh = h[:nb * step].reshape(nb, step, h.shape[1]).sum(axis=1)
        lo = np.arange(nb) * step / K
        n = hh[:, si["n"]]
        ra = 100 * hh[:, si["ra"]] / np.maximum(n, 1)
        keep = n >= min_n
        out[name] = (lo[keep], lo[keep] + step / K, n[keep], ra[keep])
    return out


def split_at(run, depth, theta):
    """RA below / above theta for cached and candidate (caption numbers)."""
    H = run["H"][f"H_t_d{depth}"]
    si = _si(run)
    kb = int(round(theta * H.shape[1]))
    res = {}
    for name, h in (("cached", H[1:].sum(axis=(0, 2, 3))),
                    ("candidate", H[0].sum(axis=(1, 2)))):
        n, ok = h[:, si["n"]], h[:, si["ra"]]
        res[name] = (100 * ok[:kb].sum() / max(n[:kb].sum(), 1),
                     100 * ok[kb:].sum() / max(n[kb:].sum(), 1),
                     100 * n[:kb].sum() / max(n.sum(), 1))
    return res


def _style_axes(ax):
    """box_axes, then open the top and right spines (as in Fig. 2(b))."""
    box_axes(ax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _plot(ax, x, y, pop, depth, label):
    color, marker, _ = POP_STYLE[pop]
    ax.plot(x, y, color=color, lw=2.6, ls=STAGE_LS.get(depth, "-"),
            marker=marker, markersize=7.0, markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=3, solid_capstyle="round",
            label=label, clip_on=False)


def _yticks(ax, ys, step):
    """The tick span is the axis range. It is widened by one step where an
    end marker would cross the frame, which would otherwise enlarge the
    plane's ink box."""
    ys = np.concatenate([np.asarray(y, float) for y in ys if len(y)])
    lo = step * np.floor(ys.min() / step)
    hi = step * np.ceil(ys.max() / step)
    h_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    for _ in range(4):
        per_pt = (hi - lo) / h_pt
        grew = False
        if ys.max() > hi - MARKER_PT * per_pt:
            hi += step
            grew = True
        if ys.min() < lo + MARKER_PT * per_pt:
            lo -= step
            grew = True
        if not grew:
            break
    y_ticks_bounded(ax, list(np.arange(lo, hi + step / 2, step)), 0.0)
    ax.set_yticklabels([f"{v:.0f}" for v in ax.get_yticks()])


def _legend(ax, loc, pad):
    ax.legend(loc=loc, borderaxespad=0.35, labelcolor=INK, framealpha=1.0,
              edgecolor=GRID, fancybox=False, borderpad=pad)


def _finish(fig, a, name, rows, header, caption):
    """Full panel (png preview + csv + caption), then the bare vector
    plane from the SAME figure so the two stay registered."""
    fmts = tuple(f.strip() for f in a.formats.split(",") if f.strip())
    written = save(fig, a.out, name, formats=fmts, rows=rows, header=header,
                   caption=caption, transparent=a.transparent, pad=a.pad,
                   close=False)
    plane_only(fig, "all", log=print)
    pbox = ink_bbox(fig, a.plane_pad)
    ppath = os.path.join(a.out, f"{name}_plane.svg")
    fig.savefig(ppath, bbox_inches=pbox, pad_inches=0.0, transparent=True)
    import matplotlib.pyplot as plt
    plt.close(fig)
    for p in written + [ppath]:
        print(f"[plot] wrote {p}")
    print(caption)


def _load(root, life):
    L = float("inf") if str(life) == "inf" else float(life)
    v = variant(L)                                     # theta = 0
    run = load_run(root, v)
    if run is None:
        raise SystemExit(f"missing run {v} under {root}: run "
                         f"scripts/reproduce/reuse_agreement.sh run")
    return run, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/reuse_agreement/"
                                      "waymo/utonia_dec")
    ap.add_argument("--out", default="figure/method/reuse_agreement")
    ap.add_argument("--age_life", default="inf",
                    help="lifetime L of the run for (b); inf = every region "
                         "stays cached")
    ap.add_argument("--max_age", type=int, default=15,
                    help="(b) x range. Past ~15 frames the L=inf curve is "
                         "a shrinking, selected population (only regions "
                         "kept in view that long; the sequences are ~30 "
                         "frames), so it is cut there")
    ap.add_argument("--conf_life", default="8",
                    help="lifetime L of the run for (a)")
    ap.add_argument("--depths", default="3",
                    help="stages to draw; 2nd stage is dashed")
    ap.add_argument("--theta", type=float, default=0.85)
    ap.add_argument("--conf_bin", type=float, default=0.05)
    ap.add_argument("--conf_min", type=float, default=0.3,
                    help="leftmost confidence shown")
    ap.add_argument("--min_n", type=int, default=500,
                    help="hide bins with fewer scored tokens")
    ap.add_argument("--ytick", type=float, default=5.0)
    # layout of the Fig. 2(b) panel
    ap.add_argument("--match_axes", default=MATCH_AXES,
                    help="take the canvas and the axes box from this "
                         ".axes.json (default: Fig. 2(b)); 'none' or a "
                         "missing file = --width/--height")
    ap.add_argument("--width", type=float, default=4.2)
    ap.add_argument("--height", type=float, default=3.4)
    ap.add_argument("--font_scale", type=float, default=2.2)
    ap.add_argument("--pad", type=float, default=0.10)
    ap.add_argument("--plane_pad", type=float, default=0.0)
    ap.add_argument("--legend_borderpad", type=float, default=0.10)
    ap.add_argument("--opaque", dest="transparent", action="store_false",
                    help="off-white ground on the png (default transparent)")
    ap.add_argument("--formats", default="png,csv,caption",
                    help="full-panel outputs; the vector plane "
                         "<name>_plane.svg is always written")
    add_font_args(ap)
    a = ap.parse_args()
    if a.font_family is None:
        a.font_family = FONT_FAMILY

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rc = scaled_rc(font_rc(PAPER_RC, a.font_family, a.font_file or ()),
                   a.font_scale)
    plt.rcParams.update(rc)
    depths = [int(d) for d in a.depths.split(",") if d.strip()]
    stage_tag = len(depths) > 1
    rect = None
    if (a.match_axes and a.match_axes != "none"
            and not os.path.isfile(a.match_axes)):
        print(f"[axes] {a.match_axes} not found (run "
              f"scripts/reproduce/motivation.sh first); using "
              f"--width/--height")
    elif a.match_axes and a.match_axes != "none":
        with open(a.match_axes) as fh:
            ref = json.load(fh)
        rect = ref["rect"]
        a.width, a.height = ref["figsize"]
        print(f"[axes] pinned to {a.match_axes}: canvas "
              f"{a.width:.3f} x {a.height:.3f} in, box "
              + " ".join(f"{v:.4f}" for v in rect))

    def new_panel():
        fig, ax = plt.subplots(figsize=(a.width, a.height))
        if rect is not None:
            ax.set_position(rect)
        else:
            fig.subplots_adjust(left=0.2, bottom=0.22, right=0.95, top=0.95)
        _style_axes(ax)
        return fig, ax

    # ---- (b) RA vs entry age, every region stays cached --------------------
    run_a, va = _load(a.root, a.age_life)
    fig, ax = new_panel()
    rows_a, cap_a, ys = [], [], []
    for d in depths:
        age, n, ra = age_curve(run_a, d, a.min_n, a.max_age)
        _plot(ax, age, ra, "cached", d, f"Stage {d}" if stage_tag
              else "Cached")
        ys.append(ra)
        rows_a += [[d, int(x), int(m), f"{y:.2f}"]
                   for x, m, y in zip(age, n, ra)]
        if len(ra):
            cap_a.append(f"Stage {d}: {ra[0]:.1f}% at age {int(age[0])} to "
                         f"{ra[-1]:.1f}% at age {int(age[-1])}")
    _yticks(ax, ys, a.ytick)
    ax.set_xlabel("Entry age (frames)")
    ax.set_ylabel("Reuse Agreement (%)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if stage_tag:
        _legend(ax, "upper right", a.legend_borderpad)
    L_txt = "∞" if str(a.age_life) == "inf" else a.age_life
    _finish(fig, a, "b_ra_age", rows_a, ["stage", "age", "n_tokens",
                                         "ra_pct"],
            f"Reuse Agreement between the matched cache content and the "
            f"fresh feature of the same region, by entry age. SpaceCache "
            f"runs with L={L_txt} and no confidence rule, so every region "
            f"stays cached, on Waymo with Utonia (representative scenes). "
            f"Reusability drops as the entry ages: " + "; ".join(cap_a) + ".")

    # ---- (a) RA vs confidence: cached / candidate / all --------------------
    run_b, vb = _load(a.root, a.conf_life)
    fig, ax = new_panel()
    rows_b, cap_b, ys = [], [], []
    for d in depths:
        cur = conf_curves(run_b, d, a.conf_bin, a.min_n)
        for pop in ("all", "cached", "candidate"):
            lo, hi, n, ra = cur[pop]
            m = lo >= a.conf_min - 1e-9
            lab = POP_STYLE[pop][2] + (f" (S{d})" if stage_tag else "")
            _plot(ax, 0.5 * (lo[m] + hi[m]), ra[m], pop, d, lab)
            ys.append(ra[m])
            rows_b += [[d, pop, f"{l:.2f}", f"{h:.2f}", int(k), f"{y:.2f}"]
                       for l, h, k, y in zip(lo, hi, n, ra)]
        sp = split_at(run_b, d, a.theta)
        cap_b.append(
            f"Stage {d}: below / above θ={a.theta:g}, cached "
            f"{sp['cached'][0]:.1f} / {sp['cached'][1]:.1f}%, candidate "
            f"{sp['candidate'][0]:.1f} / {sp['candidate'][1]:.1f}% "
            f"({sp['cached'][2]:.0f}% / {sp['candidate'][2]:.0f}% of the "
            f"tokens below θ)")
    ax.axvline(a.theta, color=INK2, lw=1.2, ls=(0, (3, 3)), zorder=1)
    ax.text(a.theta, 0.03, f" θ={a.theta:g}",
            transform=ax.get_xaxis_transform(), color=INK, ha="left",
            va="bottom", fontsize=plt.rcParams["legend.fontsize"])
    _yticks(ax, ys, a.ytick)
    ax.set_xlim(a.conf_min, 1.0)
    ax.set_xlabel("Region confidence")
    ax.set_ylabel("Reuse Agreement (%)")
    _legend(ax, "upper left", a.legend_borderpad)
    _finish(fig, a, "a_ra_conf", rows_b,
            ["stage", "population", "conf_lo", "conf_hi", "n_tokens",
             "ra_pct"],
            f"Reuse Agreement by region confidence for the cached entries "
            f"(served from SpaceCache), the candidates (regions computed "
            f"this frame on the cache misses), and all tokens, for "
            f"SpaceCache with L={a.conf_life} and no confidence rule on "
            f"Waymo with Utonia (representative scenes). "
            + "; ".join(cap_b) + ".")
    print(f"(a) {vb}\n(b) {va}")


if __name__ == "__main__":
    main()
