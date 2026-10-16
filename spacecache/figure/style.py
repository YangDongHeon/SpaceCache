"""Shared matplotlib style for the paper figures.

Colour tokens and palettes (NPG10-derived series and ramps, Okabe-Ito
dataset colours), rc dicts, argparse helpers for fonts and point sizes,
and small layout helpers (crop boxes, legend placement, bare plot planes)
used by the scripts under spacecache/figure/.
"""

from __future__ import annotations

import os


# off-white surface, primary / secondary ink, recessive grid
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdcd8"

# ---------------------------------------------------------------------------
#  Palettes
# ---------------------------------------------------------------------------

# Nature Publishing Group palette (ggsci::pal_npg("nrc")); MODEL_SERIES,
# SEQUENTIAL, RAMP_BLUE and NEUTRAL are derived from it.
NPG10 = (
    "#E64B35",   # 0  red
    "#4DBBD5",   # 1  cyan
    "#00A087",   # 2  teal
    "#3C5488",   # 3  navy
    "#F39B7F",   # 4  salmon
    "#8491B4",   # 5  slate
    "#91D1C2",   # 6  mint
    "#DC0000",   # 7  crimson
    "#7E6148",   # 8  brown
    "#B09C85",   # 9  tan
)

# Okabe-Ito qualitative set (colour-blind safe) for a categorical axis such
# as datasets or backbones; each series also carries a marker.
SERIES = (
    ("#0072B2", "o"),   # blue
    ("#D55E00", "s"),   # vermillion
    ("#009E73", "^"),   # bluish green
    ("#CC79A7", "D"),   # reddish purple
)

# Sequential ramp for magnitudes (heatmaps): near-white -> NPG cyan -> NPG
# navy. Luminance decreases monotonically, so the ramp reads as an order in
# greyscale and under colour-vision deficiencies.
SEQUENTIAL = ("#F8FCFD", "#D1EDF4", "#A3DCE9", "#74CADE", "#4CB6D1",
              "#4795B9", "#4175A0", "#3C5488")


def seq_cmap(name="seq"):
    """Colormap interpolating SEQUENTIAL."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, list(SEQUENTIAL))


def series_style(i):
    """(colour, marker) of series i, in fixed order and never cycled."""
    if i >= len(SERIES):
        raise ValueError(
            f"series {i} exceeds the validated palette ({len(SERIES)} "
            f"entries). A 5th series is not a generated hue — facet the "
            f"panel or fold the extra series into small multiples."
        )
    return SERIES[i]


# ---------------------------------------------------------------------------
#  Paper look: serif labels, a full box frame with a dashed grid (box_axes)
#  and the MODEL_SERIES / DATA_SERIES colours. Panels opt in via PAPER_RC.
# ---------------------------------------------------------------------------

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman",
                   "Liberation Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 7.5,
    "axes.labelsize": 8.5,
    "axes.labelweight": "bold",
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    # opaque legend, so the grid does not show through its text when
    # save(transparent=True) drops the background
    "legend.facecolor": "white",
    "pdf.fonttype": 42,      # embed TrueType instead of Type 3 fonts
    "ps.fonttype": 42,
}

# Model series: two NPG families, dark member first (navy, red). The light
# members (slate, salmon) mark a second variant of the same model.
MODEL_SERIES = (
    (NPG10[3], "o"),   # navy   #3C5488
    (NPG10[0], "s"),   # red    #E64B35
    (NPG10[5], "^"),   # slate  #8491B4 — light member of the navy family
    (NPG10[4], "D"),   # salmon #F39B7F — light member of the red family
)

# Dataset series: the Okabe-Ito anchors. They are more saturated than the
# NPG entries, which keeps thin lines and small markers readable, and the
# blue / vermillion / green triple is the most separable one under
# deuteranopia and protanopia.
DATA_SERIES = (
    ("#0072B2", "o"),   # blue          (Okabe-Ito)
    ("#D55E00", "s"),   # vermillion    (Okabe-Ito)
    ("#009E73", "^"),   # bluish green  (Okabe-Ito)
    ("#CC79A7", "D"),   # reddish purple(Okabe-Ito)
)


# Ordered ramp for a stack of pipeline parts: tints of NPG navy at
# 22 / 45 / 72 / 100 %, with lightness carrying the order. NEUTRAL (NPG
# tan) is the residual bucket.
RAMP_BLUE = ("#D4D9E5", "#A7B2C9", "#7384A9", "#3C5488")
NEUTRAL = NPG10[9]

# Six-step navy ramp for the six encoder stages (embedding, enc0..enc4).
# It is darker than RAMP_BLUE so that its lightest step still stands out
# against SURFACE.
RAMP_BLUE6 = ("#A6B1C8", "#8E9CB7", "#7586A6",
              "#5C7095", "#455B84", "#2E4068")


# ---------------------------------------------------------------------------
#  Dataset identity, shared by the motivation panels
# ---------------------------------------------------------------------------

# Colour per dataset name (not per rank), so adding a dataset never
# repaints another.
DATASET_SERIES = {
    "nuscenes": DATA_SERIES[0],        # blue           #0072B2
    "waymo": DATA_SERIES[1],           # vermillion     #D55E00
    "scannet": DATA_SERIES[2],         # bluish green   #009E73
    "kitti": DATA_SERIES[3],           # reddish purple #CC79A7
}

# Per-dataset six-step ordinal ramps (light -> dark; NPG navy / red / teal)
# for stacks of the six encoder stages drawn one row per dataset. They
# group rows by hue but are not separable from each other at a fixed step;
# use DATASET_SERIES to tell datasets apart by colour alone.
DATASET_RAMP6 = {
    "nuscenes": ("#A6B1C8", "#8C9AB9", "#7182A7",
                 "#566B95", "#3C5488", "#2A3C60"),
    "waymo": ("#EC9E8E", "#E47A63", "#E64B35",
              "#C13A26", "#94301F", "#6B2214"),
    "scannet": ("#72C2AF", "#4EB299", "#26A182",
                "#1B8469", "#146953", "#0D4C3B"),
}


def _dataset_key(label):
    """Dataset key of a display label ("ScanNet-video" -> "scannet"), by
    lowercase substring; None if unknown."""
    low = str(label).lower()
    for key in ("nuscenes", "waymo", "semantic_kitti", "kitti", "scannet"):
        if key in low:
            return "kitti" if key == "semantic_kitti" else key
    return None


def dataset_style(label, fallback_index=None):
    """(colour, marker) of a dataset, by name.

    An unknown label raises KeyError, or gets DATA_SERIES[fallback_index]
    when `fallback_index` is given.
    """
    key = _dataset_key(label)
    if key is not None:
        return DATASET_SERIES[key]
    if fallback_index is None:
        raise KeyError(
            f"no colour pinned for dataset {label!r}. Add it to "
            f"DATASET_SERIES in spacecache/figure/style.py so every panel "
            f"gives it the same one.")
    return DATA_SERIES[fallback_index % len(DATA_SERIES)]


def short_label(label):
    """Display form of a dataset label, without the "-video" split suffix."""
    return str(label).replace("-video", "")


def dataset_ramp(label, n=6):
    """First n steps (light -> dark) of the dataset's ramp; RAMP_BLUE6 for a
    dataset without one."""
    key = _dataset_key(label)
    ramp = DATASET_RAMP6.get(key, RAMP_BLUE6)
    if len(ramp) < n:
        raise ValueError(f"ramp for {label!r} has {len(ramp)} steps, need {n}")
    return ramp[:n]


# rc keys holding point sizes. scaled_rc() scales them; sizes set by hand go
# through pt() so that they follow the installed rc.
_RC_SIZE_KEYS = ("font.size", "axes.labelsize", "axes.titlesize",
                 "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
                 "figure.titlesize")


def scaled_rc(base, scale=1.0):
    """Copy of `base` with every point size multiplied by `scale`.

    With scale = authored width / displayed width, the text of a figure
    that LaTeX shrinks arrives on the page at the sizes given in `base`.
    """
    rc = dict(base)
    for k in _RC_SIZE_KEYS:
        if isinstance(rc.get(k), (int, float)):
            rc[k] = rc[k] * scale
    return rc


def add_font_args(ap):
    """Add the shared --font_family / --font_file options to `ap`."""
    ap.add_argument("--font_family", default=None,
                    help="font family for the whole figure (default: the "
                         "PAPER_RC serif stack)")
    ap.add_argument("--font_file", nargs="*", default=None, metavar="PATH",
                    help="TTF/OTF files, or directories of them, to "
                         "register for this run")
    return ap


def ink_bbox(fig, pad=0.0):
    """The figure's drawn extent in inches, padded by `pad` (a crop box)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    from matplotlib.transforms import Bbox
    return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad,
                             bb.x1 + pad, bb.y1 + pad)


def fit_axes_width(fig, target, tol=0.004, iters=6, log=None):
    """Narrow the first axes until the figure's ink fits inside `target`.

    For panels cropped to one shared box while labels overhang the right
    edge of the axes: only the plot width shrinks; the left edge, bottom
    and height are kept. Returns the inches taken off.
    """
    ax = fig.axes[0]
    w_in = fig.get_size_inches()[0]
    shrunk = 0.0
    for _ in range(iters):
        over = ink_bbox(fig).x1 - target.x1
        if over <= tol:
            break
        rect = list(ax.get_position().bounds)
        rect[2] = max(rect[2] - over / w_in, 0.05)
        ax.set_position(rect)
        shrunk += over
    if log:
        log(f"  [axes] narrowed the plot by {shrunk:.3f} in so the ink "
            f"fits the shared crop box")
    return shrunk


def nudge_legend(fig, dx_px=0.0, dy_px=0.0, dpi=300):
    """Shift the first axes' legend by (dx_px, dy_px) pixels at `dpi`.

    The legend is re-anchored to its own box, shifted, with
    `borderaxespad` set to zero; otherwise the pad, now measured from that
    box, would pull the legend inward.
    """
    if not dx_px and not dy_px:
        return
    ax = fig.axes[0]
    leg = ax.get_legend()
    if leg is None:
        return
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    k = fig.dpi / float(dpi)            # saved pixels -> display pixels
    dx, dy = dx_px * k, dy_px * k
    moved = Bbox.from_extents(bb.x0 + dx, bb.y0 + dy, bb.x1 + dx, bb.y1 + dy)
    leg.borderaxespad = 0.0
    leg.set_bbox_to_anchor(moved.transformed(ax.transAxes.inverted()),
                           transform=ax.transAxes)


def plane_only(fig, drop_text="outside", log=None):
    """Hide a figure's text furniture, leaving only the plot plane.

    Axis labels, tick labels, titles, legends and figure-level texts and
    artists are hidden; spines, tick marks, grid and data stay.
    drop_text="all" also hides the text inside the axes. Modifies `fig` in
    place (save the full panel first) and returns the number of artists
    hidden.
    """
    n = 0
    for ax in fig.axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.tick_params(labelbottom=False, labelleft=False,
                       labeltop=False, labelright=False)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_visible(False)
            n += 1
        if drop_text == "all":
            for t in ax.texts:
                t.set_visible(False)
                n += 1
    for group in (getattr(fig, "legends", []), fig.texts, fig.artists):
        for art in list(group):
            art.set_visible(False)
            n += 1
    if log:
        log(f"  [plane] hid {n} pieces of furniture; text inside the "
            f"axes {'dropped' if drop_text == 'all' else 'kept'}")
    return n


def drop_corner_tick(ax, tol=1e-6):
    """Hide the y tick mark at the lower y limit, keeping its label.

    A mark on the corner reads as the x spine overshooting the y spine.
    No-op if no tick sits on the lower limit; returns whether a mark was
    hidden.
    """
    lo, hi = ax.get_ylim()
    span = abs(hi - lo) or 1.0
    ticks = ax.yaxis.get_major_ticks()
    for t, v in zip(ticks, ax.get_yticks()):
        if abs(v - lo) <= tol * span:
            t.tick1line.set_visible(False)
            t.tick2line.set_visible(False)
            return True
    return False


def y_ticks_bounded(ax, ticks, margin=0.05):
    """Set the y ticks, pad the limits by `margin` of the tick span and
    bound the left spine to the outermost ticks.

    Panels using the same margin get their y spines at the same heights of
    the axes box. Returns the limits that were set.
    """
    ticks = [float(t) for t in ticks]
    lo, hi = ticks[0], ticks[-1]
    pad = margin * (hi - lo)
    ax.set_yticks(ticks)
    ax.set_ylim(lo - pad, hi + pad)
    # with no margin the spine is left untouched
    if pad:
        ax.spines["left"].set_bounds(lo, hi)
    return lo - pad, hi + pad


def snap_legend_right(fig, box, margin_in=0.0, log=None):
    """Move the first axes' legend horizontally so that its right edge is
    at `box.x1 - margin_in` (`box`: the crop box, in inches).

    Sets `borderaxespad` to zero, as `nudge_legend` does. Returns the
    distance moved in inches (positive = right).
    """
    if box is None:
        return 0.0
    ax = fig.axes[0]
    leg = ax.get_legend()
    if leg is None:
        return 0.0
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    dx = (box.x1 - margin_in) * fig.dpi - bb.x1
    moved = Bbox.from_extents(bb.x0 + dx, bb.y0, bb.x1 + dx, bb.y1)
    leg.borderaxespad = 0.0
    leg.set_bbox_to_anchor(moved.transformed(ax.transAxes.inverted()),
                           transform=ax.transAxes)
    if log:
        log(f"  [legend] snapped {dx / fig.dpi:+.3f}in to the crop's "
            f"right edge (margin {margin_in:.3f}in)")
    return dx / fig.dpi


def add_type_args(ap):
    """Add --axis_pt / --tick_pt, point sizes for figures drawn at their
    final size."""
    ap.add_argument("--axis_pt", type=float, default=None,
                    help="point size of the axis labels (default: PAPER_RC, "
                         "8.5)")
    ap.add_argument("--tick_pt", type=float, default=None,
                    help="point size of the tick labels and legend text "
                         "(default: PAPER_RC, 7.5 / 7)")
    return ap


def type_rc(base, axis_pt=None, tick_pt=None):
    """Copy of `base` with the axis-label and tick / legend sizes set."""
    rc = dict(base)
    if axis_pt:
        rc["axes.labelsize"] = rc["axes.titlesize"] = axis_pt
    if tick_pt:
        for k in ("xtick.labelsize", "ytick.labelsize", "legend.fontsize",
                  "font.size"):
            rc[k] = tick_pt
    return rc


def font_rc(base, family=None, files=(), log=None):
    """Copy of `base` that uses font `family`, after registering `files`.

    Font files (or directories of them) are registered with matplotlib for
    this process only. The fallback chain keeps only installed families
    (matplotlib warns once per missing name) and ends with DejaVu Serif,
    which ships with matplotlib.
    """
    from matplotlib import font_manager as fm

    # print when no logger is given, so a font fallback is always visible
    say = log.info if log else (lambda m, *a: print("[font] " + m % a))
    warn = log.warning if log else (lambda m, *a:
                                    print("[font] WARNING " + m % a))
    for spec in (files or ()):
        paths = ([os.path.join(spec, f) for f in sorted(os.listdir(spec))]
                 if os.path.isdir(spec) else [spec])
        for f in paths:
            if f.lower().endswith((".ttf", ".otf", ".ttc")):
                fm.fontManager.addfont(f)
                say("registered %s as %r", os.path.basename(f),
                    fm.FontProperties(fname=f).get_name())
    rc = dict(base)
    if not family:
        return rc
    have = {f.name for f in fm.fontManager.ttflist}
    if family not in have:
        warn("font %r is not installed; matplotlib will substitute it. "
             "Pass its files with --font_file.", family)
    rc["font.family"] = [family] + [
        f for f in base.get("font.serif", ())
        if f != family and f in have] + ["DejaVu Serif"]
    say("font chain: %s", " -> ".join(rc["font.family"]))
    return rc


def pt(k=1.0, key="font.size"):
    """`k` times the current rcParams point size `key`.

    Call it inside drawing code, not as a default argument value, which is
    bound at import time, before the script installs its rc.
    """
    import matplotlib.pyplot as plt
    return plt.rcParams[key] * k


def box_axes(ax, grid_axis="y"):
    """Full four-side frame with a dashed grid along `grid_axis`."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK, labelcolor=INK, length=3, width=0.9)
    ax.grid(axis=grid_axis, color=GRID, linestyle=(0, (3, 3)),
            linewidth=0.6, alpha=0.95)
    ax.set_axisbelow(True)


def ink_on(hex_color):
    """Readable text colour (white or INK) for a label on a `hex_color`
    fill, chosen by the fill's relative luminance."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if lum < 0.55 else INK


def apply_axes(ax, x_grid=False, grid_axis=None):
    """Recessive frame: no top/right spine, hairline axes, grid behind.

    `grid_axis` ("x" / "y" / "both") overrides `x_grid`; horizontal bars
    need "x".
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelcolor=INK2, length=3, width=0.8)
    ax.grid(axis=grid_axis or ("both" if x_grid else "y"), color=GRID,
            linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def save(fig, out_dir, name, formats=("png", "pdf", "csv", "caption"),
         rows=None, header=None, caption=None, bbox="tight",
         transparent=False, pad=0.02, close=True):
    """Write the figure and its data to `out_dir`; return the paths.

    Args:
        formats: any of "png", "pdf", "svg" (the figure), "csv" (`rows`
            under `header`) and "caption" (`caption` as <name>.caption.txt).
        bbox: "tight" crops to the drawn content with `pad` inches of
            margin; None keeps the exact figsize.
        transparent: save without figure / axes background.
        close: close the figure after saving.
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for ext in ("png", "pdf", "svg"):
        if ext not in formats:
            continue
        path = os.path.join(out_dir, f"{name}.{ext}")
        # no facecolor with transparent=True: an explicit one overrides it
        if transparent:
            fig.savefig(path, dpi=300, bbox_inches=bbox, pad_inches=pad,
                        transparent=True)
        else:
            fig.savefig(path, dpi=300, bbox_inches=bbox, pad_inches=pad,
                        facecolor=SURFACE)
        written.append(path)
    if close:
        plt.close(fig)
    if "csv" in formats and rows is not None:
        path = os.path.join(out_dir, f"{name}.csv")
        with open(path, "w") as f:
            if header:
                f.write(",".join(str(h) for h in header) + "\n")
            for row in rows:
                f.write(",".join(str(v) for v in row) + "\n")
        written.append(path)
    if "caption" in formats and caption:
        path = os.path.join(out_dir, f"{name}.caption.txt")
        with open(path, "w") as f:
            f.write(caption.rstrip() + "\n")
        written.append(path)
    return written
