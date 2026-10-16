"""Staleness / error-source figures and tables from the Reuse Agreement runs
(scripts/reproduce/reuse_agreement.sh). CPU only. Diagnostics beyond
Fig. 5; the paper panels come from
spacecache/figure/method/reuse_agreement/plot_reuse_agreement.py.

One figure per claim (Sec. 4.3):

  age   RA of served entries falls with entry age          (staleness)
        (a) by region: static / movable / label changed
        (b) the decomposition: total = stale x ctx, where
            stale = oracle(t_w) vs oracle(t)   (time only)
            ctx   = served vs oracle(t_w)      (reduced context only)
        age 0 = regions recomputed this frame on the miss subset.
  conf  why confidence: RA is low where the region confidence is low,
        (a) for cached entries (stale + context) AND for recomputed
            candidates (age 0 — the age limit never touches them)
        (b) at a FIXED age (age groups), where L cannot separate entries
        (c) by the confidence an entry was written with (the candidate
            test predicts its later reusability); theta marked.
  joint RA over age x confidence: the two controls cover different cells.
  time  hit RA vs frames since sequence start: L=inf keeps falling, finite
        L levels off, + confidence lifts the served RA.

Tables: per-run served RA, and the confidence rule read as a detector on the
L=inf, theta=0 run (soft AUROC of low conf for the disagreement mass
1 - RA, the share of that mass below theta vs the share of hits it evicts).

    PYTHONPATH=. python spacecache/experiments/reuse_agreement/analyze.py
    PYTHONPATH=. python spacecache/experiments/reuse_agreement/analyze.py \\
        --root output/reuse_agreement/waymo/utonia_dec --theta 0.85
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from spacecache.common import naming

POLICY = naming.SPACECACHE_S34
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
# every series also carries its own marker and a direct end label, so it
# is not identified by hue alone
RED, ORANGE, BLUE, VIOLET = "#ff0000", "#ff8700", "#147df5", "#580aff"
CAT_STYLE = {"all": (RED, "o", "all"), "static": (BLUE, "s", "static"),
             "movable": (ORANGE, "^", "movable"),
             "changed": (VIOLET, "D", "label changed")}


def variant(L, theta=0.0, mode="flash"):
    """Run name of a Reuse Agreement run (spacecache/common/naming.py)."""
    return naming.run_name(POLICY, mode, lifetime=L, confidence=theta)


def run_dirs(root, var):
    sh = sorted(glob.glob(os.path.join(root, var + "_shard*of*")))
    if sh:
        return sh
    d = os.path.join(root, var)
    return [d] if os.path.isdir(d) else []


def load_run(root, var):
    """Summed histograms + concatenated frame rows over the shard dirs."""
    dirs = run_dirs(root, var)
    H, frames, meta = {}, [], None
    for d in dirs:
        p = os.path.join(d, "reuse_agreement.npz")
        if not os.path.isfile(p):
            continue
        z = np.load(p, allow_pickle=False)
        meta = dict(stats=[str(s) for s in z["stats"]],
                    cats=[str(c) for c in z["cats"]],
                    depths=[int(x) for x in z["depths"]])
        for k in z.files:
            if k.startswith("H_"):
                H[k] = H[k] + z[k] if k in H else z[k].astype(np.float64)
        fp = os.path.join(d, "reuse_agreement_frames.jsonl")
        if os.path.isfile(fp):
            with open(fp) as f:
                frames += [json.loads(x) for x in f if x.strip()]
    if meta is None:
        return None
    return dict(H=H, frames=frames, n_dirs=len(dirs), **meta)


class Hist:
    """H[age, conf, cat, bnd, stat] with named reductions."""

    def __init__(self, H, stats, cats):
        self.H, self.si = H, {s: i for i, s in enumerate(stats)}
        self.ci = {c: i for i, c in enumerate(cats)}


def _mean(h, si, stat, axes):
    s = h.sum(axis=axes)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s[..., si[stat]] / s[..., si["n"]], s[..., si["n"]]


# --- numbers ----------------------------------------------------------------

def auroc_low_conf(h, si):
    """Soft AUROC of (low confidence) for disagreement, from conf bins: a
    token counts (1 - RA) as positive and RA as negative, so it reduces to
    the usual AUROC when RA is 0/1. h: (conf, stat) summed over the rest."""
    pos = h[:, si["n"]] - h[:, si["ra"]]
    neg = h[:, si["ra"]]
    P, N = pos.sum(), neg.sum()
    if P <= 0 or N <= 0:
        return float("nan")
    neg_above = np.cumsum(neg[::-1])[::-1] - neg      # conf strictly higher
    return float((pos * (neg_above + 0.5 * neg)).sum() / (P * N))


def detector(run, depth, theta, ages=(1, None)):
    """The theta rule on the no-filter run, read as a detector of bad
    reuse: what it evicts vs how much disagreement (1 - RA) it catches.
    ages (1, None) = cached entries, (0, 1) = recomputed candidates."""
    hs = Hist(run["H"][f"H_t_d{depth}"], run["stats"], run["cats"])
    si = hs.si
    h = hs.H[ages[0]:ages[1]].sum(axis=(0, 2, 3))      # (conf, stat)
    kb = int(round(theta * h.shape[0]))
    n, ok = h[:, si["n"]], h[:, si["ra"]]
    bad = n - ok
    low = slice(0, kb)
    return dict(
        auroc=auroc_low_conf(h, si),
        evict_share=float(n[low].sum() / max(n.sum(), 1)),
        disagree_share=float(bad[low].sum() / max(bad.sum(), 1)),
        ra_below=float(ok[low].sum() / max(n[low].sum(), 1)),
        ra_above=float(ok[kb:].sum() / max(n[kb:].sum(), 1)),
        ra_all=float(ok.sum() / max(n.sum(), 1)),
    )


def run_row(run, depth):
    key = f"H_t_d{depth}"
    if run is None or key not in run["H"]:
        return None
    hs = Hist(run["H"][key], run["stats"], run["cats"])
    si = hs.si
    hit = hs.H[1:].sum(axis=(0, 1, 2, 3))
    miss = hs.H[0].sum(axis=(0, 1, 2))
    ages = np.arange(hs.H.shape[0])
    n_age = hs.H[..., si["n"]].sum(axis=(1, 2, 3))
    fr = [f for f in run["frames"] if f.get(f"n_d{depth}")]
    n_tok = sum(f[f"n_d{depth}"] for f in fr)
    n_hit = sum(f[f"n_hit_d{depth}"] for f in fr)
    return dict(
        hit_share=n_hit / max(n_tok, 1),
        ra=hit[si["ra"]] / max(hit[0], 1),
        top1=hit[si["top1"]] / max(hit[0], 1),
        cos=hit[si["cos"]] / max(hit[0], 1),
        ra_stale=hit[si["ra_stale"]] / max(hit[0], 1),
        ra_ctx=hit[si["ra_ctx"]] / max(hit[0], 1),
        pred_agree=hit[si["pts_agree"]] / max(hit[si["pts"]], 1),
        acc_hit=hit[si["pts_ok"]] / max(hit[si["pts_valid"]], 1),
        acc_hit_oracle=hit[si["pts_ok_oracle"]] / max(hit[si["pts_valid"]], 1),
        ra_miss=miss[si["ra"]] / max(miss[0], 1),
        mean_age=float((ages[1:] * n_age[1:]).sum() / max(n_age[1:].sum(), 1)),
        n_hit=int(hit[0]))


# --- plotting ----------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=7.5, length=0)


def _line(ax, x, y, color, marker, label, end_label=True):
    ok = np.isfinite(y)
    if not ok.any():
        return
    ax.plot(x[ok], y[ok], color=color, lw=2, marker=marker, ms=4,
            markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)
    if end_label:
        i = np.where(ok)[0][-1]
        ax.annotate(label, (x[i], y[i]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7,
                    color=INK2)


def _save(fig, out, name, formats):
    for ext in formats:
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor=SURFACE)


def fig_age(run, depth, out, formats, max_age, min_n):
    import matplotlib.pyplot as plt
    hs = Hist(run["H"][f"H_t_d{depth}"], run["stats"], run["cats"])
    si = hs.si
    A = min(max_age + 1, hs.H.shape[0] - 1)      # last bin = overflow
    x = np.arange(A)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), facecolor=SURFACE)
    ax = axs[0]
    for cat in ("all", "static", "movable", "changed"):
        h = hs.H[:A] if cat == "all" else hs.H[:A, :, [hs.ci[cat]]]
        y, n = _mean(h, si, "ra", (1, 2, 3))
        y = np.where(n >= min_n, y, np.nan)
        if cat == "changed":
            y[0] = np.nan                        # no "change" at write
        c, m, lab = CAT_STYLE[cat]
        _line(ax, x, 100 * y, c, m, lab)
    ax.set_title("(a) RA by region", fontsize=8.5, color=INK, loc="left")
    ax = axs[1]
    for stat, (c, m, lab) in (("ra", (RED, "o", "total")),
                              ("ra_stale", (BLUE, "s", "stale (time)")),
                              ("ra_ctx", (ORANGE, "^", "context"))):
        y, n = _mean(hs.H[:A], si, stat, (1, 2, 3))
        _line(ax, x, 100 * np.where(n >= min_n, y, np.nan), c, m, lab)
    ax.set_title("(b) error sources", fontsize=8.5, color=INK, loc="left")
    for ax in axs:
        _style(ax)
        ax.set_xlabel("entry age (frames since computed; 0 = miss)",
                      fontsize=7.5, color=INK2)
        ax.set_ylabel("Reuse Agreement (%)", fontsize=7.5, color=INK2)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    _save(fig, out, f"ra_age_d{depth}", formats)
    plt.close(fig)


def _conf_curve(h, si, rebin, min_n):
    """h: (conf, stat) -> (x, RA) on rebinned confidence bins."""
    K = h.shape[0] // rebin
    hh = h[:K * rebin].reshape(K, rebin, h.shape[1]).sum(axis=1)
    n = hh[:, si["n"]]
    x = (np.arange(K) + 0.5) * rebin / h.shape[0]
    return x, 100 * np.where(n >= min_n, hh[:, si["ra"]] / np.maximum(n, 1),
                             np.nan)


CONF_AGE_GROUPS = (((1, 2), ORANGE, "^", "age 1-2"),
                   ((3, 8), BLUE, "s", "age 3-8"),
                   ((9, 63), VIOLET, "D", "age 9+"))


def fig_conf(run, depth, out, formats, theta, rebin, min_n):
    """Why confidence: RA falls with confidence for BOTH populations the
    Commit Gate judges, and it still does at a FIXED age, i.e. where the
    age limit cannot tell entries apart.
      (a) cached entries (age >= 1, stale + context) and recomputed
          candidates (age 0, context only; the age rule never sees them)
      (b) cached entries, split by age group
      (c) cached entries by the confidence they were WRITTEN with — the
          candidate test predicts how reusable the entry is later."""
    import matplotlib.pyplot as plt
    ht = Hist(run["H"][f"H_t_d{depth}"], run["stats"], run["cats"])
    hw = Hist(run["H"][f"H_w_d{depth}"], run["stats"], run["cats"])
    si = ht.si
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 2.6), facecolor=SURFACE)
    ax = axs[0]
    for h, (c, m, lab) in (
            (ht.H[1:].sum(axis=(0, 2, 3)), (RED, "o", "cached (reused)")),
            (ht.H[0].sum(axis=(1, 2)), (BLUE, "s", "recomputed"))):
        _line(ax, *_conf_curve(h, si, rebin, min_n), c, m, lab)
    ax.set_title("(a) cached vs recomputed", fontsize=8.5, color=INK,
                 loc="left")
    ax = axs[1]
    for (a0, a1), c, m, lab in CONF_AGE_GROUPS:
        h = ht.H[a0:a1 + 1].sum(axis=(0, 2, 3))
        _line(ax, *_conf_curve(h, si, rebin, min_n), c, m, lab)
    ax.set_title("(b) cached, at a fixed age", fontsize=8.5, color=INK,
                 loc="left")
    ax = axs[2]
    _line(ax, *_conf_curve(hw.H[1:].sum(axis=(0, 2, 3)), si, rebin, min_n),
          RED, "o", "cached")
    ax.set_title("(c) by confidence at write", fontsize=8.5, color=INK,
                 loc="left")
    for i, ax in enumerate(axs):
        ax.axvline(theta, color=INK2, lw=1, ls="--")
        ax.annotate(f"θ={theta:g}", (theta, 0.02), xycoords=("data",
                    "axes fraction"), xytext=(3, 0), textcoords="offset "
                    "points", fontsize=7, color=INK2)
        _style(ax)
        ax.set_xlabel("region confidence at write" if i == 2
                      else "region confidence c(v)", fontsize=7.5,
                      color=INK2)
        ax.set_ylabel("Reuse Agreement (%)", fontsize=7.5, color=INK2)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, out, f"ra_conf_d{depth}", formats)
    plt.close(fig)


AGE_GROUPS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 63))


def fig_joint(run, depth, out, formats, conf_lo, min_n):
    import matplotlib.pyplot as plt
    hs = Hist(run["H"][f"H_t_d{depth}"], run["stats"], run["cats"])
    si = hs.si
    K = hs.H.shape[1]
    step = K // 20                                    # 0.05-wide bins
    k0 = int(round(conf_lo * K))
    edges = list(range(k0, K + 1, step))
    M = np.full((len(AGE_GROUPS), len(edges) - 1), np.nan)
    N = np.zeros_like(M)
    for i, (a0, a1) in enumerate(AGE_GROUPS):
        h = hs.H[a0:a1 + 1].sum(axis=(0, 2, 3))
        for j in range(len(edges) - 1):
            hh = h[edges[j]:edges[j + 1]].sum(axis=0)
            N[i, j] = hh[si["n"]]
            if hh[si["n"]] >= min_n:
                M[i, j] = 100 * hh[si["ra"]] / hh[si["n"]]
    fig, ax = plt.subplots(figsize=(4.6, 2.6), facecolor=SURFACE)
    im = ax.imshow(M, aspect="auto", cmap="Blues", origin="lower",
                   vmin=np.nanmin(M) if np.isfinite(M).any() else 0,
                   vmax=100)
    ax.set_yticks(range(len(AGE_GROUPS)),
                  [f"{a}" if a == b else f"{a}-{b}" if b < 63 else f"{a}+"
                   for a, b in AGE_GROUPS])
    xt = range(0, len(edges) - 1, 2)
    ax.set_xticks(list(xt), [f"{edges[j] / K:.2f}" for j in xt])
    ax.set_xlabel("region confidence at reuse", fontsize=7.5, color=INK2)
    ax.set_ylabel("entry age", fontsize=7.5, color=INK2)
    ax.tick_params(colors=INK2, labelsize=7, length=0)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Reuse Agreement (%)", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=6.5, colors=INK2)
    fig.tight_layout()
    _save(fig, out, f"ra_joint_d{depth}", formats)
    plt.close(fig)
    return M, N


def timeline(run, depth, max_fi, min_len):
    fr = [f for f in run["frames"] if f.get(f"n_hit_d{depth}") is not None]
    seg = np.asarray([-1 if f.get("segment") is None else f["segment"]
                      for f in fr])
    fi = np.asarray([int(f["fi"]) for f in fr])
    keep = fi <= max_fi
    if min_len and len(seg):
        _, inv, cnt = np.unique(seg, return_inverse=True, return_counts=True)
        keep &= cnt[inv] >= min_len
    num, den = np.zeros(max_fi + 1), np.zeros(max_fi + 1)
    for f, i, k in zip(fr, fi, keep):
        if k:
            num[i] += f[f"hit_ra_d{depth}"]
            den[i] += f[f"n_hit_d{depth}"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def fig_timeline(curves, depth, out, formats, max_fi):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 2.6), facecolor=SURFACE)
    x = np.arange(max_fi + 1)
    for (lab, c, m), y in curves:
        _line(ax, x, 100 * y, c, m, lab)
    _style(ax)
    ax.set_xlabel("frames since sequence start", fontsize=7.5, color=INK2)
    ax.set_ylabel("Reuse Agreement of hits (%)", fontsize=7.5, color=INK2)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    _save(fig, out, f"ra_timeline_d{depth}", formats)
    plt.close(fig)


# --- main --------------------------------------------------------------------

def pct(x, nd=1):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="output/reuse_agreement/waymo/"
                            "utonia_dec")
    ap.add_argument("--out", default=None,
                    help="default: <root>/../../analysis")
    ap.add_argument("--theta", type=float, default=0.85)
    ap.add_argument("--max_age", type=int, default=30)
    ap.add_argument("--max_fi", type=int, default=29)
    ap.add_argument("--min_len", type=int, default=30,
                    help="timeline: sequences with >= this many frames only, "
                         "so every position averages the same sequences")
    ap.add_argument("--min_n", type=int, default=500,
                    help="hide bins with fewer scored tokens")
    ap.add_argument("--conf_rebin", type=int, default=5,
                    help="conf bins merged per plotted point (100 bins)")
    ap.add_argument("--conf_lo", type=float, default=0.5)
    ap.add_argument("--formats", default="png,pdf")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(
        a.root.rstrip("/"))), "analysis")
    os.makedirs(out, exist_ok=True)
    fmts = a.formats.split(",")
    inf = float("inf")

    runs = {
        "L=∞": variant(inf), "L=8": variant(8), "L=4": variant(4),
        "L=8 +conf": variant(8, a.theta),
        "L=∞ +conf": variant(inf, a.theta),
        "L=1 (sanity)": variant(1),
    }
    data = {k: load_run(a.root, v) for k, v in runs.items()}
    for k, v in runs.items():
        print(f"{'found' if data[k] else 'MISSING':8s} {k:14s} {v}")
    main_run = data["L=∞"]
    if main_run is None:
        raise SystemExit(f"no L=inf theta=0 run under {a.root}")

    summary = dict(runs={}, detector={}, theta=a.theta)
    lines = ["# Reuse Agreement — staleness and error sources", "",
             f"root: `{a.root}`", "",
             "RA = soft prototype agreement (Appendix A.2; profile overlap at "
             "the motivation run's per-stage T) between the served "
             "feature and the oracle (cache-off) feature of the same frame. "
             "stale = oracle at write vs oracle now; ctx = served vs oracle "
             "at write.", ""]
    for depth in main_run["depths"]:
        hdr = ["run", "hit share", "RA", "top-1", "cos", "RA stale",
               "RA ctx", "RA recomputed", "mean age"]
        lines += [f"## Stage {depth}", "", "| " + " | ".join(hdr) + " |",
                  "|" + "---|" * len(hdr)]
        for k in runs:
            r = run_row(data[k], depth)
            if r is None:
                continue
            summary["runs"].setdefault(k, {})[f"d{depth}"] = r
            lines.append("| " + " | ".join([
                k, pct(r["hit_share"]), pct(r["ra"]), pct(r["top1"]),
                f"{r['cos']:.3f}", pct(r["ra_stale"]), pct(r["ra_ctx"]),
                pct(r["ra_miss"]),
                f"{r['mean_age']:.1f}"]) + " |")
        lines += ["", f"Why confidence — θ={a.theta:g} read as a detector "
                  f"of low RA on the L=∞, θ=0 run:", "",
                  "| population | soft AUROC | below θ: share of tokens | "
                  "below θ: share of 1-RA | RA below θ | RA above θ |",
                  "|---|---|---|---|---|---|"]
        for pop, ages in (("cached (age ≥ 1)", (1, None)),
                          ("recomputed (age 0)", (0, 1))):
            det = detector(main_run, depth, a.theta, ages)
            summary["detector"].setdefault(f"d{depth}", {})[pop] = det
            lines.append(f"| {pop} | {det['auroc']:.3f} | "
                         f"{pct(det['evict_share'])} | "
                         f"{pct(det['disagree_share'])} | "
                         f"{pct(det['ra_below'])} | {pct(det['ra_above'])} |")
        lines.append("")
        # same age, different confidence: what the age limit cannot separate
        hs_ = Hist(main_run["H"][f"H_t_d{depth}"], main_run["stats"],
                   main_run["cats"])
        kb = int(round(a.theta * hs_.H.shape[1]))
        for (a0, a1), *_, lab in CONF_AGE_GROUPS:
            h = hs_.H[a0:a1 + 1].sum(axis=(0, 2, 3))
            n, ok = h[:, hs_.si["n"]], h[:, hs_.si["ra"]]
            lines.append(f"- {lab}: RA below θ {pct(ok[:kb].sum() / max(n[:kb].sum(), 1))}"
                         f"% vs above {pct(ok[kb:].sum() / max(n[kb:].sum(), 1))}%")
        lines.append("")

        fig_age(main_run, depth, out, fmts, a.max_age, a.min_n)
        fig_conf(main_run, depth, out, fmts, a.theta, a.conf_rebin, a.min_n)
        M, N = fig_joint(main_run, depth, out, fmts, a.conf_lo, a.min_n)
        summary.setdefault("joint", {})[f"d{depth}"] = dict(
            ra=np.where(np.isfinite(M), M, -1).tolist(), n=N.tolist(),
            age_groups=AGE_GROUPS)
        curves = []
        for k, sty in (("L=4", (ORANGE, "^")), ("L=8", (BLUE, "s")),
                       ("L=∞", (VIOLET, "D")), ("L=8 +conf", (RED, "o"))):
            if data[k] is not None:
                curves.append(((k, *sty),
                               timeline(data[k], depth, a.max_fi, a.min_len)))
        fig_timeline(curves, depth, out, fmts, a.max_fi)

    s = data["L=1 (sanity)"]
    if s is not None:
        fr = s["frames"]
        for depth in s["depths"]:
            n = sum(f.get(f"n_miss_d{depth}", 0) for f in fr)
            t1 = sum(f.get(f"miss_ra_d{depth}", 0) for f in fr)
            cs = sum(f.get(f"miss_cos_d{depth}", 0) for f in fr)
            lines.append(f"Sanity L=1 stage {depth}: RA served vs oracle "
                         f"{pct(t1 / max(n, 1), 2)}%, mean cos "
                         f"{cs / max(n, 1):.4f} (expect ~100 / ~1.0)")
    with open(os.path.join(out, "staleness.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "staleness.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n".join(lines))
    print(f"\n-> {out}/staleness.md, ra_{{age,conf,joint,timeline}}_d*.png")


if __name__ == "__main__":
    main()
