"""Lifetime analysis: error drift and entry age of SpaceCache hits.

Reads the full-val `--region_stats` runs of spacecache_s34 (region scale 1,
L in `--lifetimes`, theta = `--theta`, plus theta=0 with `--theta0`) and
compares each with the L=1 run (no reuse: the backbone's predictions on the
same voxels):

  drift   ΔmIoU, Δ movable-class mIoU and Δ dynamic-region accuracy vs the
          frame position in the sequence
  age     accuracy of cache-served voxels vs the age of the served entry,
          and the share of hits whose cell label changed since it was written
  table   one row per run: mIoU, hit fraction, mean hit age, share and
          accuracy of hits at least `--age_cut` frames old, stale hits,
          dynamic / stable accuracy

The merged `<run>/results.json` gives the totals and the age histogram
(`spacecache_region_age`); the per-frame inputs (`per_frame_counts.npz`,
`spacecache_per_frame.jsonl`) are read from the shard dirs
`<run>_shardKofN/`, so keep them.

    PYTHONPATH=. python spacecache/experiments/lifetime_analysis.py \\
        --acc_root output/ablation/accuracy/waymo/utonia_dec --theta 0.85
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os

import numpy as np

from spacecache.common import naming

MOVABLE = {"car", "truck", "bus", "other vehicle", "motorcyclist",
           "bicyclist", "pedestrian", "bicycle", "motorcycle",
           "construction_vehicle", "trailer"}

# plot colours: a light-to-dark blue ramp over L, orange for dynamic regions
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
L_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
BLUE, ORANGE = "#2a78d6", "#eb6834"


# lifetime rule of the runs to read (`--expiry`, "staggered" or "fixed")
EXPIRY = "staggered"
# attention of the accuracy runs (`--acc_mode`)
ACC_MODE = "dense"


def _lname(L):
    return "L=∞" if math.isinf(L) else f"L={int(round(L))}"


def variant(L, theta, mode=None, policy=naming.SPACECACHE_S34):
    """Run name of one lifetime / confidence point."""
    return naming.run_name(policy, mode or ACC_MODE, lifetime=L,
                           confidence=theta, expiry=EXPIRY)


def run_dirs(root, var):
    """Shard dirs of a variant (sorted by shard), else the plain dir."""
    sh = sorted(glob.glob(os.path.join(root, var + "_shard*of*")))
    if sh:
        return sh
    d = os.path.join(root, var)
    return [d] if os.path.isdir(d) else []


def load_merged(root, var):
    p = os.path.join(root, var, "results.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


# --- drift -----------------------------------------------------------------

def class_drift(dirs, max_fi, min_len=0, bin_w=1):
    """Class counts summed per frame position over the runs' frames.

    Returns dict(inter, union, target: (bins, C), n: (bins,), names), or
    None if a run dir has no per_frame_counts.npz. The bin of a frame is
    fi // bin_w; counts are summed within a bin, so its mIoU is exact.
    """
    inter = union = target = None
    nb = max_fi // bin_w + 1
    nfr = np.zeros(nb, dtype=np.int64)
    names = None
    for d in dirs:
        p = os.path.join(d, "per_frame_counts.npz")
        if not os.path.isfile(p):
            return None
        z = np.load(p, allow_pickle=False)
        keep = _keep_rows(z["seg"], z["fi"], max_fi, min_len)
        b = z["fi"][keep] // bin_w
        if inter is None:
            C = z["intersection"].shape[1]
            inter = np.zeros((nb, C))
            union = np.zeros((nb, C))
            target = np.zeros((nb, C))
            names = [str(n) for n in z["names"]]
        np.add.at(inter, b, z["intersection"][keep])
        np.add.at(union, b, z["union"][keep])
        np.add.at(target, b, z["target"][keep])
        np.add.at(nfr, b, 1)
    return dict(inter=inter, union=union, target=target, n=nfr, names=names)


def _keep_rows(seg, fi, max_fi, min_len):
    """Rows at fi <= max_fi whose sequence has >= min_len frames, so every
    position averages the same set of sequences."""
    keep = np.asarray(fi) <= max_fi
    if min_len:
        _, inv, cnt = np.unique(np.asarray(seg), return_inverse=True,
                                return_counts=True)
        keep &= cnt[inv] >= min_len
    return keep


def miou_by_fi(cd, cls_mask=None):
    """mIoU per position over the classes present in the GT there (unlike
    the union, the GT does not depend on the predictions, so every run
    averages over the same classes)."""
    iou = cd["inter"] / np.maximum(cd["union"], 1e-10)
    present = cd["target"] > 0
    if cls_mask is not None:
        present = present & cls_mask[None, :]
    with np.errstate(invalid="ignore"):
        return np.where(present.any(1),
                        (iou * present).sum(1) / np.maximum(present.sum(1), 1),
                        np.nan)


def region_drift(dirs, max_fi, min_len=0, bin_w=1):
    """Per-position sums of the --region_stats frame counters."""
    keys = ("rd_n_dyn", "rd_ok_dyn", "rd_n_stable", "rd_ok_stable",
            "rd_nh_stale", "rd_nh_fresh", "n_match", "n_d3",
            "n_match_d4", "n_d4")
    nb = max_fi // bin_w + 1
    acc = {k: np.zeros(nb) for k in keys}
    for d in dirs:
        p = os.path.join(d, "spacecache_per_frame.jsonl")
        if not os.path.isfile(p):
            return None
        with open(p) as f:
            rows = [json.loads(x) for x in f if x.strip()]
        seg = [r.get("segment") for r in rows]
        fi = [int(r.get("fi", 0)) for r in rows]
        keep = _keep_rows([-1 if s is None else s for s in seg], fi,
                          max_fi, min_len)
        for r, f_i, k_ in zip(rows, fi, keep):
            if not k_:
                continue
            b = f_i // bin_w
            for k in keys:
                acc[k][b] += float(r.get(k) or 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(
            acc_dyn=acc["rd_ok_dyn"] / acc["rd_n_dyn"],
            acc_stable=acc["rd_ok_stable"] / acc["rd_n_stable"],
            stale=acc["rd_nh_stale"] / (acc["rd_nh_stale"] + acc["rd_nh_fresh"]),
            hit=acc["n_match"] / acc["n_d3"],
            hit_d4=acc["n_match_d4"] / acc["n_d4"],
            n_dyn=acc["rd_n_dyn"])


# --- plotting --------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=7.5, length=0)


def plot_drift(curves, xs, out, formats, label_ends=("L=8", "L=∞"),
               xpos=None):
    """`xs` = bin indices to draw, `xpos` = their frame positions."""
    xpos = xs if xpos is None else xpos
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("dmiou", "ΔmIoU vs backbone (pt)"),
              ("dmov", "Δ movable-class mIoU (pt)"),
              ("ddyn", "Δ acc. on dynamic regions (pt)"),
              ("hit", "cache-hit fraction (%)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(10.0, 2.7),
                             sharex=True)
    for ax, (key, title) in zip(axes, panels):
        _style(ax)
        if key != "hit":
            ax.axhline(0, color=INK2, lw=0.8)
        for c in curves:
            y = c.get(key)
            if y is None:
                continue
            yy = y[xs]
            ax.plot(xpos, yy, color=c["color"], lw=2, ls=c["ls"],
                    label=c["label"])
            if c["label"] in label_ends and key == panels[0][0]:
                last = np.where(np.isfinite(yy))[0]
                if len(last):
                    i = last[-1]
                    ax.text(xpos[i] + 0.4, yy[i], c["label"], fontsize=7,
                            color=INK, va="center")
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("frames since sequence start", fontsize=7.5, color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=len(lab), loc="upper center", frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in formats:
        fig.savefig(f"{out}.{ext}", dpi=220, facecolor=SURFACE)
    plt.close(fig)


def plot_age(age_runs, out, formats, cut=8, min_n=2000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax in (a1, a2):
        _style(ax)
        ax.axvspan(-0.5, cut - 0.5, color="#f0efec", zorder=0)
        ax.set_xlabel("age of the served entry (frames)", fontsize=7.5,
                      color=INK2)
    a1.text(cut / 2 - 0.5, 0.02, f"reachable at L={cut}",
            transform=a1.get_xaxis_transform(), ha="center", fontsize=6.5,
            color=INK2)
    for r in age_runs:
        ag, ls, name = r["age"], r["ls"], r["label"]
        n = np.asarray(ag["n"], float)
        ok = np.asarray(ag["ok"], float)
        nd = np.asarray(ag["n_dyn"], float)
        okd = np.asarray(ag["ok_dyn"], float)
        ns = np.asarray(ag["n_stale"], float)
        a = np.arange(len(n))
        m, md = n >= min_n, nd >= min_n / 20
        with np.errstate(invalid="ignore", divide="ignore"):
            a1.plot(a[m], 100 * ok[m] / n[m], color=BLUE, lw=2, ls=ls,
                    marker="o", ms=3, label=f"all hits · {name}")
            a1.plot(a[md], 100 * okd[md] / nd[md], color=ORANGE, lw=2,
                    ls=ls, marker="o", ms=3, label=f"dynamic · {name}")
            a2.plot(a[m], 100 * ns[m] / n[m], color=BLUE, lw=2, ls=ls,
                    marker="o", ms=3, label=name)
    a1.set_title("accuracy of cache-served voxels", fontsize=8, color=INK,
                 loc="left")
    a2.set_title("hits whose cell label changed since caching (%)",
                 fontsize=8, color=INK, loc="left")
    a1.legend(frameon=False, fontsize=6.5)
    a2.legend(frameon=False, fontsize=6.5)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    for ext in formats:
        fig.savefig(f"{out}.{ext}", dpi=220, facecolor=SURFACE)
    plt.close(fig)


# --- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc_root",
                    default="output/ablation/accuracy/waymo/utonia_dec")
    ap.add_argument("--lifetimes", default="2,4,8,16,inf")
    ap.add_argument("--theta", type=float, default=0.85)
    ap.add_argument("--theta0", action="store_true",
                    help="also draw the theta=0 (lifetime-only) runs, dashed")
    ap.add_argument("--max_fi", type=int, default=59)
    ap.add_argument("--min_len", type=int, default=30,
                    help="drift: only sequences with >= this many frames, "
                         "drawn up to fi = min_len-1 (0 = all sequences)")
    ap.add_argument("--bin", type=int, default=3,
                    help="drift: frames per x bin (counts summed in-bin)")
    ap.add_argument("--min_frames", type=int, default=25,
                    help="cut the drift x-axis where fewer sequences remain")
    ap.add_argument("--age_cut", type=int, default=8)
    ap.add_argument("--out", default="output/ablation/lifetime")
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--expiry", default="staggered",
                    choices=("staggered", "fixed"),
                    help="lifetime rule of the runs to read")
    ap.add_argument("--acc_mode", default="dense",
                    choices=("dense", "flash"),
                    help="attention of the accuracy runs")
    a = ap.parse_args()
    global EXPIRY, ACC_MODE
    EXPIRY = a.expiry
    ACC_MODE = a.acc_mode

    Ls = [float(x) for x in a.lifetimes.split(",")]
    thetas = [a.theta] + ([0.0] if a.theta0 else [])
    base_var = variant(1, 0.0)
    bdirs = run_dirs(a.acc_root, base_var)
    if a.min_len:
        a.max_fi = min(a.max_fi, a.min_len - 1)
    dkw = dict(min_len=a.min_len, bin_w=max(a.bin, 1))
    bcd = class_drift(bdirs, a.max_fi, **dkw) if bdirs else None
    brd = region_drift(bdirs, a.max_fi, **dkw) if bdirs else None
    if bcd is None:
        raise SystemExit(f"backbone reference {base_var} (L=1, "
                         "--region_stats) is missing under " + a.acc_root)
    mov = np.array([n.lower() in MOVABLE for n in bcd["names"]])
    b_all, b_mov = miou_by_fi(bcd), miou_by_fi(bcd, mov)
    nfr = bcd["n"]
    bw = max(a.bin, 1)
    xs = np.arange(len(nfr))[nfr >= a.min_frames]
    xpos = xs * bw + (bw - 1) / 2.0             # bin centre, in frames

    curves, rows, age_runs = [], [], []
    mixed = len(thetas) > 1          # dashed = theta 0 only when both drawn
    longest = max(Ls)                # age figure: L=inf, else the longest L
    for th in thetas:
        for i, L in enumerate(Ls):
            var = variant(L, th)
            dirs = run_dirs(a.acc_root, var)
            res = load_merged(a.acc_root, var)
            if not dirs or res is None:
                print(f"missing: {var}")
                continue
            cd = class_drift(dirs, a.max_fi, **dkw)
            rd = region_drift(dirs, a.max_fi, **dkw)
            label = _lname(L) + (" (no conf.)" if mixed and th == 0 else "")
            ci = round(i * (len(L_RAMP) - 1) / max(len(Ls) - 1, 1))
            c = dict(label=label, color=L_RAMP[ci],
                     ls="--" if mixed and th == 0 else "-")
            if cd is not None:
                c["dmiou"] = 100 * (miou_by_fi(cd) - b_all)
                c["dmov"] = 100 * (miou_by_fi(cd, mov) - b_mov)
            if rd is not None and brd is not None:
                c["ddyn"] = 100 * (rd["acc_dyn"] - brd["acc_dyn"])
                c["hit"] = 100 * rd["hit"]
            curves.append(c)
            ag = res.get("spacecache_region_age")
            if ag and (L == longest or L == a.age_cut):
                age_runs.append(dict(age=ag, label=label,
                                     ls="-" if L == longest else ":"))
            # summary row
            reg = res.get("spacecache_region_stats") or {}
            g = reg.get
            n_age = np.asarray((ag or {}).get("n", []), float)
            ok_age = np.asarray((ag or {}).get("ok", []), float)
            k = a.age_cut

            def _r(x, y):
                return float(x) / float(y) if y else float("nan")

            tail = xs[-5:] if len(xs) >= 5 else xs
            rows.append(dict(
                run=var, L=_lname(L), theta=th,
                mIoU=100 * res["mIoU"],
                hit_d3=100 * (res.get("spacecache_hit_rate") or 0),
                hit_d4=100 * (res.get("spacecache_hit_rate_s4") or 0),
                mean_hit_age=_r((n_age * np.arange(len(n_age))).sum(),
                                n_age.sum()),
                hits_old_pct=100 * _r(n_age[k:].sum(), n_age.sum()),
                acc_hit_young=100 * _r(ok_age[:k].sum(), n_age[:k].sum()),
                acc_hit_old=100 * _r(ok_age[k:].sum(), n_age[k:].sum()),
                stale_hits_pct=100 * _r(g("rd_nh_stale", 0),
                                        g("rd_nh_stale", 0)
                                        + g("rd_nh_fresh", 0)),
                acc_stale_hit=100 * _r(g("rd_okh_stale", 0),
                                       g("rd_nh_stale", 0)),
                acc_dyn=100 * _r(g("rd_ok_dyn", 0), g("rd_n_dyn", 0)),
                acc_stable=100 * _r(g("rd_ok_stable", 0),
                                    g("rd_n_stable", 0)),
                dmiou_late=float(np.nanmean(c["dmiou"][tail]))
                if "dmiou" in c else float("nan"),
                ddyn_late=float(np.nanmean(c["ddyn"][tail]))
                if "ddyn" in c else float("nan")))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    formats = [x for x in a.formats.split(",") if x]
    if curves:
        plot_drift(curves, xs, a.out + "_drift", formats, xpos=xpos,
                   label_ends=("L=8", _lname(longest)))
    if age_runs:
        plot_age(age_runs, a.out + "_age", formats, cut=a.age_cut)

    # backbone reference numbers for the table header
    bres = load_merged(a.acc_root, base_var) or {}
    breg = bres.get("spacecache_region_stats") or {}
    bacc_dyn = 100 * (breg.get("rd_ok_dyn", 0) / breg["rd_n_dyn"]
                      if breg.get("rd_n_dyn") else float("nan"))
    cols = list(rows[0]) if rows else []
    with open(a.out + "_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = [f"# Lifetime deep dive — {a.acc_root}", "",
             f"backbone (L=1): mIoU {100 * bres.get('mIoU', float('nan')):.2f}"
             f" · acc on dynamic regions {bacc_dyn:.1f}%", "",
             f"drift: sequences with >= {a.min_len} frames "
             f"({'all' if not a.min_len else 'fixed set'}), {bw}-frame bins. "
             f"`*_late` = mean over the last {min(5, len(xs))} plotted "
             f"bins (up to fi {int(xpos[-1]) if len(xs) else '-'}, "
             f">= {a.min_frames} sequences each).", ""]
    if rows:
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r in rows:
            lines.append("| " + " | ".join(
                (f"{v:.2f}" if isinstance(v, float) else str(v))
                for v in r.values()) + " |")
    text = "\n".join(lines)
    with open(a.out + "_summary.md", "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nSaved {a.out}_{{drift,age}}.{{{a.formats}}} and "
          f"{a.out}_summary.{{md,csv}}")


if __name__ == "__main__":
    main()
