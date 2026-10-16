"""Confidence-rule ablation: the lifetime sweep with and without the rule.

Rows: L in `--lifetimes` x {lifetime only (theta=0), + confidence (theta)},
plus L=1 (no reuse, i.e. the backbone). Accuracy from the merged full-val
runs (`--acc_mode`, dense attention by default), latency from the latency-slice
runs under `--lat_name` (dense attention).
Columns: mIoU, cache-hit fraction and hit-token accuracy (level-0 voxel
accuracy of the cells served from the stage-3 / stage-4 cache), latency and
speedup. Writes `<out>.md`, `<out>.csv` and a four-panel figure.

    PYTHONPATH=. python spacecache/experiments/confidence_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import os

from spacecache.common import naming
from spacecache.experiments.ablation_table import (
    CELL, lat_ms, load, md_table, num, pct, stream_stats, vname,
)

POLICY = naming.SPACECACHE_S34
# plot colours
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
RULES = (("none", "lifetime only", ORANGE), ("conf", "+ confidence", BLUE))


def collect(a):
    acc_roots = [os.path.join(a.abl, "accuracy", CELL)]
    lat_roots = [os.path.join(a.abl, a.lat_name, CELL)]
    Ls = [int(x) for x in a.lifetimes.split(",")]
    base_acc = load(acc_roots, vname(POLICY, a.acc_mode, L=1))  # L=1 = backbone
    base_lat = load(lat_roots, vname(naming.BACKBONE, "dense"))
    base_ms = lat_ms(base_lat)
    pts, missing = [], []
    for rule, _, _ in RULES:
        th = a.theta if rule == "conf" else 0.0
        for L in Ls:
            av = vname(POLICY, a.acc_mode, L=L, thresh=th)
            lv = vname(POLICY, "dense", L=L, thresh=th, tc=True)
            acc, lat = load(acc_roots, av), load(lat_roots, lv)
            if acc is None:
                missing.append(av)
            if lat is None:
                missing.append(lv)
            if acc is None and lat is None:
                continue
            src, mark = stream_stats(acc, lat)
            ms = lat_ms(lat)
            pts.append(dict(
                rule=rule, L=L, mark=mark, miou=(acc or {}).get("mIoU"),
                hit3=src.get("spacecache_hit_rate"),
                hit4=src.get("spacecache_hit_rate_s4"),
                hacc3=src.get("spacecache_acc_hit"),
                hacc4=src.get("spacecache_acc_hit_s4"),
                ms=ms, speed=(base_ms / ms) if (ms and base_ms) else None))
    return dict(pts=pts, base_miou=(base_acc or {}).get("mIoU"),
                base_ms=base_ms, missing=missing, Ls=Ls)


def table(d, theta):
    hdr = ["L", "confidence", "mIoU", "ΔmIoU", "hit d3", "hit d4",
           "hit acc d3", "hit acc d4", "latency (ms)", "speedup"]
    rows = [["1 (backbone)", "—", pct(d["base_miou"], 2), "—", "0", "0",
             "—", "—", num(d["base_ms"]), "1.00x"]]
    for p in sorted(d["pts"], key=lambda p: (p["L"], p["rule"] == "conf")):
        dm = (None if p["miou"] is None or d["base_miou"] is None
              else 100 * (p["miou"] - d["base_miou"]))
        rows.append([
            p["L"], f"θ={theta}" if p["rule"] == "conf" else "off",
            pct(p["miou"], 2), num(dm, 2),
            pct(p["hit3"]) + p["mark"], pct(p["hit4"]) + p["mark"],
            pct(p["hacc3"]) + p["mark"], pct(p["hacc4"]) + p["mark"],
            num(p["ms"]),
            "—" if p["speed"] is None else f"{p['speed']:.2f}x"])
    return hdr, rows


def plot(d, out, formats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("miou", "mIoU (%)"), ("hacc", "hit-token accuracy (%)"),
              ("hit", "cache-hit fraction (%)"), ("ms", "latency (ms)")]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.7), sharex=True)
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=INK2, labelsize=7.5, length=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(d["Ls"])
        ax.set_xticklabels([str(L) for L in d["Ls"]])
        ax.set_xlabel("lifetime L (frames)", fontsize=7.5, color=INK2)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        for rule, lab, col in RULES:
            ps = sorted((p for p in d["pts"] if p["rule"] == rule),
                        key=lambda p: p["L"])
            # stage 3 solid, stage 4 dashed where a stage split exists
            series = ({"miou": [("miou", "-", "")],
                       "ms": [("ms", "-", "")],
                       "hacc": [("hacc3", "-", " · s3"),
                                ("hacc4", "--", " · s4")],
                       "hit": [("hit3", "-", " · s3"),
                               ("hit4", "--", " · s4")]})[key]
            for k, ls, sfx in series:
                q = [(p["L"], p[k]) for p in ps if p[k] is not None]
                if not q:
                    continue
                ys = [y if k == "ms" else 100 * y for _, y in q]
                ax.plot([x for x, _ in q], ys, color=col, ls=ls, lw=2,
                        marker="o", ms=5, label=lab + sfx)
        if key == "miou" and d["base_miou"] is not None:
            ax.axhline(100 * d["base_miou"], color=INK2, lw=0.9, ls=":",
                       label="backbone")
        if key == "ms" and d["base_ms"]:
            ax.axhline(d["base_ms"], color=INK2, lw=0.9, ls=":",
                       label="backbone")
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[1].legend(frameon=False, fontsize=6)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    for ext in formats:
        fig.savefig(f"{out}.{ext}", dpi=220, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abl", default="output/ablation")
    ap.add_argument("--lat_name", default="latency")
    ap.add_argument("--lifetimes", default="2,4,8,16,32")
    ap.add_argument("--theta", type=float, default=0.85)
    ap.add_argument("--acc_mode", default="dense",
                    choices=("dense", "flash"),
                    help="attention of the accuracy runs")
    ap.add_argument("--out", default="output/ablation/confidence")
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--expiry", default="staggered",
                    choices=("staggered", "fixed"),
                    help="lifetime rule of the runs to read")
    a = ap.parse_args()
    import spacecache.experiments.ablation_table as _abl
    _abl.EXPIRY = a.expiry

    d = collect(a)
    hdr, rows = table(d, a.theta)
    lines = ["# Confidence filtering — lifetime sweep with / without", "",
             f"accuracy: {a.abl}/accuracy (full val, {a.acc_mode}) · latency: "
             f"{a.abl}/{a.lat_name} (latency slice, dense, pass 2; includes the "
             "cache update at the Commit Gate)", "",
             "- hit d3 / d4 = cache-hit fraction of the stage-3 / stage-4 "
             "tokens (non-seed frames).",
             "- hit acc d3 / d4 = level0 voxel accuracy of the cells served "
             "from the stage-3 / stage-4 cache.",
             "- † = hit stats from the latency-slice run.", "",
             md_table(hdr, rows), ""]
    if d["missing"]:
        lines += ["## missing runs", ""] + [f"- `{m}`" for m in d["missing"]]
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".md", "w") as f:
        f.write(text + "\n")
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(text)
    if d["pts"]:
        plot(d, a.out, [x for x in a.formats.split(",") if x])
        print(f"\nSaved {a.out}.{{md,csv,{a.formats}}}")


if __name__ == "__main__":
    main()
