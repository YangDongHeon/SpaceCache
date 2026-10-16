"""Ablation tables on Waymo / Utonia: lifetime, confidence rule, placement.

Follows the outdoor main-table protocol: accuracy from the merged full-val
runs (dense attention by default, `--acc_mode`), latency / FLOPs / memory
from the latency-slice runs (dense attention; latency is the mean of passes 2..N
for multi-pass runs). Every lookup tries the ablation roots (`--abl`:
accuracy, `--lat_name`) first and falls back to the main-table roots
(`--main`: accuracy, latency), so points shared with the main table are not
re-run. The latency includes the cache update at the Commit Gate
(`spacecache.common.latency`).

    PYTHONPATH=. python spacecache/experiments/ablation_table.py

    # placement probe: hit fraction per (stage, cell scale) on the latency slice
    PYTHONPATH=. python spacecache/experiments/ablation_table.py --probe
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

from spacecache.common import naming
from spacecache.common.latency import frame_ms

ABL = "output/ablation"
MAIN = "output/outdoor"
CELL = "waymo/utonia_dec"

# movable classes (dynamic objects); every other class is static
MOVABLE = {"car", "truck", "bus", "other vehicle", "motorcyclist",
           "bicyclist", "pedestrian", "bicycle", "motorcycle",
           "construction_vehicle", "trailer"}


# --- run names (spacecache/common/naming.py) -----------------------------------

# lifetime rule of the runs to read ("staggered" or "fixed"), from --expiry
EXPIRY = "staggered"


def vname(policy, mode, mf=1.0, L=8, thresh=0.0, tc=False):
    """Run name of an ablation point (`mf` = region scale)."""
    if policy == naming.BACKBONE:
        return naming.run_name(naming.BACKBONE, mode)
    return naming.run_name(policy, mode, region_scale=mf, lifetime=L,
                           confidence=thresh, expiry=EXPIRY, triton=tc)


def load(roots, variant):
    for root in roots:
        p = os.path.join(root, variant, "results.json")
        if os.path.isfile(p):
            with open(p) as f:
                r = json.load(f)
            r["_src"] = root
            return r
    return None


# --- per-run metrics -------------------------------------------------------

def lat_ms(r):
    """Clean frame latency (passes 2..N), Commit Gate included."""
    return frame_ms(r) if r else None


def cache_oh_ms(r):
    """Per-frame cache machinery, median over frames, summed over caches:
    lookup + subset build + restore + staging + commit (`update` is
    restore + stage, so it is not added again)."""
    bd = (r or {}).get("spacecache_ms") or {}
    if not bd:
        return None
    tot = 0.0
    for d in (bd.get("median") or {}, bd.get("s4_median") or {}):
        tot += sum(float(d.get(k) or 0.0)
                   for k in ("match", "subset", "restore", "stage", "commit"))
    return tot


def gflops(r, key="flops_encoder"):
    v = ((r or {}).get("metrics") or {}).get(key)
    return float(v) / 1e9 if v else None


def class_groups(r):
    """(movable mIoU, static mIoU) from the summed per-class counts."""
    if not r or not r.get("intersection"):
        return None, None
    names = [n.lower() for n in (r.get("per_class_iou") or {})]
    inter, union = r["intersection"], r["union"]
    iou = [i / (u + 1e-10) for i, u in zip(inter, union)]
    if len(names) != len(iou):
        return None, None
    mv = [x for n, x in zip(names, iou) if n in MOVABLE]
    st = [x for n, x in zip(names, iou) if n not in MOVABLE]
    return (sum(mv) / len(mv) if mv else None,
            sum(st) / len(st) if st else None)


def _ratio(a, b):
    return (a / b) if b else None


def stream_stats(acc, lat):
    """(results dict carrying the SpaceCache aggregates, source mark).

    A merged accuracy run has exact aggregates (hit fraction, hit / miss
    accuracy) only if merge_shards pooled the shards' frame logs
    (`spacecache_merged == "frame_log"`). Otherwise the latency-slice run is
    used (marked †), or as a last resort the shard-0 values (‡)."""
    if acc and (acc.get("spacecache_merged") == "frame_log"
                or not acc.get("merged_from_shards")):
        return acc, ""
    if lat and lat.get("spacecache_hit_rate") is not None:
        return lat, "†"
    return (acc or {}), "‡"           # shard-0 approximation, last resort


def hit_cols(acc, lat, keys=("spacecache_hit_rate",)):
    src, mark = stream_stats(acc, lat)
    return [pct(src.get(k)) + mark if src.get(k) is not None else "—"
            for k in keys]


def age_stats(r, cut=8):
    """Cache-hit voxels by entry age (`spacecache_region_age`): mean age,
    share older than `cut` frames (what an L=cut lifetime would already
    have refreshed), and hit accuracy below / at-or-above the cut."""
    ag = (r or {}).get("spacecache_region_age") or {}
    n, ok = ag.get("n"), ag.get("ok")
    if not n or not sum(n):
        return {}
    tot = float(sum(n))
    return dict(
        mean_age=sum(a * c for a, c in enumerate(n)) / tot,
        share_old=sum(n[cut:]) / tot,
        acc_young=_ratio(sum(ok[:cut]), sum(n[:cut])),
        acc_old=_ratio(sum(ok[cut:]), sum(n[cut:])))


def cache_mib(r):
    m = (r or {}).get("metrics") or {}
    b = (m.get("cache_bytes_mean") or 0) + (m.get("cache_bytes_d4_mean") or 0)
    return b / 2 ** 20 if b else None


def region(r):
    d = (r or {}).get("spacecache_region_stats") or {}
    if not d:
        return {}
    g = d.get
    nh_s, nh_f = g("rd_nh_stale", 0), g("rd_nh_fresh", 0)
    return dict(
        acc_dyn=_ratio(g("rd_ok_dyn", 0), g("rd_n_dyn", 0)),
        acc_flip=_ratio(g("rd_ok_flip", 0), g("rd_n_flip", 0)),
        acc_stable=_ratio(g("rd_ok_stable", 0), g("rd_n_stable", 0)),
        acc_new=_ratio(g("rd_ok_new", 0), g("rd_n_new", 0)),
        hit_dyn=_ratio(g("rd_nh_dyn", 0), g("rd_n_dyn", 0)),
        hit_stable=_ratio(g("rd_nh_stable", 0), g("rd_n_stable", 0)),
        stale_share=_ratio(nh_s, nh_s + nh_f),
        acc_stale=_ratio(g("rd_okh_stale", 0), nh_s),
        acc_fresh=_ratio(g("rd_okh_fresh", 0), nh_f),
        bnd_all=_ratio(g("rd_cells_bnd", 0), g("rd_cells", 0)),
        bnd_evicted=_ratio(g("rd_low_bnd", 0), g("rd_low", 0)),
        acc_evicted=_ratio(g("rd_ok_low", 0), g("rd_n_low", 0)),
        evicted_resident=_ratio(g("rd_low_hit", 0), g("rd_low", 0)),
        # fresh candidates (recomputed on the miss subset = reduced context):
        # refused at insertion vs inserted
        acc_rej_cand=_ratio(g("rd_ok_low_cand", 0), g("rd_n_low_cand", 0)),
        acc_ins_cand=_ratio(g("rd_ok_keep_cand", 0), g("rd_n_keep_cand", 0)),
    )


# --- formatting ------------------------------------------------------------

def pct(x, nd=1):
    return "—" if x is None else f"{100 * x:.{nd}f}"


def num(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


class Ctx:
    def __init__(self, a):
        self.acc_roots = [os.path.join(a.abl, "accuracy", CELL),
                          os.path.join(a.main, "accuracy", CELL)]
        self.lat_roots = [os.path.join(a.abl, a.lat_name, CELL),
                          os.path.join(a.main, "latency", CELL)]
        self.acc_mode, self.lat_mode = a.acc_mode, a.lat_mode
        self.base_acc = load(self.acc_roots,
                             vname(naming.BACKBONE, self.acc_mode))
        self.base_lat = load(self.lat_roots,
                             vname(naming.BACKBONE, self.lat_mode))
        self.base_ms = lat_ms(self.base_lat)
        self.base_miou = (self.base_acc or {}).get("mIoU")
        self.missing = []

    def pair(self, policy, tc=None, **kw):
        """(accuracy run, latency run). SpaceCache latency runs
        carry `_triton` (--triton_cache) unless `tc=False`."""
        tc = (policy != naming.BACKBONE) if tc is None else tc
        av = vname(policy, self.acc_mode, **kw)
        lv = vname(policy, self.lat_mode, tc=tc, **kw)
        acc, lat = load(self.acc_roots, av), load(self.lat_roots, lv)
        if acc is None:
            self.missing.append("acc  " + av)
        if lat is None:
            self.missing.append("lat  " + lv)
        return acc, lat

    def speed(self, lat):
        ms = lat_ms(lat)
        return (self.base_ms / ms) if (ms and self.base_ms) else None

    def dmiou(self, acc):
        m = (acc or {}).get("mIoU")
        return None if (m is None or self.base_miou is None) \
            else 100 * (m - self.base_miou)


# --- (a) lifetime ------------------------------------------------------------

def table_lifetime(c, Ls, thetas):
    hdr = ["L", "θ", "hit d3", "hit d4", "mIoU", "ΔmIoU", "mIoU mov",
           "mIoU static", "acc dyn", "acc stable", "hit age", "hits ≥8f",
           "hit acc <8f", "hit acc ≥8f", "stale hits", "acc stale hit",
           "lat (ms)", "speedup", "lat-run hit d3",
           "lat-run hit d4", "cache MiB", "GFLOPs total"]
    rows = []
    bm, bs = class_groups(c.base_acc)
    rows.append(["backbone", "—", "—", "—", pct(c.base_miou, 2), "—",
                 pct(bm, 2), pct(bs, 2)] + ["—"] * 8
                + [num(c.base_ms), "1.00x", "—", "—", "0",
                   num(gflops(c.base_lat, "flops_total"), 0)])
    for th in thetas:
        for L in Ls:
            if L == 1 and th > 0:
                continue                      # L=1 reuses nothing
            acc, lat = c.pair(naming.SPACECACHE_S34, L=L, thresh=th)
            mv, st = class_groups(acc)
            rg, ag = region(acc), age_stats(acc)
            ms = lat_ms(lat)
            sp = c.speed(lat)
            rows.append([
                "∞" if math.isinf(L) else int(L), num(th, 2),
                *hit_cols(acc, lat, ("spacecache_hit_rate",
                                     "spacecache_hit_rate_s4")),
                pct((acc or {}).get("mIoU"), 2), num(c.dmiou(acc), 2),
                pct(mv, 2), pct(st, 2),
                pct(rg.get("acc_dyn")), pct(rg.get("acc_stable")),
                num(ag.get("mean_age")), pct(ag.get("share_old")),
                pct(ag.get("acc_young")), pct(ag.get("acc_old")),
                pct(rg.get("stale_share")), pct(rg.get("acc_stale")),
                num(ms),
                "—" if sp is None else f"{sp:.2f}x",
                pct((lat or {}).get("spacecache_hit_rate")),
                pct((lat or {}).get("spacecache_hit_rate_s4")),
                num(cache_mib(lat)), num(gflops(lat, "flops_total"), 0)])
    return hdr, rows


# --- (b) confidence ---------------------------------------------------------

def table_confidence(c, pairs, theta):
    hdr = ["L", "invalidation", "hit d3", "hit acc", "evicted acc",
           "evicted bnd", "bnd (all)", "evicted resident", "rejected cand acc",
           "inserted cand acc", "acc dyn", "acc flip", "mIoU", "ΔmIoU",
           "lat (ms)", "speedup"]
    rows = []
    for L, mode in pairs:
        kw = dict(L=L)
        if mode == "conf":
            kw.update(thresh=theta)
        acc, lat = c.pair(naming.SPACECACHE_S34, **kw)
        rg = region(acc)
        sp = c.speed(lat)
        label = {"none": "lifetime only",
                 "conf": f"+ confidence θ={theta}"}[mode]
        rows.append([
            L, label,
            *hit_cols(acc, lat, ("spacecache_hit_rate",
                                 "spacecache_acc_hit")),
            pct(rg.get("acc_evicted")), pct(rg.get("bnd_evicted")),
            pct(rg.get("bnd_all")), pct(rg.get("evicted_resident")),
            pct(rg.get("acc_rej_cand")), pct(rg.get("acc_ins_cand")),
            pct(rg.get("acc_dyn")), pct(rg.get("acc_flip")),
            pct((acc or {}).get("mIoU"), 2), num(c.dmiou(acc), 2),
            num(lat_ms(lat)), "—" if sp is None else f"{sp:.2f}x"])
    return hdr, rows


def table_theta(c, thetas, L=8):
    hdr = ["θ", "hit d3", "hit acc", "miss acc", "mIoU", "ΔmIoU",
           "lat (ms)", "speedup", "GFLOPs total"]
    rows = []
    for th in thetas:
        acc, lat = c.pair(naming.SPACECACHE_S34, L=L, thresh=th)
        sp = c.speed(lat)
        rows.append([
            num(th, 2),
            *hit_cols(acc, lat, ("spacecache_hit_rate",
                                 "spacecache_acc_hit", "spacecache_acc_miss")),
            pct((acc or {}).get("mIoU"), 2), num(c.dmiou(acc), 2),
            num(lat_ms(lat)), "—" if sp is None else f"{sp:.2f}x",
            num(gflops(lat, "flops_total"), 0)])
    return hdr, rows


# --- (c) placement ------------------------------------------------------------

# Utonia enc_depths (3, 3, 3, 12, 3): blocks each placement encloses
ENCLOSED = {"spacecache_s0": 3, "spacecache_s1": 3, "spacecache_s2": 3,
            "spacecache_s3": 12, "spacecache_s4": 3, "spacecache_s34": 15}


def table_placement(c, arms, theta, L=8, tc=True):
    hdr = ["placement", "blocks", "cell scale", "cell size", "hit", "mIoU",
           "ΔmIoU", "GFLOPs enc", "GFLOPs total", "lat (ms)",
           "speedup", "cache oh (ms)"]
    rows = []
    for name, policy, depth, mf in arms:
        acc, lat = c.pair(policy, tc=tc, mf=mf, L=L, thresh=theta)
        ms = lat_ms(lat)
        sp = c.speed(lat)
        rows.append([
            name, ENCLOSED.get(policy, "—"), f"×{mf:g}",
            f"2^{depth}g·{mf:g} = {2 ** depth * mf:g}g",
            *hit_cols(acc, lat),
            pct((acc or {}).get("mIoU"), 2), num(c.dmiou(acc), 2),
            num(gflops(lat), 0), num(gflops(lat, "flops_total"), 0),
            num(ms),
            "—" if sp is None else f"{sp:.2f}x", num(cache_oh_ms(lat), 2)])
    return hdr, rows


def probe(a):
    """Hit fraction of every probe run, grouped by stage, with the scale whose
    hit is closest to the stage-3 reference (mf=1)."""
    root = a.probe_root
    if not os.path.isdir(root):
        raise SystemExit(f"no probe dir {root}")
    by = {}
    for v in sorted(os.listdir(root)):
        p = os.path.join(root, v, "results.json")
        if "shard" in v or not os.path.isfile(p):
            continue
        with open(p) as f:
            r = json.load(f)
        pol, mf = r.get("policy"), r.get("region_scale")
        if not pol or mf is None or r.get("spacecache_hit_rate") is None:
            continue
        by.setdefault(pol, []).append(
            (float(mf), float(r["spacecache_hit_rate"]),
             float(r["mIoU"]), v))
    ref = [h for mf, h, _, _ in by.get(naming.SPACECACHE_S3, [])
           if abs(mf - 1.0) < 1e-9]
    target = a.target_hit if a.target_hit is not None else (
        ref[0] if ref else None)
    print(f"target hit = {pct(target)}% "
          f"({'--target_hit' if a.target_hit is not None else 'd3 @ mf=1'})")
    picks = {}
    for pol in sorted(by):
        pts = sorted(by[pol])
        best = (min(pts, key=lambda p: abs(p[1] - target))
                if target is not None else None)
        if best:
            picks[pol] = best[0]
        for mf, h, miou, v in pts:
            mark = "  <== closest" if best and v == best[3] else ""
            print(f"  {pol:24s} mf={mf:<5g} hit={100 * h:5.1f}%  "
                  f"mIoU={100 * miou:5.2f}{mark}")
    # machine-readable line: the picked cell scales of stages 0, 1, 2
    got = [picks.get(f"spacecache_s{d}") for d in (0, 1, 2)]
    if all(g is not None for g in got):
        print("PICKS " + " ".join(f"{g:g}" for g in got))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abl", default=ABL)
    ap.add_argument("--main", default=MAIN)
    ap.add_argument("--acc_mode", default="dense",
                    choices=("dense", "flash"),
                    help="attention of the accuracy runs (dense, as in "
                         "scripts/reproduce)")
    ap.add_argument("--lat_mode", default="dense")
    ap.add_argument("--lat_name", default="latency",
                    help="latency subdir under --abl, e.g. latency or "
                         "latency_p2")
    ap.add_argument("--lifetimes", default="1,2,4,8,16,inf")
    ap.add_argument("--theta", type=float, default=0.85)
    ap.add_argument("--theta_sweep",
                    default="0,0.5,0.55,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.99")
    ap.add_argument("--place_mf", default="8,4,2",
                    help="cell scale for stages 0,1,2 (iso-hit picks)")
    ap.add_argument("--native", action="store_true",
                    help="also list stages 0-2 at their native cell (mf=1)")
    ap.add_argument("--no_lifetime_conf", action="store_true",
                    help="lifetime table: only the theta=0 (confidence rule "
                         "off) runs")
    ap.add_argument("--no_place_tc", action="store_true",
                    help="placement latency runs were timed without "
                         "--triton_cache")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe_root",
                    default=os.path.join(ABL, "probe", CELL))
    ap.add_argument("--target_hit", type=float, default=None)
    ap.add_argument("--out", default=os.path.join(ABL, "ablation_table"))
    ap.add_argument("--expiry", default="staggered",
                    choices=("staggered", "fixed"),
                    help="lifetime rule of the runs to read")
    a = ap.parse_args()
    global EXPIRY
    EXPIRY = a.expiry

    if a.probe:
        probe(a)
        return

    c = Ctx(a)
    Ls = [float(x) for x in a.lifetimes.split(",")]
    k0, k1, k2 = (float(x) for x in a.place_mf.split(","))
    arms = []
    if a.native:
        arms += [("stage 0 (native)", "spacecache_s0", 0, 1.0),
                 ("stage 1 (native)", "spacecache_s1", 1, 1.0),
                 ("stage 2 (native)", "spacecache_s2", 2, 1.0)]
    arms += [("stage 0", "spacecache_s0", 0, k0),
             ("stage 1", "spacecache_s1", 1, k1),
             ("stage 2", "spacecache_s2", 2, k2),
             ("stage 3", naming.SPACECACHE_S3, 3, 1.0),
             ("stage 4", "spacecache_s4", 4, 1.0),
             ("stage 3+4 (default)", naming.SPACECACHE_S34, 3, 1.0)]

    sections = [
        ("(a) Lifetime L — Stage-3,4, region scale 1",
         table_lifetime(c, Ls, (0.0,) if a.no_lifetime_conf
                        else (0.0, a.theta))),
        ("(b) Confidence filtering — invalidation rule at matched L",
         table_confidence(c, [(2, "none"), (4, "none"), (4, "conf"),
                              (8, "none"), (8, "conf")], a.theta)),
        ("(b') Threshold sweep θ at L=8",
         table_theta(c, [float(x) for x in a.theta_sweep.split(",")])),
        (f"(c) Placement — L=8, θ={a.theta}",
         table_placement(c, arms, a.theta, tc=not a.no_place_tc)),
    ]

    lines = [f"# Ablation panel — {CELL}",
             "",
             f"backbone: mIoU {pct(c.base_miou, 2)} (full val, "
             f"{c.acc_mode}) · latency {num(c.base_ms)} ms "
             f"(latency slice, {c.lat_mode}, passes 2..N)",
             "",
             "- hit = cache-hit fraction of the stage's tokens (non-seed "
             "frames); hit acc / miss acc = level0 voxel accuracy of "
             "cache-served / recomputed cells.",
             "- mIoU mov / static = mean IoU over movable / static classes "
             "(summed counts, all shards).",
             "- acc dyn / flip / stable (`--region_stats`): voxel accuracy on "
             "cells whose GT label changed since their last observation "
             "with a movable class on either side / changed static<->static "
             "/ unchanged. Same voxel set in every run; the L=1 row is the "
             "backbone's.",
             "- stale hits = share of cache-hit voxels whose cell label "
             "changed since the entry was written; acc stale hit = their "
             "accuracy.",
             "- evicted acc / bnd = accuracy and multi-class-cell share of "
             "the cells the invalidation removed; bnd (all) = base rate; "
             "evicted resident = share that were cache entries (the rest are "
             "fresh candidates refused at insertion); rejected / inserted "
             "cand acc = accuracy of fresh candidates (computed on the miss "
             "subset, i.e. reduced context) the rule refused / let in.",
             "- lat includes the cache update at the Commit Gate.",
             "- † = hit / hit-acc from the latency-slice run (the main-table "
             "accuracy merge kept only shard 0's SpaceCache stats); "
             "‡ = shard-0 approximation.",
             ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        for title, (hdr, rows) in sections:
            lines += [f"## {title}", "", md_table(hdr, rows), ""]
            w.writerow([title])
            w.writerow(hdr)
            w.writerows(rows)
            w.writerow([])
    if c.missing:
        lines += ["## missing runs", ""]
        lines += [f"- `{m}`" for m in sorted(set(c.missing))]
    text = "\n".join(lines)
    print(text)
    with open(a.out + ".md", "w") as f:
        f.write(text + "\n")
    print(f"\nSaved {a.out}.md / .csv")


if __name__ == "__main__":
    main()
