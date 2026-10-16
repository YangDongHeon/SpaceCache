"""Table 1: each method's cheapest point within an mIoU budget (outdoor).

Accuracy comes from the merged full-val runs under `--root` (dense
attention, like the latency runs), latency and FLOPs from the latency-slice
runs under `--lat_root`. Runs are joined on (method, swept value). For each
budget (an mIoU fraction, 0.01 = 1 pt) a method's pick is its fewest-FLOPs
point (`--pick_by`, total FLOPs by default) with mIoU >= backbone mIoU -
budget, i.e. the ringed point of each curve in Fig. 6. The GFLOPs column
reports `--flops` (total by default). Latency is the per-frame mean of
passes 2..N for multi-pass runs (`--passes`), else `timing.total_ms`,
including the Commit Gate. The table is written to `--out`, by default
`table1.md` next to `--root`.

    PYTHONPATH=. python spacecache/experiments/main_table_outdoor.py \\
        [--budgets 0.01,0.02] [--flops total] [--pick_by total]
"""

from __future__ import annotations

import argparse
import json
import os

from spacecache.common import naming
from spacecache.common.latency import frame_ms


METHOD_ORDER = [
    (naming.BACKBONE,       "Utonia / LitePT-B"),
    (naming.TOME,           "+ToMe*"),
    (naming.ALGM,           "+ALGM*"),
    (naming.GITMERGE3D,     "+GitMerge3D†"),
    (naming.SPACECACHE_S3,  "+SpaceCache Stage-3 (ours)"),
    (naming.SPACECACHE_S34, "+SpaceCache Stage-3,4 (ours)"),
]
SWEPT_SYMBOL = {"merge_ratio": "r", "tau": "τ", "confidence": "θ"}


def _parse_variant(variant):
    """-> (method, swept value or None); (None, None) for shards and runs
    of other methods."""
    d = naming.parse_run_name(variant)
    if d is None or d["shard"] is not None:
        return None, None
    if d["policy"] not in {k for k, _ in METHOD_ORDER}:
        return None, None
    return d["policy"], d["swept"]


def _swept_tag(key, sweep):
    """'θ=0.85', 'r=0.025', 'τ=0.995'."""
    sym = SWEPT_SYMBOL.get(naming.SWEPT.get(key), "?")
    return f"{sym}={naming.fmt(sweep)}" if sweep is not None else "—"


def _load_acc(root, ds, bb):
    """-> {method: {swept: (mIoU, run)}}, backbone mIoU."""
    pts, base = {}, None
    cell = os.path.join(root, ds, bb)
    if not os.path.isdir(cell):
        return pts, base
    for variant in sorted(os.listdir(cell)):
        if "shard" in variant:
            continue
        rp = os.path.join(cell, variant, "results.json")
        if not os.path.isfile(rp):
            continue
        try:
            r = json.load(open(rp))
        except Exception:
            continue
        key, sweep = _parse_variant(variant)
        if key is None or r.get("mIoU") is None:
            continue
        if key == naming.BACKBONE:
            # flash and dense backbone runs may both exist: take the higher
            base = max(base or 0.0, float(r["mIoU"]))
        else:
            pts.setdefault(key, {})[sweep] = (float(r["mIoU"]), variant)
    return pts, base


def _load_lat(root, ds, bb):
    """-> {method: {swept: (ms, flops_enc_G, run, flops_total_G)}}, backbone."""
    pts, base = {}, None
    cell = os.path.join(root, ds, bb)
    if not os.path.isdir(cell):
        return pts, base
    for variant in sorted(os.listdir(cell)):
        rp = os.path.join(cell, variant, "results.json")
        if not os.path.isfile(rp):
            continue
        try:
            r = json.load(open(rp))
        except Exception:
            continue
        key, sweep = _parse_variant(variant)
        if key is None:
            continue
        # mean over passes 2..N, Commit Gate included
        ms = frame_ms(r)
        fl = (r.get("metrics") or {}).get("flops_encoder")
        ft = (r.get("metrics") or {}).get("flops_total")
        if ms is None and fl is None:
            continue
        val = (float(ms) if ms is not None else None,
               float(fl) / 1e9 if fl else None, variant,
               float(ft) / 1e9 if ft else None)
        if key == naming.BACKBONE:
            base = val
        else:
            pts.setdefault(key, {})[sweep] = val
    return pts, base


def _pick(acc_pts, lat_pts, base_miou, budget, pick_by="encoder"):
    """Cheapest point with mIoU >= base - budget, by encoder FLOPs, total
    FLOPs or latency (`pick_by`), among points whose latency run reports it.

    Returns (sweep, mIoU, ms, enc_G, acc_variant, lat_variant, total_G) or
    None.
    """
    idx = {"encoder": 3, "total": 6, "latency": 2}[pick_by]
    cands = []
    for s, (miou, av) in acc_pts.items():
        if miou < base_miou - budget:
            continue
        lat = lat_pts.get(s)
        if lat is None:
            continue
        c = (s, miou, lat[0], lat[1], av, lat[2], lat[3])
        if c[idx] is None:
            continue
        cands.append(c)
    if not cands:
        return None
    return min(cands, key=lambda c: c[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="output/outdoor/accuracy")
    ap.add_argument(
        "--lat_root",
        default="output/outdoor/latency")
    ap.add_argument("--out", default=None,
                    help="output markdown (default: table1.md next to "
                         "--root)")
    ap.add_argument("--budgets", default="0.01,0.02",
                    help="comma-separated mIoU-drop budgets, e.g. "
                         "0.005,0.01,0.02")
    ap.add_argument("--datasets", default="waymo,nuscenes")
    ap.add_argument("--backbones", default="utonia_dec,litept_base")
    ap.add_argument("--flops", default="total",
                    choices=("total", "encoder"),
                    help="FLOPs reported in the table")
    ap.add_argument("--pick_by", default="total",
                    choices=("encoder", "total", "latency"),
                    help="cost minimised by the budget pick: encoder "
                         "FLOPs, total FLOPs or latency")
    args = ap.parse_args()

    budgets = [float(x) for x in args.budgets.split(",")]
    datasets = [s.strip() for s in args.datasets.split(",")]
    backbones = [s.strip() for s in args.backbones.split(",")]

    out_lines = []
    for backbone in backbones:
        bb_name = {"utonia_dec": "Utonia",
                   "litept_base": "LitePT-B"}.get(backbone, backbone)
        for budget in budgets:
            out_lines.append(f"\n### {bb_name} — budget -{budget*100:.1f}pt"
                             f" (pick: lowest {args.pick_by}; GFLOPs:"
                             f" {args.flops})\n")
            hdr = "| Method |"
            sep = "|---|"
            for ds in datasets:
                hdr += (f" {ds} mIoU↑ | Latency↓ (ms) | GFLOPs↓ |"
                        f" Speedup↑ | pick |")
                sep += "---|---|---|---|---|"
            out_lines += [hdr, sep]
            for key, label in METHOD_ORDER:
                row = f"| {label} |"
                for ds in datasets:
                    acc, base_miou = _load_acc(args.root, ds, backbone)
                    lat, base_lat = _load_lat(args.lat_root, ds, backbone)
                    base_fl = ((base_lat[3] if args.flops == "total"
                                else base_lat[1]) if base_lat else None)
                    base_ms = base_lat[0] if base_lat else None
                    if key == naming.BACKBONE:
                        if base_miou is not None:
                            fl_s = (f"{base_fl:.0f}" if base_fl else "—")
                            ms_s = (f"{base_ms:.0f}" if base_ms else "—")
                            row += (f" {base_miou:.4f} | {ms_s} |"
                                    f" {fl_s} | — | — |")
                        else:
                            row += " — | — | — | — | — |"
                        continue
                    if base_miou is None:
                        row += " n/a | n/a | n/a | n/a | n/a |"
                        continue
                    pick = _pick(acc.get(key, {}), lat.get(key, {}),
                                 base_miou, budget, args.pick_by)
                    if pick is None:
                        row += (f" — | — | — | — |"
                                f" no point >= {base_miou - budget:.4f} |")
                        continue
                    sweep, miou, ms, fl_enc, av, lv, fl_tot = pick
                    fl = fl_tot if args.flops == "total" else fl_enc
                    fl_s = f"{fl:.0f}" if fl else "—"
                    ms_s = f"{ms:.0f}" if ms else "—"
                    sp = (f"{base_ms / ms:.2f}x"
                          if (base_ms and ms) else "—")
                    tag = _swept_tag(key, sweep)
                    row += (f" {miou:.4f} | {ms_s} |"
                            f" {fl_s} | {sp} | {tag} |")
                out_lines.append(row)

    text = "\n".join(out_lines)
    print(text)
    out_md = args.out or os.path.join(os.path.dirname(args.root.rstrip("/")),
                                      "table1.md")
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved {out_md}")


if __name__ == "__main__":
    main()
