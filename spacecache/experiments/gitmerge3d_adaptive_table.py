"""Appendix Table 4: GitMerge3D with and without its adaptive patch partition.

Both versions run at the same merge ratio (`--merge_ratio`, 0.5):
GitMerge3D-dagger is `gitmerge3d` (fixed patches, the version of the main
tables) and GitMerge3D is `gitmerge3d_adaptive` (the official adaptive
partition). mIoU comes from the full-val accuracy runs. The speedup comes
from the latency-slice runs outdoors (mean frame latency of passes 2..N, as
in Table 1) and from the Orin kernel time on ScanNet (`graph_fair.csv` of
`graph_fair_table.py --with_adaptive`, a paired ratio of means). The runs
are written by scripts/reproduce/gitmerge3d_adaptive.sh; the backbone
accuracy falls back to the main sweeps.

    PYTHONPATH=. python spacecache/experiments/gitmerge3d_adaptive_table.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

from spacecache.common import naming
from spacecache.common.latency import frame_ms

CELLS = [("waymo", "utonia_dec"), ("waymo", "litept_base"),
         ("nuscenes", "utonia_dec"), ("nuscenes", "litept_base")]
DATASETS = ["waymo", "nuscenes", "scannet"]
DS_LABEL = {"waymo": "Waymo", "nuscenes": "nuScenes", "scannet": "ScanNet"}
BACKBONES = [("utonia_dec", "Utonia"), ("litept_base", "LitePT-B")]
ORIN_GLOB = "output/gitmerge3d_adaptive/kernel_time/*/piecewise/graph_fair.csv"


def _load(*run_dirs):
    """results.json of the first run dir that has one, else None."""
    for d in run_dirs:
        rp = os.path.join(d, "results.json")
        if os.path.isfile(rp):
            with open(rp) as f:
                return json.load(f)
    return None


def _miou(r):
    return None if r is None or r.get("mIoU") is None else 100.0 * r["mIoU"]


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/gitmerge3d_adaptive")
    ap.add_argument("--main_acc",
                    default="output/outdoor/accuracy",
                    help="fallback root of the outdoor backbone accuracy")
    ap.add_argument("--main_indoor",
                    default="output/indoor/accuracy/"
                            "scannet_video/litept_base_meshnormal",
                    help="fallback dir of the ScanNet backbone accuracy")
    ap.add_argument("--orin", default=None,
                    help="graph_fair.csv of the Orin runs (default: the "
                         f"first match of {ORIN_GLOB})")
    ap.add_argument("--merge_ratio", type=float, default=0.5)
    ap.add_argument("--out", default=None,
                    help="markdown output (default: <root>/table4.md)")
    args = ap.parse_args()

    r = args.merge_ratio
    versions = [("dagger", naming.run_name(naming.GITMERGE3D, "dense",
                                           merge_ratio=r),
                 f"+ GitMerge3D† (r={r:g})"),
                ("adaptive", naming.run_name(naming.GITMERGE3D_ADAPTIVE,
                                             "dense", merge_ratio=r),
                 f"+ GitMerge3D (r={r:g})")]
    backbone = naming.run_name(naming.BACKBONE, "dense")
    # (backbone, row) -> {dataset: (mIoU, speedup)}
    cells = {}

    for ds, bb in CELLS:
        acc = os.path.join(args.root, "accuracy", ds, bb)
        lat = os.path.join(args.root, "latency", ds, bb)
        base_acc = _load(os.path.join(acc, backbone),
                         os.path.join(args.main_acc, ds, bb, backbone))
        base_lat = _load(os.path.join(lat, backbone))
        base_ms = frame_ms(base_lat) if base_lat else None
        cells.setdefault((bb, "backbone"), {})[ds] = (
            _miou(base_acc), 1.0 if base_ms else None)
        for key, var, _ in versions:
            r_lat = _load(os.path.join(lat, var))
            ms = frame_ms(r_lat) if r_lat else None
            sp = base_ms / ms if (base_ms and ms) else None
            cells.setdefault((bb, key), {})[ds] = (
                _miou(_load(os.path.join(acc, var))), sp)

    # ScanNet: LitePT-B only; speedup = paired kernel-time ratio
    orin = args.orin or next(iter(sorted(glob.glob(ORIN_GLOB))), None)
    kx = {}
    if orin and os.path.isfile(orin):
        with open(orin, newline="") as f:
            for row in csv.DictReader(f):
                kx[row.get("run")] = _float(row.get("k_frame_x"))
    ind = os.path.join(args.root, "indoor_accuracy")
    base_in = _load(os.path.join(ind, backbone),
                    os.path.join(args.main_indoor, backbone))
    cells.setdefault(("litept_base", "backbone"), {})["scannet"] = (
        _miou(base_in), 1.0 if kx else None)
    for key, var, _ in versions:
        cells.setdefault(("litept_base", key), {})["scannet"] = (
            _miou(_load(os.path.join(ind, var))), kx.get(var))

    def fmt(x, suf=""):
        return "--" if x is None else f"{x:.2f}{suf}"

    hdr = "| Method |" + "".join(
        f" {DS_LABEL[d]} mIoU (%) | {DS_LABEL[d]} Speedup |" for d in DATASETS)
    lines = [
        f"# GitMerge3D with and without the adaptive patch partition "
        f"(r={r:g})", "",
        "Speedup: latency (outdoor, mean of passes 2..N) and Orin kernel "
        f"time (ScanNet, {orin or 'no graph_fair.csv found'}).", "",
        hdr, "|---|" + "---|" * (2 * len(DATASETS))]
    for bb, label in BACKBONES:
        for key, name in [("backbone", f"**{label}**")] + [
                (k, n) for k, _, n in versions]:
            row = f"| {name} |"
            for d in DATASETS:
                miou, sp = cells.get((bb, key), {}).get(d, (None, None))
                row += f" {fmt(miou)} | {fmt(sp, '×')} |"
            lines.append(row)
    text = "\n".join(lines) + "\n"
    print(text)
    out = args.out or os.path.join(args.root, "table4.md")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
