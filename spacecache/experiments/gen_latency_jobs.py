"""Print a latency job for every run of the outdoor accuracy sweep.

Scans the merged accuracy runs under `<root>/<dataset>/<backbone>/` (shard
dirs are skipped), maps each run name back to its run.py arguments
(spacecache/common/naming.py) and prints one `latency-slice config|run.py args`
line per job (config paths are relative to spacecache/configs/), as read by
scripts/reproduce/outdoor_latency.sh.

    PYTHONPATH=. python spacecache/experiments/gen_latency_jobs.py \\
        --root output/outdoor/accuracy
"""

import argparse
import os

from spacecache.common import naming

LATENCY_CFG = {
    ("waymo", "utonia_dec"): "waymo/utonia_dec/latency_slice.py",
    ("waymo", "litept_base"): "waymo/litept_base/latency_slice.py",
    ("nuscenes", "utonia_dec"): "nuscenes/utonia_dec/latency_slice.py",
    ("nuscenes", "litept_base"): "nuscenes/litept_base/latency_slice.py",
}


def args_of(run):
    """run name -> run.py arguments (without the attention / timing flags),
    or None for shards and foreign directories."""
    d = naming.parse_run_name(run)
    if d is None or d["shard"] is not None:
        return None
    p = d["policy"]
    args = ["--policy", p]
    if p == naming.ALGM:
        args += ["--algm_tau", naming.fmt(d["tau"])]
        for key, flag in (("algm_cell", "--algm_cell"),
                          ("algm_cell_offset", "--algm_cell_offset"),
                          ("algm_window", "--algm_wsize")):
            if d[key] is not None:
                args += [flag, str(d[key])]
    elif d["merge_ratio"] is not None:
        args += ["--merge_ratio", naming.fmt(d["merge_ratio"])]
    if p in naming.SPACECACHE:
        args += ["--lifetime", naming.fmt(d["lifetime"]),
                 # always explicit: the config default enables the rule
                 "--confidence", naming.fmt(d["confidence"])]
        if d["region_scale"] != 1.0:
            args += ["--region_scale", naming.fmt(d["region_scale"])]
    if d["proportional_attn"]:
        args.append("--proportional_attn")
    if d["clamp_patch"]:
        args.append("--tome_clamp_patch")
    if d["merge_reduce"] != "scatter":
        args += ["--merge_reduce", d["merge_reduce"]]
    return " ".join(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/outdoor/accuracy")
    args = ap.parse_args()

    seen = set()
    for (ds, bb), cfg in sorted(LATENCY_CFG.items()):
        cell = os.path.join(args.root, ds, bb)
        if not os.path.isdir(cell):
            continue
        for run in sorted(os.listdir(cell)):
            if not os.path.isfile(os.path.join(cell, run, "results.json")):
                continue
            out = args_of(run)
            if out is None:
                continue
            line = cfg + "|" + out
            if line not in seen:
                seen.add(line)
                print(line)


if __name__ == "__main__":
    main()
