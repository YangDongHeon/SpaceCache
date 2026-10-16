"""Host-free latency table of the CUDA-graph runs (`run.py --graph_capture`).

Reads `<root>/<cell>/<run>/results.json` (the `graph_capture` record of
spacecache/common/model_graph.py) and writes `<root>/graph_fair.md`,
`graph_fair.csv` (one row per cell) and `graph_fair_frames.csv` (one row per
captured frame). Speedups are paired: a cell and the backbone are compared
on the frames both captured, as a ratio of per-frame means (`--stat mean`,
the default and the indoor table: total time over the picks) or of medians
(`--stat median`). Time columns use the same statistic; host-step counts
and coverage stay medians.

Columns, --graph_mode piecewise (the whole inference as a chain of graphs):
  piecewise   chain replay with every host step run for real (primary)
  wall        the same replay on the host clock
  host idle   piecewise - k_piecewise: GPU idle time inside the chain
  host steps  syncs / dynamic-shape ops / H2D copies / rulebook builds
  coverage    k_piecewise / k_frame; ~1 if the chain ran every eager kernel
Columns, --graph_mode replay (one graph with recorded host values):
  graph       replay median of the captured work
  fair        graph + (k_frame - k_graph): every eager kernel once (primary)
  +commit     fair + k_commit
Both modes:
  kernel      k_frame + k_commit, the kernel-time sum of the eager frame
              including the Commit Gate (the indoor table)
  commit      k_commit, the SpaceCache commit's kernel time (part of kernel)
  eager       eager p50 over frames that are neither a pick nor right after
              one (indicative only)

Flagged: argmax agreement below AGREE_MIN, failed picks, coverage outside
COVERAGE, host-step results that drift across replays, RNG draws not handed
back, excluded < 0 (replay mode), missing kernel sums, a graph mode other
than the backbone's, and backbone drift above DRIFT_PCT between the
`backbone` and `backbone_end` cells.

    PYTHONPATH=. python spacecache/experiments/graph_fair_table.py \\
        --root output/indoor/kernel_time/<tag>/<mode>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

from spacecache.common import naming

# method -> table row
METHOD = {
    naming.SPACECACHE_S34: "+SpaceCache Stage-3,4",
    naming.SPACECACHE_S3: "+SpaceCache Stage-3",
    # GitMerge3D with its adaptive partition is reported only with
    # --with_adaptive (Table 4)
    naming.GITMERGE3D_ADAPTIVE: "+GitMerge3D (adaptive)",
    naming.GITMERGE3D: "+GitMerge3D†",
    naming.TOME: "+ToMe*",
    naming.ALGM: "+ALGM*",
    naming.BACKBONE: "Backbone",
}
ORDER = ["Utonia", "LitePT-B"] + list(METHOD.values())[::-1]


def backbone_name(r):
    """Backbone row label of a run: Utonia or LitePT-B."""
    b = str(r.get("backbone") or r.get("config") or "").lower()
    return "LitePT-B" if "litept" in b else "Utonia"


AGREE_MIN = 0.999
DRIFT_PCT = 2.0
COVERAGE = (0.95, 1.10)   # replays run warm, so kernels can be faster
PRIMARY = {"piecewise": "piecewise_ms", "replay": "fair_ms"}
PAIRED = {"piecewise": ("piecewise_ms", "piecewise_wall_ms", "k_frame_ms"),
          "replay": ("graph_ms", "fair_ms", "fair_commit_ms", "k_frame_ms")}


def median(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


STAT = {"mean": mean, "median": median}


def method_of(variant):
    return METHOD.get(naming.policy_of(variant), variant)


def _ok(f):
    return "graph_ms" in f or "piecewise_ms" in f


def load(root, with_adaptive=False):
    cells = {}
    # <root>/cells.tsv (written by indoor_latency_orin.sh from
    # table2_points.txt) gives the mIoU-drop column of each cell
    budget_of = {}
    tsv = os.path.join(root, "cells.tsv")
    if os.path.isfile(tsv):
        for line in open(tsv).read().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 3:
                budget_of[parts[2]] = parts[0]
    for p in sorted(glob.glob(os.path.join(root, "*", "*", "results.json"))):
        cell = p.split(os.sep)[-3]
        r = json.load(open(p))
        g = r.get("graph_capture") or {}
        allf = g.get("frames") or []
        frames = {int(f["frame"]): f for f in allf if _ok(f)}
        # kernel time of a frame includes the Commit Gate: k_frame ends at
        # the seg_head, k_commit sums the commit kernels that follow
        for f in frames.values():
            if f.get("k_frame_ms") is not None and f.get("k_commit_ms"):
                f["k_frame_ms"] = f["k_frame_ms"] + f["k_commit_ms"]
        picks = set(int(x) for x in g.get("picks") or [])
        per = (r.get("timing") or {}).get("per_frame_ms") or []
        eager = [ms for i, ms in enumerate(per)
                 if i not in picks and (i - 1) not in picks]
        variant = r.get("run_name", cell)
        policy = naming.policy_of(variant)
        if policy == naming.GITMERGE3D_ADAPTIVE and not with_adaptive:
            continue                         # not reported
        # setting part of the cell name: tome_r0.11 -> r0.11,
        # s34_conf0.99 -> conf0.99
        pick = ("" if cell.startswith("backbone")
                else cell.split("_", 1)[-1])
        cells[cell] = dict(
            cell=cell, variant=variant,
            method=(backbone_name(r) if policy == naming.BACKBONE
                    else method_of(variant)),
            # budget: the mIoU-drop column from cells.tsv
            pick=pick, frames=frames,
            budget=budget_of.get(cell) or ("knob" if pick else ""),
            failed=[f for f in allf if not _ok(f)], miou=r.get("mIoU"),
            eager_p50=median(eager), device=g.get("device"),
            host=g.get("host"), replays=g.get("replays"),
            kernel_acct=g.get("kernel_acct"),
            mode=g.get("mode") or "replay",
            # with a cells.tsv, only the cells it lists and the backbone
            # belong to the table
            listed=(not budget_of or cell in budget_of
                    or cell.startswith("backbone")))
    return cells


def paired(base, cell, key, agg=mean):
    common = sorted(set(base["frames"]) & set(cell["frames"]))
    keep = [(base["frames"][i].get(key), cell["frames"][i].get(key))
            for i in common]
    keep = [(x, y) for x, y in keep if x is not None and y is not None]
    if not keep:
        return None, None, 0
    mb = agg([x for x, _ in keep])
    mc = agg([y for _, y in keep])
    return mc, (mb / mc if mc else None), len(keep)


def fmt(x, nd=2, suf=""):
    return "—" if x is None else f"{x:.{nd}f}{suf}"


def fvals(c, key):
    return [f.get(key) for f in c["frames"].values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--with_adaptive", action="store_true",
                    help="keep gitmerge3d_adaptive cells (GitMerge3D with "
                         "its adaptive patch partition); dropped by default")
    ap.add_argument("--stat", choices=tuple(STAT), default="mean",
                    help="per-frame statistic of the time columns and the "
                         "paired speedups: mean (indoor table) or median")
    args = ap.parse_args()
    agg = STAT[args.stat]
    cells = load(args.root, args.with_adaptive)
    if "backbone" not in cells:
        raise SystemExit(f"no backbone cell under {args.root}")
    base = cells["backbone"]
    mode = base["mode"]
    prim = PRIMARY[mode]

    rows, flags = [], []
    ignored = sorted(n for n, c in cells.items() if not c["listed"])
    for name, c in cells.items():
        if name == "backbone_end" or not c["listed"]:
            continue
        if c["mode"] != mode:
            flags.append(f"{name}: mode {c['mode']} != backbone's {mode} "
                         f"— not comparable, skipped")
            continue
        row = dict(cell=name, method=c["method"], budget=c["budget"],
                   pick=c["pick"] or "—", run=c["variant"],
                   n_frames=len(c["frames"]), n_failed=len(c["failed"]),
                   miou=c["miou"], n_common=0)
        for k in PAIRED[mode]:
            ms, sp, n = paired(base, c, k, agg)
            row[k] = ms
            row[k.replace("_ms", "_x")] = sp
            row["n_common"] = max(row["n_common"], n)
        for k in ("excluded_ms", "host_idle_ms", "k_commit_ms",
                  "k_piecewise_ms"):
            row[k] = agg(fvals(c, k))
        for k in ("n_host", "n_graphs", "coverage"):
            row[k] = median(fvals(c, k))
        ag = [(f.get("parity") or {}).get("argmax_agree")
              for f in c["frames"].values()]
        ag = [a for a in ag if a is not None]
        row["agree_min"] = min(ag) if ag else None
        # eager re-run of the same frame vs eager: the floor that
        # nondeterministic kernels alone produce
        rr = [(f.get("parity_rerun") or {}).get("argmax_agree")
              for f in c["frames"].values()]
        rr = [a for a in rr if a is not None]
        row["rerun_agree_min"] = min(rr) if rr else None
        row["n_excl_neg"] = sum(1 for f in c["frames"].values()
                                if f.get("excluded_negative"))
        row["rng_miss"] = sum(f.get("rng_miss") or 0
                              for f in c["frames"].values())
        row["host_drift"] = sum(f.get("host_drift") or 0
                                for f in c["frames"].values())
        # SpaceCache token cut: cache-hit fraction per cached stage (the
        # captured frame's lookup), mean over picks — cold frames included
        hits = {}
        for f in c["frames"].values():
            for li, lv in enumerate(f.get("levels") or []):
                h, sub = lv.get("n_hit"), lv.get("subset")
                if h is not None and sub is not None and h + sub > 0:
                    hits.setdefault(li, []).append(h / (h + sub))
        row["hit"] = ("/".join(f"{100 * sum(v) / len(v):.0f}"
                               for _, v in sorted(hits.items()))
                      if hits else None)
        row["eager_p50"] = c["eager_p50"]
        row["eager_x"] = (base["eager_p50"] / c["eager_p50"]
                          if c["eager_p50"] and base["eager_p50"] else None)
        rows.append(row)

        floor = row["rerun_agree_min"]
        bar = AGREE_MIN if floor is None else min(AGREE_MIN, floor - 0.005)
        if row["agree_min"] is not None and row["agree_min"] < bar:
            flags.append(f"{name}: argmax agreement down to "
                         f"{row['agree_min']:.4f} (eager re-run "
                         f"{fmt(floor, 4)}) — the capture diverged from "
                         f"the eager frame beyond kernel nondeterminism")
        elif row["agree_min"] is not None and row["agree_min"] < AGREE_MIN:
            flags.append(f"{name}: agreement {row['agree_min']:.4f} is at "
                         f"the eager re-run floor {fmt(floor, 4)} — "
                         f"nondeterministic kernels, not the capture")
        if row["n_failed"]:
            flags.append(f"{name}: {row['n_failed']} pick(s) failed — "
                         f"{c['failed'][0].get('error', '')[:160]}")
        if row["n_excl_neg"]:
            flags.append(f"{name}: excluded < 0 on {row['n_excl_neg']} "
                         f"pick(s) — the graph ran kernels the eager frame "
                         f"did not")
        cov = row["coverage"]
        if cov is not None and not (COVERAGE[0] <= cov <= COVERAGE[1]):
            flags.append(f"{name}: coverage {cov:.3f} — the chain did not "
                         f"run the eager frame's kernels one-to-one")
        if row["host_drift"]:
            flags.append(f"{name}: {row['host_drift']} host-step result(s) "
                         f"drifted across replays (nondeterministic "
                         f"kernels); captured values kept")
        if row["rng_miss"]:
            flags.append(f"{name}: {row['rng_miss']} RNG draw(s)/merge decision(s) "
                         f"not handed back from the eager frame (live)")
        k_any = [f.get("k_frame_ms") for f in
                 list(c["frames"].values()) + c["failed"]]
        if c["kernel_acct"] and k_any and all(k is None for k in k_any):
            flags.append(f"{name}: kernel accounting ON but no kernel "
                         f"sums — the profiler saw no device events here")

    drift_line = "backbone drift: no backbone_end cell"
    if "backbone_end" in cells:
        end = cells["backbone_end"]
        ms_a, _, n = paired(base, end, prim, agg)
        mb = agg([f.get(prim) for i, f in base["frames"].items()
                  if i in end["frames"]])
        drift_line = "backbone drift: no common frames"
        if ms_a and mb:
            drift = 100.0 * (ms_a - mb) / mb
            drift_line = (f"backbone drift first->last: {mb:.2f} -> "
                          f"{ms_a:.2f} ms ({drift:+.2f}%, {n} frames)")
            if abs(drift) > DRIFT_PCT:
                flags.append(drift_line + f" — above {DRIFT_PCT}%, rerun")

    rows.sort(key=lambda r: (r["budget"] or "0",
                             ORDER.index(r["method"])
                             if r["method"] in ORDER else 99, r["pick"]))

    if mode == "piecewise":
        head = ("| Method | budget | pick | frames | agree min (re-run) | coverage | "
                "host steps | hit % S3/S4 | **piecewise ms** | **×** | wall ms | "
                "host idle ms | kernel ms | × | commit ms | eager p50 ms | "
                "× |")
    else:
        head = ("| Method | budget | pick | frames | agree min | graph ms | "
                "excluded ms | **fair ms** | **×** | +commit ms | × | "
                "kernel ms | × | eager p50 ms | × |")
    lines = [
        f"# Host-free latency ({mode}) — {base['device']} "
        f"({base['host']})", "",
        f"Frames: {len(base['frames'])} picks per cell, "
        f"{base['replays']} replays each; speedups are paired with "
        f"backbone on common frames (ratio of {args.stat}s; time columns "
        f"are per-frame {args.stat}s). {drift_line}.", "",
        head, "|" + "|".join(["---"] * (head.count("|") - 1)) + "|"]
    for r in rows:
        nfr = str(r["n_common"])
        if r["n_failed"]:
            nfr += f"/{r['n_frames'] + r['n_failed']}"
        lead = (f"| {r['method']} | {r['budget'] or '—'} | {r['pick']} | "
                f"{nfr} | {fmt(r['agree_min'], 4)} "
                f"({fmt(r.get('rerun_agree_min'), 4)}) | ")
        if mode == "piecewise":
            lines.append(
                lead + f"{fmt(r['coverage'], 3)} | {fmt(r['n_host'], 0)} | "
                f"{r['hit'] or '—'} | "
                f"**{fmt(r['piecewise_ms'])}** | "
                f"**{fmt(r['piecewise_x'], 2, '×')}** | "
                f"{fmt(r['piecewise_wall_ms'])} | "
                f"{fmt(r['host_idle_ms'])} | {fmt(r['k_frame_ms'])} | "
                f"{fmt(r['k_frame_x'], 2, '×')} | "
                f"{fmt(r['k_commit_ms'])} | {fmt(r['eager_p50'], 1)} | "
                f"{fmt(r['eager_x'], 2, '×')} |")
        else:
            lines.append(
                lead + f"{fmt(r['graph_ms'])} | {fmt(r['excluded_ms'])} | "
                f"**{fmt(r['fair_ms'])}** | "
                f"**{fmt(r['fair_x'], 2, '×')}** | "
                f"{fmt(r['fair_commit_ms'])} | "
                f"{fmt(r['fair_commit_x'], 2, '×')} | "
                f"{fmt(r['k_frame_ms'])} | {fmt(r['k_frame_x'], 2, '×')} | "
                f"{fmt(r['eager_p50'], 1)} | {fmt(r['eager_x'], 2, '×')} |")
    if ignored:
        lines += ["", f"Not in cells.tsv (left out): {', '.join(ignored)}"]
    lines += ["", "Flags:" if flags else "Flags: none."]
    lines += [f"- {f}" for f in flags]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(args.root, "graph_fair.md"), "w") as f:
        f.write(md)

    keys = ["cell", "method", "budget", "pick", "run", "n_frames",
            "n_failed", "n_common", "miou", "agree_min", "coverage",
            "n_host", "n_graphs", "piecewise_ms", "piecewise_x",
            "piecewise_wall_ms", "piecewise_wall_x", "host_idle_ms",
            "k_piecewise_ms", "graph_ms", "graph_x", "excluded_ms",
            "fair_ms", "fair_x", "fair_commit_ms", "fair_commit_x",
            "k_frame_ms", "k_frame_x", "k_commit_ms", "n_excl_neg",
            "rng_miss", "host_drift", "rerun_agree_min", "hit", "eager_p50",
            "eager_x"]
    with open(os.path.join(args.root, "graph_fair.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    fkeys = ["cell", "frame", "n_input_voxels", "piecewise_ms",
             "piecewise_wall_ms", "k_piecewise_ms", "host_idle_ms",
             "n_graphs", "n_host", "coverage", "graph_ms", "graph_min_ms",
             "k_frame_ms", "k_graph_ms", "excluded_ms", "fair_ms",
             "k_commit_ms", "fair_commit_ms", "argmax_agree", "host_kinds",
             "levels"]
    with open(os.path.join(args.root, "graph_fair_frames.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=fkeys, extrasaction="ignore")
        w.writeheader()
        for name, c in cells.items():
            for i in sorted(c["frames"]):
                fr = dict(c["frames"][i])
                fr["cell"] = name
                fr["argmax_agree"] = (fr.get("parity") or {}).get(
                    "argmax_agree")
                fr["host_kinds"] = json.dumps(fr.get("host_kinds"))
                fr["levels"] = json.dumps(fr.get("levels"))
                w.writerow(fr)
    print(md)
    print(f"-> {args.root}/graph_fair.md, graph_fair.csv, "
          f"graph_fair_frames.csv")


if __name__ == "__main__":
    main()
