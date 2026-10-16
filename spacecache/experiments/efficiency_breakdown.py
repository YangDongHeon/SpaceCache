"""Latency and memory breakdown of each method at an mIoU budget.

For each method the operating point is taken from the main-table runs
(output/outdoor/{accuracy,latency}, Waymo / Utonia): the point with the
fewest encoder FLOPs (`--pick_by flops`)
or the lowest clean latency (`--pick_by latency`) whose full-val mIoU is
within `--budget` of the backbone. `--list_picks` prints the picks with the
run.py arguments that reproduce them.

The figure has three panels sharing the method rows: (a) latency per frame
stacked into compute and policy bands, (b) the policy total on a log axis,
(c) peak memory. Band shares come from `--timing_level block` runs
(`--block_dir`), bar heights and memory from clean runs (`--lat_dirs`), and
the appendix sub-steps from `--timing_level block --fine_regions` runs
(`--fine_dir`). Every time is a mean over the measured frames: the passes
after the first (warm-up) pass of a multi-pass run. See scripts/reproduce/
for the runs.

    PYTHONPATH=. python spacecache/experiments/efficiency_breakdown.py --list_picks
    PYTHONPATH=. python spacecache/experiments/efficiency_breakdown.py \\
        --fine_dir output/efficiency/fine/waymo/utonia_dec
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics

from spacecache.common import naming
from spacecache.common.latency import commit_ms, frame_ms
from spacecache.experiments.gen_latency_jobs import args_of

CELL = "waymo/utonia_dec"
MAIN = "output/outdoor"

# (method, figure label, how often its policy runs per frame)
FAMILIES = [
    (naming.BACKBONE, "Utonia", "—"),
    (naming.TOME, "+ToMe*", "× 32 blocks"),
    (naming.ALGM, "+ALGM*", "2 merge + 2 unmerge"),
    (naming.GITMERGE3D, "+GitMerge3D†", "× 32 blocks"),
    (naming.SPACECACHE_S3, "+SpaceCache Stage-3", "× 1 stage"),
    (naming.SPACECACHE_S34, "+SpaceCache Stage-3,4", "× 2 stages"),
]

# FrameProfiler regions (`attn/<name>`) of each policy band. The regions do
# not nest in each other. The GitMerge3D ones (IN_ATTN) lie
# inside module/*.attn and are subtracted from ATTN; the ToMe / ALGM ones
# lie outside the attention modules and come out of `others`.
SELECT = ("matching", "matching/high", "matching/low", "weighted/energy",
          "weighted/partition", "tome_select", "clap_select", "gbm_select")
REDUCE = ("merge", "tome_merge", "clap_merge", "gbm_merge")
RESTORE = ("unmerge", "weighted/reindex", "tome_unmerge", "clap_unmerge",
           "gbm_unmerge")
IN_ATTN = ("matching", "matching/high", "matching/low", "weighted/energy",
           "weighted/partition", "merge", "unmerge", "weighted/reindex")

# Compute bands: CPE (spconv), ATTN, FFN and `others` = frame - CPE - ATTN -
# FFN - in-forward policy time (norms, LayerScale, residual adds, embedding,
# pooling / unpooling with re-serialization, head). Policy bands:
#   select   matching | ToMe / ALGM index build | cache lookup
#   reduce   merge | miss-subset gather
#   restore  unmerge | assembly of cache hits and misses
#   retain   cache staging + commit (SpaceCache); the commit is part of the
#            frame latency and is kept out of `others` as its own band
COMPUTE_KEYS = ("cpe", "attn", "others", "ffn")
POLICY_KEYS = ("select", "reduce", "restore", "retain")

# stack order keeps the neutral `others` between the red ATTN and the green
# FFN band; the memory panel is a single neutral series
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdcd8"
COLOR = {"cpe": "#5593de", "attn": "#e96d6d", "others": "#b5b3aa",
         "ffn": "#339c33", "select": "#6741d9", "reduce": "#e8590c",
         "restore": "#0ca678", "retain": "#0ca678"}
HATCH = {"retain": "////"}
LABEL = {"cpe": "CPE (spconv)", "attn": "ATTN", "others": "others",
         "ffn": "FFN", "select": "select / match", "reduce": "reduce / build",
         "restore": "restore", "retain": "retain (cache maint.)"}
MEM_BAR = "#b5b3aa"


# --- loading ---------------------------------------------------------------

def _load(dirs, variant):
    for d in dirs:
        p = os.path.join(d, variant, "results.json")
        if os.path.isfile(p):
            with open(p) as f:
                r = json.load(f)
            r["_dir"] = os.path.join(d, variant)
            return r
    return None


def _clean_ms(r):
    """Clean frame latency without the Commit Gate (drawn as its own band)."""
    ms = frame_ms(r or {})
    return None if ms is None else ms - commit_ms(r or {})


def family(variant):
    p = naming.policy_of(variant)
    for fam, label, sites in FAMILIES:
        if fam == p:
            return fam, label, sites
    return None, None, None


def cli_args(variant):
    """run.py method arguments that reproduce `variant`.

    --config, --dense_attn and the timing / output flags are not included.
    """
    d = naming.parse_run_name(variant)
    if d is None:
        raise ValueError(f"no run.py mapping for {variant}")
    out = args_of(variant)
    return out + (" --triton_cache" if d["triton"] else "")


def auto_picks(acc_dir, lat_dir, budget, acc_mode="dense", by="flops"):
    """Per method, the cheapest point (encoder FLOPs, or clean latency with
    by="latency") whose accuracy-run mIoU is >= backbone - budget.

    Returns ([(label, variant, sites, mIoU, ms)], backbone mIoU, threshold).
    """
    base_name = naming.run_name(naming.BACKBONE, acc_mode)
    base = _load([acc_dir], base_name)
    if base is None:
        raise SystemExit(f"no {base_name} under {acc_dir}")
    thr = base["mIoU"] - budget
    best = {}
    for v in sorted(os.listdir(lat_dir)):
        fam, label, sites = family(v)
        d = naming.parse_run_name(v)
        if fam is None or d["shard"] is not None or d["attention"] != "dense":
            continue
        if fam in naming.SPACECACHE and (d["region_scale"] != 1.0
                                         or d["lifetime"] != 8):
            continue                       # main-table protocol only
        lat = _load([lat_dir], v)
        av = naming.rename(v, attention=acc_mode, triton=False)
        acc = _load([acc_dir], av)
        ms = frame_ms(lat) if lat else None
        if acc is None or ms is None:
            continue
        miou = acc["mIoU"]
        if fam != naming.BACKBONE and miou < thr:
            continue
        fl = ((lat or {}).get("metrics") or {}).get("flops_encoder")
        cost = ms if by == "latency" else fl
        if cost is None:
            continue
        if fam not in best or cost < best[fam][0]:
            best[fam] = (cost, (label, v, sites, miou, ms))
    return ([best[f][1] for f, _, _ in FAMILIES if f in best],
            base["mIoU"], thr)


# --- decomposition ---------------------------------------------------------

def _frame_rows(run_dir, warmup):
    """Per-frame FrameProfiler records (per_frame_timing.jsonl), warmup
    frames dropped, or None."""
    p = os.path.join(run_dir, "per_frame_timing.jsonl")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        rows = [json.loads(x) for x in f if x.strip()]
    return rows[warmup:] or None


def _region_sum(ab, names):
    return sum(float(ab.get("attn/" + n, 0.0)) for n in names)


def _frame_bands(mm, ab, total):
    """Compute and policy-region bands of one frame (or of the run means)."""
    def child(sfx):
        return sum(v for k, v in mm.items()
                   if k.startswith("module/") and k.endswith(sfx))
    cpe, attn, ffn = child(".cpe"), child(".attn"), child(".mlp")
    return dict(cpe=cpe, attn=attn - _region_sum(ab, IN_ATTN), ffn=ffn,
                select=_region_sum(ab, SELECT),
                reduce=_region_sum(ab, REDUCE),
                restore=_region_sum(ab, RESTORE), total=float(total))


def _clean_per_frame(cr):
    """Per-frame clean latency: median over passes 2..N of each frame."""
    if not cr:
        return None
    n = cr.get("frames_per_pass")
    pfm = (cr.get("timing") or {}).get("per_frame_ms") or []
    if not n or len(pfm) < 2 * n:
        return None
    passes = [pfm[i * n:(i + 1) * n] for i in range(1, len(pfm) // n)]
    return [statistics.median(col) for col in zip(*passes)]


def _stream_frames(run_dir):
    """SpaceCache per-frame regions (spacecache_per_frame.jsonl), summed over
    the cached stages: select / reduce / restore / retain_in / commit."""
    p = os.path.join(run_dir, "spacecache_per_frame.jsonl")
    if not os.path.isfile(p):
        return None
    out = []
    with open(p) as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)

            def g(k):
                return float(x.get(k) or 0.0) + float(x.get(k + "_d4") or 0.0)
            out.append(dict(select=g("match_ms"), reduce=g("subset_ms"),
                            restore=g("restore_ms"), retain_in=g("stage_ms"),
                            commit=g("commit_ms")))
    return out


def _measured(rows, r):
    """The frames that count: every pass after the first (the warm-up) of a
    multi-pass run, else the frames after the in-band `warmup_frames`."""
    if rows is None:
        return None
    fpp = int(r.get("frames_per_pass") or 0)
    if (r.get("passes") or 1) > 1 and fpp:
        return rows[fpp:]
    return rows[int(r.get("warmup_frames") or 0):]


def _mean_dict(dicts):
    """Key-wise mean of per-frame dicts; a key missing in a frame counts
    as 0."""
    dicts = [d or {} for d in dicts]
    acc = {}
    for d in dicts:
        for k, v in d.items():
            acc[k] = acc.get(k, 0.0) + float(v)
    return {k: v / len(dicts) for k, v in acc.items()} if dicts else {}


def _measured_frames(r):
    """Measured per-frame FrameProfiler records of a run, or None."""
    if not r or not r.get("_dir"):
        return None
    return _measured(_frame_rows(r["_dir"], 0), r) or None


_STREAM_OPS = ("match", "subset", "restore", "stage", "commit")


def _stream_stage_means(r):
    """SpaceCache ms per frame of each cache op, per cached stage
    ({"s3": {...}, "s4": {...}}): means over the measured frames of
    spacecache_per_frame.jsonl, else the run means."""
    sb = (r or {}).get("spacecache_ms") or {}
    if not sb:
        return {}
    p = os.path.join(r.get("_dir") or "", "spacecache_per_frame.jsonl")
    if os.path.isfile(p):
        with open(p) as f:
            rows = _measured([json.loads(x) for x in f if x.strip()], r)
        if rows:
            out = {}
            for tag, sfx in (("s3", ""), ("s4", "_d4")):
                if any(x.get(f"match_ms{sfx}") is not None for x in rows):
                    out[tag] = {op: statistics.fmean(
                        float(x.get(f"{op}_ms{sfx}") or 0.0) for x in rows)
                        for op in _STREAM_OPS}
            return out
    return {tag: {op: float(d.get(op) or 0.0) for op in _STREAM_OPS}
            for tag, d in (("s3", sb), ("s4", sb.get("s4") or {})) if d}


STALL = 1.20   # drop a frame whose block-level time > 1.2 x its clean time


def decompose(r, policy_stat="mean", clean=None):
    """Band ms/frame from one block-level run (unscaled).

    The measured frames are every pass after the first of a multi-pass run
    (the first pass is the warm-up), else the frames after `warmup_frames`.
    policy_stat="mean": every frame is decomposed on its own (`others` is
    that frame's remainder) and each band is the mean over the measured
    frames, so the bands add up to the mean frame. Medians are not used:
    frame sizes vary across scenes, so the median frame is not the mean
    frame. policy_stat="median": per-band medians (reference only).

    `clean` (per-frame latencies of the method's clean run, passed only
    with `--drop_stalls`) turns on the earlier stall filter: a frame whose
    block-level time exceeds STALL x the same clean frame is dropped, and a
    cache-step value above max(10 x its median, median + 1 ms) is set to
    the median.
    """
    t = r["timing"]
    rows = _measured(_frame_rows(r["_dir"], 0) or [], r)
    stream = (_measured(_stream_frames(r["_dir"]), r)
              if r.get("spacecache_ms") else None)
    fpp = int(r.get("frames_per_pass") or 0)
    if stream is not None and len(stream) != len(rows):
        stream = None                    # misaligned -> run-level fallback
    n_capped = 0
    if clean and stream is not None:
        for k in ("select", "reduce", "restore", "retain_in", "commit"):
            med = statistics.median(x[k] for x in stream)
            for x in stream:
                if x[k] > max(10.0 * med, med + 1.0):
                    x[k] = med
                    n_capped += 1
    sb = r.get("spacecache_ms") or {}
    d = (sb.get("median") if policy_stat == "median" else sb) or {}
    d4 = (sb.get("s4_median") if policy_stat == "median"
          else sb.get("s4")) or {}
    st_run = {k: sum(float(x.get(src, 0.0)) for x in (d, d4))
              for k, src in (("select", "match"), ("reduce", "subset"),
                             ("restore", "restore"), ("retain_in", "stage"),
                             ("commit", "commit"))}
    keys = ("cpe", "attn", "ffn", "select", "reduce", "restore",
            "retain_in", "commit", "others")
    per, dropped = [], 0
    for j, fr in enumerate(rows):
        tot = float(fr.get("total_ms", 0.0))
        cj = j % fpp if fpp else j
        if clean and cj < len(clean) and tot > STALL * clean[cj]:
            dropped += 1
            continue
        stf = stream[j] if stream is not None else st_run
        if r.get("commit_in_frame"):
            tot -= stf["commit"]
        b = _frame_bands(fr.get("modules_ms") or {},
                         fr.get("attn_breakdown_ms") or {}, tot)
        for k in ("select", "reduce", "restore"):
            b[k] += stf[k]
        b["retain_in"], b["commit"] = stf["retain_in"], stf["commit"]
        b["others"] = (tot - b["cpe"] - b["attn"] - b["ffn"] - b["select"]
                       - b["reduce"] - b["restore"] - b["retain_in"])
        per.append(b)
    if not per:                          # no per-frame file: run means
        tot = float(t["total_ms"])
        if r.get("commit_in_frame"):
            tot -= st_run["commit"]
        b = _frame_bands(t.get("module_ms") or {},
                         t.get("attn_breakdown_ms") or {}, tot)
        for k in ("select", "reduce", "restore"):
            b[k] += st_run[k]
        b["retain_in"], b["commit"] = st_run["retain_in"], st_run["commit"]
        b["others"] = (b["total"] - b["cpe"] - b["attn"] - b["ffn"]
                       - b["select"] - b["reduce"] - b["restore"]
                       - b["retain_in"])
        per = [b]
    agg = statistics.median if policy_stat == "median" else statistics.fmean
    out = {k: agg([b[k] for b in per]) for k in keys}
    out["total"] = sum(out[k] for k in keys if k != "commit")
    out["frames_kept"], out["frames_dropped"] = len(per), dropped
    out["cache_values_capped"] = n_capped
    out["block_total_mean"] = float(t["total_ms"])
    return out


def scaled(bands, clean_ms, mode="none", session=1.0):
    """Map the in-forward bands onto a latency axis.

    The commit is kept apart from the in-forward bands and added unscaled.
    `mode`:
      none     the profiled numbers as measured (default). The block runs
               come from one profiling session, so modules a method does
               not touch read the same as in the backbone.
      session  one common factor `session` for every method (backbone
               table latency / backbone profile), so the backbone bar
               equals the table.
      method   each method scaled onto its own table latency; totals match
               the table, but each run's noise then scales every band.
    """
    if mode == "method":
        s = (clean_ms / bands["total"]) if (clean_ms and bands["total"]) \
            else 1.0
    elif mode == "session":
        s = session
    else:
        s = 1.0
    out = {k: bands[k] * s for k in ("cpe", "attn", "others", "ffn",
                                     "select", "reduce", "restore")}
    out["retain"] = bands["retain_in"] * s + bands["commit"]
    out["frame_ms"] = (clean_ms if (mode == "method" and clean_ms)
                       else bands["total"] * s)
    out["table_ms"] = clean_ms
    out["bar_ms"] = out["frame_ms"] + bands["commit"]
    out["policy_ms"] = sum(out[k] for k in POLICY_KEYS)
    out["scale"] = s
    out["frames_kept"] = bands.get("frames_kept")
    out["frames_dropped"] = bands.get("frames_dropped")
    return out


def memory(r):
    m = (r or {}).get("metrics") or {}
    if not m.get("peak_bytes_mean"):
        return None
    mib = 1.0 / 2 ** 20
    model = float(r.get("param_bytes") or 0) * mib
    cache3 = float(m.get("cache_bytes_mean") or 0) * mib
    cache4 = float(m.get("cache_bytes_d4_mean") or 0) * mib
    peak = float(m["peak_bytes_mean"]) * mib
    return dict(model=model, cache=cache3 + cache4, cache_d3=cache3,
                cache_d4=cache4, peak=peak,
                working=peak - model - cache3 - cache4,
                peak_stage=m.get("peak_stage"))


# --- appendix: fine sub-steps + per-stage split ------------------------------

_BAND_OF_PREFIX = {"match": "select", "merge": "reduce", "unmerge": "restore"}
_COARSE_BAND = {**{n: "select" for n in SELECT},
                **{n: "reduce" for n in REDUCE},
                **{n: "restore" for n in RESTORE}}


def fine_rows(label, r, target=None):
    """Coarse policy regions and their sub-steps as rows (label, band, op,
    ms/frame, calls/frame, us/call).

    Merge engines are read from attn_breakdown_ms / attn_fine_ms of a
    --fine_regions run, SpaceCache from spacecache_ms (medians,
    one call per cached stage). `target` holds the method's figure bands
    (select / reduce / restore): the sub-step markers inflate the coarse
    regions of the fine run, so each coarse region and its sub-steps are
    scaled by figure band / fine-run band. Shares within a band and call
    counts are kept, and the merge-engine rows sum to the figure bands.
    """
    out = []
    t = (r or {}).get("timing") or {}
    frs = _measured_frames(r)
    ab = (_mean_dict(x.get("attn_breakdown_ms") for x in frs) if frs
          else t.get("attn_breakdown_ms") or {})
    fine = (_mean_dict(x.get("attn_fine_ms") for x in frs) if frs
            else t.get("attn_fine_ms") or {})
    coarse = [(_COARSE_BAND[k[len("attn/"):]], k[len("attn/"):], float(v))
              for k, v in sorted(ab.items())
              if k[len("attn/"):] in _COARSE_BAND]
    f = {}
    if target:
        for band in ("select", "reduce", "restore"):
            tot = sum(v for b, _n, v in coarse if b == band)
            if tot > 0 and target.get(band):
                f[band] = float(target[band]) / tot
    for band, name, v in coarse:
        out.append((band, name, v * f.get(band, 1.0), None, None))
    for k, v in sorted(fine.items()):
        if not k.startswith("fine/"):
            continue
        name = k[len("fine/"):]
        n = float(fine.get("n/" + name, 0) or 0)
        band = _BAND_OF_PREFIX.get(name.split("/", 1)[0], "?")
        ms = float(v) * f.get(band, 1.0)
        out.append((band, "  └ " + name, ms, n,
                    1000 * ms / n if n else None))
    ss = _stream_stage_means(r)
    for tag in ("s3", "s4"):
        d = ss.get(tag) or {}
        for op, band in (("match", "select"), ("subset", "reduce"),
                         ("restore", "restore"), ("stage", "retain"),
                         ("commit", "retain")):
            if d.get(op) is not None:
                out.append((band, f"{tag}/{op}", float(d[op]), 1.0,
                            1000 * float(d[op])))
    return [(label,) + row for row in out]


def stage_rows(label, r):
    """Per-stage ms of CPE / ATTN / FFN / cache / rest, from module labels
    `module/backbone.{enc,dec}.<stage>.blockN.<child>` and stage totals.

    Merge regions are frame-global, so a merge engine's policy time stays
    inside its stage's ATTN (GitMerge3D) or rest (ToMe / ALGM); the
    SpaceCache time is split out of enc3 / enc4.
    """
    t = (r or {}).get("timing") or {}
    st = t.get("stage_ms") or {}
    mm = t.get("module_ms") or {}
    frs = _measured_frames(r)
    if frs:
        mm = _mean_dict(x.get("modules_ms") for x in frs)
        st = {k[len("stage/"):]: v for k, v in mm.items()
              if k.startswith("stage/")}
    per = {s: dict(cpe=0.0, attn=0.0, ffn=0.0) for s in st}
    for k, v in mm.items():
        if not k.startswith("module/backbone."):
            continue
        parts = k.split(".")
        if len(parts) < 5:
            continue
        s, child = parts[2], parts[-1]
        key = {"cpe": "cpe", "attn": "attn", "mlp": "ffn"}.get(child)
        if s in per and key:
            per[s][key] += float(v)
    ss = _stream_stage_means(r)
    pol = {}
    for s, d in (("enc3", ss.get("s3") or {}), ("enc4", ss.get("s4") or {})):
        if d:
            pol[s] = sum(float(d.get(k) or 0.0)
                         for k in ("match", "subset", "restore", "stage"))
    rows = []
    for s, tot in st.items():
        p = per[s]
        cache = pol.get(s, 0.0)
        rest = float(tot) - p["cpe"] - p["attn"] - p["ffn"] - cache
        rows.append((label, s, float(tot), p["cpe"], p["attn"], p["ffn"],
                     cache, rest))
    return rows


# --- figure ----------------------------------------------------------------

STAGE_ORDER = ("embedding", "enc0", "enc1", "enc2", "enc3", "enc4",
               "dec3", "dec2", "dec1", "dec0", "seg_head")
BAND_COLOR = {"select": COLOR["select"], "reduce": COLOR["reduce"],
              "restore": COLOR["restore"], "retain": COLOR["retain"]}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["hatch.linewidth"] = 0.8
    matplotlib.rcParams["hatch.color"] = SURFACE
    return plt


def _axstyle(axis, grid="x"):
    axis.set_facecolor(SURFACE)
    axis.grid(axis=grid, color=GRID, lw=0.6)
    axis.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axis.spines[s].set_visible(False)
    axis.spines["bottom"].set_color(INK2)
    axis.tick_params(colors=INK2, labelsize=7.5, length=0)


def _save(fig, out, formats):
    fig.patch.set_facecolor(SURFACE)
    for ext in formats:
        fig.savefig(f"{out}.{ext}", dpi=220, facecolor=SURFACE,
                    bbox_inches="tight")


def plot(rows, out, formats):
    """Breakdown figure: (a) stacked latency, (b) policy overhead on a log
    axis (dot + stem, as bar lengths on a log axis are misleading), (c) peak
    memory with a model-weight tick and the cache size."""
    plt = _mpl()
    from matplotlib.patches import Patch

    n = len(rows)
    has_mem = any(r["mem"] for r in rows)
    ncol = 3 if has_mem else 2
    fig, axes = plt.subplots(
        1, ncol, figsize=(10.0 if has_mem else 8.0, 0.48 * n + 1.3),
        sharey=True,
        gridspec_kw=dict(width_ratios=[3.2, 1.6, 1.3][:ncol], wspace=0.3))
    ax, az = axes[0], axes[1]
    ys = list(range(n))[::-1]            # first method on top
    for axis in axes:
        _axstyle(axis)
    xmax = max(r["b"]["bar_ms"] for r in rows)

    # (a) stacked latency — in-bar numbers only where they fit
    for y, row in zip(ys, rows):
        b, left = row["b"], 0.0
        for k in COMPUTE_KEYS + POLICY_KEYS:
            w = b[k]
            if w <= 0:
                continue
            ax.barh(y, w, left=left, height=0.58, color=COLOR[k],
                    edgecolor=SURFACE, linewidth=1.2, hatch=HATCH.get(k))
            if w >= 0.07 * xmax:
                ax.text(left + w / 2, y, f"{w:.0f}", ha="center",
                        va="center", fontsize=6.5, color=INK)
            left += w
        ax.text(b["bar_ms"] + 0.012 * xmax, y, f"{b['bar_ms']:.0f}",
                va="center", fontsize=7, color=INK)
    ax.set_xlim(0, xmax * 1.1)
    ax.set_xlabel("latency per frame (ms)", color=INK2, fontsize=7.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], color=INK, fontsize=7.5)

    # (b) total policy overhead, log x
    pos = [r["b"]["policy_ms"] for r in rows if r["b"]["policy_ms"] > 0]
    lo = 10 ** (int(f"{min(pos):e}".split("e")[1]) - 0) / 2 if pos else 0.5
    hi = max(pos) * 40 if pos else 10
    az.set_xscale("log")
    az.set_xlim(lo, hi)
    for y, row in zip(ys, rows):
        b = row["b"]
        v = b["policy_ms"]
        if v <= 0:
            az.text(lo * 1.15, y, "no policy", va="center", fontsize=6.5,
                    color=INK2)
            continue
        az.plot([lo, v], [y, y], color=GRID, lw=1.2, zorder=1)
        az.plot([v], [y], "o", ms=7, color=INK, mec=SURFACE, mew=1.5,
                zorder=3)
        az.text(v * 1.35, y, f"{v:.1f} ms · {100 * v / b['bar_ms']:.1f}%\n"
                f"{row['sites']}", va="center", fontsize=6.2, color=INK2,
                linespacing=1.1)
    az.set_xlabel("policy overhead (ms, log)", color=INK2, fontsize=7.5)

    # (c) memory
    if has_mem:
        am = axes[2]
        gmax = max((r["mem"]["peak"] for r in rows if r["mem"]), default=1)
        for y, row in zip(ys, rows):
            m = row["mem"]
            if not m:
                continue
            gib = m["peak"] / 1024
            am.barh(y, gib, height=0.58, color=MEM_BAR, edgecolor=SURFACE,
                    linewidth=1.2)
            am.plot([m["model"] / 1024] * 2, [y - 0.29, y + 0.29],
                    color=INK, lw=1.3)
            txt = f"{gib:.2f}"
            if m["cache"] > 0:
                txt += f"\n+{m['cache']:.0f} MiB cache"
            am.text(gib + 0.03 * gmax / 1024, y, txt, va="center",
                    fontsize=6.2, color=INK2, linespacing=1.1)
        am.set_xlim(0, gmax / 1024 * 1.75)
        am.set_xlabel("peak memory (GiB)\n| = model weights", color=INK2,
                      fontsize=7.5)

    handles = [Patch(facecolor=COLOR[k], edgecolor=SURFACE,
                     hatch=HATCH.get(k), label=LABEL[k])
               for k in COMPUTE_KEYS + POLICY_KEYS]
    fig.legend(handles=handles, ncol=8, loc="lower center", frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, 0.99), handlelength=1.2,
               columnspacing=1.0)
    _save(fig, out, formats)
    plt.close(fig)


def plot_fine(fine_all, out, formats, width=9.0):
    """Appendix: every policy sub-step, one small multiple per method,
    ms / frame on a shared log axis, coloured by band, with calls x unit
    cost beside each dot. A coarse region is drawn only when its band has
    no sub-steps (sub-steps nest inside it)."""
    if not fine_all:
        return
    plt = _mpl()
    from matplotlib.lines import Line2D
    methods = []
    for r in fine_all:
        if r[0] not in methods:
            methods.append(r[0])
    per = {}
    for m in methods:
        rows = [r for r in fine_all if r[0] == m]
        subs = {r[1] for r in rows if r[2].startswith("  └")}
        # rows below 0.01 ms stay in the table but would only stretch the
        # log axis
        keep = [r for r in rows if (r[2].startswith("  └")
                                    or r[1] not in subs) and r[3] >= 0.01]
        per[m] = sorted(keep, key=lambda r: -r[3])
    vals = [r[3] for m in methods for r in per[m] if r[3] > 0]
    lo, hi = min(vals) / 3, max(vals) * 30
    nc = 2
    nr = (len(methods) + nc - 1) // nc
    heights = [max(len(per[m]) for m in methods[i * nc:(i + 1) * nc])
               for i in range(nr)]
    fig, axes = plt.subplots(
        nr, nc, figsize=(width, (0.19 * sum(heights) + 0.9 * nr)
                         * min(1.0, width / 9.0 * 1.15)),
        gridspec_kw=dict(height_ratios=heights, hspace=0.55, wspace=0.75),
        squeeze=False)
    for i, m in enumerate(methods):
        ax = axes[i // nc][i % nc]
        _axstyle(ax)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        rows = per[m]
        ys = list(range(len(rows)))[::-1]
        for y, (_m, band, op, ms, n, us) in zip(ys, rows):
            if ms <= 0:
                continue
            ax.plot([lo, ms], [y, y], color=GRID, lw=1.0, zorder=1)
            ax.plot([ms], [y], "o", ms=6, color=BAND_COLOR.get(band, INK),
                    mec=SURFACE, mew=1.2, zorder=3)
            note = (f"{ms:.2f}" + (f"  ({n:.0f} × {us:.0f} µs)"
                                   if n and us is not None else ""))
            ax.text(ms * 1.3, y, note, va="center", fontsize=5.8,
                    color=INK2)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[2].replace("  └ ", "") for r in rows],
                           fontsize=6.2, color=INK)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        tot = sum(r[3] for r in rows)
        ax.set_title(f"{m} — {tot:.1f} ms / frame", fontsize=7.5,
                     color=INK, loc="left")
        ax.set_xlabel("ms / frame (log)", fontsize=6.5, color=INK2)
    for j in range(len(methods), nr * nc):
        axes[j // nc][j % nc].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", ms=6, color=BAND_COLOR[k],
                      label=LABEL[k]) for k in POLICY_KEYS[:3]]
    handles.append(Line2D([], [], marker="o", ls="", ms=6,
                          color=BAND_COLOR["retain"],
                          label="retain / commit"))
    fig.legend(handles=handles, ncol=4, loc="lower center", frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 1.0))
    _save(fig, out, formats)
    plt.close(fig)


def plot_stages(stage_all, out, formats, base_label="Utonia", width=9.6):
    """Appendix: per-stage time, one small multiple per method, with the
    backbone drawn as an outline behind each bar."""
    if not stage_all:
        return
    plt = _mpl()
    by = {}
    for r in stage_all:
        by.setdefault(r[0], {})[r[1]] = r[2]
    base = by.get(base_label, {})
    methods = [m for m in by if m != base_label]
    stages = [s for s in STAGE_ORDER if s in base]
    nc = 3
    nr = (len(methods) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(width, 2.1 * nr * width
                                              / 9.6 * 1.1),
                             sharey=True, squeeze=False,
                             gridspec_kw=dict(hspace=0.6, wspace=0.12))
    x = list(range(len(stages)))
    ymax = max(max(v.values()) for v in by.values()) * 1.15
    for i, m in enumerate(methods):
        ax = axes[i // nc][i % nc]
        _axstyle(ax, grid="y")
        ax.spines["left"].set_visible(False)
        ax.bar(x, [by[m].get(s, 0) for s in stages], width=0.62,
               color=COLOR["cpe"], edgecolor=SURFACE, linewidth=1.0,
               zorder=2)
        ax.bar(x, [base.get(s, 0) for s in stages], width=0.62,
               facecolor="none", edgecolor=INK2, linewidth=0.9, zorder=3)
        tot_m = sum(by[m].values())
        tot_b = sum(base.values())
        ax.set_title(f"{m} — {tot_m:.0f} ms ({tot_m - tot_b:+.0f})",
                     fontsize=7.5, color=INK, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("embedding", "emb")
                            .replace("seg_head", "head") for s in stages],
                           rotation=60, fontsize=6.2)
        ax.set_ylim(0, ymax)
    for j in range(len(methods), nr * nc):
        axes[j // nc][j % nc].set_visible(False)
    axes[0][0].set_ylabel("ms / frame", fontsize=7, color=INK2)
    fig.text(0.5, 1.0, f"filled = method, outline = {base_label} backbone "
             "(block-level, unscaled)", ha="center", fontsize=7,
             color=INK2)
    _save(fig, out, formats)
    plt.close(fig)


# paper \textwidth in inches; paper figures are drawn at print size
PAPER_W = 5.5
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                   "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 6.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "hatch.linewidth": 0.6,
    "hatch.color": SURFACE,
}
PAPER_LABEL = {"cpe": "spconv (CPE)", "attn": "attention", "others": "others",
               "ffn": "FFN", "select": "match", "reduce": "build",
               "restore": "restore", "retain": "cache maint."}


def plot_paper(rows, out, formats):
    """Breakdown figure at paper text width (PAPER_W):
    (a) latency per frame, stacked;
    (b) policy overhead per frame on a log axis, with its share;
    (c) peak memory with FlashAttention (model-weight tick, cache in MiB),
    only if flash runs exist. Rows follow the table order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(rows)
    has_mem = any(r.get("mem") for r in rows)
    with matplotlib.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(PAPER_W, 0.17 * n + 0.62))
        widths = [2.45, 1.05, 1.05] if has_mem else [2.9, 1.3]
        gs = fig.add_gridspec(1, len(widths), width_ratios=widths,
                              wspace=0.12, left=0.185, right=0.995,
                              top=1 - 0.2 / (0.17 * n + 0.62),
                              bottom=0.30 / (0.17 * n + 0.62) + 0.06)
        axes = [fig.add_subplot(gs[0, 0])]
        for j in range(1, len(widths)):
            axes.append(fig.add_subplot(gs[0, j], sharey=axes[0]))
        ys = list(range(n))[::-1]
        for axis in axes:
            axis.set_facecolor("white")
            axis.grid(axis="x", color=GRID, lw=0.4)
            axis.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                axis.spines[sp].set_visible(False)
            axis.spines["bottom"].set_color(INK2)
            axis.tick_params(colors=INK2, labelsize=5.8, length=1.5,
                             pad=1.5)
            axis.tick_params(axis="y", length=0)
        for axis in axes[1:]:
            plt.setp(axis.get_yticklabels(), visible=False)

        ax = axes[0]
        xmax = max(r["b"]["bar_ms"] for r in rows)
        for y, row in zip(ys, rows):
            b, left = row["b"], 0.0
            for k in COMPUTE_KEYS + POLICY_KEYS:
                w = b[k]
                if w <= 0:
                    continue
                ax.barh(y, w, left=left, height=0.66, color=COLOR[k],
                        edgecolor="white", linewidth=0.5,
                        hatch=HATCH.get(k))
                if k in COMPUTE_KEYS + ("reduce",) and w >= 0.11 * xmax:
                    ax.text(left + w / 2, y, f"{w:.0f}", ha="center",
                            va="center", fontsize=5.2, color=INK)
                left += w
            ax.text(b["bar_ms"] + 0.01 * xmax, y, f"{b['bar_ms']:.0f}",
                    va="center", fontsize=5.8, color=INK)
        ax.set_xlim(0, xmax * 1.08)
        ax.set_yticks(ys)
        ax.set_yticklabels([r["label"] for r in rows], fontsize=6.3,
                           color=INK)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlabel("(a) latency per frame (ms)", fontsize=6.3,
                      color=INK, labelpad=1.5)

        az = axes[1]
        pos = [r["b"]["policy_ms"] for r in rows if r["b"]["policy_ms"] > 0]
        lo, hi = (min(pos) / 3, max(pos) * 100) if pos else (0.5, 10)
        az.set_xscale("log")
        az.set_xlim(lo, hi)
        for y, row in zip(ys, rows):
            b = row["b"]
            v = b["policy_ms"]
            if v <= 0:
                az.text(lo * 1.2, y, "—", va="center", fontsize=5.8,
                        color=INK2)
                continue
            az.plot([lo, v], [y, y], color=GRID, lw=0.8, zorder=1)
            az.plot([v], [y], "o", ms=3.6, color=INK, mec="white",
                    mew=0.6, zorder=3)
            p = 100 * v / b["bar_ms"]
            az.text(v * 1.4, y, f"{v:.1f} ({p:.1f}%)" if p < 10
                    else f"{v:.0f} ({p:.0f}%)", va="center", fontsize=5.4,
                    color=INK2)
        az.set_xlabel("(b) policy overhead (ms)", fontsize=6.3, color=INK,
                      labelpad=1.5)

        if has_mem:
            am = axes[2]
            gmax = max(r["mem"]["peak"] for r in rows if r.get("mem"))
            for y, row in zip(ys, rows):
                m = row.get("mem")
                if not m:
                    continue
                gib = m["peak"] / 1024
                am.barh(y, gib, height=0.66, color=MEM_BAR,
                        edgecolor="white", linewidth=0.5)
                am.plot([m["model"] / 1024] * 2, [y - 0.33, y + 0.33],
                        color=INK, lw=0.8)
                txt = f"{gib:.2f}"
                if m["cache"] > 0:
                    txt += f" (+{m['cache']:.0f} MiB)"
                am.text(gib + 0.02 * gmax / 1024, y, txt, va="center",
                        fontsize=5.4, color=INK2)
            am.set_xlim(0, gmax / 1024 * 2.1)
            am.set_xlabel("(c) peak memory (GiB)", fontsize=6.3,
                          color=INK, labelpad=1.5)

        handles = [Patch(facecolor=COLOR[k], edgecolor="white",
                         hatch=HATCH.get(k), label=PAPER_LABEL[k])
                   for k in COMPUTE_KEYS + POLICY_KEYS]
        fig.legend(handles=handles, ncol=8, loc="upper center",
                   frameon=False, fontsize=5.8, bbox_to_anchor=(0.59, 1.0),
                   handlelength=1.1, handleheight=0.8, columnspacing=0.9,
                   handletextpad=0.35, borderaxespad=0.0)
        for ext in formats:
            fig.savefig(f"{out}.{ext}", dpi=300, facecolor="white")
        plt.close(fig)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


# --- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.02,
                    help="mIoU budget below the backbone (0.02 = -2 pt)")
    ap.add_argument("--acc_dir", default=f"{MAIN}/accuracy/{CELL}")
    ap.add_argument("--acc_mode", default="dense",
                    choices=("dense", "flash"),
                    help="attention tag of the accuracy runs")
    ap.add_argument("--pick_lat_dir",
                    default=f"{MAIN}/latency/{CELL}",
                    help="clean latency runs the picks are chosen from")
    ap.add_argument("--pick_by", choices=("flops", "latency"),
                    default="flops",
                    help="cost that selects the operating point within the "
                         "budget (flops: the main_table_outdoor.py rule)")
    ap.add_argument("--list_picks", action="store_true",
                    help="print the picks as TSV (label, run name, run.py "
                         "args, mIoU, ms) and exit")
    ap.add_argument("--block_dir",
                    default=f"output/efficiency/block/{CELL}")
    ap.add_argument("--fine_dir", default=None,
                    help="--fine_regions runs for the appendix tables")
    ap.add_argument("--lat_dirs", default=f"{MAIN}/latency/{CELL}",
                    help="comma-separated clean-run dirs, first hit wins "
                         "(bar heights and memory)")
    ap.add_argument("--mem_flash_dir", default=None,
                    help="FlashAttention runs (same run names with _flash), "
                         "the memory source of the figures")
    ap.add_argument("--paper_dir", default=None,
                    help="also write the paper-size figures here, e.g. "
                         "figure/experiment/breakdown")
    ap.add_argument("--rescale", choices=("none", "session", "method"),
                    default="none",
                    help="how bars map onto latency (see `scaled`)")
    ap.add_argument("--policy_stat", choices=("mean", "median"),
                    default="mean",
                    help="per-band statistic over the kept frames (see "
                         "`decompose`)")
    ap.add_argument("--drop_stalls", action="store_true",
                    help="earlier protocol: drop a block-level frame slower "
                         "than STALL x the same frame of the clean run and "
                         "cap cache-step outliers (default: every measured "
                         "frame counts)")
    ap.add_argument("--out", default=None,
                    help="output prefix (default output/efficiency/"
                         "breakdown_m<budget>pt)")
    ap.add_argument("--formats", default="png,pdf")
    a = ap.parse_args()

    picks, base_miou, thr = auto_picks(a.acc_dir, a.pick_lat_dir, a.budget,
                                       by=a.pick_by, acc_mode=a.acc_mode)
    if a.list_picks:
        for label, v, _s, miou, ms in picks:
            print(f"{label}\t{v}\t{cli_args(v)}\t{100 * miou:.2f}\t{ms:.1f}")
        return
    tag = f"m{a.budget * 100:g}pt".replace(".", "p")
    out = a.out or f"output/efficiency/breakdown_{tag}"
    lat_dirs = [d for d in a.lat_dirs.split(",") if d]

    rows, missing = [], []
    session = 1.0      # backbone table latency / backbone profile (first row)
    for label, v, sites, miou, _ms in picks:
        br = _load([a.block_dir], v)
        cr = _load(lat_dirs, v)
        if br is None:
            missing.append(f"block  {v}")
            continue
        if cr is None:
            missing.append(f"clean  {v}")
        dec = decompose(br, a.policy_stat,
                        _clean_per_frame(cr) if a.drop_stalls else None)
        if family(v)[0] == naming.BACKBONE and dec["total"] and _clean_ms(cr):
            session = _clean_ms(cr) / dec["total"]
        b = scaled(dec, _clean_ms(cr), a.rescale, session)
        fr = (_load([a.mem_flash_dir], naming.rename(v, attention="flash"))
              if a.mem_flash_dir else None)
        if a.mem_flash_dir and memory(fr) is None:
            missing.append(f"flash  {naming.rename(v, attention='flash')}")
        rows.append(dict(label=label, var=v, sites=sites, miou=miou, b=b,
                         mem=memory(fr), mem_nf=memory(cr), block=br))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    keys = COMPUTE_KEYS + POLICY_KEYS
    hdr = (["method", "pick", "mIoU", "in-forward ms", "frame ms", "table ms"]
           + [LABEL[k] for k in keys]
           + ["policy ms", "policy %", "runs", "scale", "frames kept/dropped"])
    trows = []
    for r in rows:
        b = r["b"]
        trows.append([r["label"], r["var"], f"{100 * r['miou']:.2f}",
                      f"{b['frame_ms']:.1f}", f"{b['bar_ms']:.1f}",
                      _f(b.get("table_ms"), 1)]
                     + [f"{b[k]:.2f}" for k in keys]
                     + [f"{b['policy_ms']:.2f}",
                        f"{100 * b['policy_ms'] / b['bar_ms']:.1f}",
                        r["sites"], f"{b['scale']:.3f}",
                        f"{b.get('frames_kept')}/{b.get('frames_dropped')}"])

    mhdr = ["method", "model (MiB)", "working (MiB)", "cache (MiB)",
            "peak (MiB)", "Δpeak vs backbone", "Δworking", "peak stage"]

    def mem_rows(key):
        base = next((r[key] for r in rows
                     if r[key] and family(r["var"])[0] == naming.BACKBONE),
                    None)
        out_rows = []
        for r in rows:
            m = r[key]
            if not m:
                out_rows.append([r["label"]] + ["—"] * 7)
                continue
            cache = (f"{m['cache']:.1f}" if not m["cache_d4"]
                     else f"{m['cache']:.1f} ({m['cache_d3']:.1f}+"
                          f"{m['cache_d4']:.1f})")
            out_rows.append([
                r["label"], f"{m['model']:.0f}", f"{m['working']:.0f}",
                cache, f"{m['peak']:.0f}",
                f"{m['peak'] - base['peak']:+.0f}" if base else "—",
                f"{m['working'] - base['working']:+.0f}" if base else "—",
                m["peak_stage"] or "—"])
        return out_rows

    mrows = mem_rows("mem")            # FlashAttention (paper)
    mrows_nf = mem_rows("mem_nf")      # dense attention, reference only

    stall_note = (f"; stall filter: block-level frame > {STALL:g} x the "
                  "same clean frame dropped, cache-step values > 10x their "
                  "median set to the median" if a.drop_stalls else "")
    lines = [f"# Efficiency breakdown — {CELL}, budget -{a.budget * 100:g} pt",
             "",
             f"backbone mIoU {100 * base_miou:.2f} -> picks need >= "
             f"{100 * thr:.2f} (cheapest by {a.pick_by}); bands: per-frame "
             f"decomposition, {a.policy_stat} over the measured frames "
             f"(every pass after the warm-up pass{stall_note}); "
             f"bars: `--rescale {a.rescale}` (`scale`; `table ms` = main-"
             "table latency for reference); "
             "frame ms = in-forward time + Commit Gate (the latency of the tables).",
             "", "## Latency (ms / frame)", "", md_table(hdr, trows), "",
             "## Memory with FlashAttention (figure / paper; flash runs, "
             "pass-1 metrics)", "",
             (md_table(mhdr, mrows) if a.mem_flash_dir
              else "_no --mem_flash_dir given_"), "",
             "## Memory without FlashAttention (reference only — dense "
             "K x K attention dominates it and flatters merging)", "",
             md_table(mhdr, mrows_nf), ""]

    # --- appendix -----------------------------------------------------------
    fine_all, stage_all = [], []
    for label, v, _s, _m, _ms in picks:
        fr = _load([a.fine_dir], v) if a.fine_dir else None
        br = _load([a.block_dir], v)
        if a.fine_dir and fr is None:
            missing.append(f"fine   {v}")
        # SpaceCache sub-steps are the same regions at either level; take
        # them from the fine run when present, else the block run
        src = fr or br
        if src is not None and (fr is not None
                                or src.get("spacecache_ms")):
            tgt = next((rr["b"] for rr in rows if rr["label"] == label),
                       None) if fr is not None else None
            fine_all += fine_rows(label, src, tgt)
        if br is not None:
            stage_all += stage_rows(label, br)
    if fine_all:
        lines += ["## Appendix A — policy sub-steps (scaled per band onto "
                  "the figure)",
                  "",
                  "Coarse regions are the figure's bands; `└` rows are their "
                  "sub-steps (`--fine_regions`), scaled per band onto the "
                  "figure (fine-run band x figure / fine). calls / frame show "
                  "how often each op runs; us / call is its unit cost.", "",
                  md_table(["method", "band", "op", "ms / frame",
                            "calls / frame", "us / call"],
                           [(m, b, op, _f(ms), _f(n, 0), _f(us, 1))
                            for m, b, op, ms, n, us in fine_all]), ""]
    if stage_all:
        stages = []
        for row in stage_all:
            if row[1] not in stages:
                stages.append(row[1])
        methods = []
        for row in stage_all:
            if row[0] not in methods:
                methods.append(row[0])
        tot = {(r[0], r[1]): r for r in stage_all}
        lines += ["## Appendix B — per-stage time (block-level, unscaled)",
                  "",
                  "stage total ms, and ATTN ms in brackets. Merging only "
                  "moves ATTN (and its own overhead); SpaceCache shrinks every"
                  " band of the cached stages. Full split in the CSV.", "",
                  md_table(["method"] + stages,
                           [[m] + [(f"{tot[(m, s)][2]:.1f} "
                                    f"({tot[(m, s)][4]:.1f})"
                                    if (m, s) in tot else "—")
                                   for s in stages] for m in methods]), ""]
    if missing:
        lines += ["## missing runs", ""] + [f"- `{m}`" for m in missing]

    with open(out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(trows)
        w.writerow([])
        w.writerow(["memory_flash"] + mhdr)
        w.writerows([["flash"] + r for r in mrows])
        w.writerow([])
        w.writerow(["memory_dense_reference"] + mhdr)
        w.writerows([["dense"] + r for r in mrows_nf])
        if fine_all:
            w.writerow([])
            w.writerow(["method", "band", "op", "ms_frame", "calls_frame",
                        "us_call"])
            w.writerows(fine_all)
        if stage_all:
            w.writerow([])
            w.writerow(["method", "stage", "total_ms", "cpe_ms", "attn_ms",
                        "ffn_ms", "cache_ms", "rest_ms"])
            w.writerows(stage_all)
    text = "\n".join(lines)
    with open(out + ".md", "w") as f:
        f.write(text + "\n")
    print(text)
    fmts = [x for x in a.formats.split(",") if x]
    if rows:
        plot(rows, out, fmts)
    if fine_all:
        plot_fine(fine_all, out + "_appendix_substeps", fmts)
    if stage_all:
        plot_stages(stage_all, out + "_appendix_stages", fmts)
    print(f"\nSaved {out}.{{md,csv,{a.formats}}}, {out}_appendix_substeps.* "
          f"and {out}_appendix_stages.*")
    if a.paper_dir and rows:
        import matplotlib
        os.makedirs(a.paper_dir, exist_ok=True)
        pfmt = ["pdf", "png"]
        plot_paper(rows, os.path.join(a.paper_dir, "breakdown"), pfmt)
        with matplotlib.rc_context(PAPER_RC):
            if fine_all:
                plot_fine(fine_all, os.path.join(
                    a.paper_dir, "breakdown_appendix_substeps"), pfmt,
                    width=PAPER_W)
            if stage_all:
                plot_stages(stage_all, os.path.join(
                    a.paper_dir, "breakdown_appendix_stages"), pfmt,
                    width=PAPER_W)
        print(f"Saved paper figures -> {a.paper_dir}/breakdown*.{{pdf,png}}")


if __name__ == "__main__":
    main()
