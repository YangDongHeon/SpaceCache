"""Evaluate one method with Pointcept's `SemSegEvaluator` and save its results.

The evaluator runs on `cfg.data.val` with batch size 1 and no test-time
augmentation. One method per invocation; each run writes
`<output_dir>/<run name>/results.json` (mIoU/mAcc/allAcc, per-class IoU,
per-frame timing, FLOPs / tokens / peak memory, SpaceCache statistics).
The run name lists the method and its settings (spacecache/common/naming.py).

    PYTHONPATH=. python spacecache/experiments/run.py \\
        --config spacecache/configs/waymo/utonia_dec/latency_slice.py \\
        --policy backbone --dense_attn

    PYTHONPATH=. python spacecache/experiments/run.py \\
        --config spacecache/configs/waymo/utonia_dec/latency_slice.py \\
        --policy spacecache_s34 --lifetime 8 --confidence 0.85 --dense_attn
"""

from __future__ import annotations

import os
import sys


def _peek_timing_level(default="block"):
    if "--no_timing" in sys.argv:
        return "frame"
    for i, a in enumerate(sys.argv):
        if a == "--timing_level" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--timing_level="):
            return a.split("=", 1)[1]
    return default


# The token-merging regions are timed only at `--timing_level block`; the flag
# must be set before `token_merging_algos` is imported.
os.environ.setdefault(
    "TOME_PROFILE", "1" if _peek_timing_level() == "block" else "0")
if "--fine_regions" in sys.argv:
    os.environ.setdefault("TOME_PROFILE_FINE", "1")

# Avoid allocator fragmentation over thousands of frames of varying size.
# Not on Tegra (Jetson), whose driver does not support expandable segments.
if not os.path.exists("/etc/nv_tegra_release"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import inspect  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
))

from pointcept.datasets import build_dataset, collate_fn  # noqa: E402
import pointcept.datasets.waymo  # noqa: F401,E402
from pointcept.engines.hooks.evaluator import SemSegEvaluator  # noqa: E402
from pointcept.utils.config import Config  # noqa: E402
from pointcept.utils.events import EventStorage  # noqa: E402
from pointcept.utils.logger import get_root_logger  # noqa: E402

from spacecache.common import naming, region_key  # noqa: E402
from spacecache.common.helpers import (  # noqa: E402
    build_segmentor, prepare_input, set_seed,
)
from spacecache.common.profiling import FrameProfiler  # noqa: E402
from spacecache.common.runtime import NON_MODEL_KEYS, drop_keys  # noqa: E402
from spacecache.common.token_merging_algos import (  # noqa: E402
    MERGE_REDUCERS, set_merge_reduce, set_tome_compute_bf16,
)
from spacecache.common.litept_tome_attention import (  # noqa: E402
    install_litept_dense, is_litept_attention, set_litept_compute_dtype,
)
from spacecache.common.cost_meter import (  # noqa: E402
    CostMeter, cache_bytes, summarize_cost,
)
from spacecache.policies import build_policy  # noqa: E402
from spacecache.policies.spacecache import SpaceCachePolicy  # noqa: E402


@contextlib.contextmanager
def policy_override(model, policy, ctx, stages=None):
    """Route every encoder and decoder `Block.forward` through
    `policy.run_block`; restore on exit. `stages` (e.g. {"enc3"}) restricts
    the patch to those stages."""
    handles = []
    stage_iters = [model.backbone.enc._modules.items()]
    if hasattr(model.backbone, "dec"):
        stage_iters.append(model.backbone.dec._modules.items())
    for stage_name, stage_seq in itertools.chain(*stage_iters):
        if stages is not None and stage_name not in stages:
            continue
        for sub_name, sub_mod in stage_seq._modules.items():
            if not sub_name.startswith("block"):
                continue

            def make_bound(_policy, _ctx):
                def wrapped(self, point):
                    return _policy.run_block(self, point, _ctx)
                return wrapped

            handles.append((sub_mod, sub_mod.forward))
            sub_mod.forward = types.MethodType(make_bound(policy, ctx),
                                               sub_mod)
    try:
        yield
    finally:
        for sub_mod, orig in handles:
            sub_mod.forward = orig


def summarize_timing(records, log):
    """Mean over frames of FrameProfiler's per-frame records.
    encoder = embedding + enc*, decoder = dec* + seg_head."""
    if not records:
        return None
    n = len(records)

    def _mean(key_fn):
        return sum(key_fn(r) for r in records) / n

    labels = set()
    for r in records:
        labels.update(r.get("modules_ms", {}).keys())
    per_module = {
        lab: round(sum(r.get("modules_ms", {}).get(lab, 0.0)
                       for r in records) / n, 5)
        for lab in labels
    }
    stage_ms = {k[len("stage/"):]: v for k, v in per_module.items()
                if k.startswith("stage/")}
    enc_ms = sum(v for k, v in stage_ms.items()
                 if k == "embedding" or k.startswith("enc"))
    dec_ms = sum(v for k, v in stage_ms.items()
                 if k.startswith("dec") or k == "seg_head")
    total_ms = _mean(lambda r: r.get("total_ms", 0.0))
    totals = sorted(r.get("total_ms", 0.0) for r in records)
    total_p50 = totals[n // 2] if n % 2 else (totals[n // 2 - 1]
                                             + totals[n // 2]) / 2
    total_p95 = totals[max(0, math.ceil(0.95 * n) - 1)]

    attn_labels = set()
    for r in records:
        attn_labels.update(r.get("attn_breakdown_ms", {}).keys())
    attn_ms = {
        lab: round(sum(r.get("attn_breakdown_ms", {}).get(lab, 0.0)
                       for r in records) / n, 5)
        for lab in attn_labels
    }
    # sub-regions of the merge steps (--fine_regions), kept separate so they
    # are not double-counted with attn_breakdown_ms
    fine_labels = set()
    for r in records:
        fine_labels.update(r.get("attn_fine_ms", {}).keys())
    fine_ms = {
        lab: round(sum(r.get("attn_fine_ms", {}).get(lab, 0.0)
                       for r in records) / n, 5)
        for lab in fine_labels
    }

    log.info("timing/frame (%d frames): total=%.2f  enc=%.2f  dec=%.2f ms"
             "%s", n, total_ms, enc_ms, dec_ms,
             ("  attn=" + ",".join(f"{k.split('/')[-1]}={v:.2f}"
                                   for k, v in sorted(attn_ms.items()))
              if attn_ms else ""))
    out = dict(
        n_frames=n, total_ms=round(total_ms, 5),
        total_p50=round(total_p50, 5), total_p95=round(total_p95, 5),
        per_frame_ms=[round(t, 3) for t in
                      (r.get("total_ms", 0.0) for r in records)],
        encoder_ms=round(enc_ms, 5), decoder_ms=round(dec_ms, 5),
        stage_ms=stage_ms, module_ms=per_module, attn_breakdown_ms=attn_ms,
    )
    if fine_ms:
        out["attn_fine_ms"] = fine_ms
        log.info("  fine regions: %s", ",".join(
            f"{k[len('fine/'):]}={v:.2f}x{fine_ms.get('n/' + k[len('fine/'):], 0):.0f}"
            for k, v in sorted(
                ((k, v) for k, v in fine_ms.items() if k.startswith("fine/")),
                key=lambda kv: -kv[1])[:10]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", default=naming.BACKBONE,
                    choices=naming.POLICIES,
                    help="method: backbone | spacecache_s3 | spacecache_s34 "
                         "| spacecache_s{0,1,2,4} (stage ablation) | tome | "
                         "algm | gitmerge3d | gitmerge3d_adaptive")
    # --- SpaceCache --------------------------------------------------------
    ap.add_argument("--lifetime", type=float, default=None,
                    help="SpaceCache lifetime L in frames ('inf' allowed). "
                         "Default: config `lifetime`, else 4.")
    ap.add_argument("--confidence", type=float, default=None,
                    help="SpaceCache confidence threshold θ: regions whose "
                         "mean max-softmax probability is <= θ are not "
                         "stored and are evicted. 0 disables the rule. "
                         "Default: config `confidence`, else 0.")
    ap.add_argument("--region_scale", type=float, default=1.0,
                    help="SpaceCache region size relative to the voxel size "
                         "of the cached stage (1.0 in the paper)")
    ap.add_argument("--expiry", default="staggered",
                    choices=("staggered", "fixed"),
                    help="SpaceCache lifetime rule: staggered = expiry "
                         "staggered by the region index (0..L-1 reuses per "
                         "entry, the paper); fixed = every entry is reused "
                         "for exactly L-1 frames")
    ap.add_argument("--region_key", default="packed",
                    choices=("packed", "prime"),
                    help="SpaceCache region key: packed = one-to-one "
                         "bit-field (|vx|,|vy| < 2^23, |vz| < 2^14 cells, "
                         "the paper); prime = a linear prime hash")
    ap.add_argument("--triton_cache", action="store_true",
                    help="SpaceCache: run the cache operations with the fused "
                         "Triton kernels in spacecache/common/triton_cache.py "
                         "(falls back to PyTorch if unavailable)")
    ap.add_argument("--region_stats", action="store_true",
                    help="SpaceCache: count dynamic/static region accuracy, "
                         "hit accuracy by entry age and the evicted regions "
                         "(results.json `spacecache_region_stats`, Table 3). "
                         "Does not change predictions.")
    ap.add_argument("--reuse_agreement", action="store_true",
                    help="SpaceCache: Reuse Agreement probe (Fig. 5). A "
                         "second forward per frame with the cache disarmed "
                         "scores every served token against the backbone "
                         "feature (Appendix A.2), binned by entry age, "
                         "confidence and region type -> reuse_agreement.npz "
                         "/ _frames.jsonl. Needs --no_timing --no_metrics "
                         "and one pass; predictions are unchanged.")
    ap.add_argument("--ra_min_cone_tokens", type=int, default=20,
                    help="--reuse_agreement: pure tokens a class needs in "
                         "the frame to get a prototype")
    ap.add_argument("--ra_temp_json",
                    default="output/motivation/reuse_agreement/"
                            "waymo_utonia.json",
                    help="--reuse_agreement: cone_affinity.py output whose "
                         "settings.temp_per_stage gives the per-stage T "
                         "(scripts/reproduce/motivation.sh)")
    ap.add_argument("--ra_temps", default=None,
                    help="--reuse_agreement: explicit per-stage T, e.g. "
                         "'3:0.1035,4:0.0525' (overrides --ra_temp_json)")
    # --- token merging -----------------------------------------------------
    ap.add_argument("--merge_ratio", type=float, default=None,
                    help="ToMe / GitMerge3D: fraction of each patch's tokens "
                         "merged")
    ap.add_argument("--merge_stages", default="all",
                    help="ToMe / GitMerge3D: stages that merge, e.g. 'enc3' "
                         "or 'enc3,dec3'. 'all' = every encoder and decoder "
                         "block.")
    ap.add_argument("--proportional_attn", action="store_true",
                    help="ToMe: track token sizes and add ToMe's log(size) "
                         "proportional-attention bias (dense attention "
                         "path; off in the paper)")
    ap.add_argument("--tome_clamp_patch", action="store_true",
                    help="ToMe on LitePT: merge a stage shorter than one "
                         "attention patch as a single patch (needed on "
                         "indoor frames, Table 2)")
    ap.add_argument("--merge_reduce", choices=MERGE_REDUCERS,
                    default="scatter",
                    help="GitMerge3D merge reduction: scatter (the official "
                         "scatter_reduce, the paper) or the atomic-free "
                         "segment_csr / segment_csr_fused / cumsum variants")
    ap.add_argument("--algm_tau", type=float, default=None,
                    help="ALGM: local-merge cosine-similarity threshold τ "
                         "(default 0.99)")
    ap.add_argument("--algm_wsize", type=int, default=None,
                    help="ALGM, serial windows: window size in serialized "
                         "tokens (default 4)")
    ap.add_argument("--algm_clap_mode", default=None,
                    choices=("serial", "grid"),
                    help="ALGM local-merge windows: 'grid' (default) = "
                         "--algm_cell-sided voxel cubes; 'serial' = "
                         "--algm_wsize consecutive serialized tokens")
    ap.add_argument("--algm_cell", type=int, default=None,
                    help="ALGM, grid windows: cube side in input voxels "
                         "(default 4)")
    ap.add_argument("--algm_cell_offset", type=int, default=None,
                    help="ALGM, grid windows: phase offset in voxels, "
                         "(coord + offset) // cell (default 2)")
    # --- attention, data, measurement ---------------------------------------
    ap.add_argument("--dense_attn", action="store_true",
                    help="dense attention instead of FlashAttention (the "
                         "paper's accuracy, latency and FLOPs runs)")
    ap.add_argument("--fp32", action="store_true",
                    help="fp32 attention. Default follows the backbone: "
                         "PT-v3 bf16, LitePT fp16.")
    ap.add_argument("--upcast_softmax", action="store_true",
                    help="LitePT dense path: softmax in fp32")
    ap.add_argument("--skip_existing", action="store_true",
                    help="exit if <output_dir>/<run name>/results.json exists")
    ap.add_argument("--scene_shard", default=None, metavar="K/N",
                    help="evaluate only the scenes with index %% N == K "
                         "(whole scenes); merge with merge_shards.py")
    ap.add_argument("--shard_balance", action="store_true",
                    help="--scene_shard by frame count (largest scene to the "
                         "lightest shard) instead of round-robin")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="evaluate only the first N frames")
    ap.add_argument("--warmup_frames", type=int, default=0,
                    help="exclude the first N frames from the aggregates "
                         "(they still run)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="unprofiled forwards before the eval (GPU clocks, "
                         "kernel loading)")
    ap.add_argument("--passes", type=int, default=1,
                    help="run the val loader N times in one process; the "
                         "SpaceCache state is reset between passes, "
                         "FLOPs/memory are measured on pass 1 only, and the "
                         "latency of the tables is the mean of passes 2..N")
    ap.add_argument("--no_timing", action="store_true",
                    help="accuracy-only run (no timing instrumentation)")
    ap.add_argument("--profile_regions", action="store_true",
                    help="record_function scope per profiled region and a "
                         "per-stage GPU kernel-time table (implies "
                         "--torch_profile)")
    ap.add_argument("--export_trace", default=None,
                    help="write the torch.profiler chrome trace here "
                         "(implies --torch_profile)")
    ap.add_argument("--profile_pass", type=int, default=-1,
                    help="profile only this 0-based eval pass (-1 = all)")
    ap.add_argument("--graph_capture", action="store_true",
                    help="capture and replay whole frames as CUDA graphs "
                         "(spacecache/common/model_graph.py)")
    ap.add_argument("--graph_frame", type=int, default=-1,
                    help="eval frame to capture (-1: a late frame for "
                         "SpaceCache, the middle frame otherwise)")
    ap.add_argument("--graph_frames", default="",
                    help="several frames: '0,40,80' or 'stride:K[:OFF]'; "
                         "overrides --graph_frame")
    ap.add_argument("--graph_mode", default="replay",
                    choices=("replay", "piecewise"),
                    help="replay = one graph with host-decided values "
                         "recorded; piecewise = the whole inference as a "
                         "chain of graphs split at every host step, replayed "
                         "with those steps run for real")
    ap.add_argument("--graph_no_kernel_acct", action="store_true",
                    help="skip the per-pick CUDA-only profiler windows")
    ap.add_argument("--graph_debug", action="store_true",
                    help="keep diagnostic reductions inside the graph")
    ap.add_argument("--graph_replays", type=int, default=200,
                    help="replays per captured frame (median is reported)")
    ap.add_argument("--graph_warmup", type=int, default=3,
                    help="side-stream warmup forwards before capture")
    ap.add_argument("--timing_level", choices=("frame", "stage", "block"),
                    default="block",
                    help="frame = forward total; stage = + per stage; "
                         "block = + per block and merge regions")
    ap.add_argument("--fine_regions", action="store_true",
                    help="also time the sub-steps of each merge region "
                         "(timing.attn_fine_ms)")
    ap.add_argument("--no_metrics", action="store_true",
                    help="skip FLOPs / token / peak-memory accounting")
    ap.add_argument("--torch_profile", action="store_true",
                    help="wrap the eval in torch.profiler and print a "
                         "per-CUDA-op table")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--worker_ctx", default=None,
                    choices=["fork", "spawn", "forkserver"],
                    help="DataLoader multiprocessing context")
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg.seed (tags the run name _seed<N>)")
    ap.add_argument("--output_dir", default=None,
                    help="parent of the run directory. Default: config "
                         "`output_root`, else output/runs")
    args = ap.parse_args()
    region_key.set_mode(args.region_key)

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    cfg = Config.fromfile(args.config)
    device = torch.device(cfg.get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 37))
    set_seed(seed)

    weight = cfg.get("weight")
    assert weight, "config must specify `weight`"

    # --- method and run name -------------------------------------------------
    policy = build_policy(args.policy)
    intercept = getattr(policy, "intercept", "block")
    is_streaming = (isinstance(intercept, str)
                    and intercept.startswith("enc"))
    uses_ratio = args.policy in (naming.TOME, naming.GITMERGE3D,
                                 naming.GITMERGE3D_ADAPTIVE)
    if uses_ratio and args.merge_ratio is None:
        raise SystemExit(f"--policy {args.policy} needs --merge_ratio")
    r = args.merge_ratio if uses_ratio else None
    attention = "dense" if args.dense_attn else "flash"
    lifetime = (args.lifetime if args.lifetime is not None
                else float(cfg.get("lifetime", 4)))
    confidence = (args.confidence if args.confidence is not None
                  else float(cfg.get("confidence", 0.0)))

    def _stage_key(s):
        # "dN" is an alias for "encN"
        s = s.strip()
        return f"enc{s[1:]}" if (s[:1] == "d" and s[1:].isdigit()) else s

    stage_tags = None
    if intercept != "block" or args.merge_stages == "all":
        merge_stages = None
    else:
        stage_tags = [s.strip() for s in args.merge_stages.split(",")]
        merge_stages = {_stage_key(t) for t in stage_tags}

    name_kw = dict(merge_ratio=r, merge_stages=stage_tags,
                   proportional_attn=bool(args.proportional_attn),
                   clamp_patch=bool(args.tome_clamp_patch), seed=args.seed)
    if args.policy in (naming.GITMERGE3D, naming.GITMERGE3D_ADAPTIVE):
        name_kw["merge_reduce"] = args.merge_reduce
    if args.policy == naming.ALGM:
        name_kw["tau"] = (args.algm_tau if args.algm_tau is not None
                          else policy.tau)
        if (args.algm_clap_mode or policy.clap_mode) == "grid":
            if args.algm_cell not in (None, policy.cell):
                name_kw["algm_cell"] = args.algm_cell
            if args.algm_cell_offset not in (None, policy.cell_offset):
                name_kw["algm_cell_offset"] = args.algm_cell_offset
        else:
            name_kw["algm_window"] = (args.algm_wsize if args.algm_wsize
                                      is not None else policy.wsize)
    if is_streaming:
        triton_on = False
        if args.triton_cache:
            from spacecache.common import triton_cache as _tc
            triton_on = _tc.is_available() and device.type == "cuda"
        name_kw.update(lifetime=lifetime, confidence=confidence,
                       region_scale=args.region_scale, expiry=args.expiry,
                       triton=triton_on, region_key=args.region_key)
    shard = (tuple(int(x) for x in args.scene_shard.split("/"))
             if args.scene_shard else None)
    variant = naming.run_name(args.policy, attention, shard=shard, **name_kw)

    base_out = args.output_dir or cfg.get("output_root", "output/runs")
    output_dir = os.path.join(base_out, variant)
    if args.skip_existing and os.path.isfile(
            os.path.join(output_dir, "results.json")):
        print(f"skip_existing: {output_dir} already has results.json")
        return
    os.makedirs(output_dir, exist_ok=True)

    logger = get_root_logger(
        log_file=os.path.join(output_dir, "run.log"), file_mode="w")
    logger.info("policy=%s run=%s merge_ratio=%s attention=%s",
                policy.name, variant, r, attention)

    # --- model ----------------------------------------------------------------
    model = build_segmentor(cfg.model, weight, device)
    is_litept = any(is_litept_attention(m) for m in model.modules())
    logger.info("backbone: %s", "LitePT-v1" if is_litept else "PT-v3m3")

    if args.dense_attn:
        if is_litept:
            n = install_litept_dense(
                model, upcast_softmax=bool(args.upcast_softmax))
            logger.info("--dense_attn: LitePT dense attention installed on "
                        "%d PointROPEAttention modules", n)
        else:
            n = 0
            for m in model.modules():
                if hasattr(m, "enable_flash") and m.enable_flash:
                    m.enable_flash = False
                    if not hasattr(m, "patch_size_max"):
                        m.patch_size_max = int(m.patch_size)
                    if isinstance(getattr(m, "attn_drop", None), float):
                        m.attn_drop = torch.nn.Dropout(m.attn_drop)
                    n += 1
            logger.info("--dense_attn: disabled flash on %d attention modules",
                        n)

    # Attention dtype follows each backbone's own convention (PT-v3: bf16,
    # LitePT: fp16), so the dense path reproduces the flash baseline.
    set_merge_reduce(args.merge_reduce)
    if args.merge_reduce != "scatter":
        logger.info("GitMerge3D merge reduction: %s", args.merge_reduce)
    if args.fp32:
        set_tome_compute_bf16(False)
        set_litept_compute_dtype(None)
        logger.info("fp32 attention (--fp32)")
    elif is_litept:
        set_litept_compute_dtype(torch.float16)
        logger.info("fp16 attention (LitePT convention)")
    else:
        n = 0
        for m in model.modules():
            if hasattr(m, "compute_dtype"):
                m.compute_dtype = torch.bfloat16
                n += 1
        set_tome_compute_bf16(True)
        logger.info("bf16 attention on %d modules", n)

    # --- val loader (bs=1, no TTA) ----------------------------------------------
    full_val = build_dataset(cfg.data.val)
    n_total = len(full_val)
    loader_data = full_val
    # Scenes are never split across shards, so SpaceCache sees the same
    # per-sequence timeline as in an unsharded run.
    shard_seg_ids = None
    if args.scene_shard:
        sk, sn = (int(x) for x in args.scene_shard.split("/"))
        if not (0 <= sk < sn):
            raise SystemExit(
                f"--scene_shard needs 0 <= K < N, got {args.scene_shard}")
        seg_all = SpaceCachePolicy._build_seg_ids(full_val)
        if seg_all is None or len(seg_all) != n_total:
            raise SystemExit("--scene_shard needs per-sample scene ids")
        order = list(dict.fromkeys(seg_all))
        if args.shard_balance:
            n_of, pos = {}, {s: j for j, s in enumerate(order)}
            for s_ in seg_all:
                n_of[s_] = n_of.get(s_, 0) + 1
            load, owner = [0] * sn, {}
            for s_ in sorted(order, key=lambda x: (-n_of[x], pos[x])):
                j = min(range(sn), key=lambda q: (load[q], q))
                owner[s_] = j
                load[j] += n_of[s_]
            keep = {s_ for s_ in order if owner[s_] == sk}
            logger.info("--shard_balance: frames per shard %s", load)
        else:
            keep = {s for j, s in enumerate(order) if j % sn == sk}
        shard_idxs = [i for i, s in enumerate(seg_all) if s in keep]
        shard_seg_ids = [seg_all[i] for i in shard_idxs]
        loader_data = torch.utils.data.Subset(full_val, shard_idxs)
        n_total = len(loader_data)
        logger.info("--scene_shard %d/%d: %d/%d scenes -> %d frames (of %d)",
                    sk, sn, len(keep), len(order), n_total, len(full_val))
    max_frames = (args.max_frames if args.max_frames is not None
                  else cfg.get("max_frames", None))
    if max_frames is not None and max_frames > 0:
        seq = list(range(min(int(max_frames), n_total)))
        loader_data = torch.utils.data.Subset(full_val, seq)
        logger.info("--max_frames: frames [%d..%d] = %d forwards (first %d "
                    "dropped from the aggregates)", seq[0], seq[-1],
                    len(seq), int(args.warmup_frames))
    val_loader = torch.utils.data.DataLoader(
        loader_data, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
        multiprocessing_context=args.worker_ctx)
    logger.info("val frames: %d (of %d)", len(val_loader), n_total)

    cfg.save_path = output_dir
    if not hasattr(cfg, "empty_cache"):
        cfg.empty_cache = False

    # minimal trainer stand-in: SemSegEvaluator only reads these fields
    shim = types.SimpleNamespace(
        model=model, val_loader=val_loader, logger=logger, cfg=cfg,
        writer=None, epoch=0, comm_info={}, best_metric_value=0.0)

    evaluator = SemSegEvaluator()
    evaluator.trainer = shim

    ctx = dict(timer=None, merge_ratio=(r if r is not None else 0.0),
               merge_stages=merge_stages,
               tome_size=bool(args.proportional_attn),
               tome_clamp_patch=bool(args.tome_clamp_patch))
    if args.algm_tau is not None:
        ctx["algm_tau"] = float(args.algm_tau)
    if args.algm_wsize is not None:
        ctx["algm_wsize"] = int(args.algm_wsize)
    if args.algm_clap_mode is not None:
        ctx["algm_clap_mode"] = str(args.algm_clap_mode)
    if args.algm_cell is not None:
        ctx["algm_cell"] = int(args.algm_cell)
    if args.algm_cell_offset is not None:
        ctx["algm_cell_offset"] = int(args.algm_cell_offset)

    if is_streaming:
        prep_kw = dict(region_scale=args.region_scale, lifetime=lifetime)
        sig = inspect.signature(policy.prepare).parameters
        if "triton_cache" in sig:
            prep_kw["triton_cache"] = args.triton_cache
        if "expiry" in sig:
            prep_kw["expiry"] = args.expiry
        policy.prepare(model, cfg, full_val, device, **prep_kw)
        if args.triton_cache and not getattr(policy, "_triton", False):
            logger.warning("--triton_cache requested but triton/CUDA is "
                           "unavailable; using the torch cache path")
        if shard_seg_ids is not None:
            # the sequence table is indexed by eval position
            policy.seg_ids = shard_seg_ids
        if args.region_stats:
            policy.region_diag = True
            logger.info("region statistics on")
        if confidence > 0:
            policy.configure_invalidation(confidence, "conf", seed)
            logger.info("confidence rule: theta=%.3f", confidence)
        cell = getattr(policy, "voxel_size_d3", 0.0)
        logger.info("SpaceCache: policy=%s intercept=%s L=%s region_scale=%s "
                    "coord_scale=%g cell=%g sequences=%s exclude_unlabeled=%s",
                    policy.name, intercept,
                    lifetime, policy.mf, policy.coord_scale, cell,
                    (len(set(policy.seg_ids)) if policy.seg_ids else "?"),
                    policy.exclude_ignore)
    ra_probe = None
    if args.reuse_agreement:
        if not (is_streaming and hasattr(policy, "_cell_confidence")):
            raise SystemExit(
                f"--reuse_agreement is not supported by policy {policy.name}")
        if not (args.no_timing and args.no_metrics) or args.passes != 1:
            raise SystemExit("--reuse_agreement is an accuracy-run probe: "
                             "pass --no_timing --no_metrics and one pass")
        from spacecache.experiments.reuse_agreement.reuse_agreement import (
            ReuseAgreementProbe, load_temps)
        ra_probe = ReuseAgreementProbe(
            policy, model, cfg, device,
            temps=load_temps(args.ra_temps, args.ra_temp_json),
            min_cone_tokens=args.ra_min_cone_tokens)
        logger.info("Reuse Agreement probe on (T=%s)", ra_probe.temps)
    # ALGM / ToMe install their own multi-site intercepts; everything else
    # goes through the generic block override.
    override_cm = (policy.install_override(model, ctx)
                   if hasattr(policy, "install_override")
                   else policy_override(model, policy, ctx,
                                        stages=merge_stages))
    stream_cm = (policy.streaming_context(model) if is_streaming
                 else contextlib.nullcontext())
    # entered after stream_cm: its hooks must follow the gates
    ra_cm = (ra_probe.context() if ra_probe is not None
             else contextlib.nullcontext())

    prof = (None if args.no_timing
            else FrameProfiler(model, device, level=args.timing_level,
                               fine_regions=args.fine_regions,
                               record_regions=args.profile_regions))
    if prof is not None:
        logger.info("FrameProfiler: %s", prof.rf_note)

    def _extra_bytes():
        out = {"cache_bytes": cache_bytes(policy)}
        c4 = getattr(policy, "cache4", None)
        if c4 is not None and hasattr(c4, "nbytes"):
            out["cache_bytes_d4"] = int(c4.nbytes())
        return out

    # registered after FrameProfiler so its pre-hook does not reset a peak
    # the profiler still needs
    metrics = None if args.no_metrics else CostMeter(
        model, device,
        extra_bytes_fn=(_extra_bytes if is_streaming else None))

    if is_streaming and prof is not None:
        policy.measure_breakdown = True
        prof.add_frame_flush(policy.flush_frame_timing)

    logger.info("=> Running SemSegEvaluator with policy=%s ...", variant)
    with EventStorage() as storage, override_cm, stream_cm, ra_cm:
        shim.storage = storage
        # The gates were registered on entering stream_cm, after the
        # profiler; move the frame end behind the Commit Gate so the
        # cache update is part of the frame latency.
        if is_streaming and prof is not None:
            prof.end_frame_last()
        # Armed during the warmup so the cache kernels are compiled before
        # the timed eval; the cache is reset afterwards.
        if is_streaming:
            policy.reset_state()
            policy.arm()
        if args.warmup > 0:
            warm = full_val[0]
            warm_inp = prepare_input(warm, device)
            drop_keys(warm_inp, NON_MODEL_KEYS)
            if is_streaming and warm.get("pose") is not None:
                warm_inp["pose"] = torch.as_tensor(
                    np.asarray(warm["pose"])).to(device)
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(warm_inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            logger.info("GPU warm: %d unprofiled forwards", int(args.warmup))
            del warm, warm_inp
        if is_streaming:
            c = getattr(policy, "_committer", None)
            if c is not None:
                c.wait_pending()
                c.join()
                policy._committer = None
            policy.reset_state()
            policy.arm()
            if ra_probe is not None:
                ra_probe.reset()
        # Keep per-frame intersection/union/target so warmup frames can be
        # dropped and shards merged exactly.
        per_frame_cap = {k: [] for k in
                         ("val_intersection", "val_union", "val_target",
                          "val_loss")}
        _orig_put = storage.put_scalar

        def _cap_put(name, value, *a, **k):
            if name in per_frame_cap:
                per_frame_cap[name].append(
                    np.asarray(value).copy() if hasattr(value, "__len__")
                    else float(value))
            return _orig_put(name, value, *a, **k)

        storage.put_scalar = _cap_put

        def _mk_tprof():
            if (args.torch_profile or args.profile_regions
                    or args.export_trace):
                return torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA])
            return contextlib.nullcontext()

        if prof is not None:
            prof.start()
        if metrics is not None:
            metrics.start()

        gctx = None
        if args.graph_capture:
            from spacecache.common import model_graph
            n_eval = len(val_loader)
            spec = (args.graph_frames or "").strip()
            if spec.startswith("stride:"):
                _p = spec.split(":")[1:]
                _k = int(_p[0])
                _off = int(_p[1]) if len(_p) > 1 and _p[1] else 0
                picks = list(range(_off, n_eval, max(1, _k)))
            elif spec:
                picks = sorted({int(x) for x in spec.split(",")
                                if x.strip()})
            else:
                pick = int(args.graph_frame)
                if pick < 0:
                    pick = (min(150, n_eval - 1) if is_streaming
                            else max(0, n_eval // 2))
                picks = [pick]
            bad = [p for p in picks if not (0 <= p < n_eval)]
            if bad or not picks:
                raise SystemExit(
                    f"--graph_frame(s) {bad or spec!r} out of range for "
                    f"{n_eval} eval frames")
            _outer_prof = (args.torch_profile or args.profile_regions
                           or args.export_trace)
            if _outer_prof:
                logger.warning("--graph_capture with an outer profiler: "
                               "timings are noisy and kernel accounting "
                               "is off")
            kacct = not (args.graph_no_kernel_acct or _outer_prof)
            gctx = model_graph.attach(
                model, pick_frames=picks,
                policy=(policy if is_streaming else None),
                replays=args.graph_replays, warmup=args.graph_warmup,
                prof=prof, metrics=metrics, logger=logger,
                kernel_acct=kacct, debug=args.graph_debug,
                mode=args.graph_mode)
            logger.info("--graph_capture[%s]: %d pick(s) %s (%d replays "
                        "each, kernel accounting %s)", args.graph_mode,
                        len(picks),
                        picks if len(picks) <= 12
                        else f"{picks[:6]}...{picks[-2:]}",
                        args.graph_replays, "on" if kacct else "off")
        tprof = None
        region_stage_ms = None
        for _pass in range(args.passes):
            if _pass and is_streaming:
                # same workload every pass; keep the frame log of all passes
                policy._resolve_pending_stats()
                if getattr(policy, "_committer", None) is not None:
                    policy._committer.drain_results()
                saved_log = policy.frame_log
                policy.reset_state()
                policy.frame_log = saved_log
            if _pass and metrics is not None and metrics._active:
                # FLOPs / peak memory come from pass 1; later passes run
                # without the metric hooks
                metrics.stop()
                logger.info("metrics off for pass %d", _pass + 1)
            logger.info("=== eval pass %d/%d", _pass + 1, args.passes)
            _prof_here = (args.profile_pass < 0
                          or _pass == int(args.profile_pass))
            with (_mk_tprof() if _prof_here
                  else contextlib.nullcontext()) as _tp:
                evaluator.eval()
            if _tp is not None:
                tprof = _tp
        if prof is not None:
            prof.stop()
        if metrics is not None:
            metrics.stop()
        storage.put_scalar = _orig_put
        if is_streaming:
            policy.disarm()
        if tprof is not None and args.export_trace:
            os.makedirs(os.path.dirname(os.path.abspath(args.export_trace)),
                        exist_ok=True)
            tprof.export_chrome_trace(args.export_trace)
            logger.info("chrome trace -> %s", args.export_trace)
        if tprof is not None:
            ka = tprof.key_averages()
            table = ka.table(sort_by="cuda_time_total", row_limit=30)
            logger.info("torch.profiler (CUDA) per-op table:\n%s", table)
            print(table)

            # Kernel time per record_function scope (no host idle time).
            def _cuda_us(e):
                return (getattr(e, "cuda_time_total", None)
                        or getattr(e, "device_time_total", 0) or 0)
            _pfx = ("nonflash/", "stage/")
            # Each label appears as a CPU-side annotation (kernels launched
            # inside the scope) and a GPU-side one; use the CPU row.
            _by_key = {}
            for e in ka:
                _k = str(e.key)
                if not _k.startswith(_pfx):
                    continue
                _is_cpu = (
                    "cpu" in str(getattr(e, "device_type", "")).lower()
                    or getattr(e, "self_cpu_time_total", 0) > 0)
                if _k not in _by_key or (_is_cpu
                                         and not _by_key[_k][0]):
                    _by_key[_k] = (_is_cpu, e)
            region = [(k, _cuda_us(e) / 1000.0, getattr(e, "count", 0))
                      for k, (_c, e) in _by_key.items()]
            if not region:
                from collections import defaultdict
                acc, cnt = defaultdict(float), defaultdict(int)
                try:
                    evs = tprof.events()
                except Exception:
                    evs = []
                for e in evs:
                    n = str(getattr(e, "name", ""))
                    if (n.startswith(_pfx) and "cpu" in str(
                            getattr(e, "device_type", "")).lower()):
                        acc[n] += _cuda_us(e) / 1000.0
                        cnt[n] += 1
                region = [(k, v, cnt[k]) for k, v in acc.items()]
            region.sort(key=lambda x: -x[1])
            if region:
                # one stage/* scope per stage per frame: its count is the
                # number of profiled frames
                stage_rows = [(k, v, c) for k, v, c in region
                              if k.startswith("stage/")]
                nf = max((c for _, _, c in stage_rows), default=0) or 1
                summary = "\n".join(f"  {k:34s} {v:9.3f} ms  (x{c})"
                                    for k, v, c in region)
                logger.info("region CUDA total ms:\n%s", summary)
                print("\nregion CUDA total ms (over profiled frames):\n"
                      + summary)
                if stage_rows:
                    tot = sum(v for _, v, _ in stage_rows)
                    print(f"\nstage GPU-busy (ms/frame, x{nf} frames):")
                    for k, v, c in sorted(
                            stage_rows, key=lambda r: r[0]):
                        print(f"  {k:34s} {v / c:9.3f}  {v / tot:6.1%}")
                    print(f"  {'TOTAL':34s} {tot / nf:9.3f}")
                    region_stage_ms = {k: v / c for k, v, c in stage_rows}

    # --- aggregates (frames >= warmup_frames) ----------------------------------
    n_eval_total = len(per_frame_cap["val_intersection"])
    w = min(max(int(args.warmup_frames), 0), max(n_eval_total - 1, 0))
    inter = np.sum(per_frame_cap["val_intersection"][w:], axis=0)
    union = np.sum(per_frame_cap["val_union"][w:], axis=0)
    target = np.sum(per_frame_cap["val_target"][w:], axis=0)
    loss_vals = per_frame_cap["val_loss"][w:]
    loss_avg = float(np.mean(loss_vals)) if loss_vals else 0.0
    n_counted = n_eval_total - w
    if w:
        logger.info("warmup_frames=%d: aggregating last %d / %d frames",
                    w, n_counted, n_eval_total)

    timing = None
    frame_records = []
    if prof is not None:
        frame_records = prof.records
        prof.remove()
        timing = summarize_timing(frame_records[w:], logger)

    metric_summary = None
    metric_records = []
    if metrics is not None:
        metric_records = metrics.records
        metrics.remove()
        metric_summary = summarize_cost(metric_records[w:], logger)

    iou_class = inter / (union + 1e-10)
    acc_class = inter / (target + 1e-10)
    m_iou = float(np.mean(iou_class))
    m_acc = float(np.mean(acc_class))
    all_acc = float(inter.sum() / (target.sum() + 1e-10))

    per_frame_miou = [
        float(np.mean(np.asarray(i) / (np.asarray(u) + 1e-10)))
        for i, u in zip(per_frame_cap["val_intersection"],
                        per_frame_cap["val_union"])
    ]
    per_frame_all_acc = [
        float(np.asarray(i).sum() / (np.asarray(t).sum() + 1e-10))
        for i, t in zip(per_frame_cap["val_intersection"],
                        per_frame_cap["val_target"])
    ]
    per_frame_loss = [float(v) for v in per_frame_cap["val_loss"]]

    names = list(cfg.data.get("names", [f"class_{i}"
                 for i in range(int(cfg.data.num_classes))]))
    result = dict(
        config=args.config, weight=weight, policy=policy.name,
        run_name=variant, merge_ratio=r, attention=attention,
        compute_dtype=("fp32" if args.fp32
                       else ("fp16" if is_litept else "bf16")),
        backbone=("litept" if is_litept else "ptv3"),
        upcast_softmax=bool(args.upcast_softmax),
        merge_reduce=args.merge_reduce,
        seed=seed, n_frames=len(val_loader),
        warmup_frames=w, n_frames_counted=n_counted,
        mIoU=m_iou, mAcc=m_acc, allAcc=all_acc, val_loss=loss_avg,
        per_frame_mIoU=per_frame_miou,
        per_frame_allAcc=per_frame_all_acc,
        per_frame_loss=per_frame_loss,
        per_class_iou={names[i]: float(iou_class[i])
                       for i in range(len(names))},
        # raw per-class counts, so shards can be re-aggregated exactly
        intersection=inter.tolist(),
        union=union.tolist(),
        target=target.tolist(),
        scene_shard=args.scene_shard,
        scene_shard_mode=(("frames" if args.shard_balance else "round_robin")
                          if args.scene_shard else None),
        timing=timing,
        metrics=metric_summary,
        params=(metrics.params if metrics is not None else None),
        param_bytes=(metrics.param_bytes if metrics is not None else None),
    )
    if args.passes > 1:
        result["passes"] = int(args.passes)
        result["frames_per_pass"] = len(val_loader)
    if region_stage_ms:
        result["stage_gpu_ms_per_frame"] = region_stage_ms
    if args.policy == naming.ALGM:
        result["tau"] = name_kw["tau"]
    if is_streaming:
        result["intercept"] = intercept
        result["lifetime"] = lifetime
        result["confidence"] = confidence
        result["region_scale"] = args.region_scale
        result["expiry"] = args.expiry
        result["region_key"] = args.region_key
        result["spacecache_exclude_unlabeled"] = bool(policy.exclude_ignore)
        # False: tokens of a frame sharing a region are averaged into one
        # entry
        result["spacecache_keys_unique"] = getattr(policy, "keys_unique",
                                                   None)
        # the commit runs inside the frame timer (end_frame_last)
        result["commit_in_frame"] = prof is not None
        fl = getattr(policy, "frame_log", [])
        if fl and args.passes > 1:
            # SpaceCache statistics of the last pass
            fl = fl[-len(val_loader):]
        if fl:
            # exclude the first frame of every sequence (empty cache)
            ns = [f for f in fl if not f.get("is_seed", False)] or fl
            result["spacecache_hit_rate"] = float(
                sum(f["match_ratio"] for f in ns) / len(ns))
            result["spacecache_miss_tokens"] = float(
                sum(f["subset_size"] for f in ns) / len(ns))
            # level0 accuracy of cache-served vs recomputed cells
            hk = sum(f.get("n_hit_ok", 0) for f in ns)
            hn = sum(f.get("n_hit_l0", 0) for f in ns)
            mk = sum(f.get("n_miss_ok", 0) for f in ns)
            mn = sum(f.get("n_miss_l0", 0) for f in ns)
            if hn + mn > 0:
                result["spacecache_acc_hit"] = hk / hn if hn else None
                result["spacecache_acc_miss"] = mk / mn if mn else None
                result["spacecache_hit_fraction_input"] = hn / (hn + mn)
            d4 = [f for f in ns if "match_ratio_d4" in f]
            if d4:
                result["spacecache_hit_rate_s4"] = float(
                    sum(f["match_ratio_d4"] for f in d4) / len(d4))
                result["spacecache_miss_tokens_s4"] = float(
                    sum(f["subset_size_d4"] for f in d4) / len(d4))
                hk4 = sum(f.get("n_hit_ok_d4", 0) for f in d4)
                hn4 = sum(f.get("n_hit_l0_d4", 0) for f in d4)
                mk4 = sum(f.get("n_miss_ok_d4", 0) for f in d4)
                mn4 = sum(f.get("n_miss_l0_d4", 0) for f in d4)
                if hn4 + mn4 > 0:
                    result["spacecache_acc_hit_s4"] = (hk4 / hn4
                                                       if hn4 else None)
                    result["spacecache_acc_miss_s4"] = (mk4 / mn4
                                                        if mn4 else None)
            # --region_stats: raw counts summed over every frame of the pass
            rd_rows = [f for f in fl if "rd_n_dyn" in f]
            if rd_rows:
                rd_keys = sorted({k for f in rd_rows for k in f
                                  if k.startswith("rd_")})
                result["spacecache_region_stats"] = {
                    k: int(sum(int(f.get(k) or 0) for f in rd_rows))
                    for k in rd_keys}
            ra = getattr(policy, "_rd_age", None)
            if ra is not None:
                from spacecache.policies.spacecache import _RD_AGE_ROWS
                result["spacecache_region_age"] = {
                    nm: [int(round(x)) for x in row]
                    for nm, row in zip(_RD_AGE_ROWS, ra.cpu().tolist())}

            # Per-frame cost of the cache operations (ms). match = key
            # projection + lookup, subset = miss-subset construction,
            # restore = scatter back to full resolution, stage = append to
            # staging, commit = cache update at the commit gate. The commit
            # is inside total_ms (see `commit_in_frame`).
            def _mean(xs):
                return sum(xs) / len(xs) if xs else 0.0

            def _median(xs):
                if not xs:
                    return 0.0
                s = sorted(xs)
                h = len(s) // 2
                return s[h] if len(s) % 2 else 0.5 * (s[h - 1] + s[h])

            mm = [f["match_ms"] for f in ns if "match_ms" in f]
            um = [f["update_ms"] for f in ns if "update_ms" in f]
            sm = [f["subset_ms"] for f in ns if "subset_ms" in f]
            rm = [f["restore_ms"] for f in ns
                  if f.get("restore_ms") is not None]
            gm = [f["stage_ms"] for f in ns if f.get("stage_ms") is not None]
            cm = [f["commit_ms"] for f in ns
                  if f.get("commit_ms") is not None]
            mm4 = [f["match_ms_d4"] for f in ns
                   if f.get("match_ms_d4") is not None]
            if mm:
                def _bd(agg):
                    d = dict(match=round(agg(mm), 5), subset=round(agg(sm), 5),
                             update=round(agg(um), 5))
                    if rm:
                        d["restore"] = round(agg(rm), 5)
                    if gm:
                        d["stage"] = round(agg(gm), 5)
                    if cm:
                        d["commit"] = round(agg(cm), 5)
                    return d

                bd = _bd(_mean)
                bd["stat"] = "mean"
                bd["median"] = _bd(_median)
                bd["n_frames"] = len(mm)
                if mm4:
                    def _bd4(agg):
                        d4d = {}
                        for k_src, k_dst in (
                                ("match_ms_d4", "match"),
                                ("subset_ms_d4", "subset"),
                                ("restore_ms_d4", "restore"),
                                ("stage_ms_d4", "stage"),
                                ("commit_ms_d4", "commit"),
                                ("update_ms_d4", "update")):
                            xs = [f[k_src] for f in ns
                                  if f.get(k_src) is not None]
                            if xs:
                                d4d[k_dst] = round(agg(xs), 5)
                        return d4d
                    bd["s4"] = _bd4(_mean)
                    bd["s4_median"] = _bd4(_median)
                result["spacecache_ms"] = bd
    if policy.name == naming.GITMERGE3D_ADAPTIVE:
        from spacecache.common.tome_attention import WPATCH_STATS
        if WPATCH_STATS:
            n_two = sum(1 for s in WPATCH_STATS
                        if s["path"] == "two_bucket")
            result["gitmerge3d_adaptive_stats"] = dict(
                n_calls=len(WPATCH_STATS),
                frac_two_bucket=round(n_two / len(WPATCH_STATS), 4),
                mean_p=round(float(np.mean(
                    [s["p"] for s in WPATCH_STATS])), 4),
                mean_num_l=round(float(np.mean(
                    [s["num_l"] for s in WPATCH_STATS])), 2),
                mean_B=round(float(np.mean(
                    [s["B"] for s in WPATCH_STATS])), 1),
                score_quantiles=[round(float(np.mean(
                    [s["score_q"][i] for s in WPATCH_STATS
                     if "score_q" in s])), 4)
                    for i in range(5)] if any(
                        "score_q" in s for s in WPATCH_STATS) else None,
                energy_thresh=WPATCH_STATS[0]["thresh"],
            )
    if gctx is not None and gctx.result is not None:
        result["graph_capture"] = gctx.result
    if ra_probe is not None:
        result["reuse_agreement"] = ra_probe.save(output_dir)
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("=> %s : mIoU/mAcc/allAcc %.4f/%.4f/%.4f  (loss %.4f, %d "
                "frames)", variant, m_iou, m_acc, all_acc, loss_avg,
                len(val_loader))
    logger.info("Saved %s", json_path)

    # Per-frame class counts with the sequence id and frame index, used by
    # spacecache/experiments/lifetime_analysis.py.
    if per_frame_cap["val_intersection"]:
        try:
            if max_frames is not None and max_frames > 0:
                seg_eval = SpaceCachePolicy._build_seg_ids(full_val)
                seg_eval = seg_eval[:len(val_loader)] if seg_eval else None
            elif shard_seg_ids is not None:
                seg_eval = shard_seg_ids
            else:
                seg_eval = SpaceCachePolicy._build_seg_ids(full_val)
            n_rows = len(per_frame_cap["val_intersection"])
            if seg_eval and n_rows % len(seg_eval) == 0:
                seg_ids_map = {}
                seg_i = [seg_ids_map.setdefault(s, len(seg_ids_map))
                         for s in seg_eval]
                fi_list, prev_s, c_fi = [], None, 0
                for s in seg_i:
                    c_fi = c_fi + 1 if s == prev_s else 0
                    fi_list.append(c_fi)
                    prev_s = s
                reps = n_rows // len(seg_eval)
                np.savez_compressed(
                    os.path.join(output_dir, "per_frame_counts.npz"),
                    intersection=np.stack(
                        per_frame_cap["val_intersection"]).astype(np.int64),
                    union=np.stack(
                        per_frame_cap["val_union"]).astype(np.int64),
                    target=np.stack(
                        per_frame_cap["val_target"]).astype(np.int64),
                    seg=np.tile(np.asarray(seg_i, dtype=np.int64), reps),
                    fi=np.tile(np.asarray(fi_list, dtype=np.int64), reps),
                    names=np.asarray(names))
                logger.info("Saved per_frame_counts.npz (%d frames)", n_rows)
            else:
                logger.warning("per_frame_counts.npz skipped: %s eval rows "
                               "vs %s sequence ids", n_rows,
                               len(seg_eval) if seg_eval else None)
        except Exception as e:
            logger.warning("per_frame_counts.npz skipped: %s", e)

    if frame_records:
        frames_path = os.path.join(output_dir, "per_frame_timing.jsonl")
        with open(frames_path, "w") as f:
            for rec in frame_records:
                f.write(json.dumps(rec) + "\n")
        logger.info("Saved %s (%d frames)", frames_path, len(frame_records))

    if metric_records:
        mpath = os.path.join(output_dir, "per_frame_metrics.jsonl")
        with open(mpath, "w") as f:
            for rec in metric_records:
                f.write(json.dumps(rec) + "\n")
        logger.info("Saved %s (%d frames)", mpath, len(metric_records))

    if is_streaming and getattr(policy, "frame_log", None):
        sl_path = os.path.join(output_dir, "spacecache_per_frame.jsonl")
        with open(sl_path, "w") as f:
            for rec in policy.frame_log:
                f.write(json.dumps(rec) + "\n")
        logger.info("Saved %s (%d frames)", sl_path, len(policy.frame_log))


if __name__ == "__main__":
    main()
