"""SpaceCache policies: intermediate-feature reuse on point-cloud streams.

A SpaceCache is attached to one encoder stage (`DEPTH`). Each stage token is
mapped to a world-space cell (ego pose + floor division by the stage cell
size), and four hooks drive the cache over a sequence:

  reset gate   model forward pre-hook      read the pose, advance the frame
                                           counter, reset on a new sequence
  mask gate    enc{DEPTH}.down forward     look up the cells; the stage's
               hook                        blocks only see the cache misses
  reuse gate   last block forward hook     restore the full token set (hits
                                           from the cache, misses from the
                                           blocks) and stage the misses
  commit gate  model forward hook          invalidate ignore / low-confidence
                                           cells, apply the lifetime, fold the
                                           staged cells into the cache

`spacecache_s3` is SpaceCache at the third encoder stage, `spacecache_s34`
adds a second cache at the fourth stage, and `spacecache_s{0,1,2,4}` move
the single cache to another stage (Table 3(b)).
"""

from __future__ import annotations

import contextlib
import math
import os

import torch

from spacecache.common import triton_cache as _tc
from spacecache.common.helpers import (
    extract_grid_size, extract_random_scale, to_global, voxel_hash,
)
from spacecache.common.space_cache import AsyncCommitter, CacheView, SpaceCache
from spacecache.common.stage_cache import (
    D3_DEPTH, build_subset_point_fast, get_stage_desc,
)
from spacecache.common.timed_block import timed_block_forward

# `--lifetime inf`: a lifetime long enough that no entry expires within a
# sequence (SpaceCache stamps due = t+1 + ((key - t - 1) mod L)).
_INF_LIFE = 1 << 40


def _life_frames(lifetime):
    return (_INF_LIFE if math.isinf(lifetime)
            else max(int(round(lifetime)), 1))


def staged_keys_unique(cfg, region_scale):
    """Whether the tokens of one frame map to distinct cells of a stage.

    Only at region scale 1 on a world-anchored grid (`GridSample(
    world_anchor=True)`, the indoor configs), where each stage voxel is a
    world cell. Outdoor grids follow the ego frame, so several tokens of a
    frame can fall into one cell; their features are then averaged into one
    entry. `SPACECACHE_KEYS_UNIQUE=0/1` (env) or `keys_unique`
    (config) overrides; =1 on an outdoor config reproduces the paper
    numbers, which stored such tokens as separate rows (lookup served the
    first).
    """
    env = os.environ.get("SPACECACHE_KEYS_UNIQUE")
    if env is not None:
        return env == "1"
    if cfg is not None and cfg.get("keys_unique") is not None:
        return bool(cfg.get("keys_unique"))
    if abs(float(region_scale) - 1.0) >= 1e-9 or cfg is None:
        return False
    return any(t.get("type") == "GridSample" and bool(t.get("world_anchor"))
               for t in cfg.data.val.transform)


# --region_stats: classes whose instances move (Waymo and nuScenes names,
# lowercased). A cell whose majority label changes to or from one of these is
# counted as a dynamic region.
_MOVABLE_CLASSES = frozenset({
    "car", "truck", "bus", "other vehicle", "motorcyclist", "bicyclist",
    "pedestrian", "bicycle", "motorcycle",
    "construction_vehicle", "trailer",
})

# --region_stats label-at-write table: one int64 per cell packs
# label + write_frame * _RD_T + point_count * _RD_C.
_RD_T = 1 << 8
_RD_C = 1 << 32
_RD_AGE_BINS = 64
_RD_AGE_ROWS = ("n", "ok", "n_dyn", "ok_dyn", "n_stale", "ok_stale")


def _rd_lookup(tab_key, tab_val, q):
    """(found, value) of `q` in a sorted (key, value) table."""
    if tab_key is None or tab_key.numel() == 0:
        return (torch.zeros(q.shape, dtype=torch.bool, device=q.device),
                torch.zeros(q.shape, dtype=torch.long, device=q.device))
    pos = torch.searchsorted(tab_key, q).clamp(max=tab_key.numel() - 1)
    return tab_key[pos] == q, tab_val[pos]


def _rd_upsert(tab_key, tab_val, new_key, new_val):
    """Insert-or-overwrite into a sorted (key, value) table; newest wins."""
    if new_key.numel() == 0:
        return tab_key, tab_val
    if tab_key is None or tab_key.numel() == 0:
        k, v = new_key, new_val
    else:
        k, v = torch.cat([tab_key, new_key]), torch.cat([tab_val, new_val])
    order = torch.argsort(k, stable=True)
    k, v = k[order], v[order]
    last = torch.ones_like(k, dtype=torch.bool)
    last[:-1] = k[1:] != k[:-1]
    return k[last], v[last]


class SpaceCachePolicy:
    """SpaceCache at encoder stage `DEPTH` (3 by default).

    `region_scale` (`--region_scale`, 1.0 in the paper) scales the region:
    cell = base_voxel * 2^DEPTH * region_scale. At 1.0 the region is the
    stage's own voxel.
    """
    name = "spacecache_s3"
    streaming_kind = "d3"
    intercept = "enc3"
    DEPTH = D3_DEPTH

    def __init__(self):
        self.cache = None
        self.stage = None
        self.seg_ids = None
        self.mf = 1.0
        self.cache_life = 4.0
        self.coord_scale = 1.0
        self.voxel_size_d3 = None
        self.pose = None
        self.armed = False
        self.is_cuda = False
        self.measure_breakdown = False
        self._cur_seg = None
        self._fi = 0
        self._frame_ptr = 0
        self._frame_ev = None
        self.frame_log = []
        # Cells covering ignore-labelled points are not inserted into the
        # cache (configs may disable this with `exclude_unlabeled`).
        self.exclude_ignore = True
        self.keys_unique = None          # set in prepare (staged_keys_unique)
        self.ignore_index = -1
        self._seg0 = None
        # pooling_inverse of enc1..enc{DEPTH-1}; chained into the
        # level0 -> stage map used by the commit gate.
        self._pis = {}
        # Confidence rule: at commit, cells whose mean max-softmax
        # probability is <= inv_thresh are invalidated. `random` invalidates
        # the same number of cells chosen uniformly (a control).
        self.inv_mode = "conf"
        self.inv_seed = 1234
        self.inv_thresh = None
        self._frame = None
        self._commit_ev = None
        self.async_commit = False
        self._committer = None
        self._committer_dev = None
        self._async_rows = []
        # (frame_log row, {field: 0-dim device tensor}); resolved with a
        # single device->host copy at the end of the eval.
        self._pend_stats = []
        self._triton = False
        self.region_diag = False
        self._rd_names = None
        self._rd_movable = None
        self._rd_key = self._rd_lab = None
        self._rd_wkey = self._rd_wval = None
        self._rd_age = None

    def run_block(self, block, point, ctx=None):
        """Stock block forward, identical to the backbone policy."""
        timer = ctx.get("timer") if ctx else None
        return timed_block_forward(block, point, timer=timer)

    # -- setup ------------------------------------------------------------

    def prepare(self, model, cfg, dataset, device, *, region_scale,
                lifetime, commit_async=False, triton_cache=False,
                expiry="staggered"):
        """Per-run state: stage handle, cell size, sequence ids, empty cache."""
        self.mf = float(region_scale)
        self.cache_life = float(lifetime)
        self.cache_expiry = str(expiry)
        self.is_cuda = device is not None and device.type == "cuda"
        self._triton = (bool(triton_cache) and self.is_cuda
                        and _tc.is_available())
        self.async_commit = bool(commit_async) and self.is_cuda
        self._committer_dev = device
        # Created lazily on the first commit: a live CUDA thread at the time
        # the DataLoader forks its workers can crash them.
        self._committer = None
        self._async_rows = []
        self.stage = get_stage_desc(model.backbone, self.DEPTH)
        base_voxel = float(extract_grid_size(cfg.data.val.transform))
        self.coord_scale = float(extract_random_scale(cfg.data.val.transform))
        self.voxel_size_d3 = base_voxel * (2 ** int(self.DEPTH))
        self.ignore_index = int(cfg.data.get("ignore_index", -1))
        self._rd_names = list(cfg.data.get("names", None) or [])
        self._rd_movable = None
        if cfg.get("exclude_unlabeled") is not None:
            self.exclude_ignore = bool(cfg.get("exclude_unlabeled"))
        self.seg_ids = self._build_seg_ids(dataset)
        # Tokens of one frame that fall into the same cell are averaged into
        # one entry, except where every token owns a distinct cell
        # (staged_keys_unique: region scale 1 on a world-anchored grid).
        self.keys_unique = staged_keys_unique(cfg, self.mf)
        self.cache = SpaceCache(
            device,
            default_life=_life_frames(self.cache_life),
            keys_unique=self.keys_unique,
            triton=self._triton,
            expiry=self.cache_expiry)
        self.reset_state()

    def configure_invalidation(self, thresh, mode="conf", seed=1234):
        """Enable the confidence rule with threshold `thresh` (theta)."""
        self.inv_thresh = None if thresh is None else float(thresh)
        self.inv_mode = str(mode)
        self.inv_seed = int(seed)

    def reset_state(self):
        """Drop the cache and the frame bookkeeping."""
        if self.cache is not None:
            self.cache.reset()
        self._cur_seg = None
        self._fi = 0
        self._frame_ptr = 0
        self._frame_ev = None
        self._frame = None
        self._commit_ev = None
        self.frame_log = []
        self._seg0 = None
        self._pis = {}
        self._async_rows = []
        self._pend_stats = []
        self._rd_key = self._rd_lab = None
        self._rd_wkey = self._rd_wval = None
        self._rd_age = None

    def arm(self):
        self.armed = True

    def disarm(self):
        self.armed = False

    @contextlib.contextmanager
    def streaming_context(self, model):
        """Register the gates for the duration of the eval.

        Every gate is a no-op until `arm()`, so warmup forwards leave the
        cache untouched.
        """
        enc = model.backbone.enc._modules
        stage_mod = enc[f"enc{self.DEPTH}"]
        last_block = stage_mod._modules[f"block{self.stage.n_blocks - 1}"]

        def _cap_pi(i):
            def hook(_m, _i, out):
                if self.armed:
                    self._pis[i] = out.pooling_inverse
            return hook

        handles = [
            model.register_forward_pre_hook(
                lambda _m, args: self.reset_gate(args[0])),
            *[enc[f"enc{i}"].register_forward_hook(_cap_pi(i))
              for i in range(1, self.DEPTH)],
        ]
        # enc0 has no `down` module, so at DEPTH 0 the subset is swapped in
        # by a stage forward pre-hook instead.
        if self.DEPTH > 0:
            handles.append(stage_mod._modules["down"]
                           .register_forward_hook(self.mask_gate))
        else:
            handles.append(stage_mod.register_forward_pre_hook(
                self.mask_pre_hook))
        handles += [
            last_block.register_forward_hook(self.reuse_gate),
            # Registered after FrameProfiler's model hook, so the commit runs
            # after the per-frame timer stops; it is timed separately as
            # `commit_ms`.
            model.register_forward_hook(self.commit_gate),
        ]
        try:
            yield
        finally:
            for h in handles:
                h.remove()
            self._frame = None
            if self._committer is not None:
                self._committer.join()
            self._resolve_pending_stats()

    # -- reset gate ---------------------------------------------------------

    def reset_gate(self, input_dict):
        """New frame: read the pose and reset the cache on a new sequence."""
        if not self.armed:
            return
        pose = input_dict.get("pose")
        if pose is not None and torch.is_tensor(pose) and pose.dim() == 3:
            pose = pose[0]
        self.pose = pose
        seg0 = input_dict.get("segment")
        if seg0 is not None and torch.is_tensor(seg0) and seg0.dim() > 1:
            seg0 = seg0.reshape(-1)
        self._seg0 = seg0

        idx = self._frame_ptr
        seg = (self.seg_ids[idx]
               if (self.seg_ids is not None and idx < len(self.seg_ids))
               else None)
        if seg != self._cur_seg:
            # Cache keys are world cells, so entries never survive a scene
            # change. Drain any in-flight async commit first.
            if self._committer is not None:
                self._committer.join()
            self.cache.reset()
            self._rd_key = self._rd_lab = None
            self._rd_wkey = self._rd_wval = None
            self._cur_seg = seg
            self._fi = 0
        else:
            self._fi += 1
        self._frame_ptr += 1

    def _evt(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    # -- mask gate ----------------------------------------------------------

    def mask_pre_hook(self, _module, inputs):
        """DEPTH 0 variant of `mask_gate` (stage forward pre-hook)."""
        sub = self.mask_gate(_module, None, inputs[0])
        return None if sub is None else (sub,)

    def _level0_map(self, point_out):
        """level0 -> stage-token index map (pi_DEPTH[...pi2[pi1]])."""
        n = int(point_out.feat.shape[0])
        if self.DEPTH == 0:
            return torch.arange(n, device=point_out.feat.device)
        pi = getattr(point_out, "pooling_inverse", None)
        if pi is None:
            return None
        acc = None
        for i in range(1, self.DEPTH):
            p = self._pis.get(i)
            if p is None:
                return None
            acc = p if acc is None else p[acc]
        return pi if acc is None else pi[acc]

    def mask_gate(self, _module, _inputs, point_out):
        """Look up the pooled tokens and return the miss subset.

        Returns None on a cold or fully-missed frame, in which case the
        blocks run on the full token set.
        """
        if not self.armed:
            return None
        if self._committer is not None:
            # the previous frame's commit must be visible to this lookup
            self._committer.wait_pending()
        meas = self.measure_breakdown and self.is_cuda
        m0 = self._evt() if meas else None
        if self._triton and self.pose is not None:
            keys, slot, hit = _tc.lookup_coords(
                point_out.coord, self.pose, self.coord_scale,
                self.voxel_size_d3 * self.mf, self.cache.key)
            view = CacheView(keys, slot, hit,
                             torch.nonzero(~hit, as_tuple=True)[0])
        else:
            coord_g = to_global(point_out.coord, self.pose,
                                scale=self.coord_scale)
            keys = voxel_hash(coord_g, self.voxel_size_d3 * self.mf)
            view = self.cache.lookup(keys)
        m1 = self._evt() if meas else None

        s0 = self._evt() if meas else None
        n_d3 = int(keys.numel())
        # Built here because `down`'s pooling_inverse is gone by the time the
        # commit gate runs.
        c3 = self._level0_map(point_out)
        n_miss = view.n_miss
        if 0 < n_miss < n_d3:
            sub = (_tc.build_subset_point(point_out, view.miss_idx)
                   if self._triton
                   else build_subset_point_fast(point_out, view.miss_idx))
        else:
            sub = None
        s1 = self._evt() if meas else None

        self._frame = dict(
            full=point_out, keys=keys, view=view, c3=c3, n_d3=n_d3,
            subset=sub, m0=m0, m1=m1, s0=s0, s1=s1,
        )
        return sub

    # -- reuse gate ---------------------------------------------------------

    def reuse_gate(self, _module, _inputs, blk_out):
        """Restore the full token set and stage the recomputed cells.

        `restore_ms` covers the scatter back to full resolution, `stage_ms`
        the append to the staging buffer; `update_ms` is their sum.
        """
        if not self.armed or self._frame is None:
            return None
        fm = self._frame
        view = fm["view"]
        meas = self.measure_breakdown and self.is_cuda
        u0 = self._evt() if meas else None
        if fm["subset"] is None:
            self.cache.restore(view, blk_out.feat, None)
            result = blk_out
            subset_size = fm["n_d3"]
            rows = None
        else:
            self.cache.restore(view, fm["full"].feat, blk_out.feat)
            result = fm["full"]
            subset_size = int(blk_out.feat.shape[0])
            rows = blk_out.feat
        if "sparse_conv_feat" in result.keys():
            result.sparse_conv_feat = result.sparse_conv_feat.replace_feature(
                result.feat)
        um = self._evt() if meas else None
        self.cache.write(view, result.feat, rows=rows)
        u1 = self._evt() if meas else None

        fm["restore_ev"] = (u0, um, u1) if meas else None
        fm["subset_size"] = subset_size
        # Do not hold the Points until commit: the pooled Point keeps every
        # shallower encoder output alive through `pooling_parent`.
        fm["full"] = fm["subset"] = None
        return result

    # -- commit gate --------------------------------------------------------

    def _low_confidence_cells(self, output, c_map, n, dev):
        """Cells removed by the confidence rule (or its random control)."""
        if self.inv_thresh is None or n == 0:
            return None
        conf = self._cell_confidence(output, c_map, n)
        if self.inv_mode == "random":
            k = (int((conf <= self.inv_thresh).sum())
                 if conf is not None else n)
            if k <= 0:
                return None
            g = torch.Generator(device=dev)
            g.manual_seed(self.inv_seed + int(self._frame_ptr))
            r = torch.rand(n, generator=g, device=dev)
            low = torch.zeros(n, dtype=torch.bool, device=dev)
            low.index_fill_(0, torch.topk(r, k, largest=False).indices, True)
            return low
        if conf is None:
            return None
        return conf <= self.inv_thresh

    def commit_gate(self, _module, _args, output):
        """Decide what the next frame may read and update the cache.

        Invalidated cells are (a) newly computed cells that cover
        ignore-labelled points and (b) cells failing the confidence rule,
        resident or newly computed. The lifetime is applied by
        `SpaceCache.commit`.
        """
        if not self.armed or self._frame is None:
            return
        fm = self._frame
        self._frame = None
        meas = self.measure_breakdown and self.is_cuda
        c0 = self._evt() if meas else None

        keys = fm["keys"]
        n_d3 = fm["n_d3"]
        c3 = fm["c3"]
        dev = keys.device
        view_hit = fm["view"].hit
        inval = torch.zeros(n_d3, dtype=torch.bool, device=dev)
        # frame_log counters stay on the device until the end of the eval
        pend = {}

        if self.exclude_ignore and c3 is not None and self._seg0 is not None:
            ign = torch.bincount(c3[self._seg0 == self.ignore_index],
                                 minlength=n_d3) > 0
            ign &= ~view_hit
            inval |= ign
            pend["n_ignore_d3"] = ign.sum()

        low_mask = self._low_confidence_cells(output, c3, n_d3, dev)
        if low_mask is not None:
            inval |= low_mask
            pend["n_lowconf"] = low_mask.sum()

        # level0 accuracy split by cache hit / miss (voxel ground truth)
        if c3 is not None and self._seg0 is not None:
            logits = (output.get("seg_logits")
                      if isinstance(output, dict) else None)
            if logits is not None and int(logits.shape[0]) == int(c3.numel()):
                correct = logits.argmax(dim=1) == self._seg0
                valid = self._seg0 != self.ignore_index
                hit_l0 = view_hit[c3] & valid
                miss_l0 = ~view_hit[c3] & valid
                pend["n_hit_l0"] = hit_l0.sum()
                pend["n_hit_ok"] = correct[hit_l0].sum()
                pend["n_miss_l0"] = miss_l0.sum()
                pend["n_miss_ok"] = correct[miss_l0].sum()

        if self.region_diag:
            self._region_diag(output, keys, c3, n_d3, view_hit, low_mask,
                              inval, pend)

        if self.async_commit:
            # table stats and commit_ms are filled in when the job finishes
            self._commit_async(self.cache, keys, inval,
                               len(self.frame_log))
            n_evicted = None
        else:
            n_evicted = self.cache.commit(
                keys, self.cache.default_life, self._fi, invalidate=inval)
            c1 = self._evt() if meas else None
            if meas:
                self._commit_ev = (c0, c1, len(self.frame_log))

        pend["n_match"] = view_hit.sum()
        pend["n_invalidated"] = inval.sum()
        row = dict(
            n_hit_l0=0, n_hit_ok=0, n_miss_l0=0, n_miss_ok=0,
            cache_bytes=(None if self.async_commit
                         else int(self.cache.nbytes())),
            frame_ptr=int(self._frame_ptr - 1),
            segment=(int(self._cur_seg)
                     if isinstance(self._cur_seg, int) else None),
            fi=int(self._fi),
            is_seed=bool(self._fi == 0),
            n_d3=n_d3, n_match=None,
            subset_size=int(fm.get("subset_size", n_d3)),
            match_ratio=None,
            n_ignore_d3=0,
            n_invalidated=None,
            n_lowconf=0,
            n_evicted=(None if self.async_commit else int(n_evicted)),
            n_cells=(None if self.async_commit
                     else int(self.cache.n_cells)),
        )
        self._pend_stats.append((row, pend))
        self.frame_log.append(row)
        fm_ev = fm.get("restore_ev")
        self._frame_ev = ((fm["m0"], fm["m1"], fm["s0"], fm["s1"]) + fm_ev
                          if (meas and fm_ev) else None)

    def _resolve_pending_stats(self):
        """Copy every deferred device counter into its frame_log row."""
        if not self._pend_stats:
            return
        jobs = [(row, name, t) for row, pend in self._pend_stats
                for name, t in pend.items()]
        if jobs:
            vals = torch.stack(
                [t.reshape(()).to(torch.int64) for _, _, t in jobs]
            ).cpu().tolist()
            for (row, name, _), v in zip(jobs, vals):
                row[name] = int(v)
        for row, _ in self._pend_stats:
            for nm, nd, mr in (("n_match", "n_d3", "match_ratio"),
                               ("n_match_d4", "n_d4", "match_ratio_d4")):
                if row.get(nm) is not None and row.get(nd):
                    row[mr] = row[nm] / max(row[nd], 1)
        self._pend_stats = []

    def _commit_async(self, cache, keys, inval, row_idx, suffix=""):
        """Queue this frame's commit on the worker thread.

        The staged references are detached on this thread so the job cannot
        see the next frame's `write()`.
        """
        if self._committer is None:
            self._committer = AsyncCommitter(self._committer_dev)
        staged = cache.pop_staging()
        life, fi = cache.default_life, self._fi

        def _job():
            e0 = torch.cuda.Event(enable_timing=True)
            e0.record()
            n_ev = cache.commit(keys, life, fi, invalidate=inval,
                                staged=staged)
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
            return {"ev": (e0, e1), "suffix": suffix,
                    "n_evicted": int(n_ev), "n_cells": int(cache.n_cells),
                    "cache_bytes": int(cache.nbytes())}

        self._committer.submit(_job, inputs=(
            keys, inval, *(t for t in staged if torch.is_tensor(t))))
        self._async_rows.append(row_idx)

    def _drain_commit_results(self):
        """Attach finished async commits to their frame_log rows."""
        if self._committer is None:
            return
        drained = self._committer.drain_results()
        n = min(len(drained), len(self._async_rows))
        for idx, out in zip(self._async_rows[:n], drained):
            if 0 <= idx < len(self.frame_log):
                row = self.frame_log[idx]
                s = out.get("suffix", "")
                try:
                    e0, e1 = out["ev"]
                    row["commit_ms" + s] = float(e0.elapsed_time(e1))
                except (KeyError, RuntimeError):
                    pass
                for k in ("n_evicted", "n_cells", "cache_bytes"):
                    row[k + s] = out.get(k)
        del self._async_rows[:n]

    def _cell_confidence(self, output, c_map, n_cells):
        """Per-cell mean of the level0 max-softmax probability, or None.

        Computed in fp32: in bf16 the scores, which mostly lie in
        [0.9, 1.0], cannot be ranked reliably.
        """
        logits = output.get("seg_logits") if isinstance(output, dict) else None
        if logits is None or c_map is None:
            return None
        if int(logits.shape[0]) != int(c_map.numel()):
            raise RuntimeError(
                f"commit_gate: seg_logits has {int(logits.shape[0])} rows but "
                f"the level0 -> stage map has {int(c_map.numel())}")
        lg = logits.float()
        conf = (lg.amax(dim=-1) - lg.logsumexp(dim=-1)).exp()
        acc = torch.zeros(n_cells, dtype=torch.float32, device=conf.device)
        cnt = torch.zeros(n_cells, dtype=torch.float32, device=conf.device)
        acc.scatter_add_(0, c_map, conf)
        cnt.scatter_add_(0, c_map, torch.ones_like(conf))
        return acc / cnt.clamp(min=1.0)

    def _region_diag(self, output, keys, c3, n_d3, view_hit, low, inval,
                     pend):
        """Counters for `--region_stats` (ablation metrics, Appendix C.3).

        Every cell is classified from the ground truth alone, relative to its
        last observation in the sequence: `dyn` (majority label changed and
        one side is a movable class), `flip` (changed between static
        classes), `stable` (unchanged) or `new` (first observation). Hits are
        further split by the label at the time their entry was written
        (`stale` / `fresh`) and by entry age. `low` is the set removed by the
        confidence rule and `inval` the full commit mask.
        """
        logits = output.get("seg_logits") if isinstance(output, dict) else None
        seg0 = self._seg0
        if (logits is None or c3 is None or seg0 is None or n_d3 == 0
                or int(logits.shape[0]) != int(c3.numel())
                or int(seg0.numel()) != int(c3.numel())):
            return
        dev = keys.device
        nc = int(logits.shape[1])
        seg0 = seg0.long()
        valid = (seg0 != self.ignore_index) & (seg0 >= 0) & (seg0 < nc)
        cnt = torch.bincount(c3[valid] * nc + seg0[valid],
                             minlength=n_d3 * nc).view(n_d3, nc)
        n_pts = cnt.sum(dim=1)
        has = n_pts > 0
        lab = cnt.argmax(dim=1)
        bnd = (cnt > 0).sum(dim=1) >= 2
        if self._rd_movable is None or int(self._rd_movable.numel()) != nc:
            names = [str(n).lower() for n in (self._rd_names or [])]
            self._rd_movable = torch.tensor(
                [i < len(names) and names[i] in _MOVABLE_CLASSES
                 for i in range(nc)], dtype=torch.bool, device=dev)
        mov = self._rd_movable

        # change since the cell's last observation
        f_prev, l_prev = _rd_lookup(self._rd_key, self._rd_lab, keys)
        chg = f_prev & has & (l_prev != lab)
        dyn = chg & (mov[lab] | mov[l_prev])
        cats = (("dyn", dyn), ("flip", chg & ~dyn),
                ("stable", f_prev & has & ~chg), ("new", has & ~f_prev))
        # change since the resident entry was written (hits only)
        f_w, v_w = _rd_lookup(self._rd_wkey, self._rd_wval, keys)
        n_w = torch.div(v_w, _RD_C, rounding_mode="floor")
        rem = v_w - n_w * _RD_C
        t_w = torch.div(rem, _RD_T, rounding_mode="floor")
        l_w = rem - t_w * _RD_T
        served = view_hit & f_w & has
        stale = served & (l_w != lab)
        hcats = (("stale", stale), ("fresh", served & (l_w == lab)),
                 ("dens2", served & (n_pts >= 2 * n_w)))

        correct = (logits.argmax(dim=1) == seg0) & valid
        hit_l0 = view_hit[c3]
        for name, cm in cats:
            m0 = cm[c3] & valid
            mh = m0 & hit_l0
            pend[f"rd_n_{name}"] = m0.sum()
            pend[f"rd_ok_{name}"] = (m0 & correct).sum()
            pend[f"rd_nh_{name}"] = mh.sum()
            pend[f"rd_okh_{name}"] = (mh & correct).sum()
        for name, cm in hcats:
            m0 = cm[c3] & valid
            pend[f"rd_nh_{name}"] = m0.sum()
            pend[f"rd_okh_{name}"] = (m0 & correct).sum()
        # hit accuracy by entry age (frames since the entry was computed)
        if self._rd_age is None:
            self._rd_age = torch.zeros(len(_RD_AGE_ROWS), _RD_AGE_BINS,
                                       dtype=torch.float64, device=dev)
        a0 = (int(self._fi) - t_w).clamp(min=0, max=_RD_AGE_BINS - 1)[c3]
        m_s = served[c3] & valid
        m_d = m_s & dyn[c3]
        m_t = m_s & stale[c3]
        for row, m in enumerate((m_s, m_s & correct, m_d, m_d & correct,
                                 m_t, m_t & correct)):
            self._rd_age[row] += torch.bincount(
                a0, weights=m.to(torch.float64), minlength=_RD_AGE_BINS)
        pend["rd_cells"] = has.sum()
        pend["rd_cells_bnd"] = (has & bnd).sum()
        pend["rd_cells_hit"] = (has & view_hit).sum()
        pend["rd_cells_dyn"] = dyn.sum()
        if low is not None:
            lw = low & has
            m0 = lw[c3] & valid
            pend["rd_low"] = lw.sum()
            pend["rd_low_bnd"] = (lw & bnd).sum()
            pend["rd_low_hit"] = (lw & view_hit).sum()
            pend["rd_n_low"] = m0.sum()
            pend["rd_ok_low"] = (m0 & correct).sum()
            for name, cm in (("low_cand", lw & ~view_hit),
                             ("keep_cand", has & ~view_hit & ~inval)):
                mc = cm[c3] & valid
                pend[f"rd_n_{name}"] = mc.sum()
                pend[f"rd_ok_{name}"] = (mc & correct).sum()

        # every labelled cell refreshes its last-observed label; only misses
        # (the cells written this frame) refresh the label-at-write table
        self._rd_key, self._rd_lab = _rd_upsert(
            self._rd_key, self._rd_lab, keys[has], lab[has])
        wm = has & ~view_hit
        wval = (lab + int(self._fi) * _RD_T
                + n_pts.clamp(max=_RD_C // 2 - 1) * _RD_C)
        self._rd_wkey, self._rd_wval = _rd_upsert(
            self._rd_wkey, self._rd_wval, keys[wm], wval[wm])

    def flush_frame_timing(self):
        """Resolve this frame's gate events into frame_log (ms).

        Called by FrameProfiler right after its per-frame sync. The commit
        runs after that sync, so `commit_ms` is resolved one frame late.
        """
        self._drain_commit_results()
        pend = self._commit_ev
        if pend is not None:
            p0, p1, idx = pend
            if 0 <= idx < len(self.frame_log):
                try:
                    self.frame_log[idx]["commit_ms"] = float(
                        p0.elapsed_time(p1))
                except RuntimeError:
                    pass
                self._commit_ev = None

        ev = self._frame_ev
        self._frame_ev = None
        if ev is None or not self.frame_log:
            return
        m0, m1, s0, s1, u0, um, u1 = ev
        self.frame_log[-1]["match_ms"] = float(m0.elapsed_time(m1))
        self.frame_log[-1]["subset_ms"] = float(s0.elapsed_time(s1))
        self.frame_log[-1]["restore_ms"] = float(u0.elapsed_time(um))
        self.frame_log[-1]["stage_ms"] = float(um.elapsed_time(u1))
        self.frame_log[-1]["update_ms"] = float(u0.elapsed_time(u1))

    @staticmethod
    def _build_seg_ids(dataset):
        """Per-sample sequence id; the cache is reset whenever it changes.

        Uses `dataset.sequence_index` when available (Waymo; nuScenes with
        scene tokens in the info pkl), otherwise groups timestamp-contiguous
        frames, otherwise the per-frame directory (ScanNet video).
        """
        seq_idx = getattr(dataset, "sequence_index", None)
        if seq_idx is not None:
            return [int(s) for s in seq_idx]
        data_list = getattr(dataset, "data_list", None)
        if not data_list:
            return None
        if isinstance(data_list[0], dict) and "timestamp" in data_list[0]:
            ts = [float(d["timestamp"]) for d in data_list]
            diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
            pos = sorted(d for d in diffs if d > 0)
            med = pos[len(pos) // 2] if pos else 1.0
            thr = 4.0 * med
            seg, cur = [0], 0
            for d in diffs:
                if d <= 0 or d > thr:
                    cur += 1
                seg.append(cur)
            return seg
        return [os.path.dirname(d) for d in data_list]


class SpaceCacheS34Policy(SpaceCachePolicy):
    """SpaceCache at the third and fourth encoder stages.

    The enc3 reuse gate returns the restored full-resolution point, so the
    enc4 gates see the complete stage-3 output and run the same machinery
    one level deeper with their own cache and cell size. The enc4 commit
    hook is registered after the enc3 one and extends the same frame_log
    row with `*_d4` columns.
    """
    name = "spacecache_s34"
    streaming_kind = "d3d4"
    D4_DEPTH = 4

    def __init__(self):
        super().__init__()
        self.cache4 = None
        self.stage4 = None
        self.voxel_size_d4 = None
        self._frame4 = None
        self._frame_ev4 = None
        self._commit_ev4 = None

    def prepare(self, model, cfg, dataset, device, *, region_scale,
                lifetime, commit_async=False, triton_cache=False,
                expiry="staggered"):
        super().prepare(model, cfg, dataset, device,
                        region_scale=region_scale, lifetime=lifetime,
                        commit_async=commit_async,
                        triton_cache=triton_cache,
                        expiry=expiry)
        self.stage4 = get_stage_desc(model.backbone, self.D4_DEPTH)
        base_voxel = float(extract_grid_size(cfg.data.val.transform))
        self.voxel_size_d4 = base_voxel * (2 ** int(self.D4_DEPTH))
        self.cache4 = SpaceCache(
            device,
            default_life=_life_frames(self.cache_life),
            keys_unique=self.keys_unique,
            triton=self._triton,
            expiry=self.cache_expiry)

    def reset_state(self):
        super().reset_state()
        if self.cache4 is not None:
            self.cache4.reset()
        self._frame4 = None
        self._frame_ev4 = None
        self._commit_ev4 = None

    def reset_gate(self, input_dict):
        super().reset_gate(input_dict)
        # `_fi == 0` exactly on the frames where the stage-3 cache was reset
        if self.armed and self._fi == 0 and self.cache4 is not None:
            self.cache4.reset()

    @contextlib.contextmanager
    def streaming_context(self, model):
        enc = model.backbone.enc._modules
        enc4 = enc[f"enc{self.D4_DEPTH}"]
        with super().streaming_context(model):
            handles = [
                enc4._modules["down"].register_forward_hook(
                    self.mask_gate4),
                enc4._modules[f"block{self.stage4.n_blocks - 1}"]
                .register_forward_hook(self.reuse_gate4),
                model.register_forward_hook(self.commit_gate4),
            ]
            try:
                yield
            finally:
                for h in handles:
                    h.remove()
                self._frame4 = None

    def mask_gate4(self, _module, _inputs, point_out):
        """enc4 mask gate; the level0 map extends the enc3 one."""
        if not self.armed:
            return None
        meas = self.measure_breakdown and self.is_cuda
        m0 = self._evt() if meas else None
        if self._triton and self.pose is not None:
            keys, slot, hit = _tc.lookup_coords(
                point_out.coord, self.pose, self.coord_scale,
                self.voxel_size_d4 * self.mf, self.cache4.key)
            view = CacheView(keys, slot, hit,
                             torch.nonzero(~hit, as_tuple=True)[0])
        else:
            coord_g = to_global(point_out.coord, self.pose,
                                scale=self.coord_scale)
            keys = voxel_hash(coord_g, self.voxel_size_d4 * self.mf)
            view = self.cache4.lookup(keys)
        m1 = self._evt() if meas else None

        s0 = self._evt() if meas else None
        n_d4 = int(keys.numel())
        c3 = self._frame["c3"] if self._frame is not None else None
        pi4 = getattr(point_out, "pooling_inverse", None)
        c4 = pi4[c3] if (c3 is not None and pi4 is not None) else None
        n_miss = view.n_miss
        if 0 < n_miss < n_d4:
            sub = (_tc.build_subset_point(point_out, view.miss_idx)
                   if self._triton
                   else build_subset_point_fast(point_out, view.miss_idx))
        else:
            sub = None
        s1 = self._evt() if meas else None

        self._frame4 = dict(
            full=point_out, keys=keys, view=view, c4=c4, n_d4=n_d4,
            subset=sub, m0=m0, m1=m1, s0=s0, s1=s1,
        )
        return sub

    def reuse_gate4(self, _module, _inputs, blk_out):
        """enc4 reuse gate, identical to `reuse_gate`."""
        if not self.armed or self._frame4 is None:
            return None
        fm = self._frame4
        view = fm["view"]
        meas = self.measure_breakdown and self.is_cuda
        u0 = self._evt() if meas else None
        if fm["subset"] is None:
            self.cache4.restore(view, blk_out.feat, None)
            result = blk_out
            subset_size = fm["n_d4"]
            rows = None
        else:
            self.cache4.restore(view, fm["full"].feat, blk_out.feat)
            result = fm["full"]
            subset_size = int(blk_out.feat.shape[0])
            rows = blk_out.feat
        if "sparse_conv_feat" in result.keys():
            result.sparse_conv_feat = result.sparse_conv_feat.replace_feature(
                result.feat)
        um = self._evt() if meas else None
        self.cache4.write(view, result.feat, rows=rows)
        u1 = self._evt() if meas else None

        fm["restore_ev"] = (u0, um, u1) if meas else None
        fm["subset_size"] = subset_size
        fm["full"] = fm["subset"] = None
        return result

    def commit_gate4(self, _module, _args, output):
        """enc4 commit gate; same rules as `commit_gate`, pooled through c4."""
        if not self.armed or self._frame4 is None:
            return
        fm = self._frame4
        self._frame4 = None
        meas = self.measure_breakdown and self.is_cuda
        c0 = self._evt() if meas else None

        keys = fm["keys"]
        n_d4 = fm["n_d4"]
        c4 = fm["c4"]
        dev = keys.device
        view_hit = fm["view"].hit
        inval = torch.zeros(n_d4, dtype=torch.bool, device=dev)

        pend = {}
        if self.exclude_ignore and c4 is not None and self._seg0 is not None:
            ign = torch.bincount(c4[self._seg0 == self.ignore_index],
                                 minlength=n_d4) > 0
            ign &= ~view_hit
            inval |= ign
            pend["n_ignore_d4"] = ign.sum()

        low = self._low_confidence_cells(output, c4, n_d4, dev)
        if low is not None:
            inval |= low
            pend["n_lowconf_d4"] = low.sum()

        if c4 is not None and self._seg0 is not None:
            logits = (output.get("seg_logits")
                      if isinstance(output, dict) else None)
            if logits is not None and int(logits.shape[0]) == int(c4.numel()):
                correct = logits.argmax(dim=1) == self._seg0
                valid = self._seg0 != self.ignore_index
                hit_l0 = view_hit[c4] & valid
                miss_l0 = ~view_hit[c4] & valid
                pend["n_hit_l0_d4"] = hit_l0.sum()
                pend["n_hit_ok_d4"] = correct[hit_l0].sum()
                pend["n_miss_l0_d4"] = miss_l0.sum()
                pend["n_miss_ok_d4"] = correct[miss_l0].sum()

        if self.async_commit:
            self._commit_async(self.cache4, keys, inval,
                               len(self.frame_log) - 1, suffix="_d4")
        else:
            n_evicted = self.cache4.commit(
                keys, self.cache4.default_life, self._fi, invalidate=inval)
            c1 = self._evt() if meas else None
            if meas:
                self._commit_ev4 = (c0, c1, len(self.frame_log) - 1)

        pend["n_match_d4"] = view_hit.sum()
        pend["n_invalidated_d4"] = inval.sum()
        if self.frame_log:
            row = self.frame_log[-1]
            row.update(dict(
                n_hit_l0_d4=0, n_hit_ok_d4=0,
                n_miss_l0_d4=0, n_miss_ok_d4=0,
                cache_bytes_d4=(None if self.async_commit
                                else int(self.cache4.nbytes())),
                n_d4=n_d4, n_match_d4=None,
                subset_size_d4=int(fm.get("subset_size", n_d4)),
                match_ratio_d4=None,
                n_ignore_d4=0,
                n_invalidated_d4=None,
                n_lowconf_d4=0,
                n_evicted_d4=(None if self.async_commit
                              else int(n_evicted)),
                n_cells_d4=(None if self.async_commit
                            else int(self.cache4.n_cells)),
            ))
            self._pend_stats.append((row, pend))
        fm_ev = fm.get("restore_ev")
        self._frame_ev4 = ((fm["m0"], fm["m1"], fm["s0"], fm["s1"]) + fm_ev
                           if (meas and fm_ev) else None)

    def flush_frame_timing(self):
        super().flush_frame_timing()
        pend = self._commit_ev4
        if pend is not None:
            p0, p1, idx = pend
            if 0 <= idx < len(self.frame_log):
                try:
                    self.frame_log[idx]["commit_ms_d4"] = float(
                        p0.elapsed_time(p1))
                except RuntimeError:
                    pass
                self._commit_ev4 = None

        ev = self._frame_ev4
        self._frame_ev4 = None
        if ev is None or not self.frame_log:
            return
        m0, m1, s0, s1, u0, um, u1 = ev
        self.frame_log[-1]["match_ms_d4"] = float(m0.elapsed_time(m1))
        self.frame_log[-1]["subset_ms_d4"] = float(s0.elapsed_time(s1))
        self.frame_log[-1]["restore_ms_d4"] = float(u0.elapsed_time(um))
        self.frame_log[-1]["stage_ms_d4"] = float(um.elapsed_time(u1))
        self.frame_log[-1]["update_ms_d4"] = float(u0.elapsed_time(u1))


# Placement ablation: the same single cache at another encoder stage.

class SpaceCacheS0Policy(SpaceCachePolicy):
    name = "spacecache_s0"
    streaming_kind = "d0"
    intercept = "enc0"
    DEPTH = 0


class SpaceCacheS1Policy(SpaceCachePolicy):
    name = "spacecache_s1"
    streaming_kind = "d1"
    intercept = "enc1"
    DEPTH = 1


class SpaceCacheS2Policy(SpaceCachePolicy):
    name = "spacecache_s2"
    streaming_kind = "d2"
    intercept = "enc2"
    DEPTH = 2


class SpaceCacheS4Policy(SpaceCachePolicy):
    name = "spacecache_s4"
    streaming_kind = "d4"
    intercept = "enc4"
    DEPTH = 4
