"""ALGM (Norouzi et al., CVPR 2024) as a persistent point-level merge.

Follows the official implementation (tue-mps/algm-segmenter): CLAP local
merging at enc0.block0 and GBM global merging at enc3.block0, both on the
post-attention residual features, restored at dec3.block0 (GBM) and
dec0.block0 (CLAP). On LitePT (block-less decoder) GBM dissolves through
the unpooling trace and CLAP is restored after dec0.up.
"""

from __future__ import annotations

import contextlib
import logging
import types

import torch

# "pointcept.*" so run.py's get_root_logger FileHandler picks it up
logger = logging.getLogger("pointcept.algm")

from spacecache.common.block_timer import _resolve_timer
from spacecache.common.litept_block import (
    is_litept_block, litept_timed_block_forward,
)
from spacecache.common.point_merge import (
    clap_grid_indices, clap_window_indices, gbm_indices, merge_point,
    unmerge_point,
)
from spacecache.common.timed_block import timed_block_forward
from spacecache.common.token_merging_algos import tome_prof_region
from spacecache.policies.base import MergePolicy

# Policy-overhead regions (TOME_PROFILE=1, i.e. run.py --timing_level
# block). FrameProfiler reports them as attn/<suffix>, so suffixes must be
# unique across policies.
_R = {k: f"algm/{k}" for k in (
    "clap_select", "clap_merge", "clap_unmerge",
    "gbm_select", "gbm_merge", "gbm_unmerge")}


@contextlib.contextmanager
def _serialized_attn(block):
    """Set `attn.serialized_out` for the context: the stock forward (flash
    or dense) returns feat in serialized order and stashes `_ser_ctx`
    instead of applying the inverse. Unlike `tome_attn`, the attention
    path itself is not modified."""
    attn = block.attn
    had = hasattr(attn, "serialized_out")
    prev = getattr(attn, "serialized_out", None)
    attn.serialized_out = True
    try:
        yield
    finally:
        if had:
            attn.serialized_out = prev
        else:
            del attn.serialized_out
        if hasattr(attn, "_ser_ctx"):
            del attn._ser_ctx


# ---------------------------------------------------------------------------
#  Shared scaffolds
# ---------------------------------------------------------------------------


def _attn_residual_serialized(block, point, timer):
    """CPE + attn sub-block with the residual completed in serialized
    space. Returns (point, res_s, ser): point.feat is still serialized
    (P,C); ser = the attention's {order, inverse, patch_size}."""
    shortcut = point.feat
    with timer.section("CPE"):
        point = block.cpe(point)
    point.feat = shortcut + point.feat

    shortcut = point.feat
    if block.pre_norm:
        with timer.section("LN"):
            point = block.norm1(point)
    with timer.section("Attn"):
        point = block.attn(point)      # point.feat: serialized (P,C)
        ser = block.attn._ser_ctx
    feat_s = point.feat
    with timer.section("LS"):
        feat_s = block.ls1[0](feat_s)
    with timer.section("Attn"):
        feat_s = block.drop_path[0](feat_s)
    with timer.section("Ordering"):
        res_s = shortcut[ser["order"]] + feat_s
    if not block.pre_norm:
        with timer.section("LN"):
            res_s = block.norm1[0](res_s)
    return point, res_s, ser


def _mlp_subblock(block, sub, timer):
    """Stock MLP sub-block on a (possibly merged) Point."""
    shortcut = sub.feat
    if block.pre_norm:
        with timer.section("LN"):
            sub = block.norm2(sub)
    with timer.section("MLP"):
        sub = block.mlp(sub)
    with timer.section("LS"):
        sub = block.ls2(sub)
    with timer.section("MLP"):
        sub = block.drop_path(sub)
    sub.feat = shortcut + sub.feat
    if not block.pre_norm:
        with timer.section("LN"):
            sub = block.norm2(sub)
    sub.sparse_conv_feat = sub.sparse_conv_feat.replace_feature(sub.feat)
    return sub


# ---------------------------------------------------------------------------
#  Per-site block forwards
# ---------------------------------------------------------------------------


def _enc0_clap_forward(block, point, policy, ctx):
    """enc0.block0: CLAP local merge (windows whose mean pairwise cosine
    similarity reaches tau average into one token), persistent — the
    merged Point flows through the rest of the network until dec0.block0.
    """
    timer = _resolve_timer(ctx.get("timer"))
    policy._clap = None
    with _serialized_attn(block):
        point, res_s, ser = _attn_residual_serialized(block, point, timer)

        with timer.section("Ordering"):
            point.feat = res_s[ser["inverse"]]
        with timer.section("Others"), tome_prof_region(_R["clap_select"]):
            if policy.clap_mode == "grid":
                src, dst = clap_grid_indices(
                    point.feat, point.grid_coord, point.batch,
                    ser["inverse"], policy.tau, policy.cell,
                    policy.cell_offset)
            else:
                src, dst = clap_window_indices(
                    res_s, policy.wsize, policy.tau, ser["patch_size"])
        policy._clap_frames += 1
        policy._clap_tokens += point.feat.shape[0]
        policy._clap_merged += int(src.numel())
        if src.numel() > 0:
            with timer.section("Ordering"), tome_prof_region(
                    _R["clap_merge"]):
                sub, policy._clap = merge_point(
                    point, src, dst, ser["order"], ser["inverse"])
        else:
            sub = point
        return _mlp_subblock(block, sub, timer)


def _enc3_gbm_forward(block, point, policy, ctx):
    """enc3.block0: GBM global merge (alternating bipartite matching; an
    edge is kept iff its similarity >= mean + std / layer_idx of the
    best-match similarities), persistent — merges into the
    already-CLAP-reduced Point and stays merged until dec3.block0."""
    timer = _resolve_timer(ctx.get("timer"))
    policy._gbm = None
    with _serialized_attn(block):
        point, res_s, ser = _attn_residual_serialized(block, point, timer)

        with timer.section("Ordering"):
            point.feat = res_s[ser["inverse"]]
            batch_pos = point.batch[ser["order"]]
            canon_pos = torch.zeros(
                res_s.shape[0], dtype=torch.bool,
                device=res_s.device)
            canon_pos.index_fill_(0, ser["inverse"], True)
        with timer.section("Others"), tome_prof_region(_R["gbm_select"]):
            src, dst = gbm_indices(
                res_s, batch_pos, canon_pos, policy.layer_idx)
        policy._gbm_frames += 1
        policy._gbm_tokens += res_s.shape[0]
        policy._gbm_merged += int(src.numel())
        if src.numel() > 0:
            with timer.section("Ordering"), tome_prof_region(
                    _R["gbm_merge"]):
                sub, policy._gbm = merge_point(
                    point, src, dst, ser["order"], ser["inverse"])
        else:
            sub = point
        return _mlp_subblock(block, sub, timer)


def _dec3_gbm_restore_forward(block, point, policy, ctx):
    """dec3.block0: unmerge GBM. Must happen here — dec2.up already maps
    back to the CLAP-level token set, so GBM rows only exist between
    dec3.up and dec2.up."""
    timer = _resolve_timer(ctx.get("timer"))
    if policy._gbm is not None:
        with timer.section("Ordering"), tome_prof_region(_R["gbm_unmerge"]):
            point = unmerge_point(point, policy._gbm)
        policy._gbm = None
    return timed_block_forward(block, point, timer=ctx.get("timer"))


def _dec0_clap_restore_forward(block, point, policy, ctx):
    """dec0.block0: unmerge CLAP, then run the stock block."""
    timer = _resolve_timer(ctx.get("timer"))
    if policy._clap is not None:
        with timer.section("Ordering"), tome_prof_region(
                _R["clap_unmerge"]):
            point = unmerge_point(point, policy._clap)
        policy._clap = None
    return timed_block_forward(block, point, timer=ctx.get("timer"))


# ---------------------------------------------------------------------------
#  LitePT scaffolds — same sites (enc0 CLAP, enc3 GBM), LitePT anatomy:
#  conv/norm0 position encoder, no LayerScale, drop_path wraps the MLP
#  call itself. The decoder has no blocks (dec_depths=0), so the GBM
#  merge dissolves implicitly at dec2.up (the remapped pooling_inverse
#  broadcasts merged enc3 rows back into the enc2 parent) and only CLAP
#  needs an explicit restore — after dec0.up, the last unpool.
# ---------------------------------------------------------------------------


def _litept_attn_residual_serialized(block, point, timer):
    """LitePT twin of `_attn_residual_serialized`."""
    if block.enable_conv:
        shortcut = point.feat
        with timer.section("CPE"):
            point = block.conv(point)
        point.feat = shortcut + point.feat
    else:
        with timer.section("LN"):
            point = block.norm0(point)

    shortcut = point.feat
    if block.pre_norm:
        with timer.section("LN"):
            point = block.norm1(point)
    with timer.section("Attn"):
        point = block.attn(point)          # point.feat: serialized (P,C)
        ser = block.attn._ser_ctx
    with timer.section("Attn"):
        feat_s = block.drop_path[0](point.feat)
    with timer.section("Ordering"):
        res_s = shortcut[ser["order"]] + feat_s
    if not block.pre_norm:
        with timer.section("LN"):
            res_s = block.norm1[0](res_s)
    return point, res_s, ser


def _litept_mlp_subblock(block, sub, timer):
    """Stock LitePT MLP sub-block on a (possibly merged) Point."""
    shortcut = sub.feat
    if block.pre_norm:
        with timer.section("LN"):
            sub = block.norm2(sub)
    with timer.section("MLP"):
        sub = block.drop_path(block.mlp(sub))
    sub.feat = shortcut + sub.feat
    if not block.pre_norm:
        with timer.section("LN"):
            sub = block.norm2(sub)
    sub.sparse_conv_feat = sub.sparse_conv_feat.replace_feature(sub.feat)
    return sub


def _litept_enc0_clap_forward(block, point, policy, ctx):
    """enc0.block0 on LitePT: the block is conv-only (enc_attn[0]=False)
    so CLAP runs on the post-block feature — the same feature `x` the
    Utonia site matches on. LitePT keeps no serialization at this level,
    so the grid variant of CLAP is forced and merge_point runs under an
    identity order (original indices are already the canonical set)."""
    timer = _resolve_timer(ctx.get("timer"))
    policy._clap = None
    point = litept_timed_block_forward(block, point,
                                     timer=ctx.get("timer"))
    ident = torch.arange(point.feat.shape[0], device=point.feat.device)
    with timer.section("Others"), tome_prof_region(_R["clap_select"]):
        src, dst = clap_grid_indices(
            point.feat, point.grid_coord, point.batch, ident,
            policy.tau, policy.cell, policy.cell_offset)
    policy._clap_frames += 1
    policy._clap_tokens += point.feat.shape[0]
    policy._clap_merged += int(src.numel())
    if src.numel() == 0:
        return point
    with timer.section("Ordering"), tome_prof_region(_R["clap_merge"]):
        sub, policy._clap = merge_point(point, src, dst, ident, ident)
    return sub


def _litept_enc3_gbm_forward(block, point, policy, ctx):
    """enc3.block0 on LitePT — the same site as Utonia: GBM on the
    post-attention residual feature in serialized space. enc3 is the
    first attention stage, so `enc3.down` has already re-serialized and
    `_serialized_attn` works unchanged. The merge rides the unpooling
    trace; no explicit restore is needed (block-less decoder)."""
    timer = _resolve_timer(ctx.get("timer"))
    policy._gbm = None
    with _serialized_attn(block):
        point, res_s, ser = _litept_attn_residual_serialized(
            block, point, timer)

        with timer.section("Ordering"):
            point.feat = res_s[ser["inverse"]]
            batch_pos = point.batch[ser["order"]]
            canon_pos = torch.zeros(
                res_s.shape[0], dtype=torch.bool,
                device=res_s.device)
            canon_pos.index_fill_(0, ser["inverse"], True)
        with timer.section("Others"), tome_prof_region(_R["gbm_select"]):
            src, dst = gbm_indices(
                res_s, batch_pos, canon_pos, policy.layer_idx)
        policy._gbm_frames += 1
        policy._gbm_tokens += res_s.shape[0]
        policy._gbm_merged += int(src.numel())
        if src.numel() > 0:
            with timer.section("Ordering"), tome_prof_region(
                    _R["gbm_merge"]):
                sub, _ = merge_point(
                    point, src, dst, ser["order"], ser["inverse"])
        else:
            sub = point
        return _litept_mlp_subblock(block, sub, timer)


def _litept_dec0_clap_restore_unpool(mod, point, policy, ctx, orig):
    """dec0.up on LitePT: the last unpool returns the merged enc0-level
    parent — unmerge CLAP after it, before the seg head."""
    parent = orig(point)
    timer = _resolve_timer(ctx.get("timer"))
    if policy._clap is not None:
        with timer.section("Ordering"), tome_prof_region(
                _R["clap_unmerge"]):
            parent = unmerge_point(parent, policy._clap)
        policy._clap = None
    return parent


# ---------------------------------------------------------------------------
#  Policy
# ---------------------------------------------------------------------------


class AlgmPolicy(MergePolicy):
    """ALGM (policy `algm`): CLAP at enc0.block0 and GBM at enc3.block0,
    both restored in the decoder. GBM's threshold is adaptive (no fixed
    r); CLAP is set by `--algm_tau` (the swept τ), `--algm_clap_mode`,
    `--algm_wsize`, `--algm_cell` and `--algm_cell_offset`."""

    name = "algm"
    intercept = "algm"

    # CLAP knobs; overridable via ctx / CLI.
    tau = 0.99
    # clap_mode "grid" (default): cell^3 voxel windows on the input
    # grid — the 3D analogue of the paper's 2x2 image-grid window.
    # "serial": windows of `wsize` consecutive serialized tokens; `cell`
    # / `cell_offset` are then unused.
    clap_mode = "grid"
    wsize = 4
    cell = 4
    # grid window phase, in voxels: (grid_coord + cell_offset) // cell.
    # With offset 0, cell-4 windows coincide with the composed 4x4x4 cells
    # of the next two stride-2 poolings, which would absorb the merges;
    # offset 2 straddles those cells, so merges reduce the enc2/enc3
    # token counts.
    cell_offset = 2
    # GBM adaptive-threshold divisor; set from the backbone in
    # install_override (index of enc3.block0 among encoder blocks).
    layer_idx = 5

    def __init__(self):
        self._clap = None
        self._gbm = None

    def run_block(self, block, point, ctx):
        raise RuntimeError(
            "algm does not use the generic per-block override — it "
            "installs its own three-site intercept via install_override")

    @contextlib.contextmanager
    def install_override(self, model, ctx):
        bb = model.backbone
        litept = is_litept_block(
            getattr(getattr(bb.enc, "enc0", None), "block0", None))
        if litept:
            # LitePT: same encoder sites (enc0 CLAP, enc3 GBM), but the
            # block-less decoder dissolves GBM implicitly at dec2.up —
            # only CLAP needs an explicit restore, after dec0.up.
            paths = ("enc.enc0.block0", "enc.enc3.block0", "dec.dec0.up")
        else:
            paths = ("enc.enc0.block0", "enc.enc3.block0",
                     "dec.dec3.block0", "dec.dec0.block0")
        targets = []
        for path in paths:
            mod = bb
            for attr in path.split("."):
                if not hasattr(mod, attr):
                    raise SystemExit(
                        f"algm: backbone has no '{path}' — needs a "
                        f"U-shaped encoder + decoder (enc_mode "
                        f"backbones lack dec)")
                mod = getattr(mod, attr)
            targets.append(mod)

        self.tau = float(ctx.get("algm_tau", self.tau))
        self.wsize = int(ctx.get("algm_wsize", self.wsize))
        self.clap_mode = str(ctx.get("algm_clap_mode", self.clap_mode))
        self.cell = int(ctx.get("algm_cell", self.cell))
        self.cell_offset = int(
            ctx.get("algm_cell_offset", self.cell_offset))
        if self.clap_mode not in ("serial", "grid"):
            raise SystemExit(
                f"algm: unknown clap_mode '{self.clap_mode}' "
                f"(expected 'serial' or 'grid')")
        if litept:
            # enc0 carries no serialization on LitePT — serial-window
            # CLAP is undefined there; grid CLAP is the only variant.
            self.clap_mode = "grid"
            if not hasattr(targets[1], "attn"):
                raise SystemExit(
                    "algm: LitePT enc3.block0 has no attention "
                    "(enc_attn[3]=False) — GBM needs a post-attention "
                    "residual; pick a config with enc3 attention")
        # GBM's layer_idx = enc3.block0's index among encoder blocks
        # (the official code divides std by the layer number).
        self.layer_idx = 1 + sum(
            len([k for k in getattr(bb.enc, f"enc{i}")._modules
                 if k.startswith("block")])
            for i in range(3))

        if litept:
            entries = [_litept_enc0_clap_forward,
                       _litept_enc3_gbm_forward,
                       _litept_dec0_clap_restore_unpool]
            needs_orig = [False, False, True]
        else:
            entries = [_enc0_clap_forward,
                       _enc3_gbm_forward,
                       _dec3_gbm_restore_forward,
                       _dec0_clap_restore_forward]
            needs_orig = [False] * 4
        self._clap_frames = self._clap_tokens = self._clap_merged = 0
        self._gbm_frames = self._gbm_tokens = self._gbm_merged = 0
        saved = [(m, m.forward) for m in targets]
        for mod, entry, orig_flag in zip(targets, entries, needs_orig):
            if orig_flag:
                def bound(self, point, _e=entry, _p=self, _c=ctx,
                          _o=mod.forward):
                    return _e(self, point, _p, _c, _o)
            else:
                def bound(self, point, _e=entry, _p=self, _c=ctx):
                    return _e(self, point, _p, _c)
            mod.forward = types.MethodType(bound, mod)
        # the other blocks run the shared timed scaffold (on the merged
        # point between the merge and restore sites)
        containers = [bb.enc] + ([bb.dec] if hasattr(bb, "dec") else [])
        for container in containers:
            for stage_seq in container._modules.values():
                for sub_name, sub_mod in stage_seq._modules.items():
                    if (not sub_name.startswith("block")
                            or sub_mod in targets):
                        continue

                    def bound(self, point, _c=ctx):
                        return timed_block_forward(
                            self, point, timer=_c.get("timer"))
                    saved.append((sub_mod, sub_mod.forward))
                    sub_mod.forward = types.MethodType(bound, sub_mod)
        try:
            yield
        finally:
            for mod, fwd in saved:
                mod.forward = fwd
            if getattr(self, "_clap_frames", 0):
                logger.info(
                    "algm merge counts over %d frames — CLAP(enc0, "
                    "cell=%d off=%d, tau=%.3f): %d/%d tokens merged "
                    "(%.3f%%) | GBM(enc3): %d/%d (%.3f%%)",
                    self._clap_frames, self.cell, self.cell_offset,
                    self.tau,
                    self._clap_merged, self._clap_tokens,
                    100.0 * self._clap_merged / max(self._clap_tokens, 1),
                    self._gbm_merged, self._gbm_tokens,
                    100.0 * self._gbm_merged / max(self._gbm_tokens, 1))
