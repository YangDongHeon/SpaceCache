"""ToMe (Bolya et al., ICLR 2023) on the residual stream of point tokens.

The stock attention (flash included) runs unmodified with `serialized_out`
set, and a hook on `attn.rope` captures the keys used for matching; tokens
merge between the attention and MLP sub-blocks. The merge persists until
the unpooling that exits the level (or the end of dec0 for enc0-level
merges). Policy `tome`, swept over `--merge_ratio` (Appendix B.2).
"""

from __future__ import annotations

import contextlib
import itertools
import types

import torch

from pointcept.models.utils.misc import offset2bincount

from spacecache.common.block_timer import _resolve_timer
from spacecache.common.cost_meter import note_attention
from spacecache.common.litept_block import is_litept_block
from spacecache.common.point_merge import (
    merge_point, tome_patch_indices, unmerge_point,
)
from spacecache.common.timed_block import timed_block_forward
from spacecache.common.token_merging_algos import (
    tome_prof_region,
)

# Policy-overhead regions (TOME_PROFILE=1, i.e. run.py --timing_level
# block). FrameProfiler reports them as attn/<suffix>, so suffixes must be
# unique across policies.
_R_SELECT = "tome/tome_select"      # K metric + per-patch bipartite indices
_R_MERGE = "tome/tome_merge"        # merge_point on the residual stream
_R_UNMERGE = "tome/tome_unmerge"    # explicit enc0-level restore


# ---------------------------------------------------------------------------
#  Attention hooks
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _serialized_attn(block):
    """Set `attn.serialized_out` for the context: the stock forward (flash
    or dense) returns feat in serialized order and stashes `_ser_ctx`
    instead of applying the inverse. The attention path itself is not
    modified."""
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


@contextlib.contextmanager
def _capture_keys(attn):
    """Record the post-rope keys of one `attn` call.

    `SerializedAttention.forward` calls `q, k = self.rope(q, k, coord)`
    before branching into flash or dense, so a forward hook on `attn.rope`
    sees the keys in padded serialized order, shape (P, H, Cd). Backbones
    built with `rope_base=0` have no `rope`; the hook then takes the
    pre-rope keys from `attn.qkv`.

    Yields a dict that holds "k" once the attention has run.
    """
    box = {}

    if getattr(attn, "rope", None) is not None:
        def hook(_mod, _inp, out):
            box["k"] = out[1]                     # (q, k) -> k
        handle = attn.rope.register_forward_hook(hook)
    else:
        H = attn.num_heads

        def hook(_mod, _inp, out):
            # Linear output (P, 3C) -> (P, 3, H, Cd), keys are index 1.
            box["k"] = out.reshape(out.shape[0], 3, H, -1)[:, 1]
        handle = attn.qkv.register_forward_hook(hook)

    try:
        yield box
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
#  Block forward
# ---------------------------------------------------------------------------


def _mlp_subblock(block, point, timer):
    """Stock MLP sub-block (the tail of `Block.forward`) on a (possibly
    merged) Point."""
    shortcut = point.feat
    if block.pre_norm:
        with timer.section("LN"):
            point = block.norm2(point)
    with timer.section("MLP"):
        point = block.mlp(point)
    with timer.section("LS"):
        point = block.ls2(point)
    with timer.section("MLP"):
        point = block.drop_path(point)
    point.feat = shortcut + point.feat
    if not block.pre_norm:
        with timer.section("LN"):
            point = block.norm2(point)
    point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(
        point.feat)
    return point



# ---------------------------------------------------------------------------
#  ToMe — persistent per-block merging (the dense-prediction form)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _capture_keys_litept(attn):
    """LitePT twin of `_capture_keys`. `PointROPEAttention` calls
    `self.rope` twice, q first and k second, each returning one
    (1, H, P, Cd) tensor. Keep the second output, transposed to the
    (P, H, Cd) layout of `_capture_keys`."""
    box = {}
    outs = []

    def hook(_mod, _inp, out):
        outs.append(out)

    handle = attn.rope.register_forward_hook(hook)
    try:
        yield box
    finally:
        handle.remove()
        if len(outs) >= 2:
            box["k"] = outs[1].squeeze(0).transpose(0, 1)   # (P, H, Cd)


def _litept_tome_forward(block, point, policy, ctx, level=None, fresh=False):
    """`_tome_forward` for a LitePT `Block`.

    Same persistent merge on LitePT's block anatomy: `conv`/`norm0`
    position encoder instead of `cpe`, no LayerScale, and the MLP residual
    goes through `drop_path(mlp(.))` as one PointSequential call. Only
    attention blocks are patched, so `level` is "enc0" only on configs
    with attention at enc0 or dec0.
    """
    if fresh:
        policy._stack.clear()
    timer = _resolve_timer(ctx.get("timer"))
    attn = block.attn

    # === position encoder + residual (stock) ===
    if block.enable_conv:
        shortcut = point.feat
        with timer.section("CPE"):
            point = block.conv(point)
        point.feat = shortcut + point.feat
    else:
        with timer.section("LN"):
            point = block.norm0(point)

    # === Attn sub-block, residual completed in serialized space ===
    shortcut = point.feat
    if block.pre_norm:
        with timer.section("LN"):
            point = block.norm1(point)
    with timer.section("Attn"):
        with _serialized_attn(block), _capture_keys_litept(attn) as cap:
            point = block.attn(point)
            ser = attn._ser_ctx
    with timer.section("Attn"):
        feat_s = block.drop_path[0](point.feat)
    with timer.section("Ordering"):
        res_s = shortcut[ser["order"]] + feat_s
    if not block.pre_norm:
        with timer.section("LN"):
            res_s = block.norm1[0](res_s)

    K = ser["patch_size"]
    P, C = res_s.shape
    if ctx.get("tome_clamp_patch") and 0 < P < K:
        # LitePT pads only stages longer than a patch, so a shorter stage
        # is one unpadded patch of P tokens; --tome_clamp_patch merges it
        # as such (as PT-v3m3's dense K = min(N, patch_size))
        K = P
    r = int(K * float(ctx["merge_ratio"]))
    sub = point
    if r > 0 and P % K == 0:
        with timer.section("Ordering"):
            point.feat = res_s[ser["inverse"]]
            with tome_prof_region(_R_SELECT):
                metric = cap["k"].mean(dim=1)          # (P, Cd)
                src, dst = tome_patch_indices(metric, r, K)
        if src.numel() > 0:
            if ctx.get("tome_size") and "tome_size" not in point.keys():
                point["tome_size"] = torch.ones(
                    point.feat.shape[0], dtype=torch.float32,
                    device=point.feat.device)
            with timer.section("Ordering"), tome_prof_region(_R_MERGE):
                sub, state = merge_point(
                    point, src, dst, ser["order"], ser["inverse"])
            if level == "enc0":
                policy._stack.append(state)
    else:
        with timer.section("Ordering"):
            point.feat = res_s[ser["inverse"]]

    # === MLP sub-block on the (possibly merged) Point ===
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


def _size_attn_forward(attn, point):
    """Dense serialized attention with ToMe's proportional-attention bias.

    Mirrors `SerializedAttention.forward`'s non-flash branch and adds
    `log(size)` to the logits per key position, which the flash kernel
    cannot express. Honours the `serialized_out`/`_ser_ctx` contract, so
    `_tome_forward` runs it in place of the stock attention when
    size tracking is on.

    `point["tome_size"]` (original order) counts how many original
    tokens each row represents; absent = all-ones = baseline attention.
    """
    if "min_bincount" not in point.keys():
        point["min_bincount"] = int(
            offset2bincount(point.offset).min())
    # patch_size_max exists only on non-flash builds; a flash-built
    # module keeps the configured patch_size, which plays the same role.
    attn.patch_size = min(
        point["min_bincount"],
        getattr(attn, "patch_size_max", attn.patch_size))

    H, K, C = attn.num_heads, attn.patch_size, attn.channels
    pad, unpad, cu_seqlens = attn.get_padding_and_inverse(point)
    order = point.serialized_order[attn.order_index][pad]
    inverse = unpad[point.serialized_inverse[attn.order_index]]

    qkv = attn.qkv(point.feat[order])
    size_s = (point.tome_size[order].float()
              if "tome_size" in point.keys() else None)

    if attn.rope_base:
        qkv = qkv.reshape(-1, 3, H, C // H)
        q, k, v = qkv.unbind(dim=1)
        # training-only coord augs are skipped (eval path); the
        # _capture_keys hook still fires on this rope call.
        q, k = attn.rope(q, k, point.coord[order].clone())
        q = q.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
        k = k.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
        v = v.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
    else:
        q, k, v = (qkv.reshape(-1, K, 3, H, C // H)
                   .permute(2, 0, 3, 1, 4).unbind(dim=0))

    if attn.compute_dtype is not None:
        q = q.to(attn.compute_dtype)
        k = k.to(attn.compute_dtype)
        v = v.to(attn.compute_dtype)
    if attn.upcast_attention:
        q = q.float()
        k = k.float()

    # exact shapes for the CostMeter; this runs as `attn.forward`, so the
    # module's pre/post hooks are live
    note_attention(q, k)
    logits = (q * attn.scale) @ k.transpose(-2, -1)   # (B, H, K, K)
    if size_s is not None:
        logits = logits + size_s.reshape(-1, K)[
            :, None, None, :].log().to(logits.dtype)
    if attn.enable_rpe:
        logits = logits + attn.rpe(attn.get_rel_pos(point, order))
    if attn.upcast_softmax:
        logits = logits.float()
    logits = attn.softmax(logits)
    # flash builds keep attn_drop as a float rate; the Dropout module
    # exists only on non-flash builds (identity under eval anyway).
    if isinstance(attn.attn_drop, torch.nn.Dropout):
        logits = attn.attn_drop(logits)
    logits = logits.to(v.dtype)
    feat = (logits @ v).transpose(1, 2).reshape(-1, C).to(qkv.dtype)

    if getattr(attn, "serialized_out", False):
        attn._ser_ctx = dict(order=order, inverse=inverse,
                             patch_size=K)
    else:
        feat = feat[inverse]
    feat = attn.proj(feat)
    feat = attn.proj_drop(feat)
    point.feat = feat
    return point


def _size_attn_module(mod, point, policy, ctx):
    """`attn.forward` replacement for size mode — routed through the
    module so the CostMeter's pre/post hooks fire and `note_attention`
    inside `_size_attn_forward` has a live stack entry."""
    return _size_attn_forward(mod, point)


def _size_pool_forward(mod, point, policy, ctx, orig):
    """GridPooling wrapper: a pooled token's size is the sum of the sizes
    of the rows pooled into it."""
    sz = point.tome_size if "tome_size" in point.keys() else None
    sub = orig(point)
    if sz is not None:
        m = torch.zeros(sub.feat.shape[0], dtype=torch.float32,
                        device=sz.device)
        m.index_add_(0, sub.pooling_inverse, sz.float())
        sub["tome_size"] = m
    return sub


def _size_unpool_forward(mod, point, policy, ctx, orig):
    """GridUnpooling wrapper: restore per-row sizes.

    The unpool returns the saved pre-pool point object, so if it carried
    `tome_size` that exact vector already came back. Otherwise the
    child's size is split evenly over the rows pooled into it."""
    sz = point.tome_size if "tome_size" in point.keys() else None
    inv = point.pooling_inverse if sz is not None else None
    parent = orig(point)
    if (sz is not None and inv is not None
            and "tome_size" not in parent.keys()):
        cnt = torch.bincount(inv, minlength=sz.numel()).to(sz.dtype)
        parent["tome_size"] = (sz[inv] / cnt[inv].clamp(min=1)).float()
    return parent


def _tome_forward(block, point, policy, ctx, level=None, fresh=False):
    """Persistent ToMe for one block — the paper's dense-prediction form.

    Stock attention, then the post-rope K metric and the merge on the
    serialized residual. The merge is applied to the Point itself via
    `merge_point` and not undone at the block's end: the shrunk point
    flows through the rest of the network, so
    every subsequent block, pooling and sparse conv runs on it.

    The unmerge is mostly implicit: `merge_point` remaps each merge's
    `pooling_inverse`, so the next GridUnpooling exiting that level
    scatters the merged features back to the parent rows (the paper's
    "duplicate the merged embeddings at the original indices"). enc0 has
    no unpooling below it (its rows feed the seg head), so enc0-level
    merges, made in enc0 and dec0 blocks, are pushed on `policy._stack`
    and restored by `_tome_restore_forward` at the end of dec0.

    level: "enc0" when this block operates on enc0-level rows (stages
        enc0 and dec0), else None — merges at any other level dissolve
        at the next unpool and their state is dropped.
    fresh: True for the first enc0-level merge site in execution order,
        resetting the stack each frame.
    """
    if fresh:
        policy._stack.clear()
    timer = _resolve_timer(ctx.get("timer"))
    attn = block.attn

    shortcut = point.feat
    with timer.section("CPE"):
        point = block.cpe(point)
    point.feat = shortcut + point.feat

    shortcut = point.feat
    if block.pre_norm:
        with timer.section("LN"):
            point = block.norm1(point)
    with timer.section("Attn"):
        with _serialized_attn(block), _capture_keys(attn) as cap:
            # under --proportional_attn, attn.forward is `_size_attn_module`
            # (see install_override)
            point = block.attn(point)
            ser = attn._ser_ctx
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

    K = ser["patch_size"]
    P, C = res_s.shape
    r = int(K * float(ctx["merge_ratio"]))
    if r <= 0 or P % K != 0:
        with timer.section("Ordering"):
            point.feat = res_s[ser["inverse"]]
        return _mlp_subblock(block, point, timer)

    with timer.section("Ordering"):
        point.feat = res_s[ser["inverse"]]
        with tome_prof_region(_R_SELECT):
            metric = cap["k"].mean(dim=1)              # (P, Cd)
            src, dst = tome_patch_indices(metric, r, K)
    if src.numel() > 0:
        if ctx.get("tome_size") and "tome_size" not in point.keys():
            # first merge of the frame: every original token counts 1
            point["tome_size"] = torch.ones(
                point.feat.shape[0], dtype=torch.float32,
                device=point.feat.device)
        with timer.section("Ordering"), tome_prof_region(_R_MERGE):
            sub, state = merge_point(
                point, src, dst, ser["order"], ser["inverse"])
        if level == "enc0":
            policy._stack.append(state)
    else:
        sub = point
    return _mlp_subblock(block, sub, timer)


def _tome_restore_forward(block, point, policy, ctx, do_merge):
    """dec0's last block: optionally merge like the others, then restore
    every enc0-level merge in reverse order before the seg head.

    do_merge tracks whether dec0 is inside --merge_stages: if only enc0
    was selected the dec0 blocks run stock and this site only restores.
    """
    if do_merge:
        point = _tome_forward(
            block, point, policy, ctx, level="enc0")
    else:
        point = timed_block_forward(
            block, point, timer=ctx.get("timer"))
    timer = _resolve_timer(ctx.get("timer"))
    while policy._stack:
        with timer.section("Ordering"), tome_prof_region(_R_UNMERGE):
            point = unmerge_point(point, policy._stack.pop())
    return point


def _tome_restore_unpool(mod, point, policy, ctx, orig):
    """enc0-level restore on `dec0.up` for block-less decoders (LitePT
    dec_depths=0): the unpool returns the merged enc0-level parent —
    drain the merge stack after it, before the seg head."""
    parent = orig(point)
    timer = _resolve_timer(ctx.get("timer"))
    while policy._stack:
        with timer.section("Ordering"), tome_prof_region(_R_UNMERGE):
            parent = unmerge_point(parent, policy._stack.pop())
    return parent


class TomePolicy:
    """ToMe for dense prediction (policy `tome`): every patched block
    merges r tokens per serialized patch on the residual stream and the
    merge persists through the rest of the network.
    Merge maps ride the unpooling trace (remapped `pooling_inverse`), so
    merges at every level above enc0 are broadcast back to parent rows
    by the next GridUnpooling; enc0-level merges are unmerged explicitly
    at the end of dec0 before the seg head.

    `--merge_ratio` is the per-block fraction of each patch's tokens
    removed; the merges compound over blocks.

    `ctx["tome_size"]` (CLI `--proportional_attn`) enables the paper's
    proportional attention: each row carries a `tome_size` count,
    merges sum it, pooling sums it into the child token, unpooling
    restores the saved parent's exact vector, and attention adds
    `log(size)` to key logits via `_size_attn_forward`. That bias needs
    the eager attention path — with the flag on, patched blocks run
    dense; without it they keep flash.

    Remaining differences from the paper: matching is per serialized
    patch (the codebase's locality partition) rather than global over
    the whole token set, and r is a per-patch ratio rather than an
    absolute token count.
    """

    name = "tome"

    def __init__(self):
        self._stack = []

    def run_block(self, block, point, ctx):
        raise RuntimeError(
            "tome does not use the generic per-block override — "
            "it installs its own intercept via install_override")

    @contextlib.contextmanager
    def install_override(self, model, ctx):
        """Patch every enc/dec block with the persistent-merge forward,
        plus the enc0-level restore at dec0's last block.

        `ctx["merge_stages"]` restricts which stages merge; whenever
        enc0-level merges can exist (stages unset, or enc0/dec0 in the
        set) the restore is installed — even if dec0 itself is not a
        merge site, in which case its last block runs stock + restore.
        """
        bb = model.backbone
        if not hasattr(bb, "dec"):
            raise SystemExit(
                "tome needs the decoder (enc_mode backbones "
                "lack dec)")
        stages = ctx.get("merge_stages")
        self._stack = []

        def blocks_of(seq):
            return [(n, m) for n, m in seq._modules.items()
                    if n.startswith("block") and hasattr(m, "attn")]

        saved = []

        def patch(mod, fn, *fargs):
            def bound(self, point, _f=fn, _a=fargs, _p=self, _c=ctx):
                return _f(self, point, _p, _c, *_a)
            saved.append((mod, mod.forward))
            mod.forward = types.MethodType(bound, mod)

        # LitePT: only blocks with `enable_attn` carry .attn, so
        # `blocks_of` limits merging to attention stages. Its decoder has
        # no blocks; enc0-level restores fall back to dec0.up below.
        litept = any(
            is_litept_block(m)
            for stage_seq in bb.enc._modules.values()
            for m in stage_seq._modules.values())
        block_fwd = (_litept_tome_forward if litept
                     else _tome_forward)

        fresh_taken = False
        dec0_last = None
        dec0_patched = False
        for container_name, container in (("enc", bb.enc), ("dec", bb.dec)):
            for stage_name, stage_seq in container._modules.items():
                if stages is not None and stage_name not in stages:
                    continue
                bl = blocks_of(stage_seq)
                for sub_name, sub_mod in bl:
                    level = ("enc0" if stage_name in ("enc0", "dec0")
                             else None)
                    if (container_name == "dec" and stage_name == "dec0"
                            and sub_name == bl[-1][0]):
                        dec0_last = sub_mod
                        dec0_patched = True
                        continue          # patched with restore below
                    fresh = level == "enc0" and not fresh_taken
                    fresh_taken = fresh_taken or fresh
                    patch(sub_mod, block_fwd, level, fresh)

        # stages excluded by --merge_stages run the shared timed scaffold
        if stages is not None:
            for stage_name, stage_seq in itertools.chain(
                    bb.enc._modules.items(), bb.dec._modules.items()):
                if stage_name in stages:
                    continue
                for sub_name, sub_mod in stage_seq._modules.items():
                    if not sub_name.startswith("block"):
                        continue

                    def bound(self, point, _c=ctx):
                        return timed_block_forward(
                            self, point, timer=_c.get("timer"))
                    saved.append((sub_mod, sub_mod.forward))
                    sub_mod.forward = types.MethodType(bound, sub_mod)

        # enc0-level merges exist iff enc0 or dec0 blocks merge.
        enc0_active = (stages is None
                       or "enc0" in stages or "dec0" in stages)
        if enc0_active:
            if dec0_last is None:
                bl = blocks_of(bb.dec.dec0)
                if bl:
                    patch(bl[-1][1], _tome_restore_forward,
                          dec0_patched)
                elif "up" in bb.dec.dec0._modules:
                    # block-less decoder (LitePT): drain the merge stack
                    # on the enc0-level parent the last unpool returns.
                    up = bb.dec.dec0.up
                    patch(up, _tome_restore_unpool, up.forward)
                else:
                    raise SystemExit(
                        "tome: dec0 has no attention blocks to "
                        "carry the enc0-level restore")
            else:
                patch(dec0_last, _tome_restore_forward,
                      dec0_patched)

        # Proportional attention: patch `attn.forward` on every block,
        # since merged tokens also flow through stages that are not merge
        # sites (unmerged rows have size 1, i.e. no bias). Patching
        # attn.forward keeps the CostMeter's module hooks live. Pooling
        # sums sizes into the child token; unpooling restores the saved
        # parent's exact vector.
        if ctx.get("tome_size"):
            for container in (bb.enc, bb.dec):
                for stage_seq in container._modules.values():
                    for _n, sub_mod in blocks_of(stage_seq):
                        # LitePT is flash-only: keep the stock flash call
                        # and drop the +log(size) key bias
                        if not litept:
                            patch(sub_mod.attn, _size_attn_module)
                    for name, fn in (("down", _size_pool_forward),
                                     ("up", _size_unpool_forward)):
                        mod = stage_seq._modules.get(name)
                        if mod is not None:
                            patch(mod, fn, mod.forward)

        try:
            yield
        finally:
            for mod, fwd in saved:
                mod.forward = fwd
            self._stack = []
