"""Timing-instrumented LitePT-v1 block forward.

LitePT blocks use `conv` (or `norm0`) as position encoder, have no
LayerScale, and skip the attention/MLP half when `enable_attn` is off.
The scaffold uses the same `BlockTimer` categories as `timed_block.py`.
"""

from __future__ import annotations

import torch

from spacecache.common.block_timer import _resolve_timer


def is_litept_block(block):
    """True for a LitePT-v1 `Block`, False for a PT-v3m3 one.

    Duck-typed on the position encoder: PT-v3m3 blocks have `cpe`; LitePT
    blocks have `conv` (enable_conv=True) or `norm0` instead."""
    return (not hasattr(block, "cpe")
            and (hasattr(block, "conv") or hasattr(block, "norm0")))


@torch.no_grad()
def litept_timed_block_forward(block, point, timer=None):
    """LitePT `Block.forward` with `BlockTimer` sections.

      CPE   `block.conv` (SubMConv3d position encoder)
      LN    `block.norm0` / `norm1` / `norm2`
      Attn  attention forward and its drop_path
      MLP   `block.mlp` and its drop_path
      LS    empty (no LayerScale)
    """
    timer = _resolve_timer(timer)

    # === position encoder (conv + residual) or plain norm0 ===
    if block.enable_conv:
        shortcut = point.feat
        with timer.section("CPE"):
            point = block.conv(point)
        point.feat = shortcut + point.feat
    else:
        with timer.section("LN"):
            point = block.norm0(point)

    if block.enable_attn:
        # === Attn: norm1 -> attn -> drop_path -> residual ===
        shortcut = point.feat
        if block.pre_norm:
            with timer.section("LN"):
                point = block.norm1(point)
        with timer.section("Attn"):
            point = block.drop_path(block.attn(point))
        point.feat = shortcut + point.feat
        if not block.pre_norm:
            with timer.section("LN"):
                point = block.norm1(point)

        # === MLP: norm2 -> mlp -> drop_path -> residual ===
        shortcut = point.feat
        if block.pre_norm:
            with timer.section("LN"):
                point = block.norm2(point)
        with timer.section("MLP"):
            point = block.drop_path(block.mlp(point))
        point.feat = shortcut + point.feat
        if not block.pre_norm:
            with timer.section("LN"):
                point = block.norm2(point)

    point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
    return point
