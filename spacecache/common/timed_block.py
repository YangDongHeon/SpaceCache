"""Timing-instrumented block forward shared by all policies.

`timed_block_forward` mirrors the stock `Block.forward` with every step in
a `BlockTimer` section and calls `block.attn(point)`, so whichever
attention module is live (stock or reparented by `tome_attn`) runs.
LitePT blocks dispatch to `litept_block.litept_timed_block_forward`.
"""

from __future__ import annotations

import torch

from spacecache.common.block_timer import _resolve_timer
from spacecache.common.litept_block import (
    is_litept_block, litept_timed_block_forward,
)


def timed_block_forward(block, point, timer=None):
    """Dispatch to the block scaffold matching `block`'s backbone."""
    if is_litept_block(block):
        return litept_timed_block_forward(block, point, timer=timer)
    return ptv3_timed_block_forward(block, point, timer=timer)


@torch.no_grad()
def ptv3_timed_block_forward(block, point, timer=None):
    """PT-v3m3 `Block.forward` with `BlockTimer` sections.

    The whole attention forward (qkv, rope, optional merge/unmerge, proj)
    is timed as `Attn`.
    """
    timer = _resolve_timer(timer)

    # === CPE + residual ===
    shortcut = point.feat
    with timer.section("CPE"):
        point = block.cpe(point)
    point.feat = shortcut + point.feat

    # === Attn block: norm1 -> attn -> ls1 -> drop_path -> residual ===
    shortcut = point.feat
    if block.pre_norm:
        with timer.section("LN"):
            point = block.norm1(point)
    with timer.section("Attn"):
        point = block.attn(point)
    with timer.section("LS"):
        point = block.ls1(point)
    with timer.section("Attn"):
        point = block.drop_path(point)
    point.feat = shortcut + point.feat
    if not block.pre_norm:
        with timer.section("LN"):
            point = block.norm1(point)

    # === MLP block: norm2 -> mlp -> ls2 -> drop_path -> residual ===
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

    point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
    return point
