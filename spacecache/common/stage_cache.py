"""Encoder-stage handles and the miss-subset Point used by SpaceCache."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pointcept.models.utils.structure import Point


D3_DEPTH = 3


@dataclass
class StageDesc:
    """Handle of `backbone.enc.enc{depth}`: `enc_seq` holds `down` (the
    pooling, absent at depth 0) and `block0..block{n_blocks-1}`."""
    depth: int
    enc_seq: object
    n_blocks: int


def get_stage_desc(backbone, depth):
    enc_seq = backbone.enc._modules[f"enc{depth}"]
    n_blocks = sum(
        1 for k in enc_seq._modules.keys() if k.startswith("block")
    )
    return StageDesc(depth=depth, enc_seq=enc_seq, n_blocks=n_blocks)


def build_subset_point_fast(point_full, sub_idx):
    """Subset Point of `point_full` at `sub_idx`, without re-serializing.

    The serialized orders of the subset are derived from the parent's with a
    fixed-shape gather/scatter (no boolean indexing, no host sync), and the
    sparse tensor reuses the parent's spatial shape, so the construction is
    also legal under CUDA-graph capture.
    """
    device = point_full.feat.device
    n_full = int(point_full.feat.shape[0])
    n_sub = int(sub_idx.numel())

    parent_to_sub = torch.full(
        (n_full,), -1, dtype=torch.long, device=device,
    )
    parent_to_sub[sub_idx] = torch.arange(n_sub, device=device)

    pd = {
        "feat": point_full.feat[sub_idx].clone(),
        "coord": point_full.coord[sub_idx].clone(),
        "batch": point_full.batch[sub_idx].clone(),
        "offset": torch.full((1,), n_sub, dtype=torch.long, device=device),
    }
    if "serialized_order" in point_full.keys():
        # Every serialized order is a permutation of all tokens, so each row
        # keeps exactly n_sub entries: rank the survivors with cumsum and
        # scatter them into (n_orders, n_sub); the rest go to a junk column.
        parent_order = point_full.serialized_order   # (n_orders, n_full)
        n_orders = parent_order.shape[0]
        vals = parent_to_sub[parent_order]
        in_sub_at_order = vals >= 0
        rank = torch.where(
            in_sub_at_order, in_sub_at_order.cumsum(-1) - 1,
            torch.full_like(vals, n_sub))
        sub_order = torch.zeros(
            n_orders, n_sub + 1, dtype=torch.long, device=device)
        sub_order.scatter_(1, rank, vals.clamp(min=0))
        sub_order = sub_order[:, :n_sub]
        sub_inverse = torch.zeros_like(sub_order).scatter_(
            dim=1, index=sub_order,
            src=torch.arange(n_sub, device=device).repeat(n_orders, 1),
        )
        pd["serialized_order"] = sub_order
        pd["serialized_inverse"] = sub_inverse
    if "order" in point_full.keys():
        # GridPooling re-serializes pooled points from `point.order`
        pd["order"] = point_full["order"]
    if "grid_coord" in point_full.keys():
        pd["grid_coord"] = point_full.grid_coord[sub_idx].clone()
    if "grid_size" in point_full.keys():
        pd["grid_size"] = point_full.grid_size
    if "sparse_shape" in point_full.keys():
        pd["sparse_shape"] = point_full.sparse_shape
    sub = Point(pd)
    if "sparse_conv_feat" in point_full.keys() and "grid_coord" in pd:
        # Point.sparsify() reads batch[-1] on the host; reuse the parent's
        # spatial shape and batch size instead.
        psct = point_full.sparse_conv_feat
        sub["sparse_conv_feat"] = type(psct)(
            features=pd["feat"],
            indices=torch.cat(
                [pd["batch"].unsqueeze(-1).int(),
                 pd["grid_coord"].int()], dim=1).contiguous(),
            spatial_shape=psct.spatial_shape,
            batch_size=psct.batch_size,
        )
    else:
        sub.sparsify()
    return sub
