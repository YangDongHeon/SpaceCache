"""Persistent point-level token merging (policies `tome` and `algm`).

Matchers return (src_pos, dst_pos) pairs of flat padded-serialized
positions (`order[p]` is the original token at position p, `inverse[i]`
the canonical position of token i). `merge_point` shrinks the Point
itself, so later blocks, pooling and sparse convs run on the merged
cloud; `unmerge_point` restores the pre-merge cloud.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spacecache.common.cost_meter import note_matmul
from spacecache.common.token_merging_algos import tome_fine_region

from pointcept.models.utils.structure import Point

from spacecache.common.stage_cache import build_subset_point_fast


# ---------------------------------------------------------------------------
#  Matchers — return (src_pos, dst_pos) in flat padded-serialized positions
# ---------------------------------------------------------------------------


def clap_window_indices(feat_s, w, tau, K):
    """CLAP — ALGM's local merge (local_merge.conditional_pooling).

    Non-overlapping windows of `w` consecutive serialized tokens inside
    each patch (serialized adjacency ~= spatial adjacency; windows never
    cross batch boundaries because each batch occupies whole patches).
    A window is merged iff its mean pairwise cosine similarity (self
    excluded) >= tau; all `w` tokens then average into one token whose
    slot is the window's last position (paper's dst convention).

    feat_s: (P, C) padded serialized features — the post-attention
    residual x (the paper matches on token embeddings, not on K).

    Returns (src_pos, dst_pos): flat padded positions, len S*(w-1).
    """
    P, C = feat_s.shape
    B = P // K
    Kc = K - K % w
    x = F.normalize(feat_s.view(B, K, C)[:, :Kc], dim=-1)
    x = x.view(B, Kc // w, w, C)
    sims = x @ x.transpose(-1, -2)                      # (B, nwin, w, w)
    mu = (sims.sum((-1, -2)) - w) / (w * (w - 1))       # (B, nwin)
    sel = (mu >= tau).nonzero()                         # (S, 2): [patch, win]
    if sel.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=feat_s.device)
        return empty, empty
    base = sel[:, 0] * K + sel[:, 1] * w                # (S,)
    pos = base[:, None] + torch.arange(w, device=feat_s.device)
    dst = pos[:, -1]                                    # (S,)
    src = pos[:, :-1].reshape(-1)                       # (S*(w-1),)
    return src, dst.repeat_interleave(w - 1)


def clap_grid_indices(feat, grid_coord, batch, inverse, tau, cell=2,
                      offset=0):
    """CLAP on cell^3 voxel windows — the 3D analogue of ALGM's 2x2
    local window (local_merge.conditional_pooling).

    ALGM pools non-overlapping 2x2 windows of the image token grid. Here
    a window is a `cell`-sided cube of the input voxel grid (tokens
    sharing `(grid_coord + offset) // cell` within a batch), i.e. the
    same non-overlapping spatial partition in 3D; `clap_window_indices`
    is the 1D serialized-order alternative.

    A window is merged iff its mean pairwise cosine similarity (self
    excluded) >= tau; all its tokens then average into one token whose
    slot is the window's last serialized position (paper's dst
    convention). The sparse grid gives windows of 1..cell^3 tokens;
    single-token windows never merge.

    The mean pairwise similarity avoids the (n, n) Gram matrix: for unit
    vectors, sum_{i,j} cos(x_i, x_j) = ||sum_i x_i||^2, so
    mu = (||S||^2 - n) / (n(n-1)) with S the per-window sum.

    feat: (N, C) post-attention residual x in original token order (the
        paper matches on token embeddings, not on K).
    grid_coord / batch: (N, 3) / (N,) from the Point.
    inverse: the attention module's (N,) canonical serialized position
        per original token.

    Returns (src_pos, dst_pos): flat padded-serialized positions, the
    same convention as `clap_window_indices`.
    """
    device = feat.device
    N, C = feat.shape

    # window id = (batch, (grid_coord + offset) // cell), packed into
    # one int64 so the grouping is a 1D unique. `offset` (voxels) shifts
    # the window phase off the cells of the following GridPooling.
    with tome_fine_region("match/clap_group"):
        gc = torch.div(grid_coord.long() + offset, cell,
                       rounding_mode="floor")
        gc = gc - gc.min(dim=0, keepdim=True).values
        # dims stay on-device: a python int here is a D2H sync (and a
        # CUDA-graph capture violation); 0-dim tensor scalars broadcast
        # the same way.
        dims = gc.max(dim=0).values + 1
        key = batch.long()
        for d in range(3):
            key = key * dims[d] + gc[:, d]
        _, gid = torch.unique(key, return_inverse=True)
    # At most N distinct windows — use N as the host-known upper bound
    # instead of int(gid.max())+1, another D2H sync.
    G = N

    with tome_fine_region("match/clap_sim"):
        x = F.normalize(feat.float(), dim=-1)
        S = torch.zeros(G, C, device=device)
        S.index_add_(0, gid, x)
        n = torch.zeros(G, device=device)
        n.index_add_(0, gid, torch.ones(N, device=device))
        mu = (S.pow(2).sum(-1) - n) / (n * (n - 1.0)).clamp(min=1.0)
        ok = (n >= 2) & (mu >= tau)                      # (G,)

    # dst = the window's last serialized position. `inverse` is injective
    # over original tokens, so `pos == rep[gid]` selects exactly one.
    with tome_fine_region("match/clap_select"):
        pos = inverse
        rep = torch.full((G,), -1, dtype=torch.long, device=device)
        rep.scatter_reduce_(0, gid, pos, reduce="amax",
                            include_self=False)
        is_src = ok[gid] & (pos != rep[gid])
        idx = is_src.nonzero().squeeze(-1)  # empty when nothing merges
        return pos[idx], rep[gid[idx]]


def tome_patch_indices(metric_s, r, K):
    """ToMe BSM on serialized patches — index-pair form for merge_point.

    Per patch of K serialized tokens: alternating split (src = even
    positions, dst = odd), per-src argmax cosine edge over dst, keep the
    top-`r` edges by similarity. Same decision rule as
    ToMe's bipartite soft matching (which returns
    merge/unmerge closures) but yields flat padded positions.

    metric_s: (P, Cd) matching metric in padded serialized order — ToMe
        matches on the attention keys, head-averaged.
    """
    P, Cd = metric_s.shape
    B = P // K
    device = metric_s.device
    empty = torch.empty(0, dtype=torch.long, device=device)
    x = F.normalize(metric_s.view(B, K, Cd).float(), dim=-1)
    a, b = x[:, ::2], x[:, 1::2]                       # src, dst split
    rr = min(int(r), a.shape[1])
    if rr <= 0:
        return empty, empty
    note_matmul(a, b)          # matching cost, same convention as BSM
    with tome_fine_region("match/tome_scores"):
        scores = a @ b.transpose(-1, -2)               # (B, Ka, Kb)
        node_max, node_idx = scores.max(dim=-1)
    with tome_fine_region("match/tome_argsort"):
        edge = node_max.argsort(dim=-1, descending=True)[:, :rr]
        dst_b = node_idx.gather(1, edge)
    base = torch.arange(B, device=device)[:, None] * K
    src = base + edge * 2
    dst = base + dst_b * 2 + 1
    return src.reshape(-1), dst.reshape(-1)


def gbm_indices(feat_s, batch_pos, canon_pos, layer_idx):
    """GBM — ALGM's global merge (global_merge.turbo_matching).

    Alternating A/B split over each batch's serialized token sequence —
    canonical positions only (pad slots never participate). Fully-
    connected bipartite cosine-similarity graph, per-A argmax edge kept,
    and edges retained iff sim >= mean(node_max) + std(node_max) /
    layer_idx — the official code's adaptive threshold. Cross-batch edges
    are masked.

    feat_s: (P, C) padded serialized post-residual features.
    batch_pos: (P,) batch index per serialized position
        (`point.batch[order]`).
    canon_pos: (P,) bool — real (non-pad) serialized positions
        (`canon_pos[inverse] = True`).

    The (P/2) x (P/2) score matrix never materializes: argmax is computed
    in row chunks. Returns (src_pos, dst_pos): flat padded positions.
    """
    device = feat_s.device
    _split = tome_fine_region("match/gbm_split")
    _split.__enter__()
    xn = F.normalize(feat_s, dim=-1)
    cpos = canon_pos.nonzero().squeeze(-1)              # serialized order
    cb = batch_pos[cpos]
    n = cpos.numel()
    # Within-batch parity: each batch's token range is contiguous in
    # canonical positions, so rank minus its batch's start gives parity.
    bounds = torch.cat([
        cpos.new_zeros(1),
        (cb[1:] != cb[:-1]).nonzero().squeeze(-1) + 1])
    bid = torch.searchsorted(
        bounds, torch.arange(n, device=device), right=True) - 1
    wpar = (torch.arange(n, device=device) - bounds[bid]) % 2
    # bool-mask indexing is an implicit dynamic-shape nonzero — illegal
    # under CUDA-graph capture — so go through nonzero explicitly (the
    # model_graph recorder replays it).
    a_pos = cpos[torch.nonzero(wpar == 0, as_tuple=False).squeeze(-1)]
    b_pos = cpos[torch.nonzero(wpar == 1, as_tuple=False).squeeze(-1)]
    a, b = xn[a_pos], xn[b_pos]
    ba, bb = batch_pos[a_pos], batch_pos[b_pos]

    Pa, Pb = a.shape[0], b.shape[0]
    chunk = max(1, (1 << 26) // max(Pb, 1))
    node_max = torch.full((Pa,), -float("inf"), device=device)
    node_idx = torch.zeros(Pa, dtype=torch.long, device=device)
    _split.__exit__(None, None, None)
    with torch.no_grad(), tome_fine_region("match/gbm_scores"):
        for s0 in range(0, Pa, chunk):
            s1 = min(s0 + chunk, Pa)
            note_matmul(a[s0:s1], b)     # GBM matching cost
            scores = a[s0:s1] @ b.T                            # (c, Pb)
            scores[ba[s0:s1, None] != bb[None, :]] = -float("inf")
            mx, ix = scores.max(dim=-1)
            better = mx > node_max[s0:s1]
            node_max[s0:s1] = torch.where(
                better, mx, node_max[s0:s1])
            node_idx[s0:s1] = torch.where(
                better, ix, node_idx[s0:s1])
    with torch.no_grad(), tome_fine_region("match/gbm_select"):
        thr = node_max.mean() + node_max.std() / layer_idx
        sel = (node_max >= thr).nonzero().squeeze(-1)
        return a_pos[sel], b_pos[node_idx[sel]]


# ---------------------------------------------------------------------------
#  Point shrink / restore
# ---------------------------------------------------------------------------


def merge_point(point, src_pos, dst_pos, order, inverse):
    """Shrink `point` to the surviving tokens of a serialized-space merge.

    Pad slots are mapped to their canonical token, so the merge map is a
    plain (N,) -> (M,) index in original-token space; the reduced Point
    is built with `stage_cache.build_subset_point_fast`.

    point: pre-merge Point, `point.feat` in original order (N, C).
    src_pos/dst_pos: flat padded-serialized position pairs (src merges
        into dst), e.g. from `clap_window_indices` / `gbm_indices`.
    order/inverse: the attention module's padded order + inverse.

    Returns (sub, state):
      sub   — Point restricted to the surviving tokens, in serialized
              order; feat = per-cluster mean over original tokens;
              serialized order/inverse + sparse_conv_feat rebuilt.
      state — {"parent", "src_to_dst": (N,) merged-row per original
              token} for `unmerge_point`.
    """
    device = point.feat.device
    N, C = point.feat.shape
    P = order.numel()
    _map = tome_fine_region("merge/pm_map")
    _map.__enter__()
    canon = inverse[order]                               # (P,)

    src_c = canon[src_pos]
    dst_c = canon[dst_pos]

    # Canonical merge map with transitive resolution: a pad slot can alias
    # a canonical token that is itself a src, producing chains (x->c->d);
    # follow them so every token lands on a surviving dst.
    ser_map = torch.arange(P, device=device)
    ser_map[src_c] = dst_c
    # Fixpoint in ~2 hops: canonical src/dst sets are disjoint in every
    # matcher here, so chains only form through pad aliases and stay
    # short. A fixed count avoids the per-iteration torch.equal sync.
    for _ in range(4):
        ser_map = ser_map[ser_map]

    canon_pos = torch.zeros(P, dtype=torch.bool, device=device)
    # index_fill_ bakes the scalar into the kernel; `t[i] = True` wraps it
    # as a CPU tensor -> pageable H2D memcpy, illegal under graph capture.
    canon_pos.index_fill_(0, inverse, True)
    # Pathological pad-alias cycles (x->y->x) leave non-fixpoint entries;
    # treat them as unmerged rather than dropping their tokens.
    # `torch.where` keeps this fixed-shape — bool index_put is an
    # implicit dynamic-shape nonzero, illegal under graph capture.
    bad = canon_pos & (ser_map != ser_map[ser_map])
    ser_map = torch.where(
        bad, torch.arange(P, device=device), ser_map)

    keep_c = (canon_pos & (ser_map == torch.arange(P, device=device))
              ).nonzero().squeeze(-1)                    # sorted survivors
    M = keep_c.numel()

    dst_rank = torch.full((P,), -1, dtype=torch.long, device=device)
    dst_rank[keep_c] = torch.arange(M, device=device)
    src_to_dst = dst_rank[ser_map][inverse]              # (N,)
    _map.__exit__(None, None, None)

    # ToMe token size: with "tome_size" (original tokens per row), the
    # merged feature is the size-weighted mean (the paper's merge_wavg)
    # and the dst size is the cluster sum; otherwise a plain mean.
    with tome_fine_region("merge/pm_reduce"):
        w = (point.tome_size.unsqueeze(-1).to(point.feat.dtype)
             if "tome_size" in point.keys()
             else torch.ones(N, 1, dtype=point.feat.dtype, device=device))
        merged_feat = torch.zeros(M, C, dtype=point.feat.dtype,
                                  device=device)
        merged_feat.index_add_(0, src_to_dst, point.feat * w)
        counts = torch.zeros(M, 1, dtype=point.feat.dtype, device=device)
        counts.index_add_(0, src_to_dst, w)
        merged_feat = merged_feat / counts.clamp(min=1e-6)

    _rb = tome_fine_region("merge/pm_rebuild")
    _rb.__enter__()
    dst_gidx = order[keep_c]                             # (M,) orig idx
    sub = build_subset_point_fast(point, dst_gidx)
    sub.feat = merged_feat
    sub.sparse_conv_feat = sub.sparse_conv_feat.replace_feature(
        merged_feat)
    if "tome_size" in point.keys():
        sub["tome_size"] = counts.squeeze(-1).float()

    # Carry the unpooling trace this level arrived with: GridUnpooling
    # reads pooling_parent/pooling_inverse off the incoming point to
    # rebuild it. pooling_inverse maps parent rows -> this level's rows,
    # so remap it through src_to_dst into merged-row space.
    if "pooling_parent" in point.keys():
        sub["pooling_parent"] = point.pooling_parent
        sub["pooling_inverse"] = src_to_dst[point.pooling_inverse]
    for k in ("origin_coord", "color", "condition", "context", "name",
              "split"):
        if k in point.keys():
            v = point[k]
            sub[k] = (v[dst_gidx].clone()
                      if torch.is_tensor(v) and v.shape[0] == N else v)

    _rb.__exit__(None, None, None)
    state = {"parent": point, "src_to_dst": src_to_dst}
    return sub, state


# Keys not carried from the pre-merge point into the restored one: the
# attention pad caches depend on patch_size (dec blocks recompute them)
# and feat/sparse are rebuilt explicitly. pooling_* keys are kept, since
# the saved parent's trace is what the next GridUnpooling needs.
_RESTORE_EXCLUDE = {
    "feat", "sparse_conv_feat", "pad", "unpad", "cu_seqlens_key",
    "min_bincount", "unpooling_parent",
}


def unmerge_point(point_merged, state):
    """Restore a merge produced by `merge_point`.

    point_merged: the merged Point as it arrives at the restore site
    (feat may have been updated in place by blocks/unpooling since).
    Every original token gets its cluster's dst feature back. All other
    fields (coord/batch/serialized_order/...) are taken from the saved
    pre-merge point, so the restored cloud is identical to the pre-merge
    one except for features.
    """
    parent = state["parent"]
    with tome_fine_region("unmerge/pm_gather"):
        full_feat = point_merged.feat[state["src_to_dst"]]
    with tome_fine_region("unmerge/pm_rebuild"):
        newp = Point({k: v for k, v in parent.items()
                      if k not in _RESTORE_EXCLUDE})
        newp.feat = full_feat
        newp.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(
            full_feat)
    return newp
