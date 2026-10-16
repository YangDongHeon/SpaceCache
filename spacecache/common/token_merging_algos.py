"""Token-merging primitives on (B, H, K, C) patch tensors.

`patch_based_matching` and `globally_informed_energy` are adapted from
GitMerge3D (https://github.com/anhtuanhsgs/GitMerge3D), Copyright (c) 2025
GitMerge3d, MIT License.
Also holds the TOME_PROFILE region timers and the merge-reduction and
bf16 switches set by `spacecache/experiments/run.py`.
"""
import os
import torch
from contextlib import nullcontext
from typing import Callable, Tuple
import torch.nn.functional as F

# FLOPs accounting for the matching matmuls (no-op unless a CostMeter is
# active).
from spacecache.common.cost_meter import note_matmul

try:
    from torch.profiler import record_function as _record_function
except Exception:  # pragma: no cover - profiler always available on torch>=1.8
    _record_function = None


def do_nothing(x, mode=None):
    return x


# ---------------------------------------------------------------------------
# Per-region profiling of the merge path.
#
# With TOME_PROFILE=1, `tome_prof_region(name)` records a CUDA-event pair
# around the region and opens a torch.profiler `record_function`; otherwise
# it returns a shared nullcontext. Events are only enqueued (no mid-stream
# sync); `tome_prof_flush()` synchronizes once per forward and folds them
# into the per-region totals returned by `tome_prof_results()`.
# ---------------------------------------------------------------------------
TOME_PROFILE = os.environ.get("TOME_PROFILE", "0") == "1"

_TOME_EVENTS = []   # pending [(name, start_event, end_event)] for the current window
_TOME_TIMES = {}    # name -> accumulated GPU milliseconds
_TOME_COUNTS = {}   # name -> number of recorded calls
_NULLCTX = nullcontext()  # reused singleton so the disabled path allocates nothing


class _RegionTimer:
    __slots__ = ("name", "start", "rf")

    def __init__(self, name):
        self.name = name
        self.start = None
        self.rf = None

    def __enter__(self):
        if _record_function is not None:
            self.rf = _record_function(self.name)
            self.rf.__enter__()
        if torch.cuda.is_available():
            self.start = torch.cuda.Event(enable_timing=True)
            self.start.record()
        return self

    def __exit__(self, *exc):
        if self.start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            _TOME_EVENTS.append((self.name, self.start, end))
        if self.rf is not None:
            self.rf.__exit__(*exc)
        return False


def tome_prof_region(name):
    if TOME_PROFILE:
        return _RegionTimer(name)
    return _NULLCTX


# Fine sub-step regions (match/*, merge/*, unmerge/*), enabled by
# TOME_PROFILE_FINE=1 on top of TOME_PROFILE (run.py --fine_regions). They
# nest inside the coarse regions, so FrameProfiler reports them separately
# (attn_fine_ms) and never adds them to the coarse totals.
TOME_PROFILE_FINE = TOME_PROFILE and os.environ.get(
    "TOME_PROFILE_FINE", "0") == "1"


def tome_fine_region(name):
    if TOME_PROFILE_FINE:
        return _RegionTimer(name)
    return _NULLCTX


def tome_prof_flush():
    """Synchronize once and fold pending CUDA events into per-region totals."""
    if not _TOME_EVENTS:
        return
    torch.cuda.synchronize()
    for name, start, end in _TOME_EVENTS:
        _TOME_TIMES[name] = _TOME_TIMES.get(name, 0.0) + start.elapsed_time(end)
        _TOME_COUNTS[name] = _TOME_COUNTS.get(name, 0) + 1
    _TOME_EVENTS.clear()


def tome_prof_reset():
    """Drop pending events and clear accumulated totals."""
    _TOME_EVENTS.clear()
    _TOME_TIMES.clear()
    _TOME_COUNTS.clear()


def tome_prof_results():
    """Return (times_ms, counts) dicts keyed by region name."""
    return dict(_TOME_TIMES), dict(_TOME_COUNTS)


# ---------------------------------------------------------------------------
# bf16 token mixing, independent of autocast. When enabled, the ToMe
# attention forward casts q/k/v to bf16 right after RoPE, so matching,
# merge, the attention matmuls and unmerge run in bf16 while the
# projections, MLP and norms stay in fp32.
# ---------------------------------------------------------------------------
_COMPUTE_BF16 = {"on": False}


def set_tome_compute_bf16(enable):
    _COMPUTE_BF16["on"] = bool(enable)


def tome_compute_bf16():
    return _COMPUTE_BF16["on"]


# ---------------------------------------------------------------------------
# Merge reduction backend of `patch_based_matching` (run.py --merge_reduce).
#
#   scatter            stock `scatter_reduce`: one global atomic per src row.
#   segment_csr        src rows grouped by destination (one argsort per
#                      matching, shared by the size/v/q/k merges of a
#                      block), then `torch_scatter.segment_csr` over each
#                      destination's contiguous run; no atomics.
#   cumsum             the same grouping, then an fp32 prefix sum
#                      differenced at the segment bounds; no atomics and no
#                      torch_scatter. fp32 because a bf16 prefix sum over
#                      hundreds of rows cancels catastrophically.
#   segment_csr_fused  an index built once per block lays out each
#                      destination's segment as [dst row, its src rows...];
#                      per call one gather fills it and `segment_csr`
#                      returns the merged dst directly (no split, add or
#                      divide passes).
#
# All four return the include_self mean (or sum) of `scatter_reduce`; only
# the accumulation order differs, so outputs agree up to bf16 rounding.
# ---------------------------------------------------------------------------
MERGE_REDUCERS = ("scatter", "segment_csr", "segment_csr_fused", "cumsum")
_MERGE_REDUCE = {"impl": "scatter"}


def set_merge_reduce(impl):
    if impl not in MERGE_REDUCERS:
        raise ValueError(f"merge reduce {impl!r} not in {MERGE_REDUCERS}")
    if impl.startswith("segment_csr"):
        import torch_scatter  # noqa: F401  (fail at setup, not mid-eval)
    _MERGE_REDUCE["impl"] = impl


def merge_reduce():
    return _MERGE_REDUCE["impl"]


class _DstGroups:
    """src rows grouped by destination. Depends only on dst_idx, so it is
    built once per matching and reused by every merge() of the block.

    perm    (BH, r)          src row order, destination-major
    indptr  (BH, num_dst+1)  segment bounds into the permuted rows, found
                             by searchsorted on the sorted keys (no atomic
                             histogram)
    counts  (BH, num_dst)    src rows per destination (size - 1)
    """
    __slots__ = ("perm", "indptr", "counts")

    def __init__(self, dst_idx, num_dst):
        n1, n2, r, _ = dst_idx.shape
        key = dst_idx.reshape(n1 * n2, r)
        self.perm = key.argsort(dim=-1, stable=True)
        skey = key.gather(1, self.perm)
        bounds = torch.arange(num_dst + 1, device=key.device,
                              dtype=key.dtype).expand(n1 * n2, -1)
        self.indptr = torch.searchsorted(skey, bounds.contiguous())
        self.counts = self.indptr[:, 1:] - self.indptr[:, :-1]


class _FusedMergeIndex:
    """Index plan of segment_csr_fused. Depends only on the matching, so it
    is built once per block and shared by the size/v/q/k merges.

    unm_pos  (n1, n2, n_unm, 1)  x positions of the unmerged src tokens, in
                                 unm_idx order (the stock output order)
    buf_idx  (n1, n2, n_buf, 1)  x positions laid out segment by segment:
                                 [dst j, its src rows...], n_buf = r + num_dst
    indptr   (n1*n2, num_dst+1)  segment bounds in buf. Segment j is one row
                                 longer than its src count (dst's own row).
    """
    __slots__ = ("unm_pos", "buf_idx", "indptr", "n_unm", "n_buf")

    def __init__(self, a_idx, b_idx, unm_idx, src_idx, dst_idx, num_dst):
        n1, n2, r, _ = dst_idx.shape
        bh, dev = n1 * n2, dst_idx.device
        self.unm_pos = a_idx.gather(2, unm_idx)
        src_pos = a_idx.gather(2, src_idx).reshape(bh, r)
        dst_pos = b_idx.reshape(bh, num_dst)
        key = dst_idx.reshape(bh, r)
        perm = key.argsort(dim=-1, stable=True)
        skey = key.gather(1, perm)            # sorted row i -> its segment
        bounds = torch.arange(num_dst + 1, device=dev,
                              dtype=key.dtype).expand(bh, -1)
        # src-only bounds, then + j: every segment gains its dst row
        self.indptr = torch.searchsorted(skey, bounds.contiguous()) + bounds
        # sorted src row i of segment s = skey[i] lands at
        # indptr[s] + 1 + (i - src_start[s]) = i + s + 1
        slot_src = torch.arange(r, device=dev).expand(bh, -1) + skey + 1
        buf = torch.empty(bh, r + num_dst, dtype=torch.long, device=dev)
        buf.scatter_(1, self.indptr[:, :-1], dst_pos)   # every slot written
        buf.scatter_(1, slot_src, src_pos.gather(1, perm))  # exactly once
        self.buf_idx = buf.reshape(n1, n2, r + num_dst, 1)
        self.n_unm = int(unm_idx.shape[2])
        self.n_buf = r + num_dst


def _grouped_reduce(dst, src, g, mode, impl):
    """scatter_reduce(-2, dst_idx, src, mode, include_self=True) without
    atomics. dst (n1, n2, num_dst, c), src (n1, n2, r, c) in src_idx order."""
    n1, n2, nd, c = dst.shape
    bh, r = n1 * n2, src.shape[2]
    rows = src.reshape(bh, r, c).gather(
        1, g.perm[..., None].expand(bh, r, c))
    if impl == "segment_csr":
        import torch_scatter
        seg = torch_scatter.segment_csr(rows, g.indptr, reduce="sum")
        out = dst.reshape(bh, nd, c) + seg
        if mode == "mean":
            out = out / (g.counts + 1).unsqueeze(-1).to(out.dtype)
    else:
        cs = F.pad(torch.cumsum(rows, dim=1, dtype=torch.float32),
                   (0, 0, 1, 0))                     # (bh, r+1, c), row 0 = 0
        hi = cs.gather(1, g.indptr[:, 1:, None].expand(bh, nd, c))
        lo = cs.gather(1, g.indptr[:, :-1, None].expand(bh, nd, c))
        out = dst.reshape(bh, nd, c).float() + (hi - lo)
        if mode == "mean":
            out = out / (g.counts + 1).unsqueeze(-1)
        out = out.to(dst.dtype)
    return out.reshape(n1, n2, nd, c)


def patch_based_matching(
    metric: torch.Tensor,
    r: int,
    stride: int,
    class_token: bool = False,
    distill_token: bool = False,
    no_rand: bool = False,
    mean_head: bool = False,
) -> Tuple[Callable, Callable]:
    """GitMerge3D patch-based matching.

    metric: (B, H, T, C). Each bin of `stride` consecutive tokens holds one
    dst token (random, or a fixed per-head offset with `no_rand`); all other
    tokens are src, and the `r` src tokens most similar to their best dst
    are merged into it. `mean_head` shares one matching across heads.
    Class/distillation tokens are not supported.
    """
    protected = 0
    if class_token:
        raise
    if distill_token:
        raise

    t = metric.shape[2]

    gather = torch.gather

    if r <= 0:
        raise

    with torch.no_grad():
        n_tokens = metric.shape[2]
        n_segments = metric.shape[0]
        n_heads = metric.shape[1]

        n_bins = n_tokens // stride
        if n_bins == 0:
            # patch shorter than one stride: no dst bin, keep every token
            return do_nothing, do_nothing
        # one dst per bin of `stride` tokens, the rest are src
        if no_rand:
            rand_idx = torch.zeros(n_segments, n_heads, n_bins, 1, device=metric.device, dtype=torch.int64)
            for h in range(0, n_heads):
                rand_idx[:,h] = (max(stride // n_heads, 1) * h) % stride
        else:
            rand_idx = torch.randint(stride, size=(n_segments, n_heads, n_bins, 1), device=metric.device)

        # T may not divide by stride: bin the leading n_bins * stride tokens,
        # the remainder stays src
        idx_buffer_view = torch.zeros(n_segments, n_heads, n_bins, stride, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=3, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype, device=metric.device))
        idx_buffer_view = idx_buffer_view.reshape(n_segments, n_heads, n_bins * stride)

        if (n_bins * stride) < n_tokens:
            idx_buffer = torch.zeros(n_segments, n_heads, n_tokens, device=metric.device, dtype=torch.int64)
            idx_buffer[..., :(n_bins * stride)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        rand_idx = idx_buffer.reshape(n_segments, n_heads, -1, 1).argsort(dim=2)

        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = n_bins
        a_idx = rand_idx[:,:, num_dst:, :] # src
        b_idx = rand_idx[:,:, :num_dst, :] # dst

        def split(x):
            B, H, T, C = x.shape
            src = gather(x, dim=2, index=a_idx.expand(B, H, T - num_dst, C))
            dst = gather(x, dim=2, index=b_idx.expand(B, H, num_dst, C))
            return src, dst

        if mean_head:
            metric = metric.mean(dim=1, keepdim=True).expand(-1, n_heads, -1, -1)
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)

        note_matmul(a, b)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[2], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1) # Score max value / Score max index
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens

        # dst slot (index into b_idx) of each merged src token
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    groups = [None]      # _DstGroups, built by the first atomic-free merge
    fused = [None]       # _FusedMergeIndex, built by the first fused merge

    def merge_fused(x, mode):
        import torch_scatter
        n1, n2, _, c = x.shape
        if fused[0] is None:
            with tome_fine_region("merge/group"):
                fused[0] = _FusedMergeIndex(a_idx, b_idx, unm_idx, src_idx,
                                            dst_idx, num_dst)
        f = fused[0]
        with tome_fine_region("merge/gather_unm"):
            unm = x.gather(2, f.unm_pos.expand(n1, n2, f.n_unm, c))
        with tome_fine_region("merge/gather_seg"):
            buf = x.gather(2, f.buf_idx.expand(n1, n2, f.n_buf, c))
        with tome_fine_region("merge/segment_csr"):
            dst = torch_scatter.segment_csr(
                buf.view(n1 * n2, f.n_buf, c), f.indptr,
                reduce=mode).view(n1, n2, num_dst, c)
        with tome_fine_region("merge/cat"):
            return torch.cat([unm, dst], dim=-2)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if (_MERGE_REDUCE["impl"] == "segment_csr_fused"
                and mode in ("mean", "sum")):
            return merge_fused(x, mode)
        with tome_fine_region("merge/split"):
            src, dst = split(x)
        n1, n2, t1, c = src.shape

        with tome_fine_region("merge/gather_unm"):
            unm = src.gather(dim=-2, index=unm_idx.expand(n1, n2, t1 - r, c))
        with tome_fine_region("merge/gather_src"):
            src = src.gather(dim=-2, index=src_idx.expand(n1, n2, r, c))
        impl = _MERGE_REDUCE["impl"]
        if mode != "prune" and (impl == "scatter"
                                or mode not in ("mean", "sum")):
            with tome_fine_region("merge/scatter_reduce"):
                dst = dst.scatter_reduce(-2, dst_idx.expand(n1, n2, r, c), src, reduce=mode)
        elif mode != "prune":
            if groups[0] is None:
                with tome_fine_region("merge/group"):
                    groups[0] = _DstGroups(dst_idx, num_dst)
            with tome_fine_region(f"merge/{impl}"):
                dst = _grouped_reduce(dst, src, groups[0], mode, impl)

        if distill_token:
            raise
        else:
            with tome_fine_region("merge/cat"):
                return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[-2]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n1, n2, _, c = unm.shape

        with tome_fine_region("unmerge/gather_src"):
            src = dst.gather(dim=-2, index=dst_idx.expand(n1, n2,  r, c))
        with tome_fine_region("unmerge/alloc"):
            out = torch.zeros(n1, n2, n_tokens, c, device=x.device, dtype=x.dtype)
        with tome_fine_region("unmerge/copy_dst"):
            out.scatter_(dim=-2, index=b_idx.expand(n1, n2, num_dst, c), src=dst)
        with tome_fine_region("unmerge/scatter_unm"):
            out.scatter_(dim=-2, index=gather(a_idx.expand(n1, n2, a_idx.shape[2], 1), dim=2, index=unm_idx).expand(n1, n2, unm_len, c), src=unm)
        with tome_fine_region("unmerge/scatter_src"):
            out.scatter_(dim=-2, index=gather(a_idx.expand(n1, n2, a_idx.shape[2], 1), dim=2, index=src_idx).expand(n1, n2, r, c), src=src)
        return out

    return merge, unmerge


def globally_informed_energy(metric: torch.Tensor) -> torch.Tensor:
    """GitMerge3D globally informed merging energy (Alg. 1, steps 1-2).

    Each token's energy is -mean_j cos(x_i, P_j) over all patch centroids
    P_j (the global bipartite graph); a patch's energy is the mean over its
    tokens. High energy means globally distinctive (merge less).

    metric: (B, H, K, C) -> (B,). Materializes a (B*K, B) similarity matrix.
    """
    B = metric.shape[0]
    x = metric.mean(dim=1)                                # [B, K, C] collapse heads
    centroids = x.mean(dim=1)                             # [B, C]   patch centroid P_j
    xn = F.normalize(x.reshape(-1, x.shape[-1]), dim=-1)  # [B*K, C] all points
    cn = F.normalize(centroids, dim=-1)                   # [B, C]   all centroids
    note_matmul(xn, cn)
    cos = xn @ cn.transpose(0, 1)                         # [B*K, B] global bipartite graph
    e_point = -cos.mean(dim=-1)                           # Step 2: -mean over all centroids
    return e_point.reshape(B, -1).mean(dim=-1)            # [B] patch energy E(P_j)
