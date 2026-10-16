"""Fused Triton kernels for the SpaceCache gates (`--triton_cache`).

  lookup_coords       to_global + voxel_hash + sorted-table lookup
  lookup_keys         sorted-table lookup of precomputed keys
  restore             table rows for hits, subset rows for misses
  build_subset_point  subset Point of the misses, without re-serializing
  stage_classify, mark_bad, overwrite_resident, surv_keep, scatter_merge
                      steps of `SpaceCache.commit`

The gates involve data-dependent sizes (hit and miss counts) that a static
CUDA graph cannot cover, so their steps are fused to cut kernel launches.
Results follow the torch path: the world transform and hash division stay
in float64 (see `spacecache.common.helpers.voxel_hash`), floor-mod follows
`torch.remainder` (Triton `%` on int64 truncates), and table membership is
a lower_bound with a fixed iteration count. Triton may contract mul+add
into FMA, so a point within ~1 ulp of a cell boundary can hash to the
neighbouring cell. Without Triton or CUDA, `is_available()` is False and
the callers use torch.
"""

from __future__ import annotations

import math

import torch

from spacecache.common import region_key as _rk

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                                # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False


def is_available():
    return _HAS_TRITON and torch.cuda.is_available()


# float64 scalars (voxel_size, coord_scale) are passed as 1-element device
# tensors: Triton passes Python floats as fp32 kernel arguments.
_SCALAR_F64 = {}


def _scalar_f64(val, device):
    key = (device, float(val))
    t = _SCALAR_F64.get(key)
    if t is None:
        # torch.full fills on-device; torch.tensor(...) does a pageable
        # H2D memcpy, illegal under CUDA-graph capture.
        t = torch.full((1,), float(val), dtype=torch.float64,
                       device=device)
        _SCALAR_F64[key] = t
    return t


def _lb_iters(nw):
    """Fixed lower_bound iteration count for `nw` sorted entries."""
    return max(1, int(math.ceil(math.log2(nw + 1))))


if _HAS_TRITON:

    @triton.jit
    def _key_phase(key, PACKED: tl.constexpr):
        """Expiry phase of a key: the prime dot product of its cell
        (region_key.phase)."""
        ph = key
        if PACKED:
            vx = (key >> 39) - 8388608
            vy = ((key >> 15) & 16777215) - 8388608
            vz = (key & 32767) - 16384
            ph = vx * 73856093 + vy * 19349663 + vz * 83492791
        return ph

    # -- mask gate: coord -> world key -> table lookup ---------------------

    @triton.jit
    def _lookup_coords_kernel(
        coord_ptr, pose_ptr, scale_ptr, vs_ptr,
        tab_ptr, keys_ptr, slot_ptr, hit_ptr,
        n, nw, iters, frame_t, life, ps,
        WITH_DUE: tl.constexpr, PACKED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # pose: (4,4) or (3,4) sensor-to-world matrix, row-major with row
        # stride `ps`: R[i,j] = pose[i*ps + j], t[i] = pose[i*ps + 3]
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n

        x = tl.load(coord_ptr + offs * 3 + 0, mask=m, other=0.0)
        y = tl.load(coord_ptr + offs * 3 + 1, mask=m, other=0.0)
        z = tl.load(coord_ptr + offs * 3 + 2, mask=m, other=0.0)
        s = tl.load(scale_ptr)                    # fp64 scalar
        vs = tl.load(vs_ptr)                      # fp64 scalar
        t0 = tl.load(pose_ptr + 3).to(tl.float64) * s
        t1 = tl.load(pose_ptr + ps + 3).to(tl.float64) * s
        t2 = tl.load(pose_ptr + 2 * ps + 3).to(tl.float64) * s
        gx = (x.to(tl.float64) * tl.load(pose_ptr + 0).to(tl.float64)
              + y.to(tl.float64) * tl.load(pose_ptr + 1).to(tl.float64)
              + z.to(tl.float64) * tl.load(pose_ptr + 2).to(tl.float64)) + t0
        gy = (x.to(tl.float64) * tl.load(pose_ptr + ps).to(tl.float64)
              + y.to(tl.float64) * tl.load(pose_ptr + ps + 1).to(tl.float64)
              + z.to(tl.float64) * tl.load(pose_ptr + ps + 2).to(tl.float64)) + t1
        gz = (x.to(tl.float64) * tl.load(pose_ptr + 2 * ps).to(tl.float64)
              + y.to(tl.float64) * tl.load(pose_ptr + 2 * ps + 1).to(tl.float64)
              + z.to(tl.float64) * tl.load(pose_ptr + 2 * ps + 2).to(tl.float64)) + t2

        # voxel_hash: fp64 floor-div, then the region key (region_key.py)
        vx = tl.floor(gx / vs).to(tl.int64)
        vy = tl.floor(gy / vs).to(tl.int64)
        vz = tl.floor(gz / vs).to(tl.int64)
        if PACKED:
            key = (((vx + 8388608) << 39) | ((vy + 8388608) << 15)
                   | (vz + 16384))
        else:
            key = vx * 73856093 + vy * 19349663 + vz * 83492791
        tl.store(keys_ptr + offs, key, mask=m)

        # lower_bound over the sorted table, fixed iteration count
        lo = tl.zeros([BLOCK], dtype=tl.int64)
        hi = lo + nw
        for _ in range(iters):
            mid = (lo + hi) // 2
            ok = mid < nw
            mv = tl.load(tab_ptr + mid, mask=m & ok, other=0)
            gr = (mv < key) & ok
            lo = tl.where(gr, mid + 1, lo)
            hi = tl.where(gr, hi, mid)
        slot = tl.minimum(lo, tl.maximum(nw - 1, 0))
        tv = tl.load(tab_ptr + slot, mask=m & (nw > 0), other=0)
        hit = (nw > 0) & (tv == key)
        if WITH_DUE:
            # torch.remainder is floor-mod; Triton % truncates, so fix the
            # sign for negative (wrapped) hashes
            r = (frame_t - _key_phase(key, PACKED)) % life
            r = tl.where(r < 0, r + life, r)
            hit = hit & (r != 0)
        tl.store(slot_ptr + offs, slot, mask=m)
        tl.store(hit_ptr + offs, hit.to(tl.int8), mask=m)

    # -- key -> table lookup (SpaceCache.lookup) ---------------------------

    @triton.jit
    def _lookup_keys_kernel(
        tab_ptr, keys_ptr, slot_ptr, hit_ptr,
        n, nw, iters, frame_t, life,
        WITH_DUE: tl.constexpr, PACKED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        key = tl.load(keys_ptr + offs, mask=m, other=0)
        lo = tl.zeros([BLOCK], dtype=tl.int64)
        hi = lo + nw
        for _ in range(iters):
            mid = (lo + hi) // 2
            ok = mid < nw
            mv = tl.load(tab_ptr + mid, mask=m & ok, other=0)
            gr = (mv < key) & ok
            lo = tl.where(gr, mid + 1, lo)
            hi = tl.where(gr, hi, mid)
        slot = tl.minimum(lo, tl.maximum(nw - 1, 0))
        tv = tl.load(tab_ptr + slot, mask=m & (nw > 0), other=0)
        hit = (nw > 0) & (tv == key)
        if WITH_DUE:
            r = (frame_t - _key_phase(key, PACKED)) % life
            r = tl.where(r < 0, r + life, r)
            hit = hit & (r != 0)
        tl.store(slot_ptr + offs, slot, mask=m)
        tl.store(hit_ptr + offs, hit.to(tl.int8), mask=m)

    # -- reuse gate: hit <- table row, miss <- subset row ------------------

    @triton.jit
    def _restore_kernel(
        full_ptr, cache_ptr, sub_ptr, hit_ptr, slot_ptr, miss_ptr,
        n, m, iters, C,
        IDENTITY_MISS: tl.constexpr,
        BN: tl.constexpr, BC: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BN + tl.arange(0, BN)
        rm = rows < n
        hit = tl.load(hit_ptr + rows, mask=rm, other=0)
        slot = tl.load(slot_ptr + rows, mask=rm, other=0)
        # miss rank = lower_bound(miss_idx, row): exact, since miss_idx is
        # sorted and contains every miss row. Without a subset the miss
        # side is the identity (the computed value is already in `full`).
        if IDENTITY_MISS:
            rank = rows.to(tl.int64)
        else:
            lo = tl.zeros([BN], dtype=tl.int64)
            hi = lo + m
            for _ in range(iters):
                mid = (lo + hi) // 2
                ok = mid < m
                mv = tl.load(miss_ptr + mid, mask=rm & ok, other=0)
                gr = (mv < rows) & ok
                lo = tl.where(gr, mid + 1, lo)
                hi = tl.where(gr, hi, mid)
            rank = lo
        cols = tl.arange(0, BC)
        for c0 in range(0, C, BC):
            cm = (c0 + cols) < C
            vh = tl.load(
                cache_ptr + slot[:, None] * C + (c0 + cols)[None, :],
                mask=rm[:, None] & (hit != 0)[:, None] & cm[None, :],
                other=0.0)
            vm = tl.load(
                sub_ptr + rank[:, None] * C + (c0 + cols)[None, :],
                mask=rm[:, None] & (hit == 0)[:, None] & cm[None, :],
                other=0.0)
            v = tl.where((hit != 0)[:, None], vh, vm)
            tl.store(
                full_ptr + rows[:, None] * C + (c0 + cols)[None, :],
                v.to(full_ptr.dtype.element_ty),
                mask=rm[:, None] & cm[None, :])

    # -- subset build: parent map + serialized-order remap ----------------

    @triton.jit
    def _p2s_kernel(sub_idx_ptr, out_ptr, n_full, m, iters,
                    BLOCK: tl.constexpr):
        """parent_to_sub[i] = position of i in sorted `sub_idx`, else -1."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        rm = offs < n_full
        lo = tl.zeros([BLOCK], dtype=tl.int64)
        hi = lo + m
        for _ in range(iters):
            mid = (lo + hi) // 2
            ok = mid < m
            mv = tl.load(sub_idx_ptr + mid, mask=rm & ok, other=0)
            gr = (mv < offs) & ok
            lo = tl.where(gr, mid + 1, lo)
            hi = tl.where(gr, hi, mid)
        found = (lo < m) & (
            tl.load(sub_idx_ptr + lo, mask=rm & (lo < m), other=-1)
            == offs)
        tl.store(out_ptr + offs,
                 tl.where(found, lo, -1).to(out_ptr.dtype.element_ty),
                 mask=rm)

    @triton.jit
    def _gather_fields_kernel(idx_ptr,
                              feat_ptr, coord_ptr, grid_ptr, batch_ptr,
                              of_ptr, oc_ptr, og_ptr, ob_ptr,
                              C, HAS_GRID: tl.constexpr,
                              HAS_BATCH: tl.constexpr, BC: tl.constexpr):
        """Gather subset rows of every Point field in one launch:
        of[j, :] = feat[idx[j], :] over column tiles; coord / grid_coord /
        batch rows are copied by the cb == 0 programs."""
        j = tl.program_id(0)
        cb = tl.program_id(1)
        src = tl.load(idx_ptr + j).to(tl.int64)
        cols = cb * BC + tl.arange(0, BC)
        m = cols < C
        tl.store(of_ptr + j * C + cols,
                 tl.load(feat_ptr + src * C + cols, mask=m), mask=m)
        if cb == 0:
            c3 = tl.arange(0, 4)
            m3 = c3 < 3
            tl.store(oc_ptr + j * 3 + c3,
                     tl.load(coord_ptr + src * 3 + c3, mask=m3), mask=m3)
            if HAS_GRID:
                tl.store(og_ptr + j * 3 + c3,
                         tl.load(grid_ptr + src * 3 + c3, mask=m3), mask=m3)
            if HAS_BATCH:
                tl.store(ob_ptr + j, tl.load(batch_ptr + src))

    @triton.jit
    def _emit_insub_kernel(po_ptr, p2s_ptr, out_ptr, total,
                           BLOCK: tl.constexpr):
        """in_sub[o,p] = parent_to_sub[parent_order[o,p]] >= 0, flat."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        rm = offs < total
        po = tl.load(po_ptr + offs, mask=rm, other=0)
        j = tl.load(p2s_ptr + po, mask=rm, other=-1)
        tl.store(out_ptr + offs, (j >= 0).to(tl.int8), mask=rm)

    @triton.jit
    def _remap_kernel(po_ptr, p2s_ptr, pos_ptr, order_ptr, inv_ptr,
                      total, n_full, n_sub, BLOCK: tl.constexpr):
        """For each serialized slot (o,p): j = p2s[order[o,p]]; when the
        token survives, sub_order[o, pos-1] = j and sub_inverse[o, j] =
        pos-1 where pos is the 1-based cumsum rank of the in_sub flag."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        rm = offs < total
        po = tl.load(po_ptr + offs, mask=rm, other=0)
        j = tl.load(p2s_ptr + po, mask=rm, other=-1)
        ins = j >= 0
        pos = tl.load(pos_ptr + offs, mask=rm, other=0) - 1
        o = offs // n_full
        tl.store(order_ptr + o * n_sub + pos, j, mask=rm & ins)
        tl.store(inv_ptr + o * n_sub + j, pos, mask=rm & ins)

    # -- commit internals --------------------------------------------------

    @triton.jit
    def _stage_classify_kernel(
        keys_ptr, st_idx_ptr, life_ptr, inv_ptr,
        k_ptr, due_ptr, keep_ptr,
        m, base, t1,
        HAS_INV: tl.constexpr, STRICT: tl.constexpr, PACKED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Per staged cell: key, due (staggered: base + floormod(phase(key) - base,
        life); fixed: base + life - 1), keep = (due > t+1) & ~inv[st_idx]."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        rm = offs < m
        si = tl.load(st_idx_ptr + offs, mask=rm, other=0)
        k = tl.load(keys_ptr + si, mask=rm, other=0)
        lf = tl.load(life_ptr + si, mask=rm, other=1)
        if STRICT:
            due = base + lf - 1
        else:
            r = (_key_phase(k, PACKED) - base) % lf
            r = tl.where(r < 0, r + lf, r)          # floor-mod, not srem
            due = base + r
        keep = due > t1
        if HAS_INV:
            iv = tl.load(inv_ptr + si, mask=rm, other=0)
            keep = keep & (iv == 0)
        tl.store(k_ptr + offs, k, mask=rm)
        tl.store(due_ptr + offs, due, mask=rm)
        tl.store(keep_ptr + offs, keep.to(tl.int8), mask=rm)

    @triton.jit
    def _mark_bad_kernel(
        res_ptr, keys_ptr, inv_ptr, bad_ptr,
        n, nw, iters, BLOCK: tl.constexpr,
    ):
        """bad[slot] = 1 for resident keys hit by this frame's invalidate
        mask (duplicate writes store the same value)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        rm = offs < n
        key = tl.load(keys_ptr + offs, mask=rm, other=0)
        iv = tl.load(inv_ptr + offs, mask=rm, other=0)
        lo = tl.zeros([BLOCK], dtype=tl.int64)
        hi = lo + nw
        for _ in range(iters):
            mid = (lo + hi) // 2
            ok = mid < nw
            mv = tl.load(res_ptr + mid, mask=rm & ok, other=0)
            gr = (mv < key) & ok
            lo = tl.where(gr, mid + 1, lo)
            hi = tl.where(gr, hi, mid)
        slot = tl.minimum(lo, tl.maximum(nw - 1, 0))
        ex = (nw > 0) & (tl.load(res_ptr + slot,
                                 mask=rm & (nw > 0), other=0) == key)
        one = tl.zeros([BLOCK], dtype=tl.int8) + 1
        tl.store(bad_ptr + slot, one, mask=rm & ex & (iv != 0))

    @triton.jit
    def _overwrite_kernel(
        res_key_ptr, res_due_ptr, res_feat_ptr,
        new_key_ptr, new_due_ptr, new_feat_ptr, fresh_ptr,
        n_ins, nw, iters, C,
        BN: tl.constexpr, BC: tl.constexpr,
    ):
        """Overwrite feat/due in place for staged keys that are already
        resident (so the expiry check sees the refreshed `due`) and write
        `fresh` = not resident, for BN staged rows per program."""
        pid = tl.program_id(0)
        rows = pid * BN + tl.arange(0, BN)
        rm = rows < n_ins
        k = tl.load(new_key_ptr + rows, mask=rm, other=0)
        lo = tl.zeros([BN], dtype=tl.int64)
        hi = lo + nw
        for _ in range(iters):
            mid = (lo + hi) // 2
            ok = mid < nw
            mv = tl.load(res_key_ptr + mid, mask=rm & ok, other=0)
            gr = (mv < k) & ok
            lo = tl.where(gr, mid + 1, lo)
            hi = tl.where(gr, hi, mid)
        slot = tl.minimum(lo, tl.maximum(nw - 1, 0))
        ex = rm & (nw > 0) & (
            tl.load(res_key_ptr + slot, mask=rm & (nw > 0), other=0) == k)
        tl.store(fresh_ptr + rows, (ex == 0).to(tl.int8), mask=rm)
        tl.store(res_due_ptr + slot,
                 tl.load(new_due_ptr + rows, mask=rm, other=0),
                 mask=ex)
        cols = tl.arange(0, BC)
        for c0 in range(0, C, BC):
            cm = (c0 + cols) < C
            w = rm[:, None] & ex[:, None] & cm[None, :]
            v = tl.load(new_feat_ptr + rows[:, None] * C
                        + (c0 + cols)[None, :], mask=w, other=0.0)
            tl.store(res_feat_ptr + slot[:, None] * C
                     + (c0 + cols)[None, :],
                     v.to(res_feat_ptr.dtype.element_ty), mask=w)

    @triton.jit
    def _surv_keep_kernel(due_ptr, bad_ptr, keep_ptr, nw, t1,
                          HAS_BAD: tl.constexpr, BLOCK: tl.constexpr):
        """keep = (due > t+1) & ~bad."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        rm = offs < nw
        keep = tl.load(due_ptr + offs, mask=rm, other=0) > t1
        if HAS_BAD:
            keep = keep & (tl.load(bad_ptr + offs, mask=rm, other=0) == 0)
        tl.store(keep_ptr + offs, keep.to(tl.int8), mask=rm)

    @triton.jit
    def _scatter_merge_kernel(
        a_key_ptr, a_due_ptr, a_feat_ptr, pa_ptr,
        b_key_ptr, b_due_ptr, b_feat_ptr, pb_ptr,
        o_key_ptr, o_due_ptr, o_feat_ptr,
        n_a, C,
        BC: tl.constexpr,
    ):
        """Write the merged sorted table: each source row (A rows, then B
        rows) scatters key/due/feat to its precomputed output position."""
        pid = tl.program_id(0)
        is_a = pid < n_a
        if is_a:
            src = pid
            pos = tl.load(pa_ptr + src)
            k = tl.load(a_key_ptr + src)
            d = tl.load(a_due_ptr + src)
        else:
            src = pid - n_a
            pos = tl.load(pb_ptr + src)
            k = tl.load(b_key_ptr + src)
            d = tl.load(b_due_ptr + src)
        tl.store(o_key_ptr + pos, k)
        tl.store(o_due_ptr + pos, d)
        cols = tl.arange(0, BC)
        for c0 in range(0, C, BC):
            cm = (c0 + cols) < C
            if is_a:
                v = tl.load(a_feat_ptr + src * C + c0 + cols, mask=cm)
            else:
                v = tl.load(b_feat_ptr + src * C + c0 + cols, mask=cm)
            tl.store(o_feat_ptr + pos * C + c0 + cols,
                     v.to(o_feat_ptr.dtype.element_ty), mask=cm)


# ---------------------------------------------------------------------------
#  Python wrappers (same results as the torch path they replace)
# ---------------------------------------------------------------------------


def lookup_coords(coord, pose, coord_scale, voxel_size, table_key,
                  life=None, frame_t=0):
    """Fused `to_global` + `voxel_hash` + sorted-table lookup.

    coord (N, 3) float; pose (4, 4) or (3, 4) sensor-to-world;
    table_key (Nw,) sorted int64 resident keys.

    With `life` given, a hit additionally requires
    `(frame_t - phase(key)) % life != 0`; SpaceCache leaves `life=None`
    (pure membership).

    Returns (keys int64 (N,), slot int64 (N,), hit bool (N,)).
    """
    dev = coord.device
    n = int(coord.shape[0])
    nw = int(table_key.numel())
    keys = torch.empty(n, dtype=torch.long, device=dev)
    slot = torch.empty(n, dtype=torch.long, device=dev)
    hit_i8 = torch.empty(n, dtype=torch.int8, device=dev)
    if n == 0:
        return keys, slot, hit_i8.view(torch.bool)
    coord = coord.contiguous()
    pose = pose.contiguous()
    vs = _scalar_f64(voxel_size, dev)
    sc = _scalar_f64(coord_scale, dev)
    iters = _lb_iters(nw) if nw else 0
    with_due = life is not None
    L = 1 if not with_due else int(life)
    _lookup_coords_kernel[(triton.cdiv(n, 256),)](
        coord, pose, sc, vs, table_key, keys, slot, hit_i8,
        n, nw, iters, int(frame_t), L, int(pose.stride(0)),
        WITH_DUE=with_due, PACKED=_rk.is_packed(), BLOCK=256)
    return keys, slot, hit_i8.view(torch.bool)


def lookup_keys(keys, table_key, life=None, frame_t=0):
    """Fused searchsorted + membership test on precomputed keys.
    Returns (slot int64 (N,), hit bool (N,))."""
    dev = keys.device
    n = int(keys.numel())
    nw = int(table_key.numel())
    slot = torch.empty(n, dtype=torch.long, device=dev)
    hit_i8 = torch.empty(n, dtype=torch.int8, device=dev)
    if n == 0:
        return slot, hit_i8.view(torch.bool)
    iters = _lb_iters(nw) if nw else 0
    with_due = life is not None
    L = 1 if not with_due else int(life)
    _lookup_keys_kernel[(triton.cdiv(n, 256),)](
        table_key, keys.contiguous(), slot, hit_i8,
        n, nw, iters, int(frame_t), L,
        WITH_DUE=with_due, PACKED=_rk.is_packed(), BLOCK=256)
    return slot, hit_i8.view(torch.bool)


def restore(full_feat, cache_feat, sub_feat, hit, slot, miss_idx):
    """Fused `SpaceCache.restore`, in place: full[i] = hit[i] ?
    cache[slot[i]] : sub[rank_i], rank_i = position of i in sorted
    `miss_idx`.

    `cache_feat=None` (empty table) and `sub_feat=None` (the stage ran on
    the full token set) are allowed; a missing sub leaves the miss rows
    of `full` unchanged.
    """
    dev = full_feat.device
    n, C = full_feat.shape
    if n == 0:
        return full_feat
    m = int(miss_idx.numel()) if miss_idx is not None else 0
    hit_i8 = hit.view(torch.int8) if hit.dtype == torch.bool else hit
    cache_ptr = cache_feat if cache_feat is not None else full_feat
    identity = sub_feat is None
    sub_ptr = full_feat if identity else sub_feat
    _restore_kernel[(triton.cdiv(n, 64),)](
        full_feat, cache_ptr, sub_ptr, hit_i8, slot,
        miss_idx if miss_idx is not None else slot,
        n, m, _lb_iters(max(m, 1)), C,
        IDENTITY_MISS=identity, BN=64, BC=128)
    return full_feat


def build_subset_point(point_full, sub_idx):
    """Fused `spacecache.common.stage_cache.build_subset_point_fast`.

    The parent map, the field gathers and the serialized-order remap run
    as four kernels plus one cumsum (the rank of each surviving token).
    """
    from pointcept.models.utils.structure import Point

    device = point_full.feat.device
    n_full = int(point_full.feat.shape[0])
    n_sub = int(sub_idx.numel())

    sub_idx = sub_idx.contiguous()
    p2s = torch.empty(n_full, dtype=torch.long, device=device)
    _p2s_kernel[(triton.cdiv(n_full, 256),)](
        sub_idx, p2s, n_full, n_sub,
        _lb_iters(max(n_sub, 1)), BLOCK=256)

    # one launch gathers every field into freshly allocated tensors
    C = int(point_full.feat.shape[1])
    has_grid = "grid_coord" in point_full.keys()
    batch_full = point_full.batch          # derived from offset if absent
    has_batch = batch_full is not None
    feat_s = torch.empty(n_sub, C, device=device,
                         dtype=point_full.feat.dtype)
    coord_s = torch.empty(n_sub, 3, device=device,
                          dtype=point_full.coord.dtype)
    grid_s = (torch.empty(n_sub, 3, device=device,
                          dtype=point_full.grid_coord.dtype)
              if has_grid else feat_s)
    batch_s = (torch.empty(n_sub, device=device, dtype=batch_full.dtype)
               if has_batch else feat_s)
    _gather_fields_kernel[(n_sub, triton.cdiv(C, 256))](
        sub_idx,
        point_full.feat, point_full.coord,
        point_full.grid_coord if has_grid else point_full.feat,
        batch_full if has_batch else point_full.feat,
        feat_s, coord_s, grid_s, batch_s,
        C, HAS_GRID=has_grid, HAS_BATCH=has_batch, BC=256)

    pd = {
        "feat": feat_s,
        "coord": coord_s,
        "offset": torch.full((1,), n_sub, dtype=torch.long,
                             device=device),
    }
    if has_batch:
        pd["batch"] = batch_s
    if "serialized_order" in point_full.keys():
        parent_order = point_full.serialized_order   # (n_orders, n_full)
        n_orders, nf = parent_order.shape
        in_sub = torch.empty(n_orders * nf, dtype=torch.int8,
                             device=device)
        _emit_insub_kernel[(triton.cdiv(n_orders * nf, 256),)](
            parent_order.contiguous(), p2s, in_sub, n_orders * nf,
            BLOCK=256)
        pos = torch.cumsum(in_sub.view(n_orders, nf), dim=1)
        sub_order = torch.empty(n_orders, n_sub, dtype=torch.long,
                                device=device)
        sub_inv = torch.empty(n_orders, n_sub, dtype=torch.long,
                              device=device)
        _remap_kernel[(triton.cdiv(n_orders * nf, 256),)](
            parent_order.contiguous(), p2s, pos.contiguous(),
            sub_order, sub_inv, n_orders * nf, nf, n_sub, BLOCK=256)
        pd["serialized_order"] = sub_order
        pd["serialized_inverse"] = sub_inv
    if "order" in point_full.keys():
        pd["order"] = point_full["order"]
    if has_grid:
        pd["grid_coord"] = grid_s
    if "grid_size" in point_full.keys():
        pd["grid_size"] = point_full.grid_size
    if "sparse_shape" in point_full.keys():
        pd["sparse_shape"] = point_full.sparse_shape
    sub = Point(pd)
    if "sparse_conv_feat" in point_full.keys() and "grid_coord" in pd:
        # Point.sparsify() reads batch[-1].tolist() (a device->host sync);
        # reuse the parent's spatial_shape and batch_size, which bound any
        # subset
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


def stage_classify(keys, st_idx, life, inv, frame_t, strict=False):
    """Fused classification of the staged cells in `SpaceCache.commit`.

    `life` is the per-cell lifetime (long, clamped to >= 1 by the caller).
    Returns (k_raw, due_raw, keep int8), aligned with `st_idx`."""
    dev = keys.device
    m = int(st_idx.numel())
    k_raw = torch.empty(m, dtype=torch.long, device=dev)
    due_raw = torch.empty(m, dtype=torch.long, device=dev)
    keep = torch.empty(m, dtype=torch.int8, device=dev)
    if m == 0:
        return k_raw, due_raw, keep
    _stage_classify_kernel[(triton.cdiv(m, 256),)](
        keys, st_idx, life,
        inv.view(torch.int8) if inv is not None else st_idx,
        k_raw, due_raw, keep,
        m, int(frame_t) + 1, int(frame_t) + 1,
        HAS_INV=inv is not None, STRICT=bool(strict),
        PACKED=_rk.is_packed(), BLOCK=256)
    return k_raw, due_raw, keep


def mark_bad(res_key, keys, inv):
    """Mark resident rows hit by the invalidate mask `inv` over `keys`.
    Returns (Nw,) int8."""
    dev = res_key.device
    nw = int(res_key.numel())
    n = int(keys.numel())
    bad = torch.zeros(nw, dtype=torch.int8, device=dev)
    if nw == 0 or n == 0:
        return bad
    _mark_bad_kernel[(triton.cdiv(n, 256),)](
        res_key, keys.contiguous(), inv.view(torch.int8), bad,
        n, nw, _lb_iters(nw), BLOCK=256)
    return bad


def overwrite_resident(res_key, res_due, res_feat, new_key, new_due,
                       new_feat):
    """Fused step (a) of `SpaceCache._compact`: overwrite the feat row
    and due of staged keys that are already resident. Returns `fresh`
    (n_ins,) int8, 1 where the key is not resident."""
    dev = res_key.device
    nw = int(res_key.numel())
    n_ins = int(new_key.numel())
    C = int(new_feat.shape[1])
    fresh = torch.empty(n_ins, dtype=torch.int8, device=dev)
    if n_ins == 0:
        return fresh
    if nw == 0:
        fresh.fill_(1)
        return fresh
    _overwrite_kernel[(triton.cdiv(n_ins, 64),)](
        res_key, res_due, res_feat, new_key, new_due, new_feat, fresh,
        n_ins, nw, _lb_iters(nw), C, BN=64, BC=128)
    return fresh


def surv_keep(due, bad, frame_t):
    """Fused survivor mask: keep = (due > t+1) & ~bad. Returns int8."""
    dev = due.device
    nw = int(due.numel())
    keep = torch.empty(nw, dtype=torch.int8, device=dev)
    if nw == 0:
        return keep
    _surv_keep_kernel[(triton.cdiv(nw, 256),)](
        due, bad if bad is not None else due, keep, nw,
        int(frame_t) + 1, HAS_BAD=bad is not None, BLOCK=256)
    return keep


def scatter_merge(a_key, a_due, a_feat, pos_a, b_key, b_due, b_feat,
                  pos_b):
    """Fused write of two sorted runs into one table. `pos_a` / `pos_b`
    are the precomputed output positions of the A and B rows.
    Returns (out_key, out_feat, out_due)."""
    dev = a_key.device if a_key is not None else b_key.device
    n_a = int(a_key.numel())
    n_b = int(b_key.numel())
    C = int(b_feat.shape[1])
    out_key = torch.empty(n_a + n_b, dtype=torch.long, device=dev)
    out_due = torch.empty(n_a + n_b, dtype=torch.long, device=dev)
    out_feat = b_feat.new_empty((n_a + n_b, C))
    if n_a + n_b == 0:
        return out_key, out_feat, out_due
    a_feat = a_feat if a_feat is not None else b_feat[:0]
    a_key = a_key if a_key is not None else b_key[:0]
    a_due = a_due if a_due is not None else b_due[:0]
    pos_a = pos_a if pos_a is not None else pos_b[:0]
    _scatter_merge_kernel[(n_a + n_b,)](
        a_key, a_due, a_feat, pos_a, b_key, b_due, b_feat, pos_b,
        out_key, out_due, out_feat, n_a, C, BC=128)
    return out_key, out_feat, out_due
