"""SpaceCache: a sorted world-cell feature table.

Every resident entry is servable, so a read is a pure membership test. All
decisions about what the next frame may read are made once per frame, in
`commit`:

  expire      the entry's lifetime runs out at t+1 (`due <= t+1`)
  invalidate  the caller marks the cell unreadable (ignore labels or low
              prediction confidence)

A resident entry is evicted for either reason; a staged (just computed) cell
is simply never inserted.

Per frame the pipeline calls
  1. `lookup(keys)`             after the stage's down-pooling (mask gate)
  2. `restore(...)`, `write()`  after the stage's last block (reuse gate);
                                `write` only appends to a staging buffer
  3. `commit(keys, L, t, ...)`  after the network output (commit gate); one
                                pass that stamps lifetimes, merges the staged
                                cells into the sorted table and evicts
and `reset()` at every sequence boundary.

Lifetime. An entry written at frame t gets

    due = (t+1) + ((key - (t+1)) mod L),

the first frame after t with (f - key) % L == 0. The phase is keyed on the
cell (`region_key.phase`, the prime dot product of the cell index), so about
1/L of the cells expire per frame instead of all at once.
Resident entries are never re-stamped: a cell is refreshed only when it is
recomputed, and cells that leave the field of view expire by the same rule.

With `expiry="fixed"` every entry instead gets due = t + L, i.e. it is
reused for exactly L-1 frames and then recomputed.
"""

from __future__ import annotations

import queue
import threading

import torch

from spacecache.common import region_key


class CacheView:
    """Result of one `lookup`, passed on to `restore` / `write`.

      keys      (n,)  long   per-token cell keys, in token order
      slot      (n,)  long   row in the resident table (valid where `hit`)
      hit       (n,)  bool   resident, hence servable
      miss_idx  (m,)  long   token indices that must be recomputed
    """

    __slots__ = ("keys", "slot", "hit", "miss_idx")

    def __init__(self, keys, slot, hit, miss_idx):
        self.keys = keys
        self.slot = slot
        self.hit = hit
        self.miss_idx = miss_idx

    @property
    def n_miss(self):
        return int(self.miss_idx.numel())


class SpaceCache:
    """Sorted world-cell store.

    Resident table (sorted by `key`, on `device`):
      key   (Nw,)    int64   cell hash, sorted and unique
      feat  (Nw, C)          cached stage output of the cell
      due   (Nw,)    int64   first frame at which the entry is no longer
                             servable (read only by `commit`)

    `keys_unique`: the caller guarantees that the keys staged in one frame
    are distinct (true at match_factor 1 on a world-anchored grid, where each
    stage token has its own cell); otherwise the features of equal keys are
    averaged into one entry. See `staged_keys_unique` in
    spacecache/policies/spacecache.py.
    `triton`: run the table operations through the fused kernels in
    `spacecache/common/triton_cache.py` (falls back to torch when unavailable).
    """

    def __init__(self, device, *, default_life=4, keys_unique=False,
                 triton=False, expiry="staggered"):
        self.device = device
        self.keys_unique = bool(keys_unique)
        self.default_life = int(default_life)
        # staggered: expiry staggered by the region hash, 0..L-1 reuses per
        # entry; fixed: every entry is reused for exactly L-1 frames
        if expiry not in ("staggered", "fixed"):
            raise ValueError(f"unknown expiry {expiry!r}")
        self.strict = expiry == "fixed"
        from spacecache.common import triton_cache as _tc
        self._triton = (bool(triton) and _tc.is_available()
                        and getattr(device, "type", None) == "cuda")
        self.reset()

    def reset(self):
        """Drop every entry (call at a sequence boundary)."""
        self.key = torch.empty(0, dtype=torch.long, device=self.device)
        self.feat = None                 # shaped (0, C) on first commit
        self.due = torch.empty(0, dtype=torch.long, device=self.device)
        self._st_src = None              # staged features (or the full tensor)
        self._st_idx = None              # staged indices into this frame's keys
        self._st_compact = False         # _st_src row i <-> key _st_idx[i]

    @property
    def n_cells(self):
        return int(self.key.numel())

    @property
    def n_staged(self):
        return 0 if self._st_idx is None else int(self._st_idx.numel())

    def nbytes(self):
        if self.feat is None:
            return 0
        return (self.key.numel() * self.key.element_size()
                + self.feat.numel() * self.feat.element_size()
                + self.due.numel() * self.due.element_size())

    # -- mask gate ----------------------------------------------------------

    def lookup(self, keys):
        """Binary-search `keys` in the resident table."""
        n = int(keys.numel())
        if self.n_cells == 0 or n == 0:
            return CacheView(
                keys,
                torch.zeros(n, dtype=torch.long, device=self.device),
                torch.zeros(n, dtype=torch.bool, device=self.device),
                torch.arange(n, device=self.device))
        if self._triton:
            from spacecache.common import triton_cache as _tc
            slot, hit = _tc.lookup_keys(keys, self.key)
            return CacheView(keys, slot, hit,
                             torch.nonzero(~hit, as_tuple=True)[0])
        slot = torch.searchsorted(self.key, keys).clamp(max=self.n_cells - 1)
        hit = self.key[slot] == keys
        return CacheView(keys, slot, hit,
                         torch.nonzero(~hit, as_tuple=True)[0])

    # -- reuse gate ---------------------------------------------------------

    def restore(self, view, full_feat, sub_feat=None):
        """Fill `full_feat` in place: hits from the table, misses from
        `sub_feat` (the stage output on the miss subset). `sub_feat` is None
        when the stage already ran on the full token set."""
        if self._triton:
            from spacecache.common import triton_cache as _tc
            return _tc.restore(full_feat, self.feat, sub_feat,
                               view.hit, view.slot, view.miss_idx)
        if self.feat is not None and self.feat.shape[0] > 0:
            full_feat[view.hit] = self.feat[view.slot[view.hit]].to(
                full_feat.dtype)
        if sub_feat is not None and view.miss_idx.numel() > 0:
            full_feat[view.miss_idx] = sub_feat.to(full_feat.dtype)
        return full_feat

    def write(self, view, feats, mask=None, rows=None):
        """Stage the recomputed cells for the next `commit` (append only).

        Only references are kept; the gather happens in `commit`. `feats` is
        the frame's full per-token feature tensor and `mask` selects the
        cells to stage (default: every miss). `rows`, if given, are the miss
        rows already in `miss_idx` order (the miss-subset output), which
        avoids holding the full stage output until the commit.
        """
        idx = view.miss_idx if mask is None else torch.nonzero(
            mask, as_tuple=True)[0]
        if idx.numel() == 0:
            self._st_src = self._st_idx = None
            self._st_compact = False
            return 0
        if rows is not None and mask is None:
            self._st_src = (rows if rows.dtype == feats.dtype
                            else rows.to(feats.dtype))
            self._st_compact = True
        else:
            self._st_src = feats
            self._st_compact = False
        self._st_idx = idx
        return int(idx.numel())

    # -- commit gate --------------------------------------------------------

    def pop_staging(self):
        """Detach this frame's staged references (for an async commit)."""
        s = (self._st_src, self._st_idx, self._st_compact)
        self._st_src = self._st_idx = None
        self._st_compact = False
        return s

    def commit(self, keys, lifetime, frame_t, invalidate=None, staged=None):
        """Make the table exactly what frame t+1 may read.

        keys        this frame's per-token keys (as passed to `lookup`)
        lifetime    lifetime in frames for the staged cells (int or per-key
                    tensor aligned with `keys`), clamped to >= 1
        frame_t     index of this frame within the sequence
        invalidate  optional bool mask over `keys`; these cells are dropped
                    whether resident or staged
        staged      staging refs from `pop_staging` (async commit)

        Returns the number of evicted resident entries (expired or
        invalidated).
        """
        t = int(frame_t)
        n = int(keys.numel())
        dev = self.device
        _tc = None
        if self._triton:
            from spacecache.common import triton_cache as _tc
        if torch.is_tensor(lifetime):
            life = lifetime.to(device=dev, dtype=torch.long).clamp(min=1)
        else:
            life = torch.full((n,), max(int(lifetime), 1),
                              dtype=torch.long, device=dev)

        if staged is not None:
            src, st_idx = staged[0], staged[1]
            compact = bool(staged[2]) if len(staged) > 2 else False
        else:
            src, st_idx = self._st_src, self._st_idx
            compact = self._st_compact
            self._st_src = self._st_idx = None
            self._st_compact = False
        inv = None if invalidate is None else invalidate.to(torch.bool)

        # Staged cells: decide survivors on the key/due columns, then sort
        # and gather the feature rows in a single pass.
        new_key = new_feat = new_due = None
        if st_idx is not None and st_idx.numel() > 0:
            if self._triton:
                k_raw, due_raw, keep8 = _tc.stage_classify(
                    keys, st_idx, life, inv, t, strict=self.strict)
                sel = torch.nonzero(keep8, as_tuple=True)[0]
            else:
                k_raw = keys[st_idx]
                due_raw = self._next_due(k_raw, life[st_idx], t, self.strict)
                keep = due_raw > t + 1
                if inv is not None:
                    keep &= ~inv[st_idx]
                sel = torch.nonzero(keep, as_tuple=True)[0]
            if sel.numel() > 0:
                k = k_raw[sel]
                order = torch.argsort(k, stable=True)
                take = sel[order] if compact else st_idx[sel][order]
                new_key = k[order]
                new_due = due_raw[sel][order]
                new_feat = src[take]
                if not self.keys_unique:
                    new_key, new_feat, new_due = self._pool_runs(
                        new_key, new_feat, new_due)

        # Resident entries hit by `invalidate`, marked without a host sync:
        # non-invalidated keys scatter into a scratch slot past the end.
        bad = None
        if inv is not None and self.n_cells > 0 and n > 0:
            if self._triton:
                bad = _tc.mark_bad(self.key, keys, inv)
            else:
                slot = torch.searchsorted(self.key, keys).clamp(
                    max=self.n_cells - 1)
                hit_and_inv = (self.key[slot] == keys) & inv
                tgt = torch.where(hit_and_inv, slot,
                                  torch.full_like(slot, self.n_cells))
                bad = torch.zeros(self.n_cells + 1, dtype=torch.bool,
                                  device=dev)
                bad.index_fill_(0, tgt, True)
                bad = bad[:self.n_cells]

        if self._triton:
            return self._compact_triton(new_key, new_feat, new_due, t, bad)
        return self._compact(new_key, new_feat, new_due, t, bad)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _next_due(key, life, frame_t, strict=False):
        """Smallest frame > frame_t with (f - phase(key)) % life == 0."""
        base = frame_t + 1
        if strict:
            return base + life - 1
        return base + torch.remainder(region_key.phase(key) - base, life)

    def _pool_runs(self, k, f, d):
        """Mean-pool equal keys in a sorted staged run (match_factor > 1)."""
        if k.numel() < 2:
            return k, f, d
        new_run = torch.ones_like(k, dtype=torch.bool)
        new_run[1:] = k[1:] != k[:-1]
        run = torch.cumsum(new_run, 0) - 1
        n_run = int(run[-1].item()) + 1
        if n_run == k.numel():
            return k, f, d
        cnt = torch.zeros(n_run, dtype=torch.float32, device=k.device)
        cnt.scatter_add_(0, run, torch.ones_like(run, dtype=torch.float32))
        acc = torch.zeros((n_run, f.shape[1]), dtype=torch.float32,
                          device=f.device)
        acc.scatter_add_(0, run.unsqueeze(1).expand(-1, f.shape[1]), f.float())
        uf = (acc / cnt.clamp(min=1).unsqueeze(1)).to(f.dtype)
        return k[new_run], uf, d[new_run]

    def _compact(self, new_key, new_feat, new_due, frame_t, bad=None):
        """Rewrite the resident table once: overwrite, evict, insert.

        Insertion merges two sorted runs: survivor A[i] lands at
        i + searchsorted(B, A[i]) and new key B[j] at j + searchsorted(A, B[j]),
        so no full re-sort of the table is needed.
        """
        n_ins_in = 0 if new_key is None else int(new_key.numel())

        # (a) staged keys already resident: overwrite in place, so the expiry
        # test below sees the refreshed `due`
        if n_ins_in and self.n_cells > 0:
            ins = torch.searchsorted(self.key, new_key).clamp(
                max=self.n_cells - 1)
            exists = self.key[ins] == new_key
            rows = ins[exists]
            self.feat[rows] = new_feat[exists].to(self.feat.dtype)
            self.due[rows] = new_due[exists]
            fresh = ~exists
            new_key = new_key[fresh]
            new_feat = new_feat[fresh]
            new_due = new_due[fresh]
        n_ins = 0 if new_key is None else int(new_key.numel())

        # (b) survivors: not expired and not invalidated
        if self.n_cells > 0:
            keep = self.due > frame_t + 1
            if bad is not None:
                keep &= ~bad
            surv = torch.nonzero(keep, as_tuple=True)[0]
            n_a = int(surv.numel())
            n_evicted = self.n_cells - n_a
        else:
            surv = None
            n_a = 0
            n_evicted = 0

        if n_ins == 0 and n_evicted == 0:
            return 0

        if surv is not None and n_evicted:
            a_key = self.key[surv]
            a_feat = (self.feat.index_select(0, surv)
                      if self.feat is not None else None)
            a_due = self.due[surv]
        else:
            a_key, a_feat, a_due = self.key, self.feat, self.due
            n_a = int(a_key.numel())

        if n_ins == 0:
            self.key, self.feat, self.due = a_key, a_feat, a_due
            return n_evicted

        # (c) merge positions of the two sorted runs
        dev = self.device
        pos_a = torch.arange(n_a, device=dev) + torch.searchsorted(
            new_key, a_key)
        pos_b = torch.arange(n_ins, device=dev) + torch.searchsorted(
            a_key, new_key)

        total = n_a + n_ins
        c = new_feat.shape[1]
        out_key = torch.empty(total, dtype=torch.long, device=dev)
        out_due = torch.empty(total, dtype=torch.long, device=dev)
        out_feat = new_feat.new_empty((total, c))

        out_key.index_copy_(0, pos_a, a_key)
        out_key.index_copy_(0, pos_b, new_key)
        out_due.index_copy_(0, pos_a, a_due)
        out_due.index_copy_(0, pos_b, new_due)
        if n_a and a_feat is not None:
            out_feat.index_copy_(0, pos_a, a_feat.to(out_feat.dtype))
        out_feat.index_copy_(0, pos_b, new_feat.to(out_feat.dtype))

        self.key, self.feat, self.due = out_key, out_feat, out_due
        return n_evicted

    def _compact_triton(self, new_key, new_feat, new_due, frame_t,
                        bad=None):
        """`_compact` with the overwrite, survivor mask and merge writes run
        as fused Triton kernels (`bad` arrives as int8 from `mark_bad`)."""
        from spacecache.common import triton_cache as _tc
        dev = self.device
        n_ins_in = 0 if new_key is None else int(new_key.numel())

        if n_ins_in and self.n_cells > 0:
            fresh = _tc.overwrite_resident(
                self.key, self.due, self.feat,
                new_key, new_due, new_feat)
            sel = torch.nonzero(fresh, as_tuple=True)[0]
            if int(sel.numel()) == 0:
                new_key = new_feat = new_due = None
            elif int(sel.numel()) < n_ins_in:
                new_key = new_key[sel]
                new_feat = new_feat[sel]
                new_due = new_due[sel]
        n_ins = 0 if new_key is None else int(new_key.numel())

        if self.n_cells > 0:
            surv = torch.nonzero(
                _tc.surv_keep(self.due, bad, frame_t), as_tuple=True)[0]
            n_a = int(surv.numel())
            n_evicted = self.n_cells - n_a
        else:
            surv = None
            n_a = 0
            n_evicted = 0

        if n_ins == 0 and n_evicted == 0:
            return 0

        if surv is not None and n_evicted:
            a_key = self.key[surv]
            a_feat = (self.feat.index_select(0, surv)
                      if self.feat is not None else None)
            a_due = self.due[surv]
        else:
            a_key, a_feat, a_due = self.key, self.feat, self.due
            n_a = int(a_key.numel())

        if n_ins == 0:
            self.key, self.feat, self.due = a_key, a_feat, a_due
            return n_evicted

        pos_a = torch.arange(n_a, device=dev) + torch.searchsorted(
            new_key, a_key)
        pos_b = torch.arange(n_ins, device=dev) + torch.searchsorted(
            a_key, new_key)
        self.key, self.feat, self.due = _tc.scatter_merge(
            a_key, a_due, a_feat, pos_a,
            new_key, new_due, new_feat, pos_b)
        return n_evicted


class AsyncCommitter:
    """Run `commit` on a worker thread and a side CUDA stream (opt-in,
    `--commit_async`), overlapping it with the next frame's input path.

    submit(fn, inputs)  queue a job; `inputs` are record_stream'ed on the
                        side stream so their memory is not reused early
    wait_pending()      called before each lookup: wait for every queued job
                        and order the current stream after it
    join()              drain the queue (before a reset or teardown)
    """

    def __init__(self, device):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.done = torch.cuda.Event()
        self.error = None
        self.results = queue.Queue()
        self._q = queue.Queue()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while True:
            job = self._q.get()
            try:
                if job is None:
                    return
                ev, fn, inputs = job
                self.stream.wait_event(ev)
                with torch.cuda.stream(self.stream):
                    for t in inputs:
                        if torch.is_tensor(t):
                            t.record_stream(self.stream)
                    out = fn()
                self.done.record(self.stream)
                self.results.put(out)
            except Exception as e:                       # pragma: no cover
                self.error = e
            finally:
                self._q.task_done()

    def submit(self, fn, inputs=()):
        # the ordering event is recorded on the submitting thread's stream
        if self.error is not None:
            raise self.error
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        self._q.put((ev, fn, inputs))

    def wait_pending(self):
        """Wait on the host for every submitted job, then make the current
        stream wait for the last one. Waiting on `done` alone could see the
        previous job's event and let a lookup race the running commit."""
        self._q.join()
        if self.error is not None:
            raise self.error
        torch.cuda.current_stream().wait_event(self.done)

    def drain_results(self):
        """Return values of all finished jobs, in submission order."""
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                break
        return out

    def join(self):
        self._q.join()
        if self.error is not None:
            raise self.error

    def close(self):
        self._q.put(None)
