"""Piecewise CUDA-graph capture of one frame's whole inference.

The forward is captured as a chain of graphs split at every point that
needs the host,

    g0 -> host0 -> g1 -> host1 -> ... -> gN

and a replay launches each graph and runs each host step for real (a sync,
a dynamic-shape op, an H2D copy, a spconv rulebook build); the Python glue
in between (module calls, hooks, dispatch) is not re-run. Host steps are
found by a TorchDispatchMode for aten ops plus patches for non-aten calls
(`boundaries`); CUDA RNG draws are handed back from the eager frame by
`RngTape`. The graphs keep the shapes each host step produced at capture,
which the same frame reproduces on every replay; a step whose result
changes anyway (nondeterministic kernels) is counted in `drift` while the
chain keeps the captured values. All segments share one memory pool, which
is safe because they always replay in capture order.
"""
from __future__ import annotations

import contextlib
import warnings
from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

aten = torch.ops.aten


def _overloads(*names):
    """Every overload of the named aten ops (missing ops are skipped)."""
    out = set()
    for n in names:
        pkt = getattr(aten, n, None)
        if pkt is None:
            continue
        for ov in pkt.overloads():
            out.add(getattr(pkt, ov))
    return out


# host reads a device value / an output size it cannot know in advance
_SYNC = _overloads("_local_scalar_dense", "is_nonzero", "equal", "nonzero",
                   "nonzero_numpy", "argwhere", "_unique", "_unique2",
                   "unique_dim", "unique_consecutive",
                   "unique_dim_consecutive", "bincount", "masked_select",
                   "histc")
_REPEAT = _overloads("repeat_interleave")
_INDEX = _overloads("index", "index_put", "index_put_", "_index_put_impl",
                    "_index_put_impl_")
_COPY = _overloads("_to_copy", "copy_", "_copy_from")
_RNG = _overloads("randint", "rand", "randn", "randperm", "normal",
                  "normal_", "uniform_", "bernoulli", "bernoulli_",
                  "multinomial", "random_", "exponential_")


def _any_cuda(args, kwargs):
    for x in tree_flatten((args, kwargs))[0]:
        if torch.is_tensor(x) and x.is_cuda:
            return True
    return False


def _dev_type(d):
    try:
        return torch.device(d).type
    except Exception:
        return None


def _classify(func, args, kwargs):
    """Host-step kind of an aten call, or None to capture it."""
    if func in _SYNC:
        return "sync" if _any_cuda(args, kwargs) else None
    if func in _REPEAT:
        if kwargs.get("output_size") is None and _any_cuda(args, kwargs):
            # `.self_int` (python-int repeats) never syncs
            if "self_int" not in str(func):
                return "sync"
        return None
    if func in _INDEX:
        self_t = args[0] if args else None
        idx = args[1] if len(args) > 1 else kwargs.get("indices")
        if not (torch.is_tensor(self_t) and self_t.is_cuda and idx):
            return None
        for t in idx:
            if torch.is_tensor(t) and (
                    t.dtype in (torch.bool, torch.uint8) or not t.is_cuda):
                return "index"
        # index_put_ with host-side values: copied to the device inside
        val = args[2] if len(args) > 2 else kwargs.get("values")
        if (torch.is_tensor(val) and not val.is_cuda
                and val.numel() > 1):
            return "index"
        return None
    if func in _COPY:
        name = str(func)
        if "_to_copy" in name:
            dst = kwargs.get("device")
            src = args[0] if args else None
            if (dst is not None and torch.is_tensor(src)
                    and _dev_type(dst) != src.device.type):
                return "copy"
            return None
        a = args[0] if args else None
        b = args[1] if len(args) > 1 else None
        if (torch.is_tensor(a) and torch.is_tensor(b)
                and a.device.type != b.device.type):
            return "copy"
        return None
    if func in _RNG:
        dev = kwargs.get("device")
        if dev is not None:
            return "rng" if _dev_type(dev) == "cuda" else None
        return "rng" if _any_cuda(args, kwargs) else None
    return None


def _drift(ref, new):
    """Number of leaves of a re-run host step whose shape or host value
    differs from the captured result.

    The new result is not written back: the chain keeps computing on the
    captured, mutually consistent values (writing back only some leaves
    could mix the indices of one run with the counts of another). Tensors
    are compared by shape only, since a value compare would add a sync.
    """
    if torch.is_tensor(ref):
        if new is ref:
            return 0
        return int(not torch.is_tensor(new)
                   or tuple(ref.shape) != tuple(new.shape))
    if isinstance(ref, (tuple, list)):
        if not isinstance(new, (tuple, list)) or len(ref) != len(new):
            return 1
        return sum(_drift(a, b) for a, b in zip(ref, new))
    if isinstance(ref, dict):
        return sum(_drift(ref[k], new.get(k)) for k in ref)
    if isinstance(ref, (bool, int, float)):
        same = ref == new or (ref != ref and new != new)   # NaN == NaN
        return 0 if same else 1
    return 0      # None, spconv descriptors: no device data


class Piecewise:
    """The captured chain and its replay."""

    def __init__(self, pool=None, error_mode="thread_local"):
        self.pool = (pool if pool is not None
                     else torch.cuda.graph_pool_handle())
        # thread_local: a global-mode capture forbids CUDA calls from every
        # thread, which would kill the loader's pin_memory thread
        self.error_mode = error_mode
        self.steps = []            # ("graph", g, None) | ("host", rec, kind)
        self.g = None
        self.kinds = Counter()
        self.ops = Counter()
        self.drift = 0             # drifted host-step leaves, all replays

    @property
    def capturing(self):
        return self.g is not None

    def begin(self):
        g = torch.cuda.CUDAGraph()
        g.capture_begin(pool=self.pool, capture_error_mode=self.error_mode)
        self.g = g

    def end(self):
        """Close the open segment and run it once, so the host step that
        follows reads real values."""
        g, self.g = self.g, None
        if g is None:
            return
        with warnings.catch_warnings():
            # back-to-back host steps leave an empty segment between them
            warnings.simplefilter("ignore")
            g.capture_end()
        self.steps.append(("graph", g, None))
        g.replay()

    def abort(self):
        """End an open capture without keeping it."""
        g, self.g = self.g, None
        if g is not None:
            try:
                g.capture_end()
            except Exception:
                pass

    def host(self, fn, args, kwargs, kind, name=None):
        """Close the segment, run `fn` as a host step, open the next one."""
        self.end()
        out = fn(*args, **kwargs)
        self.steps.append(("host", (fn, args, kwargs, out), kind))
        self.kinds[kind] += 1
        self.ops[name or getattr(fn, "__name__", str(fn))] += 1
        self.begin()
        return out

    def freeze(self, fn, args, kwargs, name=None):
        """Run a merge decision producer inside the graph (its cost stays
        in the timing) but hand downstream a copy taken once at capture.

        Atomic scatters change the scores in the last bits between replays;
        a live decision could then pick other pairs than the ones the host
        steps of merge_point were captured with, and index_add_ would run
        out of bounds.
        """
        out = fn(*args, **kwargs)
        self.end()                         # materialise `out`
        frozen = tree_map(
            lambda t: t.clone() if torch.is_tensor(t) else t, out)
        # the step record keeps the copy (outside the graph pool) alive;
        # otherwise its memory could go to a later host step while the
        # graphs still read it
        self.steps.append(("host", (None, None, None, frozen), "freeze"))
        self.kinds["freeze"] += 1
        self.ops[name or getattr(fn, "__name__", str(fn))] += 1
        self.begin()
        return frozen

    @property
    def n_graphs(self):
        return sum(1 for s in self.steps if s[0] == "graph")

    @property
    def n_host(self):
        return sum(1 for s in self.steps
                   if s[0] == "host" and s[2] != "freeze")

    def replay(self):
        # inference_mode: some captured outputs are inference tensors
        # (serialization's `encode` runs under @torch.inference_mode), and
        # in-place writes into them are refused outside that mode
        with torch.inference_mode():
            for kind, x, k in self.steps:
                if kind == "graph":
                    x.replay()
                elif k in ("rng", "freeze"):
                    continue       # frozen: the captured value stays put
                else:
                    fn, args, kwargs, ref = x
                    self.drift += _drift(ref, fn(*args, **kwargs))


class _BoundaryMode(TorchDispatchMode):
    """Send every host-needing aten call to `pw.host` while capturing."""

    def __init__(self, pw):
        super().__init__()
        self.pw = pw

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.pw.capturing:
            kind = _classify(func, args, kwargs)
            if kind is not None:
                return self.pw.host(func, args, kwargs, kind, str(func))
        return func(*args, **kwargs)


def _seq_index(idx):
    """True if an index holds a Python list, range or numpy array.

    PyTorch turns those into a device tensor inside __getitem__ /
    __setitem__ with the Python dispatch key excluded, i.e. an H2D copy the
    dispatch mode never sees (e.g. `grid_coord[:, [1, 0, 2]]`). A list made
    only of tensors is a tuple of indices.
    """
    items = idx if isinstance(idx, tuple) else (idx,)
    for it in items:
        if isinstance(it, range):
            return True
        if isinstance(it, list) and any(not torch.is_tensor(e) for e in it):
            return True
        if (type(it).__module__ == "numpy"
                and getattr(it, "ndim", 0) > 0):
            return True
    return False


def _host_value(val):
    """A __setitem__ value that has to be copied to the device first
    (python scalars and 0-dim tensors are filled on the device)."""
    if isinstance(val, (list, tuple)):
        return True
    if type(val).__module__ == "numpy" and getattr(val, "ndim", 0) > 0:
        return True
    return torch.is_tensor(val) and not val.is_cuda and val.numel() > 1


@contextlib.contextmanager
def boundaries(pw):
    """Route every host-needing call through `pw.host` while capturing."""
    patched = []

    def put(owner, name, fn):
        # remember whether the attribute was the owner's own or inherited
        # (Tensor.__getitem__ lives on torch._C.TensorBase), so the undo
        # leaves the class exactly as it was
        own = name in getattr(owner, "__dict__", {})
        patched.append((owner, name, getattr(owner, name), own))
        setattr(owner, name, fn)

    def wrap(owner, name, kind):
        orig = getattr(owner, name, None)
        if orig is None:
            return

        def w(*a, **k):
            if pw.capturing:
                return pw.host(orig, a, k, kind, name)
            return orig(*a, **k)
        put(owner, name, w)

    T = torch.Tensor
    _gi, _si, _nt = T.__getitem__, T.__setitem__, T.new_tensor

    def getitem(self, idx):
        if pw.capturing and self.is_cuda and _seq_index(idx):
            return pw.host(_gi, (self, idx), {}, "h2d", "Tensor[list]")
        return _gi(self, idx)

    def setitem(self, idx, val):
        if (pw.capturing and self.is_cuda
                and (_seq_index(idx) or _host_value(val))):
            return pw.host(_si, (self, idx, val), {}, "h2d",
                           "Tensor[list]=")
        return _si(self, idx, val)

    def new_tensor(self, data, *a, **k):
        dev = k.get("device", self.device)
        if pw.capturing and _dev_type(dev) == "cuda":
            return pw.host(_nt, (self, data) + a, k, "h2d",
                           "Tensor.new_tensor")
        return _nt(self, data, *a, **k)

    # merge decision producers: in-graph, output frozen (Piecewise.freeze)
    import sys
    for mod_name, names in (
            ("spacecache.policies.tome", ("tome_patch_indices",)),
            ("spacecache.policies.algm", ("clap_grid_indices",
                                        "clap_window_indices",
                                        "gbm_indices")),
            ("spacecache.common.point_merge", ("tome_patch_indices",
                                             "clap_grid_indices",
                                             "clap_window_indices",
                                             "gbm_indices"))):
        mod = sys.modules.get(mod_name)
        for nm in names:
            orig = getattr(mod, nm, None) if mod is not None else None
            if orig is None:
                continue

            def fz(*a, _o=orig, _n=nm, **k):
                if pw.capturing:
                    return pw.freeze(_o, a, k, _n)
                return _o(*a, **k)
            put(mod, nm, fz)

    put(T, "__getitem__", getitem)
    put(T, "__setitem__", setitem)
    put(T, "new_tensor", new_tensor)

    try:
        import spconv.pytorch.ops as _spops
        wrap(_spops, "get_indice_pairs_implicit_gemm", "spconv_pairs")
        wrap(_spops, "get_indice_pairs", "spconv_pairs")
    except Exception:
        pass
    wrap(torch.cuda, "synchronize", "synchronize")
    for nm in ("tensor", "as_tensor"):
        orig = getattr(torch, nm)

        def w(*a, _o=orig, _n=nm, **k):
            dev = k.get("device")
            if (pw.capturing and dev is not None
                    and _dev_type(dev) == "cuda"):
                return pw.host(_o, a, k, "h2d", f"torch.{_n}")
            return _o(*a, **k)
        put(torch, nm, w)
    try:
        with _BoundaryMode(pw):
            yield pw
    finally:
        for owner, name, orig, own in reversed(patched):
            if own:
                setattr(owner, name, orig)
            else:
                delattr(owner, name)


_EXT_OK = {}


def check_extension_streams():
    """Refuse to time a chain that silently drops an extension's kernels.

    A CUDA extension that launches without a stream runs on the legacy
    default stream; during a capture on a non-blocking stream that kernel
    neither errors nor enters the graph, so every replay computes without
    it. Probe LitePT's `pointrope`: capture one call on a known buffer,
    reset the buffer and replay; the replay must change the buffer again.
    """
    import sys
    lp = sys.modules.get("pointcept.models.litept.litept_v1")
    kern = getattr(lp, "_kernels", None) if lp is not None else None
    if kern is None or "pointrope" in _EXT_OK:
        return
    dev = torch.device("cuda", torch.cuda.current_device())
    tok = torch.ones(1, 2, 1, 6, device=dev)
    pos = torch.ones(1, 2, 3, dtype=torch.long, device=dev)
    ok = False
    try:
        kern.pointrope(tok, pos, 100.0, 1.0)          # load/JIT off-capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            kern.pointrope(tok, pos, 100.0, 1.0)
        torch.cuda.synchronize()
        tok.fill_(1.0)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        ok = not bool(torch.all(tok == 1.0))
        del g
    except Exception:
        ok = False
    _EXT_OK["pointrope"] = ok
    if not ok:
        raise RuntimeError(
            "piecewise: pointrope launches on the legacy default stream, so "
            "its kernel runs outside the CUDA graph and every replay "
            "computes without RoPE. libs/pointrope/kernels.cu launches "
            "on the current stream — rebuild the installed extension: "
            "`cd libs/pointrope && pip install --no-build-isolation .`")


class RngTape:
    """Record the eager frame's RNG draws and hand them back in call order
    during warmup and capture (CUDA draws as in-graph copies).

    `randperm` is taped on the CPU too: Utonia keeps shuffle_orders=True at
    eval, so a fresh permutation would make the capture a different forward.
    """

    NAMES = ("randint", "rand", "randn", "randperm")

    def __init__(self):
        self.rec = []
        self.pos = 0
        self.mode = None           # "record" | "replay" | None
        self.miss = 0
        self._orig = {}

    def install(self, mode="record"):
        self.mode = mode
        if self._orig:
            return
        for nm in self.NAMES:
            orig = getattr(torch, nm)
            self._orig[nm] = orig

            def w(*a, _o=orig, _n=nm, **k):
                dev = k.get("device")
                on_gpu = dev is not None and _dev_type(dev) == "cuda"
                if self.mode is None or not (on_gpu or _n == "randperm"):
                    return _o(*a, **k)
                if self.mode == "record":
                    out = _o(*a, **k)
                    self.rec.append(out)
                    return out
                if self.pos < len(self.rec):
                    r = self.rec[self.pos]
                    self.pos += 1
                    size = k.get("size")
                    if size is None or tuple(size) == tuple(r.shape):
                        return r.clone()
                self.miss += 1
                return _o(*a, **k)
            setattr(torch, nm, w)

        self._install_decisions()

    # merge decision producers share the tape: the eager frame's (src, dst)
    # pairs are recorded; warmup and capture still run the producer (its
    # kernels stay in the graph and in the timing) but return the recorded
    # pairs, so the capture makes the eager frame's merge.
    DECISIONS = (
        ("spacecache.policies.tome", ("tome_patch_indices",)),
        ("spacecache.policies.algm", ("clap_grid_indices",
                                    "clap_window_indices", "gbm_indices")),
        ("spacecache.common.point_merge", ("tome_patch_indices",
                                         "clap_grid_indices",
                                         "clap_window_indices",
                                         "gbm_indices")),
    )

    def _install_decisions(self):
        """Wrap the merge decision producers listed in DECISIONS."""
        import sys
        if getattr(self, "_dorig", None):
            return
        self._dorig = []
        self.drec = {}
        self.dpos = {}
        for mod_name, names in self.DECISIONS:
            mod = sys.modules.get(mod_name)
            for nm in names:
                orig = getattr(mod, nm, None) if mod is not None else None
                if orig is None:
                    continue
                self._dorig.append((mod, nm, orig))
                key = nm

                def w(*a, _o=orig, _k=key, **k):
                    out = _o(*a, **k)
                    if self.mode == "record":
                        self.drec.setdefault(_k, []).append(out)
                        return out
                    if self.mode != "replay":
                        return out
                    lst = self.drec.get(_k, [])
                    i = self.dpos.get(_k, 0)
                    self.dpos[_k] = i + 1
                    if i >= len(lst):
                        self.miss += 1
                        return out
                    rec = lst[i]
                    fl_o, fl_r = tree_flatten(out)[0], tree_flatten(rec)[0]
                    if len(fl_o) != len(fl_r) or any(
                            torch.is_tensor(x) and x.shape != y.shape
                            for x, y in zip(fl_o, fl_r)):
                        self.miss += 1
                        return out
                    return tree_map(
                        lambda t: t.clone() if torch.is_tensor(t) else t, rec)
                setattr(mod, nm, w)

    def rewind(self):
        """Switch to replay and restart from the first recorded draw."""
        self.mode = "replay"
        self.pos = 0
        self.dpos = {}

    def uninstall(self):
        """Restore every wrapped function."""
        for nm, orig in self._orig.items():
            setattr(torch, nm, orig)
        self._orig = {}
        for mod, nm, orig in reversed(getattr(self, "_dorig", None) or []):
            setattr(mod, nm, orig)
        self._dorig = []
        self.mode = None
