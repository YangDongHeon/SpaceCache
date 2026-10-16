"""Whole-frame CUDA-graph capture for host-free latency (`--graph_capture`).

At every pick frame the eager forward runs as usual, then the same frame is
captured and replayed in one of two modes:

  replay     backbone + seg_head as one graph. Host-decided values of the
             eager pass (dynamic-shape op outputs, serialization, padding,
             spconv indice pairs, RNG draws, merge decisions, the SpaceCache
             miss subset) are recorded and fed back during capture.
  piecewise  the whole inference as a chain of graphs split at every host
             step; each replay runs those steps for real
             (spacecache/common/piecewise_graph.py).

Kernel accounting sums GPU activity per pick in CUDA-only profiler windows:
k_frame (eager frame, backbone entry to seg_head exit), k_commit (streaming
commit gates) and k_graph (one replay). excluded_ms = k_frame - k_graph is
the eager GPU work whose result the graph reads from the record, and
fair_ms = graph_ms + max(excluded_ms, 0) counts every eager kernel once: at
replay speed if it was captured, at its eager duration otherwise
(fair_commit_ms adds k_commit). In piecewise mode, coverage =
k_piecewise / k_frame.
"""
from __future__ import annotations

import inspect
import sys
import torch

try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops as sp_ops
    from spconv.pytorch import functional as sp_fsp
    from spconv.pytorch.core import ImplicitGemmIndiceData, IndiceData
    _HAS_SPCONV = True
except Exception:  # pragma: no cover
    spconv = None
    _HAS_SPCONV = False

import importlib

# importlib, not `import ... as`: `from .point_transformer import *` in
# pointcept/models/__init__.py rebinds the package's `utils` attribute.
_structure = importlib.import_module("pointcept.models.utils.structure")
Point = _structure.Point


class _SiteReplay:
    """Keyed FIFO replay of dynamic-shape op outputs.

    For patched torch ops the key is (op, caller file, line). Capture pops
    one entry per call at that site; sites bypassed at capture (inside a
    function-level patch, or a disarmed gate) never consume their entries.
    """

    def __init__(self):
        self.lists = {}
        self.pos = {}

    def reset(self):
        self.pos = {}

    def record(self, key, out):
        self.lists.setdefault(key, []).append(out)
        return out

    def replay(self, key):
        lst = self.lists.get(key)
        if lst is None:
            raise RuntimeError(
                f"model_graph: {key} called during capture but never "
                f"recorded — capture pass diverged from record pass")
        i = self.pos.get(key, 0)
        if i >= len(lst):
            raise RuntimeError(
                f"model_graph: site {key} called more times during "
                f"capture ({i + 1}) than recorded ({len(lst)})")
        self.pos[key] = i + 1
        return lst[i]


def _caller_key(op):
    """(op, file, line) of the code that called the patched op."""
    f = sys._getframe(2)
    return (op, f.f_code.co_filename.split("/")[-1], f.f_lineno)


def _new_kprof():
    """CUDA-only profiler session: every kernel, memcpy and memset, no CPU
    op events."""
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA])


def _gpu_busy_ms(tp):
    """Sum of device-activity durations in a stopped CUDA-only session.

    Runtime calls (cudaLaunchKernel, cudaGraphLaunch) are CPU events and are
    skipped; CUPTI reports a replayed graph's kernels individually, so the
    same sum applies to an eager frame and to a replay. Returns
    (ms, n_events); ms is None when no device event was seen (no CUPTI), so
    "unmeasured" never reads as zero.
    """
    tot_us, n = 0.0, 0
    try:
        evs = tp.events()
    except Exception:
        evs = []
    for e in evs:
        if "cuda" not in str(getattr(e, "device_type", "")).lower():
            continue
        if getattr(e, "is_user_annotation", False):
            continue
        tr = getattr(e, "time_range", None)
        if tr is None:
            continue
        tot_us += float(tr.elapsed_us())
        n += 1
    return (tot_us / 1000.0 if n else None), n


def _median(v):
    v = sorted(v)
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


class ModelGraphSession:
    """Model hooks that record, capture and time every pick frame."""

    def __init__(self, model, pick_frame=None, policy=None, replays=200,
                 warmup=3, prof=None, metrics=None, logger=None,
                 pick_frames=None, kernel_acct=True, debug=False,
                 mode="replay"):
        self.model = model
        if mode not in ("replay", "piecewise"):
            raise ValueError(f"model_graph: unknown mode {mode!r}")
        self.mode = mode
        picks = (pick_frames if pick_frames is not None
                 else [pick_frame])
        self.picks = sorted({int(f) for f in picks})
        self._pick_set = set(self.picks)
        self.policy = policy
        self.replays = int(replays)
        self.warmup = int(warmup)
        self.prof = prof
        self.metrics = metrics
        self.logger = logger
        self.kernel_acct = bool(kernel_acct)
        # debug: the stream-restore diff reductions are captured into the
        # graph (extra kernels), so they stay off in timed runs
        self.debug = bool(debug)
        try:
            import socket
            self.host = socket.gethostname()
            self.device = torch.cuda.get_device_name()
        except Exception:
            self.host = self.device = None

        self._fi = 0
        self.frames = []                # one result dict per pick
        self._handles = []
        self.result = None
        self._reset_pick_state()

        self._handles.append(
            model.register_forward_pre_hook(self._frame_pre))
        # prepended: runs before FrameProfiler's frame end and the commit
        # gates, so the commit window opens where the frame closes
        self._handles.append(model.register_forward_hook(
            self._frame_end, prepend=True))
        self._handles.append(model.register_forward_hook(self._frame_post))
        # k_frame window = backbone entry -> seg_head exit, the boundary of
        # `_compute`; the segmentor's loss runs after seg_head
        bb = getattr(model, "backbone", None)
        head = getattr(model, "seg_head", None)
        if bb is not None:
            self._handles.append(bb.register_forward_pre_hook(
                self._win_open, prepend=True))
        if head is not None:
            self._handles.append(head.register_forward_hook(
                self._win_close))

    def _reset_pick_state(self):
        """Per-pick record/replay state. Reset after each capture, so the
        recorded tensors and the graph's pool are freed before the eval
        moves on."""
        self.rep = _SiteReplay()        # keyed dynamic-op outputs
        self.sers = []                  # serialization outputs, FIFO
        self.sparse = []                # (sparse_shape, batch_size), FIFO
        self.padinv = []                # (pad, unpad, cu_seqlens), FIFO
        self.rope_ms = []               # LitePT RoPE max_seqlen ints, FIFO
        self.convs = {}                 # SubMConv3d module -> IndiceData
        self._lv = []                   # per-level streaming snapshots
        self._rec_handles = []          # record-pass hooks, this pick only
        self._cap_handles = []
        self._origs = {}
        self._armed = False
        self._cap = False               # True while a capture pass runs
        self._dbg = {}                  # streaming restore diagnostics
        self._live = False              # pick pre-hook -> pick post-hook
        self._kp = None                 # open kernel-accounting session
        self._k = {}                    # window -> (ms, n_events)
        self._pre_snap = {}             # piecewise: cache state at frame start
        if self.mode == "piecewise":
            from spacecache.common.piecewise_graph import RngTape
            self._rng = RngTape()       # piecewise: the eager frame's draws
        else:
            self._rng = None

    # ------------------------------------------------ frame plumbing
    def _frame_pre(self, _m, inputs):
        if self._fi in self._pick_set:
            self._reset_pick_state()
            self._live = True
            if self.mode == "piecewise":
                # piecewise keeps only the eager frame's RNG draws and the
                # cache state it reads
                self._rng.install("record")
                self._pre_snap = {id(c): {
                    k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in vars(c).items()} for c in self._caches()}
            else:
                self._install_recorders()

    def _caches(self):
        """The streaming policy's SpaceCache objects (empty otherwise)."""
        pol = self.policy
        if pol is None:
            return []
        return [c for c in (getattr(pol, "cache", None),
                            getattr(pol, "cache4", None)) if c is not None]

    # kernel-accounting windows: sync at both edges so no kernel of the
    # previous region leaks in and every kernel launched inside has
    # finished (and been recorded) when the session stops.
    def _kopen(self):
        torch.cuda.synchronize()
        self._kp = _new_kprof()
        self._kp.start()

    def _kclose(self, key):
        if self._kp is None:
            return
        torch.cuda.synchronize()
        tp, self._kp = self._kp, None
        tp.stop()
        self._k[key] = _gpu_busy_ms(tp)

    def _win_open(self, _m, _args):
        if (self._live and not self._cap and self.kernel_acct
                and self._kp is None and "frame" not in self._k):
            self._kopen()

    def _win_close(self, _m, _args, _out):
        if self._live and not self._cap and "frame" not in self._k:
            self._kclose("frame")

    def _frame_end(self, _m, _args, _out):
        if not self._live:
            return
        # no seg_head on this model: close k_frame here (loss included)
        if "frame" not in self._k:
            self._kclose("frame")
        if self.kernel_acct and self.policy is not None:
            self._kopen()               # k_commit: the gates fire next

    def _frame_post(self, _m, inputs, output):
        if not self._live:
            self._fi += 1
            return
        rec = {"frame": self._fi}
        try:
            self._kclose("commit")
            # eager kernel sums first, so a failed capture still reports
            # whether the profiler sees device events
            kf = (self._k.get("frame") or (None, 0))[0]
            kc = ((self._k.get("commit") or (None, 0))[0]
                  if self.policy is not None else 0.0)
            rec["k_frame_ms"] = None if kf is None else round(kf, 4)
            rec["k_commit_ms"] = None if kc is None else round(kc, 4)
            for h in self._rec_handles:
                h.remove()
            self._rec_handles.clear()
            self._uninstall()
            if self.mode == "piecewise":
                self._rng.rewind()
                g = self._capture_piecewise(inputs[0], output)
            else:
                g = self._capture(inputs[0], output)
            rec.update(g)
            kg = g.get("k_graph_ms")
            if self.mode == "piecewise":
                kp = g.get("k_piecewise_ms")
                # coverage ~1: the chain ran every kernel the eager frame
                # ran (slightly above 1 from the host-step copy_ writes)
                if kf and kp is not None:
                    rec["coverage"] = round(kp / kf, 4)
                if self.logger:
                    par = rec.get("parity") or {}
                    self.logger.info(
                        "graph_capture[piecewise]: frame=%d %.3f ms "
                        "(wall %.3f) graphs=%d host=%d %s coverage=%s "
                        "agree=%s", self._fi, g["piecewise_ms"],
                        g["piecewise_wall_ms"], g["n_graphs"], g["n_host"],
                        g["host_kinds"], rec.get("coverage"),
                        par.get("argmax_agree"))
            elif kf is not None and kg is not None:
                exc_ms = kf - kg
                rec["excluded_ms"] = round(exc_ms, 4)
                # negative: the graph ran more than the eager frame (a
                # capture-only kernel); keep and flag the raw value, but
                # never let it discount graph_ms
                if exc_ms < 0:
                    rec["excluded_negative"] = True
                rec["fair_ms"] = round(g["graph_ms"] + max(exc_ms, 0.0), 4)
                rec["fair_commit_ms"] = round(
                    rec["fair_ms"] + (kc or 0.0), 4)
            if self.logger and self.mode != "piecewise":
                par = rec.get("parity") or {}
                self.logger.info(
                    "graph_capture: frame=%d graph=%.3f ms k_frame=%s "
                    "k_graph=%s fair=%s agree=%s", self._fi,
                    rec["graph_ms"], rec.get("k_frame_ms"), kg,
                    rec.get("fair_ms"), par.get("argmax_agree"))
        except Exception as exc:          # never break the eval
            rec["error"] = repr(exc)
            if self.logger:
                self.logger.warning(
                    "graph_capture failed at frame %d: %r dbg=%s",
                    self._fi, exc, self._dbg, exc_info=True)
        finally:
            if self._kp is not None:      # a failure left a window open
                try:
                    self._kp.stop()
                except Exception:
                    pass
                self._kp = None
            for h in self._rec_handles:
                h.remove()
            if self._rng is not None:
                self._rng.uninstall()
            try:
                self._uninstall()         # idempotent; eval must run clean
            except Exception:
                pass
            self.frames.append(rec)
            self.result = self._summary()
            self._reset_pick_state()
            self._fi += 1

    def _summary(self):
        """Aggregate over picks: mean and median across frames (per-frame
        `graph_ms` is itself the median over replays)."""
        ok = [f for f in self.frames
              if "graph_ms" in f or "piecewise_ms" in f]
        s = {"mode": self.mode, "picks": self.picks, "n_ok": len(ok),
             "n_failed": len(self.frames) - len(ok),
             "replays": self.replays, "kernel_acct": self.kernel_acct,
             "host": self.host, "device": self.device}
        for k in ("graph_ms", "k_frame_ms", "k_graph_ms", "excluded_ms",
                  "fair_ms", "k_commit_ms", "fair_commit_ms",
                  "piecewise_ms", "piecewise_wall_ms", "k_piecewise_ms",
                  "host_idle_ms", "n_graphs", "n_host", "coverage"):
            v = [f[k] for f in ok if f.get(k) is not None]
            p50 = (k[:-3] + "_p50_ms") if k.endswith("_ms") else k + "_p50"
            s[k] = round(sum(v) / len(v), 4) if v else None
            s[p50] = round(_median(v), 4) if v else None
        ag = [f["parity"]["argmax_agree"] for f in ok
              if (f.get("parity") or {}).get("argmax_agree") is not None]
        s["argmax_agree_min"] = min(ag) if ag else None
        s["n_excluded_negative"] = sum(
            1 for f in ok if f.get("excluded_negative"))
        if len(self.picks) == 1 and ok:
            # single pick: the frame's fields also at the top level
            for k in ("frame", "n_input_voxels", "enc3_subset", "levels",
                      "parity"):
                s[k] = ok[0].get(k)
        if self.frames and not ok:
            s["error"] = self.frames[-1].get("error")
        s["frames"] = self.frames
        return s

    # ------------------------------------------------ record pass
    def _install_recorders(self):
        """Patch every host-decided site to record its value."""
        O = self._origs
        rep = self.rep

        O["serialization"] = Point.serialization
        O["sparsify"] = Point.sparsify

        def rec_ser(pself, order="z", depth=None, shuffle_orders=False):
            r = O["serialization"](pself, order=order, depth=depth,
                                   shuffle_orders=shuffle_orders)
            self.sers.append((pself["serialized_code"],
                              pself["serialized_order"],
                              pself["serialized_inverse"],
                              pself["serialized_depth"],
                              pself["grid_coord"]))
            return r

        def rec_spar(pself, pad=96):
            r = O["sparsify"](pself, pad=pad)
            sct = pself["sparse_conv_feat"]
            self.sparse.append(
                (list(pself["sparse_shape"]), int(sct.batch_size),
                 sct.indices))
            return r

        Point.serialization = rec_ser
        Point.sparsify = rec_spar

        # keyed recorders for dynamic-shape ops and RNG draws. The Tensor
        # methods are patched separately: they do not route through
        # torch.nonzero / torch.unique. RNG output is replayed so the
        # capture matches the eager frame (randperm drives serialization's
        # shuffle_orders; an in-graph draw would freeze at capture anyway,
        # and torch.randint fails on the generator's capture bookkeeping).
        O["unique"] = torch.unique
        O["nonzero"] = torch.nonzero
        O["tnonzero"] = torch.Tensor.nonzero
        O["tunique"] = getattr(torch.Tensor, "unique", None)
        O["arange"] = torch.arange
        O["tensor"] = torch.tensor
        O["randperm"] = torch.randperm
        O["randint"] = torch.randint
        O["rand"] = torch.rand
        O["randn"] = torch.randn
        O["bincount"] = torch.bincount

        def rec_unique(*a, **k):
            return rep.record(_caller_key("unique"),
                              O["unique"](*a, **k))

        def rec_nonzero(*a, **k):
            return rep.record(_caller_key("nonzero"),
                              O["nonzero"](*a, **k))

        def rec_tnonzero(tself, *a, **k):
            return rep.record(_caller_key("tnonzero"),
                              O["tnonzero"](tself, *a, **k))

        def rec_tunique(tself, *a, **k):
            return rep.record(_caller_key("tunique"),
                              O["tunique"](tself, *a, **k))

        def rec_arange(*a, **k):
            out = O["arange"](*a, **k)
            if any(torch.is_tensor(x) for x in a):
                rep.record(_caller_key("arange"), out)
            return out

        def rec_tensor(*a, **k):
            out = O["tensor"](*a, **k)
            if out.is_cuda:
                rep.record(_caller_key("tensor"), out)
            return out

        def rec_rng(name):
            def rec(*a, **k):
                return rep.record(_caller_key(name), O[name](*a, **k))
            return rec

        torch.unique = rec_unique
        torch.nonzero = rec_nonzero
        torch.Tensor.nonzero = rec_tnonzero
        if O["tunique"] is not None:
            torch.Tensor.unique = rec_tunique
        torch.arange = rec_arange
        torch.tensor = rec_tensor
        torch.randperm = rec_rng("randperm")
        torch.randint = rec_rng("randint")
        torch.rand = rec_rng("rand")
        torch.randn = rec_rng("randn")
        torch.bincount = rec_rng("bincount")

        # weighted GitMerge3D merge: `cal_score`'s `p` drives Python control
        # flow (`if p >= 0.5`, `math.ceil(B - B*p)`), so record it as a host
        # float; `entropy` stays a tensor for the in-graph argsort.
        from spacecache.common.tome_attention import _ToMeForwardMixin
        O["cal_score"] = _ToMeForwardMixin.cal_score

        def rec_cal_score(saself, q, k, threshold=0.9):
            e, p = O["cal_score"](saself, q, k, threshold)
            rep.record(("cal_score",), (e, float(p)))
            return e, p
        _ToMeForwardMixin.cal_score = rec_cal_score

        # merge-index producers (tome_patch_indices, clap_*, gbm_*): replay
        # the recorded (src, dst) decision. Recomputed decisions can flip
        # near-tie argsorts between passes and give merge indices that do
        # not resolve (index_add_ device assert). Their matching kernels are
        # therefore not in the graph; excluded_ms accounts for them.
        import spacecache.policies.tome as _tome_mod
        try:
            import spacecache.policies.algm as _algm_mod
        except ImportError:
            _algm_mod = None

        def rec_idx(mod, name):
            orig = getattr(mod, name)
            O[f"idxfn::{name}"] = orig

            def rec(*a, **k):
                out = orig(*a, **k)
                rep.record(("idxfn", name),
                           tuple(out) if isinstance(out, tuple) else out)
                return out
            return rec

        setattr(_tome_mod, "tome_patch_indices",
                rec_idx(_tome_mod, "tome_patch_indices"))
        if _algm_mod is not None:
            for _n in ("clap_grid_indices", "clap_window_indices",
                       "gbm_indices"):
                setattr(_algm_mod, _n, rec_idx(_algm_mod, _n))

        # offset<->batch conversions hide dynamic-size ops
        O["offset2batch"] = _structure.offset2batch
        O["batch2offset"] = _structure.batch2offset
        _structure.offset2batch = lambda off: rep.record(
            ("o2b",), O["offset2batch"](off))
        _structure.batch2offset = lambda b: rep.record(
            ("b2o",), O["batch2offset"](b))

        # get_padding_and_inverse records its (pad, unpad, cu_seqlens)
        from pointcept.models.point_transformer_v3. \
            point_transformer_v3m3_utonia import SerializedAttention
        O["padinv"] = SerializedAttention.get_padding_and_inverse

        def rec_padinv(saself, point):
            r = O["padinv"](saself, point)
            self.padinv.append(r)
            return r
        SerializedAttention.get_padding_and_inverse = rec_padinv

        # LitePT's get_padding_and_inverse slices and branches on device
        # values per batch (implicit syncs). It shares the padinv FIFO: the
        # call order across both attention classes is deterministic.
        try:
            from pointcept.models.litept.litept_v1 import (
                PointROPEAttention, PointROPE)
        except Exception:
            PointROPEAttention = PointROPE = None
        if PointROPEAttention is not None:
            O["lpadinv"] = PointROPEAttention.get_padding_and_inverse

            def rec_lpadinv(saself, point):
                r = O["lpadinv"](saself, point)
                self.padinv.append(r)
                return r
            PointROPEAttention.get_padding_and_inverse = rec_lpadinv

            # the pure-PyTorch PointROPE fallback syncs on
            # int(positions.max()); the CUDA version takes no max_seqlen
            # and needs no patch
            if "max_seqlen" in inspect.signature(
                    PointROPE.forward).parameters:
                O["lrope"] = PointROPE.forward

                def rec_lrope(rself, tokens, positions, max_seqlen=None):
                    if max_seqlen is None:
                        max_seqlen = int(positions.max())
                        self.rope_ms.append(max_seqlen)
                    return O["lrope"](rself, tokens, positions, max_seqlen)
                PointROPE.forward = rec_lrope

        # conv IndiceData recording (post-hook sees the output SCT's dict)
        if _HAS_SPCONV:
            for m in self.model.modules():
                if isinstance(m, spconv.SubMConv3d):
                    self._rec_handles.append(m.register_forward_hook(
                        self._conv_rec_hook))

        # streaming: snapshot the policy-realised tensors
        if self.policy is not None and getattr(self.policy, "DEPTH",
                                               None) is not None:
            self._lv = []
            self._install_stream_hooks(record=True)

    def _conv_rec_hook(self, m, _i, out):
        """Keep the conv's IndiceData from the output's indice_dict."""
        if self._cap or m.indice_key is None:
            return
        try:
            d = out.find_indice_pair(m.indice_key)
        except Exception:
            d = None
        if d is not None:
            self.convs[m] = d

    def _levels(self):
        """SpaceCache levels as (depth, cache_attr, frame_attr, stage_attr).

        The policy's stage (enc{DEPTH}), plus enc4 for spacecache_s34.
        """
        pol = self.policy
        lv = [(pol.DEPTH, "cache", "_frame", "stage")]
        if getattr(pol, "cache4", None) is not None:
            lv.append((pol.D4_DEPTH, "cache4", "_frame4", "stage4"))
        return lv

    def _install_stream_hooks(self, record):
        """SpaceCache hooks: snapshot the subset and the restored point on
        the record pass; substitute them (frozen restore) during capture."""
        pol = self.policy
        enc = self.model.backbone.enc._modules
        for li, (depth, cache_attr, fm_attr, stage_attr) in enumerate(
                self._levels()):
            stage = enc[f"enc{depth}"]
            n_blocks = getattr(
                getattr(pol, stage_attr, None), "n_blocks", None)
            last = (stage._modules.get(f"block{n_blocks - 1}")
                    if n_blocks else None)
            if last is None:
                last = stage[-1]
            down = stage._modules.get("down", stage)

            if record:
                self._lv.append(dict(sub=None, full=None, view=None,
                                     cf=None))

                def _down_pre(_m, args, li=li):
                    self._dbg[f"lv{li}_downin"] = (
                        id(args[0]), tuple(args[0].feat.shape))
                self._rec_handles.append(
                    down.register_forward_pre_hook(_down_pre))

                def down_hook(_m, _i, out, li=li):
                    # mask_gate already ran: `out` is the subset point (or
                    # the full pooled point on a cold frame)
                    if not self._cap:
                        self._lv[li]["sub"] = out
                        # blocks rebind (never mutate) `sub.feat`, so this
                        # reference stays the recorded input
                        self._lv[li]["sub_feat"] = \
                            out.feat if out is not None else None
                        self._lv[li]["sub_sct"] = getattr(
                            out, "sparse_conv_feat", None) \
                            if out is not None else None
                self._rec_handles.append(
                    down.register_forward_hook(down_hook))

                def last_hook(_m, _i, out, li=li, fa=fm_attr,
                              ca=cache_attr):
                    if self._cap:
                        return
                    self._lv[li]["full"] = out
                    # the decoder pops `pooling_parent` and rebinds `feat`
                    # in place on this point and on every ancestor, so
                    # snapshot the feats and the parent chain before it runs
                    self._lv[li]["full_feat"] = out.feat
                    self._lv[li]["full_parent"] = getattr(
                        out, "pooling_parent", None)
                    chain = []
                    p = (out.pooling_parent
                         if "pooling_parent" in out.keys() else None)
                    while p is not None:
                        nxt = (p.pooling_parent
                               if "pooling_parent" in p.keys() else None)
                        chain.append((p, p.feat, nxt))
                        p = nxt
                    self._lv[li]["chain"] = chain
                    self._dbg[f"lv{li}_rec"] = (
                        tuple(out.feat.shape), id(out),
                        id(getattr(out, "pooling_parent", None)),
                        getattr(out.pooling_parent, "feat", None)
                        is not None and tuple(
                            out.pooling_parent.feat.shape))
                    fm = getattr(self.policy, fa, None)
                    if fm:
                        self._lv[li]["view"] = fm.get("view")
                    self._lv[li]["cf"] = getattr(
                        getattr(self.policy, ca, None), "feat", None)
                self._rec_handles.append(
                    last.register_forward_hook(last_hook))
            else:
                lv = self._lv[li]
                sub, view = lv["sub"], lv["view"]
                full, cf = lv["full"], lv["cf"]
                full_feat = lv.get("full_feat")
                # debug only: reference copy of the recorded restored feats
                # (the replay splices into `full_feat` in place); the diffs
                # against it are computed inside the graph
                feat_ref = (full_feat.clone()
                            if full_feat is not None and self.debug
                            else None)
                sub_feat_ref = None
                chain = lv.get("chain") or []
                # d3d4: enc4's `down` pooled the previous level's `full`,
                # while the recorded pooling_parent is a copy; link directly
                prev_full = (self._lv[li - 1]["full"] if li > 0 else None)
                # resolve bool masks to indices here, outside the graph
                # (in-graph mask indexing calls nonzero), with the unpatched
                # Tensor.nonzero: nothing was recorded at this site
                tnz = self._origs.get("tnonzero") or torch.Tensor.nonzero
                hit_rows = hit_src = miss_rows = None
                if view is not None:
                    hit = view.hit
                    hit_rows = (tnz(hit, as_tuple=False).squeeze(-1)
                                if hit.dtype == torch.bool else hit)
                    hit_src = view.slot.index_select(0, hit_rows)
                    miss = view.miss_idx
                    miss_rows = (tnz(miss, as_tuple=False).squeeze(-1)
                                 if miss.dtype == torch.bool else miss)

                sub_feat = lv.get("sub_feat")
                sub_sct = lv.get("sub_sct")

                def down_hook(_m, _i, out, sub=sub, sub_feat=sub_feat,
                              sub_sct=sub_sct):
                    # blocks rebind `sub.feat` / `sub.sparse_conv_feat`
                    # every pass; restore the recorded inputs
                    if sub is not None:
                        if (sub_feat is not None
                                and sub.feat is not sub_feat):
                            sub.feat = sub_feat
                        if (sub_sct is not None
                                and sub.sparse_conv_feat is not sub_sct):
                            sub.sparse_conv_feat = sub_sct
                    return sub
                self._cap_handles.append(
                    down.register_forward_hook(down_hook))

                def last_hook(_m, _i, blk_out, li=li, sub=sub, full=full,
                              cf=cf, view=view, hit_rows=hit_rows,
                              hit_src=hit_src, miss_rows=miss_rows,
                              full_feat=full_feat, chain=chain,
                              prev_full=prev_full):
                    # frozen reuse_gate: hits from the cache, misses from
                    # blk_out; fixed-shape index ops only, so capturable
                    res = blk_out if sub is None else full
                    if sub is not None and full_feat is not None:
                        # `full` is the record pass's point, which the
                        # decoder mutated: restore its feat and its parent
                        # link (prev_full under d3d4, else the recorded
                        # ancestor chain with the ancestors' feats)
                        if res.feat is not full_feat:
                            res.feat = full_feat
                        if prev_full is not None:
                            res["pooling_parent"] = prev_full
                        elif chain:
                            node = res
                            for p, pfeat, _ppp in chain:
                                node["pooling_parent"] = p
                                if p.feat is not pfeat:
                                    p.feat = pfeat
                                node = p
                            # the deepest ancestor had no pooling_parent
                            # at record time
                            if "pooling_parent" in node.keys():
                                del node["pooling_parent"]
                    _pp = getattr(res, "pooling_parent", None)
                    self._dbg[f"lv{li}_rep"] = (
                        "full" if res is full else "blk_out",
                        tuple(res.feat.shape),
                        tuple(full_feat.shape)
                        if full_feat is not None else None,
                        tuple(_pp.feat.shape) if _pp is not None else None,
                        id(_pp),
                        "pp_res0" if self._lv
                        and _pp is self._lv[0].get("full") else "pp_oth")
                    try:
                        if (cf is not None and cf.shape[0] > 0
                                and hit_rows is not None
                                and hit_rows.numel()):
                            _hv = cf.index_select(0, hit_src).to(
                                res.feat.dtype)
                            res.feat.index_copy_(0, hit_rows, _hv)
                            if feat_ref is not None:
                                # keep the tensor: a host read would sync
                                # during capture; _capture reads it later
                                self._dbg[f"lv{li}_hitdiff"] = (
                                    _hv - feat_ref.index_select(
                                        0, hit_rows).to(_hv.dtype)
                                    ).abs().max()
                        if (sub is not None and miss_rows is not None
                                and miss_rows.numel()):
                            _mv = blk_out.feat.to(res.feat.dtype)
                            res.feat.index_copy_(0, miss_rows, _mv)
                            if feat_ref is not None:
                                self._dbg[f"lv{li}_missdiff"] = (
                                    _mv - feat_ref.index_select(
                                        0, miss_rows).to(_mv.dtype)
                                    ).abs().max()
                    except RuntimeError as e:
                        raise RuntimeError(
                            f"model_graph stream-restore diverged: "
                            f"level={li} res={tuple(res.feat.shape)} "
                            f"hit_rows={hit_rows.numel() if hit_rows is not None else -1} "
                            f"hit_src={hit_src.numel() if hit_src is not None else -1} "
                            f"cf={tuple(cf.shape) if cf is not None else None} "
                            f"miss_rows={miss_rows.numel() if miss_rows is not None else -1} "
                            f"blk_out={tuple(blk_out.feat.shape)} "
                            f"sub={tuple(sub.feat.shape) if sub is not None else None} "
                            f"full={tuple(full.feat.shape) if full is not None else None} "
                            f"view_keys={view.keys.numel() if view is not None else -1} "
                            f"({e})") from e
                    if "sparse_conv_feat" in res.keys():
                        res.sparse_conv_feat = \
                            res.sparse_conv_feat.replace_feature(
                                res.feat)
                    return res
                self._cap_handles.append(
                    last.register_forward_hook(last_hook))

    # ------------------------------------------------ capture pass
    def _uninstall(self):
        """Undo every patch and remove the capture hooks (idempotent)."""
        O = self._origs
        if O.get("serialization") is not None:
            Point.serialization = O["serialization"]
        if O.get("sparsify") is not None:
            Point.sparsify = O["sparsify"]
        for name in ("unique", "nonzero", "arange", "tensor", "randperm",
                     "randint", "rand", "randn", "bincount"):
            if O.get(name) is not None:
                setattr(torch, name, O[name])
        if O.get("tnonzero") is not None:
            torch.Tensor.nonzero = O["tnonzero"]
        if O.get("tunique") is not None:
            torch.Tensor.unique = O["tunique"]
        if (O.get("cal_score") is not None
                or O.get("tome_attn_fwd") is not None):
            from spacecache.common.tome_attention import _ToMeForwardMixin
            if O.get("cal_score") is not None:
                _ToMeForwardMixin.cal_score = O["cal_score"]
            if O.get("tome_attn_fwd") is not None:
                _ToMeForwardMixin.forward = O["tome_attn_fwd"]
        if O.get("size_attn_fwd") is not None:
            import spacecache.policies.tome as _tome_pol
            _tome_pol._size_attn_forward = O["size_attn_fwd"]
        for _k in list(O):
            if _k.startswith("idxfn::"):
                _name = _k.split("::", 1)[1]
                for _mod in (sys.modules.get("spacecache.policies.tome"),
                             sys.modules.get("spacecache.policies.algm")):
                    if (_mod is not None and hasattr(_mod, _name)
                            and getattr(_mod, _name) is not O[_k]):
                        setattr(_mod, _name, O[_k])
        if O.get("offset2batch") is not None:
            _structure.offset2batch = O["offset2batch"]
            _structure.batch2offset = O["batch2offset"]
        if O.get("pairgen") is not None:
            import spconv.pytorch.ops as _spops
            _spops.get_indice_pairs_implicit_gemm = O["pairgen"]
            _spops.get_indice_pairs = O["pairgen_native"]
        if O.get("padinv") is not None or O.get("attn_fwd") is not None:
            from pointcept.models.point_transformer_v3. \
                point_transformer_v3m3_utonia import SerializedAttention
            if O.get("padinv") is not None:
                SerializedAttention.get_padding_and_inverse = O["padinv"]
            if O.get("attn_fwd") is not None:
                SerializedAttention.forward = O["attn_fwd"]
        if (O.get("lpadinv") is not None or O.get("lrope") is not None
                or O.get("litept_attn_fwd") is not None):
            try:
                from pointcept.models.litept.litept_v1 import (
                    PointROPEAttention, PointROPE)
                if O.get("lpadinv") is not None:
                    PointROPEAttention.get_padding_and_inverse = (
                        O["lpadinv"])
                if O.get("lrope") is not None:
                    PointROPE.forward = O["lrope"]
            except Exception:
                pass
            if O.get("litept_attn_fwd") is not None:
                from spacecache.common.litept_tome_attention import (
                    _LitePTToMeForwardMixin)
                _LitePTToMeForwardMixin.forward = O["litept_attn_fwd"]
        for m in list(self.convs):
            if "forward" in m.__dict__:
                del m.__dict__["forward"]
        for h in self._cap_handles:
            h.remove()
        self._cap_handles.clear()

    def _install_replays(self):
        """Patch every recorded site to return its recorded value."""
        O = self._origs
        rep = self.rep
        ser_it = iter(self.sers)
        sparse_it = iter(self.sparse)
        padinv_it = iter(self.padinv)

        def cap_ser(pself, order="z", depth=None, shuffle_orders=False):
            # replay the recorded serialization: its encode/argsort kernels
            # stay out of the graph, which also avoids the encoder's H2D
            # copies and keeps randperm consistent
            code, o, inv, d, gc = next(ser_it)
            pself["order"] = order
            if "grid_coord" not in pself.keys():
                pself["grid_coord"] = gc
            pself["serialized_code"] = code
            pself["serialized_order"] = o
            pself["serialized_inverse"] = inv
            pself["serialized_depth"] = d

        # recorded IndiceData keyed by the recorded indices tensor: a
        # SparseConvTensor built at capture starts with an empty indice_dict,
        # so seed it and a conv without a frozen forward finds its pairs
        # instead of running pair generation (device sort + host sync,
        # illegal during capture)
        seeds = {}
        for _m, _d in self.convs.items():
            _di = getattr(_d, "indices", None)
            _k = getattr(_m, "indice_key", None)
            if _di is not None and _k is not None:
                seeds.setdefault(id(_di), {})[_k] = _d

        def cap_spar(pself, pad=96):
            ss, bs, idx = next(sparse_it)
            if "grid_coord" not in pself.keys():
                pself["grid_coord"] = torch.div(
                    pself.coord - pself.coord.min(0)[0],
                    pself.grid_size, rounding_mode="trunc").int()
            pself["sparse_shape"] = ss
            sct = spconv.SparseConvTensor(
                features=pself.feat, indices=idx,
                spatial_shape=ss, batch_size=bs)
            sm = seeds.get(id(idx))
            if sm:
                sct.indice_dict.update(sm)
            pself["sparse_conv_feat"] = sct

        Point.serialization = cap_ser
        Point.sparsify = cap_spar

        def cap_unique(*a, **k):
            return rep.replay(_caller_key("unique"))

        def cap_nonzero(*a, **k):
            return rep.replay(_caller_key("nonzero"))

        def cap_arange(*a, **k):
            if any(torch.is_tensor(x) for x in a):
                return rep.replay(_caller_key("arange"))
            return O["arange"](*a, **k)

        def cap_tensor(*a, **k):
            dev = k.get("device")
            if dev is not None and torch.device(dev).type == "cuda":
                return rep.replay(_caller_key("tensor"))
            return O["tensor"](*a, **k)

        def cap_tnz(tself, *a, **k):
            return rep.replay(_caller_key("tnonzero"))

        def cap_tunique(tself, *a, **k):
            return rep.replay(_caller_key("tunique"))

        def cap_rng(name):
            def cap(*a, **k):
                key = _caller_key(name)
                out = rep.replay(key)
                # a shape mismatch means the capture pass took another
                # branch than the record pass; name the site here instead
                # of failing later in a kernel
                req = k.get("size")
                if req is None and a:
                    if isinstance(a[-1], (tuple, list)):
                        req = a[-1]
                    elif all(isinstance(x, int) for x in a):
                        req = a
                try:
                    if req is not None and torch.is_tensor(out) \
                            and not torch.is_tensor(req) \
                            and tuple(out.shape) != tuple(req):
                        raise RuntimeError(
                            f"model_graph: RNG replay divergence at {key}: "
                            f"recorded {tuple(out.shape)} vs live request "
                            f"{tuple(req)} — a caller-side branch flipped "
                            f"between record and replay")
                except TypeError:
                    pass
                return out
            return cap

        torch.unique = cap_unique
        torch.nonzero = cap_nonzero
        torch.Tensor.nonzero = cap_tnz
        if O.get("tunique") is not None:
            torch.Tensor.unique = cap_tunique
        torch.arange = cap_arange
        torch.tensor = cap_tensor
        torch.randperm = cap_rng("randperm")
        torch.randint = cap_rng("randint")
        torch.rand = cap_rng("rand")
        torch.randn = cap_rng("randn")
        torch.bincount = cap_rng("bincount")
        _structure.offset2batch = lambda off: rep.replay(("o2b",))
        _structure.batch2offset = lambda b: rep.replay(("b2o",))

        # pair generation cannot be captured (device sort + host copy):
        # fail with the conv whose indice_dict lookup missed
        import spconv.pytorch.ops as _spops

        def _pairgen_guard(_orig):
            def _g(indices, *a, **k):
                key, have = "?", "?"
                fr = inspect.currentframe()
                try:
                    f = fr.f_back
                    while f is not None:
                        self_ = f.f_locals.get("self")
                        if self_ is not None and hasattr(
                                self_, "indice_key"):
                            key = self_.indice_key
                            inp = f.f_locals.get("input")
                            have = (list(inp.indice_dict)
                                    if inp is not None else "<no input>")
                            break
                        f = f.f_back
                finally:
                    del fr
                raise RuntimeError(
                    "model_graph: real get_indice_pairs during capture — "
                    f"conv indice_key={key!r}, input.indice_dict keys="
                    f"{have}, indices.shape={tuple(indices.shape)} — the "
                    "recorded indice_dict seeds missed this layout")
            return _g

        if O.get("pairgen") is None:
            O["pairgen"] = _spops.get_indice_pairs_implicit_gemm
            O["pairgen_native"] = _spops.get_indice_pairs
        _spops.get_indice_pairs_implicit_gemm = _pairgen_guard(
            O["pairgen"])
        _spops.get_indice_pairs = _pairgen_guard(O["pairgen_native"])

        # merge-index producers return the recorded decision (see
        # _install_recorders)
        for _k in list(O):
            if _k.startswith("idxfn::"):
                _name = _k.split("::", 1)[1]
                for _mod in (sys.modules.get("spacecache.policies.tome"),
                             sys.modules.get("spacecache.policies.algm")):
                    if _mod is not None and hasattr(_mod, _name):
                        setattr(_mod, _name,
                                (lambda _n: lambda *a, **k:
                                 rep.replay(("idxfn", _n)))(_name))

        from pointcept.models.point_transformer_v3. \
            point_transformer_v3m3_utonia import SerializedAttention

        def cap_padinv(saself, point):
            return next(padinv_it)
        SerializedAttention.get_padding_and_inverse = cap_padinv

        # LitePT: same padinv FIFO; the PointROPE fallback gets the
        # recorded max_seqlen, so int(positions.max()) never runs in-graph
        try:
            from pointcept.models.litept.litept_v1 import (
                PointROPEAttention, PointROPE)
        except Exception:
            PointROPEAttention = PointROPE = None
        if PointROPEAttention is not None:
            if "lpadinv" not in O:
                O["lpadinv"] = PointROPEAttention.get_padding_and_inverse

            def cap_lpadinv(saself, point):
                return next(padinv_it)
            PointROPEAttention.get_padding_and_inverse = cap_lpadinv

            if "max_seqlen" in inspect.signature(
                    PointROPE.forward).parameters:
                if "lrope" not in O:
                    O["lrope"] = PointROPE.forward
                rope_ms_it = iter(self.rope_ms)

                def cap_lrope(rself, tokens, positions, max_seqlen=None):
                    return O["lrope"](
                        rself, tokens, positions,
                        next(rope_ms_it) if max_seqlen is None
                        else max_seqlen)
                PointROPE.forward = cap_lrope

        if "attn_fwd" not in O:
            O["attn_fwd"] = SerializedAttention.forward

        def cap_attn_fwd(saself, point):
            if (not saself.enable_flash
                    and "min_bincount" not in point.keys()):
                # bs=1: offset2bincount(offset).min() == N, known from the
                # shape without the `.min()` sync
                point["min_bincount"] = int(point.feat.shape[0])
            return O["attn_fwd"](saself, point)
        SerializedAttention.forward = cap_attn_fwd

        # the merge attentions bypass SerializedAttention.forward (the
        # `tome_attn` __class__ reparenting, or the ToMe policy's
        # `_size_attn_forward`): inject the same bs=1 min_bincount there,
        # and replay cal_score's `p` as a host float so its branches do not
        # sync
        from spacecache.common.tome_attention import _ToMeForwardMixin
        if "tome_attn_fwd" not in O:
            O["tome_attn_fwd"] = _ToMeForwardMixin.forward

        def cap_tome_fwd(saself, point):
            if "min_bincount" not in point.keys():
                point["min_bincount"] = int(point.feat.shape[0])
            return O["tome_attn_fwd"](saself, point)
        _ToMeForwardMixin.forward = cap_tome_fwd

        if "cal_score" not in O:
            O["cal_score"] = _ToMeForwardMixin.cal_score

        def cap_cal_score(saself, q, k, threshold=0.9):
            return rep.replay(("cal_score",))
        _ToMeForwardMixin.cal_score = cap_cal_score

        # the LitePT mixin overrides forward (so cap_tome_fwd never runs
        # for it) but inherits cal_score
        from spacecache.common.litept_tome_attention import (
            _LitePTToMeForwardMixin)
        if "litept_attn_fwd" not in O:
            O["litept_attn_fwd"] = _LitePTToMeForwardMixin.forward

        def cap_litept_fwd(saself, point):
            if "min_bincount" not in point.keys():
                point["min_bincount"] = int(point.feat.shape[0])
            return O["litept_attn_fwd"](saself, point)
        _LitePTToMeForwardMixin.forward = cap_litept_fwd

        import spacecache.policies.tome as _tome_pol
        if "size_attn_fwd" not in O:
            O["size_attn_fwd"] = _tome_pol._size_attn_forward

        def cap_size_attn(attn, point):
            if "min_bincount" not in point.keys():
                point["min_bincount"] = int(point.feat.shape[0])
            return O["size_attn_fwd"](attn, point)
        _tome_pol._size_attn_forward = cap_size_attn

        # conv bypass: run the recorded IndiceData through the conv kernel
        # directly, without the pair-generation syncs
        if _HAS_SPCONV:
            for m, datas in self.convs.items():
                m.forward = self._frozen_conv(m, datas)

        if (self.policy is not None
                and getattr(self.policy, "DEPTH", None) is not None):
            self._install_stream_hooks(record=False)

        # diagnostics (logged on failure): the shapes each GridUnpooling sees
        from pointcept.models.point_transformer_v3. \
            point_transformer_v3m3_utonia import GridUnpooling as _GU
        for _i, _m in enumerate(
                mod for mod in self.model.modules()
                if isinstance(mod, _GU)):
            def _gu_pre(_mod, _args, _i=_i):
                _p = _args[0]
                _pp = getattr(_p, "pooling_parent", None)
                self._dbg[f"unpool{_i}"] = (
                    tuple(_p.feat.shape),
                    tuple(_pp.feat.shape) if _pp is not None else None)
            self._cap_handles.append(
                _m.register_forward_pre_hook(_gu_pre))

        # diagnostics: level-0 restored feat shape at every enc4 module
        if self._lv and self._lv[0].get("full") is not None:
            _r0 = self._lv[0]["full"]
            _enc4 = self.model.backbone.enc._modules.get("enc4")
            if _enc4 is not None:
                for _name, _m in _enc4._modules.items():
                    def _watch(_mod, _args, _n=_name, _r=_r0):
                        self._dbg[f"enc4.{_n}"] = tuple(_r.feat.shape)
                    self._cap_handles.append(
                        _m.register_forward_pre_hook(_watch))
                    for _n2, _m2 in getattr(_m, "_modules", {}).items():
                        def _watch2(_mod, _args, _n=_name + "." + _n2,
                                    _r=_r0):
                            self._dbg[f"enc4.{_n}"] = tuple(_r.feat.shape)
                        self._cap_handles.append(
                            _m2.register_forward_pre_hook(_watch2))

    def _frozen_conv(self, conv, datas):
        """Conv forward on the recorded IndiceData (no pair generation)."""
        if isinstance(datas, ImplicitGemmIndiceData):
            def fwd(sct):
                out, _, _ = sp_ops.implicit_gemm(
                    sct.features, conv.weight, datas.pair_fwd,
                    datas.pair_mask_fwd_splits,
                    datas.mask_argsort_fwd_splits,
                    datas.out_indices.shape[0], datas.masks,
                    False, True, getattr(sct, "_timer", None),
                    conv.fp32_accum, conv.bias, conv.act_alpha,
                    conv.act_beta, conv.act_type, 1.0, None,
                    output_dtype=conv.weight.dtype)
                nxt = sct.replace_feature(out)
                # indice_dict is shared across replace_feature: expose the
                # recorded pairs so later lookups under this key hit
                nxt.indice_dict[conv.indice_key] = datas
                return nxt
        elif isinstance(datas, IndiceData):
            def fwd(sct):
                pairs = datas.indice_pairs
                if pairs.device != sct.features.device:
                    pairs = pairs.to(sct.features.device)
                out = sp_fsp.indice_subm_conv(
                    sct.features, conv.weight, pairs,
                    datas.indice_pair_num, datas.out_indices.shape[0],
                    datas.algo, getattr(sct, "_timer", None), conv.bias,
                    conv.act_alpha, conv.act_beta, conv.act_type)
                nxt = sct.replace_feature(out)
                nxt.indice_dict[conv.indice_key] = datas
                return nxt
        else:
            raise RuntimeError(
                f"model_graph: unrecognised IndiceData type "
                f"{type(datas)} on {conv.indice_key}")
        return fwd

    # ------------------------------------------------ the bench
    def _compute(self, d):
        """The captured callable: backbone, unpooling and seg_head."""
        model = self.model
        point = model.backbone(d)
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat(
                [parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return model.seg_head(point.feat)

    def _static_input(self, input_dict):
        """Private copy of the frame's input with both batch and offset.

        The segmentor derives the missing one before the backbone, i.e.
        outside both the k_frame window and `_compute`; filling both here
        keeps the captured boundary equal to k_frame's.
        """
        dev = next(self.model.parameters()).device
        static = {}
        for k, v in input_dict.items():
            static[k] = v.clone() if torch.is_tensor(v) else v
        if "offset" in static and "batch" not in static:
            static["batch"] = torch.repeat_interleave(
                torch.arange(len(static["offset"]), device=dev),
                torch.diff(static["offset"],
                           prepend=torch.zeros(
                               1, dtype=torch.long, device=dev)))
        if "batch" in static and "offset" not in static:
            static["offset"] = torch.cumsum(
                static["batch"].bincount(), dim=0).long()
        return static

    @staticmethod
    def _parity(out, eager_out):
        """max/mean |dlogit| and argmax agreement vs the eager frame."""
        if eager_out is None or out is None:
            return None
        eo = eager_out
        if isinstance(eo, dict):
            eo = eo.get("seg_logits", eo.get("logits"))
        elif isinstance(eo, (tuple, list)):
            eo = eo[0] if eo else None
        if not (torch.is_tensor(eo) and eo.shape == out.shape):
            return None
        d = (out.float() - eo.float()).abs()
        agree = (out.argmax(-1) == eo.argmax(-1)).float().mean()
        return {"max_abs_diff": round(float(d.max()), 6),
                "mean_abs_diff": round(float(d.mean()), 8),
                "argmax_agree": round(float(agree), 6)}

    def _capture_piecewise(self, input_dict, eager_out=None):
        """Capture the frame's whole inference as a graph chain split at
        the host steps and time the chain end to end.

        A streaming policy runs armed (lookup, subset and restore are part
        of the inference) against the cache as it stood at frame start: the
        commit gates have already run, and the post-commit table would hit
        this frame's own cells. The policy and cache attributes are restored
        afterwards, so the next frame sees the post-commit state.
        """
        import time
        from spacecache.common import piecewise_graph as pg
        from spacecache.common import tome_attention as _ta

        pg.check_extension_streams()
        static = self._static_input(input_dict)
        for mod in (self.prof, self.metrics):
            if mod is not None:
                mod._active = False
        pol = self.policy
        caches = self._caches()
        saved = ([(o, dict(vars(o))) for o in [pol] + caches]
                 if pol is not None else [])
        n_wp = len(_ta.WPATCH_STATS)

        def enter_state():
            self._rng.rewind()
            if pol is None:
                return
            for c in caches:
                c.__dict__.update(self._pre_snap.get(id(c), {}))
            pol.armed = True
            pol.measure_breakdown = False     # no timing events in capture
            for a in ("_frame", "_frame4"):
                if hasattr(pol, a):
                    setattr(pol, a, None)

        pw = None
        try:
            self._cap = True
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                # lazy inits (cuBLAS, spconv tuner, LUT caches) happen here,
                # never inside a capture
                rerun = None
                for i in range(max(1, self.warmup)):
                    enter_state()
                    o = self._compute(static)
                    if i == 0:
                        rerun = o
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            # parity floor: the same frame re-run eagerly with the same
            # state and taped draws; nondeterministic kernels (atomic
            # merges) alone can move it below 1
            ref_rerun = self._parity(rerun, eager_out)
            del rerun, o

            pw = pg.Piecewise()
            enter_state()
            with torch.cuda.stream(s):
                with pg.boundaries(pw):
                    pw.begin()
                    try:
                        out = self._compute(static)
                    except Exception:
                        pw.abort()
                        raise
                    pw.end()                  # last segment, run once
            torch.cuda.synchronize()

            levels = None
            if pol is not None:
                levels = []
                for a in ("_frame", "_frame4"):
                    fm = getattr(pol, a, None)
                    if not fm:
                        continue
                    # the reuse gate drops fm["subset"], so the subset size
                    # is the miss count of the view
                    view = fm.get("view")
                    if view is None:
                        continue
                    levels.append({
                        "subset": int(view.miss_idx.numel()),
                        "n_hit": int(view.hit.sum())})

            with torch.cuda.stream(s):
                for _ in range(3):
                    pw.replay()
                torch.cuda.synchronize()
                ref = self._parity(out, eager_out)

                # each replay starts on an idle GPU, as a new frame does
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                per, wall = [], []
                for _ in range(max(1, self.replays)):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    e0.record()
                    pw.replay()
                    e1.record()
                    torch.cuda.synchronize()
                    wall.append((time.perf_counter() - t0) * 1000.0)
                    per.append(e0.elapsed_time(e1))

                k_pw = None
                if self.kernel_acct:
                    tp = _new_kprof()
                    torch.cuda.synchronize()
                    tp.start()
                    pw.replay()
                    torch.cuda.synchronize()
                    tp.stop()
                    k_pw = _gpu_busy_ms(tp)[0]

            ms = _median(per)
            return {
                "piecewise_ms": round(ms, 4),
                "piecewise_mean_ms": round(sum(per) / len(per), 4),
                "piecewise_wall_ms": round(_median(wall), 4),
                "k_piecewise_ms": (None if k_pw is None
                                   else round(k_pw, 4)),
                # GPU idle inside the chain: host steps + graph launches
                "host_idle_ms": (None if k_pw is None
                                 else round(ms - k_pw, 4)),
                "n_graphs": pw.n_graphs,
                "n_host": pw.n_host,
                "host_kinds": dict(pw.kinds),
                "host_ops": dict(pw.ops.most_common(12)),
                "rng_miss": self._rng.miss,
                "host_drift": pw.drift,
                "frame": self._fi,
                "n_input_voxels": (int(static["coord"].shape[0])
                                   if "coord" in static else None),
                "levels": levels or None,
                "parity": ref,
                "parity_rerun": ref_rerun,
            }
        finally:
            self._cap = False
            if pw is not None:
                pw.abort()
                pw.steps.clear()              # frees the graphs + the pool
            for o, d in saved:
                o.__dict__.clear()
                o.__dict__.update(d)
            del _ta.WPATCH_STATS[n_wp:]       # drop warmup/capture entries
            for mod in (self.prof, self.metrics):
                if mod is not None:
                    mod._active = True

    def _capture(self, input_dict, eager_out=None):
        """Capture backbone + seg_head as one graph (replay mode) and time
        its replays."""
        static = self._static_input(input_dict)

        for mod in (self.prof, self.metrics):
            if mod is not None:
                mod._active = False
        if self.policy is not None:
            self._armed = getattr(self.policy, "armed", False)
            self.policy.armed = False

        try:
            self._cap = True
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(self.warmup):
                    self._replay_reset()
                    self._compute(static)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            self._replay_reset()
            with torch.cuda.graph(g):
                out = self._compute(static)
            torch.cuda.synchronize()

            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            # one event pair per replay: the frame's value is the median,
            # so a single stalled replay cannot move it
            ev = [(torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True))
                  for _ in range(max(1, self.replays))]
            for e0, e1 in ev:
                e0.record()
                g.replay()
                e1.record()
            torch.cuda.synchronize()
            per = [e0.elapsed_time(e1) for e0, e1 in ev]
            ms = _median(per)

            # k_graph: kernel sum of one replay, measured after the timed
            # replays so the profiler never perturbs graph_ms
            k_graph = None
            if self.kernel_acct:
                tp = _new_kprof()
                tp.start()
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                tp.stop()
                kb, _n = _gpu_busy_ms(tp)
                k_graph = None if kb is None else kb / 3.0

            # parity vs the eager output of the record pass (same input and
            # state); `out` holds the last replay's logits. argmax agreement
            # is the gate, since max|dlogit| is set by a single token.
            ref = None
            if eager_out is not None:
                eo = eager_out
                if isinstance(eo, dict):
                    eo = eo.get("seg_logits", eo.get("logits"))
                elif isinstance(eo, (tuple, list)):
                    eo = eo[0] if eo else None
                if torch.is_tensor(eo) and eo.shape == out.shape:
                    d = (out.float() - eo.float()).abs()
                    agree = (out.argmax(-1) == eo.argmax(-1)).float().mean()
                    ref = {"max_abs_diff": round(float(d.max()), 6),
                           "mean_abs_diff": round(float(d.mean()), 8),
                           "argmax_agree": round(float(agree), 6)}
            result = {
                "graph_ms": round(ms, 4),
                "graph_mean_ms": round(sum(per) / len(per), 4),
                "graph_min_ms": round(min(per), 4),
                "k_graph_ms": (None if k_graph is None
                               else round(k_graph, 4)),
                "frame": self._fi,
                "n_input_voxels": (int(static["coord"].shape[0])
                                   if "coord" in static else None),
                "enc3_subset": (int(self._lv[0]["sub"].feat.shape[0])
                                if self._lv and self._lv[0]["sub"]
                                is not None else None),
                "levels": [
                    {"depth": d,
                     "subset": (int(lv["sub"].feat.shape[0])
                                if lv["sub"] is not None else None),
                     "n_hit": (int(lv["view"].hit.sum())
                               if lv["view"] is not None else None)}
                    for (d, _c, _f, _s), lv in
                    zip(self._levels(), self._lv)
                ] if self._lv else None,
                "parity": ref,
            }
            if self.debug:
                # the _dbg tensors were computed inside the graph and hold
                # the last replay's values
                dbg = {}
                for _k, _v in self._dbg.items():
                    dbg[_k] = (round(float(_v), 6)
                               if torch.is_tensor(_v) and _v.numel() == 1
                               else _v)
                result["dbg"] = dbg
            return result
        finally:
            self._cap = False
            self._uninstall()
            for mod in (self.prof, self.metrics):
                if mod is not None:
                    mod._active = True
            if self.policy is not None:
                self.policy.armed = self._armed

    def _replay_reset(self):
        # per-forward replay state: fresh FIFO iterators + site cursors
        self.rep.reset()
        self._uninstall()
        self._install_replays()

    def close(self):
        for h in self._handles + self._rec_handles:
            h.remove()
        self._handles.clear()
        self._rec_handles.clear()


def attach(model, pick_frame=None, policy=None, replays=200, warmup=3,
           prof=None, metrics=None, logger=None, pick_frames=None,
           kernel_acct=True, debug=False, mode="replay"):
    """Register a `ModelGraphSession` on `model`.

    `pick_frames` (list) overrides `pick_frame`. `mode` is "replay" (one
    graph, host-decided values from the record, plus kernel accounting) or
    "piecewise" (a graph chain split at every host step, see
    spacecache/common/piecewise_graph.py).
    """
    return ModelGraphSession(model, pick_frame, policy=policy,
                             replays=replays, warmup=warmup,
                             prof=prof, metrics=metrics, logger=logger,
                             pick_frames=pick_frames,
                             kernel_acct=kernel_acct, debug=debug,
                             mode=mode)
