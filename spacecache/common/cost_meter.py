"""Per-frame FLOPs, token and peak-memory accounting (`CostMeter`).

FLOPs come from the shapes each forward actually produced: nn.Linear
(2 * rows * in * out), sparse convolutions (2 * gathered pairs * Cin * Cout),
attention (QK^T and attn @ V) and token-matching matmuls (`note_matmul`).
One multiply-accumulate is 2 FLOPs; norms, activations, softmax, RoPE,
residual adds and gathers are not counted. Peak memory is the allocator
high-water mark per stage. `*_encoder` fields cover embedding + enc*.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn


# --- explicit notes from the token-merge paths -----------------------------
#
# Post-merge q/k shapes and the matching matmuls are not visible to module
# hooks, so those code paths report them through the functions below. Both
# are no-ops while no CostMeter is active.

_ACTIVE = None


def note_attention(q, k):
    """Report the tensors entering an attention matmul: q [B,H,Kq,D],
    k [B,H,Kkv,D]. Called by the token-merge attention paths, once per
    bucket."""
    m = _ACTIVE
    if m is not None and m._stack:
        m._stack[-1][1].append((int(q.shape[0]), int(q.shape[1]),
                                int(q.shape[2]), int(k.shape[2]),
                                int(q.shape[3])))


def note_matmul(a, b):
    """Report a matmul `a @ b.transpose(-1, -2)` issued outside any
    nn.Module (token-matching scores): 2 * rows * K * cols FLOPs with
    rows = prod(a.shape[:-1]), K = a.shape[-1] and cols = b.shape[-2].

    Booked as `flops_attn_overhead` and folded into the totals, under the
    stage of the enclosing attention call, or of the running stage when
    called between blocks (ALGM's GBM)."""
    m = _ACTIVE
    if m is None:
        return
    rows = 1
    for s in tuple(a.shape)[:-1]:
        rows *= int(s)
    m._note_overhead(2 * rows * int(a.shape[-1]) * int(b.shape[-2]))


def _is_attention(mod):
    """Duck-typed: PT-v3m3 SerializedAttention, LitePT PointROPEAttention
    and their token-merge subclasses."""
    return (hasattr(mod, "qkv") and hasattr(mod, "proj")
            and hasattr(mod, "num_heads"))


def _is_sparse_conv(mod):
    """spconv convs carry a 5-D weight plus in/out channel counts; matching
    on that avoids importing spconv here."""
    w = getattr(mod, "weight", None)
    return (w is not None and getattr(w, "ndim", 0) == 5
            and hasattr(mod, "in_channels") and hasattr(mod, "out_channels"))


def _pair_count(out, indice_key):
    """Number of (in, out) index pairs the conv gathered, as a device
    scalar (resolved once per frame, so counting never syncs mid-forward).

    spconv 2.x stores one of two payloads per `indice_key`:
      native algo        -> `indice_pair_num`, per-offset counts
      implicit gemm      -> `pair_fwd` [kvol, n_out], -1 where absent
    Returns None if neither is present, so the caller can fall back and
    flag the frame as inexact."""
    d = getattr(out, "indice_dict", None)
    if not d or indice_key not in d:
        return None
    data = d[indice_key]
    n = getattr(data, "indice_pair_num", None)
    if n is not None:
        return n.sum()
    pf = getattr(data, "pair_fwd", None)
    if pf is not None:
        return (pf >= 0).sum()
    return None


# --- stage attribution -----------------------------------------------------


def stage_roots(model):
    """Stage name -> stage module, matching FrameProfiler's `stage/<name>`
    labels so the FLOPs, memory and timing breakdowns line up."""
    bb = getattr(model, "backbone", model)
    roots = {}
    if getattr(bb, "embedding", None) is not None:
        roots["embedding"] = bb.embedding
    for attr in ("enc", "dec"):
        seq = getattr(bb, attr, None)
        if seq is None:
            continue
        for name, mod in seq._modules.items():
            roots[name] = mod
    if getattr(model, "seg_head", None) is not None:
        roots["seg_head"] = model.seg_head
    return roots


def build_stage_map(model):
    """id(module) -> stage name, for every descendant of a stage root."""
    out = {}
    for stage, root in stage_roots(model).items():
        for sub in root.modules():
            out[id(sub)] = stage
    return out


def is_encoder_stage(name):
    return name == "embedding" or name.startswith("enc")


# FLOPs roles: the block's position encoding, qkv, attention matmuls, proj
# and MLP, the remaining layers, and the token-matching matmuls.
ROLES = ("cpe_conv", "cpe_linear", "qkv", "qk_av", "proj", "mlp",
         "other_conv", "other_linear", "matching")


def op_role(name, is_conv):
    """Map a module's dotted name to its role. Suffix-based, so it holds
    through the PointSequential wrappers (cpe.0 = SubMConv3d, cpe.1 =
    Linear, mlp.0.fc1/fc2, attn.qkv, attn.proj). LitePT's position encoder
    `conv` inside a block also books as `cpe_*`, while
    `embedding.stem.conv` stays `other_conv`."""
    parts = name.split(".")
    in_block = any(p.startswith("block") for p in parts)
    if "cpe" in parts or (in_block and "conv" in parts):
        return "cpe_conv" if is_conv else "cpe_linear"
    if "attn" in parts:
        if parts[-1] == "qkv":
            return "qkv"
        if parts[-1] == "proj":
            return "proj"
    if "mlp" in parts:
        return "mlp"
    return "other_conv" if is_conv else "other_linear"


def build_role_map(model):
    """id(module) -> role, for every Linear and sparse conv in the model."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            out[id(mod)] = op_role(name, False)
        elif _is_sparse_conv(mod):
            out[id(mod)] = op_role(name, True)
    return out


# --- main collector --------------------------------------------------------


class CostMeter:
    """Per-frame FLOPs / token / peak-memory collector.

    Frames are delimited by the top-level model forward, as in
    FrameProfiler; `records` holds one dict per frame."""

    def __init__(self, model, device, extra_bytes_fn=None):
        self.device = device
        self.is_cuda = device is not None and device.type == "cuda"
        self.model = model
        self.records = []
        self.handles = []
        self._active = False
        self._stage_of = build_stage_map(model)
        self._role_of = build_role_map(model)
        # persistent policy state (the SpaceCache table), recorded next to
        # peak_bytes rather than folded into it
        self._extra_bytes_fn = extra_bytes_fn

        self.params = sum(p.numel() for p in model.parameters())
        self.param_bytes = (
            sum(p.numel() * p.element_size() for p in model.parameters())
            + sum(b.numel() * b.element_size() for b in model.buffers()))

        self._reset_frame()
        self._stack = []            # open attention calls -> list of notes
        self._pending_qkv_rows = None
        self._cur_stage = None      # stage currently running

        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                self.handles.append(
                    mod.register_forward_hook(self._hook_linear))
            elif _is_sparse_conv(mod):
                self.handles.append(
                    mod.register_forward_hook(self._hook_sparse_conv))
            if _is_attention(mod):
                self.handles.append(
                    mod.register_forward_pre_hook(self._hook_attn_pre))
                self.handles.append(
                    mod.register_forward_hook(self._hook_attn_post))
        # Per-stage peak memory: the allocator peak is reset at every stage
        # entry, so the stage that sets the frame peak can be identified
        # and the encoder peak is not masked by the decoder.
        for name, root in stage_roots(model).items():
            self.handles.append(
                root.register_forward_pre_hook(self._mk_stage_pre(name)))
            self.handles.append(
                root.register_forward_hook(self._mk_stage_post(name)))

        self.handles.append(model.register_forward_pre_hook(self._frame_pre))
        self.handles.append(model.register_forward_hook(self._frame_post))

    def _mk_stage_pre(self, name):
        def hook(_m, _i):
            if self._active:
                # attribution for note_matmul calls between blocks (GBM)
                self._cur_stage = name
                if self.is_cuda:
                    torch.cuda.reset_peak_memory_stats(self.device)
        return hook

    def _mk_stage_post(self, name):
        def hook(_m, _i, _o):
            if self._active and self.is_cuda:
                self.stage_peak[name] = max(
                    self.stage_peak.get(name, 0),
                    int(torch.cuda.max_memory_allocated(self.device)))
        return hook

    # -- per-frame state --
    def _reset_frame(self):
        self.stage_peak = {}
        self.f_linear = 0
        self.f_conv = 0
        self.f_attn = 0
        self.f_overhead = 0
        # role -> flops, and role -> flops restricted to the encoder.
        self.f_role = {}
        self.f_role_enc = {}
        self.f_by_stage = {}
        self.f_attn_by_stage = {}
        self.tokens_attn = 0        # tokens entering the attention matmul
        self.tokens_qkv = 0         # padded tokens entering qkv (pre-merge)
        self.tok_by_stage = {}      # stage -> attention tokens
        self.n_attn_calls = 0
        self.conv_exact = True
        self.cpu_ns = 0
        self._conv_pairs = None
        # deferred sparse-conv terms: (pairs, 2*cin*cout, stage, role)
        self._conv_terms = []

    def _note_overhead(self, flops):
        """Matmuls issued around the attention (token matching, ALGM's
        GBM). Attributed to the stage of the attention module on the stack,
        or to the running stage when called between blocks."""
        self.f_overhead += flops
        if self._stack:
            stage = self._stage_of.get(id(self._stack[-1][0]))
        else:
            stage = self._cur_stage
        self._add(stage, flops)
        self._add_role("matching", stage, flops)

    def _add_role(self, role, stage, flops):
        self.f_role[role] = self.f_role.get(role, 0) + flops
        if stage is not None and is_encoder_stage(stage):
            self.f_role_enc[role] = self.f_role_enc.get(role, 0) + flops

    def _add(self, stage, flops, is_attn=False):
        if stage is not None:
            self.f_by_stage[stage] = self.f_by_stage.get(stage, 0) + flops
            if is_attn:
                self.f_attn_by_stage[stage] = (
                    self.f_attn_by_stage.get(stage, 0) + flops)

    # -- hooks --
    def _hook_linear(self, mod, inputs, output):
        # 2 * rows * in * out, rows = tokens actually fed to this layer
        if not self._active:
            return
        t0 = time.perf_counter_ns()
        x = inputs[0]
        rows = int(x.numel() // x.shape[-1])
        flops = 2 * rows * mod.in_features * mod.out_features
        self.f_linear += flops
        stage = self._stage_of.get(id(mod))
        self._add(stage, flops)
        self._add_role(
            self._role_of.get(id(mod), "other_linear"), stage, flops)
        # The qkv of the open attention call sees the padded token count,
        # from which the stock attention FLOPs are derived. Matched by
        # identity: a shape test (out == 3 * in) would also fire on fc1
        # at mlp_ratio 3.
        if self._stack and mod is self._stack[-1][0].qkv:
            self._pending_qkv_rows = rows
            self.tokens_qkv += rows
        self.cpu_ns += time.perf_counter_ns() - t0

    def _hook_sparse_conv(self, mod, inputs, output):
        # 2 * pairs * Cin * Cout, with the (in, out) pairs the kernel
        # gathered read from the output's `indice_dict`
        if not self._active:
            return
        t0 = time.perf_counter_ns()
        cin = int(mod.in_channels)
        cout = int(mod.out_channels)
        key = getattr(mod, "indice_key", None)
        # Count the pairs of this call. Encoder and decoder blocks of a
        # level share one indice_key but may run on different token sets
        # (merged points, cache-miss subsets), so a count cannot be
        # reused by key.
        pairs = _pair_count(output, key)
        if pairs is None:
            # no pair map: dense upper bound n_out * kernel_volume, and the
            # frame is flagged inexact
            feats = getattr(output, "features", None)
            n_out = int(feats.shape[0]) if feats is not None else 0
            kvol = 1
            for s in (getattr(mod, "kernel_size", None)
                      or tuple(mod.weight.shape)[:3]):
                kvol *= int(s)
            flops = 2 * n_out * kvol * cin * cout
            self.f_conv += flops
            stage = self._stage_of.get(id(mod))
            self._add(stage, flops)
            self._add_role(
                self._role_of.get(id(mod), "other_conv"), stage, flops)
            self.conv_exact = False
        else:
            # `pairs` is a device tensor; reading it here would sync inside
            # the forward, so the product is resolved in `_frame_post`
            self._conv_terms.append(
                (pairs, 2 * cin * cout, self._stage_of.get(id(mod)),
                 self._role_of.get(id(mod), "other_conv")))
        self.cpu_ns += time.perf_counter_ns() - t0

    def _hook_attn_pre(self, mod, inputs):
        if not self._active:
            return
        self._stack.append((mod, []))
        self._pending_qkv_rows = None

    def _hook_attn_post(self, mod, inputs, output):
        # The two attention matmuls only (qkv / proj are Linears):
        #   QK^T   [B,H,Kq,D] x [B,H,D,Kkv] -> 2*B*H*Kq*Kkv*D
        #   attn@V [B,H,Kq,Kkv] x [B,H,Kkv,D] -> same again
        # Shapes come from `note_attention` on the token-merge paths;
        # the stock path is derived from the qkv row count and patch_size.
        if not self._active:
            return
        t0 = time.perf_counter_ns()
        notes = self._stack.pop()[1] if self._stack else []
        stage = self._stage_of.get(id(mod))
        h = int(mod.num_heads)
        c = int(getattr(mod, "channels", 0)) or int(mod.qkv.in_features)
        if notes:
            # token-merge path: exact post-merge shapes, one per bucket
            for b, hh, kq, kkv, d in notes:
                flops = 2 * (2 * b * hh * kq * kkv * d)
                self.f_attn += flops
                self._add(stage, flops, is_attn=True)
                self._add_role("qk_av", stage, flops)
                self.tokens_attn += b * kq
                self.tok_by_stage[stage] = (
                    self.tok_by_stage.get(stage, 0) + b * kq)
        else:
            # Stock path: dense K x K within each patch. A sequence shorter
            # than one patch is not padded (`get_padding_and_inverse` pads
            # only when bincount > patch_size), so it forms a single patch
            # of its true length.
            rows = self._pending_qkv_rows
            k = int(getattr(mod, "patch_size", 0) or 0)
            if rows and k:
                b = rows // k
                if b == 0:
                    b, k = 1, rows
                flops = 2 * (2 * b * h * k * k * (c // h))
                self.f_attn += flops
                self._add(stage, flops, is_attn=True)
                self._add_role("qk_av", stage, flops)
                self.tokens_attn += b * k
                self.tok_by_stage[stage] = (
                    self.tok_by_stage.get(stage, 0) + b * k)
        self.n_attn_calls += 1
        self._pending_qkv_rows = None
        self.cpu_ns += time.perf_counter_ns() - t0

    def _frame_pre(self, _m, _i):
        if not self._active:
            return
        self._reset_frame()
        self._stack = []
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def _frame_post(self, _m, _i, output):
        if not self._active:
            return
        t0 = time.perf_counter_ns()
        # resolve the deferred sparse-conv terms with one host transfer
        if self._conv_terms:
            per = torch.stack([
                (p * w).reshape(()).to(torch.float64)
                for p, w, _, _ in self._conv_terms]).tolist()
            for (_p, _w, stage, role), v in zip(self._conv_terms, per):
                self.f_conv += int(v)
                self._add(stage, int(v))
                self._add_role(role, stage, int(v))

        rec = dict(
            frame=len(self.records),
            # encoder: embedding + enc*
            flops_encoder=int(sum(v for k, v in self.f_by_stage.items()
                                  if is_encoder_stage(k))),
            flops_attn_encoder=int(
                sum(v for k, v in self.f_attn_by_stage.items()
                    if is_encoder_stage(k))),
            tokens_attn_encoder=int(sum(v for k, v in self.tok_by_stage.items()
                                        if is_encoder_stage(k))),
            # whole model
            flops_total=int(self.f_linear + self.f_conv + self.f_attn
                            + self.f_overhead),
            flops_linear=int(self.f_linear),
            flops_sparse_conv=int(self.f_conv),
            flops_attn=int(self.f_attn),
            flops_attn_overhead=int(self.f_overhead),
            flops_by_role={k: int(v) for k, v in self.f_role.items()},
            flops_by_role_encoder={
                k: int(v) for k, v in self.f_role_enc.items()},
            flops_by_stage={k: int(v) for k, v in self.f_by_stage.items()},
            flops_attn_by_stage={k: int(v)
                                 for k, v in self.f_attn_by_stage.items()},
            tokens_attn=int(self.tokens_attn),
            tokens_attn_by_stage={k: int(v)
                                  for k, v in self.tok_by_stage.items()},
            tokens_qkv=int(self.tokens_qkv),
            n_attn_calls=int(self.n_attn_calls),
            sparse_conv_exact=bool(self.conv_exact),
        )
        if self.is_cuda:
            # The stage pre-hooks reset the allocator peak, so the frame
            # peak is the max over the stage peaks and the tail after the
            # last stage.
            rec["peak_by_stage"] = dict(self.stage_peak)
            enc_peaks = [v for k, v in self.stage_peak.items()
                         if is_encoder_stage(k)]
            rec["peak_bytes"] = max(
                list(self.stage_peak.values())
                + [int(torch.cuda.max_memory_allocated(self.device))])
            rec["peak_bytes_encoder"] = max(enc_peaks) if enc_peaks else 0
            rec["peak_stage"] = (
                max(self.stage_peak, key=self.stage_peak.get)
                if self.stage_peak else None)
            rec["reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(self.device))
            # Allocator diagnostics: retries (a device-wide sync plus cache
            # release) and OOMs are cumulative counters, segments the
            # current count; per-frame differences locate stalls.
            ms = torch.cuda.memory_stats(self.device)
            rec["num_alloc_retries"] = int(ms.get("num_alloc_retries", 0))
            rec["num_ooms"] = int(ms.get("num_ooms", 0))
            rec["num_segments"] = int(ms.get("segment.all.current", 0))
        if self._extra_bytes_fn is not None:
            try:
                extra = self._extra_bytes_fn()
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        rec[k] = None if v is None else int(v)
                else:
                    rec["cache_bytes"] = int(extra)
            except Exception:
                rec["cache_bytes"] = None
        self.cpu_ns += time.perf_counter_ns() - t0
        rec["metrics_cpu_ms"] = round(self.cpu_ns / 1e6, 4)
        self.records.append(rec)

    # -- control --
    def start(self):
        global _ACTIVE
        _ACTIVE = self
        self._active = True

    def stop(self):
        global _ACTIVE
        self._active = False
        if _ACTIVE is self:
            _ACTIVE = None

    def remove(self):
        self.stop()
        for h in self.handles:
            h.remove()
        self.handles.clear()


# --- aggregation -----------------------------------------------------------


def cache_bytes(policy):
    """Bytes held by a streaming policy's cache: persistent state that is
    reported next to the activation peak, never folded into it."""
    cache = getattr(policy, "cache", None)
    if cache is None:
        return 0
    # prefer the cache's own count: the object walk below would also count
    # borrowed tensors such as SpaceCache's staging buffer
    nbytes = getattr(cache, "nbytes", None)
    if nbytes is not None:
        return int(nbytes())
    total = 0
    seen = set()
    stack = [cache]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if torch.is_tensor(obj):
            total += obj.numel() * obj.element_size()
        elif isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set)):
            stack.extend(obj)
        elif hasattr(obj, "__dict__"):
            stack.extend(vars(obj).values())
    return total


def summarize_cost(records, log=None):
    """Mean over frames of the per-frame records; peak memory is also
    reported as p90 and max. Logs a one-line summary if `log` is given."""
    if not records:
        return None
    import numpy as np

    n = len(records)

    def mean(key):
        return float(sum(r.get(key, 0) or 0 for r in records) / n)

    def pct(key, q):
        vals = [r[key] for r in records if r.get(key) is not None]
        return float(np.percentile(vals, q)) if vals else None

    stages = set()
    for r in records:
        stages.update(r.get("flops_by_stage", {}))
    by_stage = {
        s: float(sum(r.get("flops_by_stage", {}).get(s, 0)
                     for r in records) / n)
        for s in stages
    }
    attn_by_stage = {
        s: float(sum(r.get("flops_attn_by_stage", {}).get(s, 0)
                     for r in records) / n)
        for s in stages
    }
    enc_keys = [s for s in stages if is_encoder_stage(s)]

    # per-role FLOPs; the encoder roles sum to flops_encoder
    roles, roles_enc = set(), set()
    for r in records:
        roles.update(r.get("flops_by_role", {}))
        roles_enc.update(r.get("flops_by_role_encoder", {}))
    by_role = {
        k: float(sum(r.get("flops_by_role", {}).get(k, 0)
                     for r in records) / n)
        for k in roles
    }
    by_role_enc = {
        k: float(sum(r.get("flops_by_role_encoder", {}).get(k, 0)
                     for r in records) / n)
        for k in roles_enc
    }

    peak_stages = set()
    for r in records:
        peak_stages.update(r.get("peak_by_stage", {}))
    peak_by_stage = {
        s: float(sum(r.get("peak_by_stage", {}).get(s, 0)
                     for r in records) / n)
        for s in peak_stages
    }

    out = dict(
        n_frames=n,
        flops_total=mean("flops_total"),
        flops_linear=mean("flops_linear"),
        flops_sparse_conv=mean("flops_sparse_conv"),
        flops_attn=mean("flops_attn"),
        flops_attn_overhead=mean("flops_attn_overhead"),
        flops_encoder=float(sum(by_stage.get(s, 0.0) for s in enc_keys)),
        flops_attn_encoder=float(
            sum(attn_by_stage.get(s, 0.0) for s in enc_keys)),
        flops_by_stage={k: round(v, 1) for k, v in by_stage.items()},
        flops_attn_by_stage={k: round(v, 1) for k, v in attn_by_stage.items()},
        flops_by_role={k: round(v, 1) for k, v in by_role.items()},
        flops_by_role_encoder={k: round(v, 1) for k, v in by_role_enc.items()},
        tokens_attn_encoder=mean("tokens_attn_encoder"),
        # query tokens entering attention per stage, summed over its blocks
        tokens_attn_by_stage={
            s: round(sum(r.get("tokens_attn_by_stage", {}).get(s, 0)
                         for r in records) / n, 2)
            for s in sorted({k for r in records
                             for k in r.get("tokens_attn_by_stage", {})})},
        tokens_attn=mean("tokens_attn"),
        tokens_qkv=mean("tokens_qkv"),
        peak_bytes_mean=mean("peak_bytes"),
        peak_bytes_p90=pct("peak_bytes", 90),
        peak_bytes_max=max((r.get("peak_bytes", 0) or 0) for r in records),
        peak_bytes_encoder_mean=mean("peak_bytes_encoder"),
        peak_bytes_encoder_p90=pct("peak_bytes_encoder", 90),
        peak_by_stage={k: round(v, 1) for k, v in peak_by_stage.items()},
        peak_stage=(max(peak_by_stage, key=peak_by_stage.get)
                    if peak_by_stage else None),
        cache_bytes_mean=mean("cache_bytes"),
        cache_bytes_d4_mean=(mean("cache_bytes_d4")
                             if any("cache_bytes_d4" in r
                                    for r in records) else None),
        metrics_cpu_ms=mean("metrics_cpu_ms"),
        sparse_conv_exact=all(r.get("sparse_conv_exact", True)
                              for r in records),
    )
    if log is not None:
        # encoder stages first, embedding leading
        enc_break = " ".join(
            f"{s}={by_stage[s] / 1e9:.0f}"
            for s in sorted(enc_keys, key=lambda x: (x != "embedding", x)))
        log.info(
            "cost/frame (%d frames): ENC GFLOPs=%.1f (attn=%.1f) [%s] | "
            "total=%.1f (conv=%.1f lin=%.1f attn=%.1f) | enc tokens "
            "attn=%.0f qkv=%.0f | peak MiB enc=%.0f all=%.0f (set by %s) | "
            "hooks %.2f ms cpu%s",
            n, out["flops_encoder"] / 1e9, out["flops_attn_encoder"] / 1e9,
            enc_break, out["flops_total"] / 1e9,
            out["flops_sparse_conv"] / 1e9, out["flops_linear"] / 1e9,
            out["flops_attn"] / 1e9,
            out["tokens_attn_encoder"], out["tokens_qkv"],
            (out["peak_bytes_encoder_mean"] or 0) / 2**20,
            (out["peak_bytes_mean"] or 0) / 2**20,
            out["peak_stage"], out["metrics_cpu_ms"],
            "" if out["sparse_conv_exact"] else "  [!] sparse conv ESTIMATED")
    return out
