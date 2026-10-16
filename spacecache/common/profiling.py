"""CUDA-event profilers for the evaluation harness.

`FrameProfiler` keeps one timing record per evaluated frame (whole forward,
stages, blocks, block children and attention-path regions).
"""

from __future__ import annotations

import os

import torch


class FrameProfiler:
    """Per-frame hierarchical CUDA timing of the whole segmentor.

    `records` receives one dict per evaluated frame (times in ms):

      total_ms           whole model forward
      modules_ms         {label: ms}; labels are `stage/<name>` (embedding,
                         enc*, dec*, seg_head), `block/<path>`,
                         `module/<path>` (block children) and
                         `pool/<path>` (the down / up module of a stage)
      attn_breakdown_ms  attention-path regions (`tome_prof_region`) as
                         `attn/<step>`; empty unless TOME_PROFILE=1
      attn_fine_ms       with `fine_regions`: the nested match/ merge/
                         unmerge/ sub-regions as `fine/<name>`, with call
                         counts as `n/<name>`
      n_voxels, frame

    A frame is one top-level model forward: its pre-hook opens the frame,
    its post-hook synchronizes once and resolves (then drops) the frame's
    events, so memory stays bounded over a full validation split.
    """

    def __init__(self, model, device, level="block", fine_regions=False,
                 record_regions=False):
        """`level` selects what is timed:

          frame  the model forward only
          stage  + embedding / enc* / dec* / seg_head
          block  + every block, its children and each stage's down / up

        `stage` keeps the encoder / decoder split with far fewer events
        than `block`, which is meant for breakdowns. `fine_regions` also
        times the children of each block's `attn` and keeps the nested
        match/ merge/ unmerge/ regions. `record_regions` (or
        PROFILE_REGIONS=1) opens a `torch.profiler.record_function` scope
        per label.
        """
        assert level in ("frame", "stage", "block"), level
        self.level = level
        self.fine_regions = fine_regions
        self.is_cuda = device is not None and device.type == "cuda"
        self.records = []
        self.handles = []
        self._active = False
        self._cur = None            # current frame: list[[label, pre, post]]
        self._frame_start = None
        self._frame_idx = 0
        # run after each frame's synchronize, see `add_frame_flush`
        self._frame_flush_cbs = []

        try:
            from spacecache.common.token_merging_algos import (
                tome_prof_flush, tome_prof_reset, tome_prof_results,
            )
            self._tome = (tome_prof_reset, tome_prof_flush, tome_prof_results)
        except Exception:
            self._tome = None

        bb = getattr(model, "backbone", model)
        to_time = {}                # label -> module (timed every frame)
        if level in ("stage", "block"):
            if getattr(bb, "embedding", None) is not None:
                to_time["stage/embedding"] = bb.embedding
            if getattr(bb, "enc", None) is not None:
                for n, m in bb.enc._modules.items():
                    to_time[f"stage/{n}"] = m
            if getattr(bb, "dec", None) is not None:
                for n, m in bb.dec._modules.items():
                    to_time[f"stage/{n}"] = m
            if getattr(model, "seg_head", None) is not None:
                to_time["stage/seg_head"] = model.seg_head
        # Transformer blocks of both backbones: `pre_norm` together with
        # `cpe` (PT-v3m3), or `conv` / `norm0` (LitePT, including its
        # conv-only blocks). Pooling and embedding modules have no
        # `pre_norm`.
        if level == "block":
            for name, m in model.named_modules():
                if hasattr(m, "pre_norm") and (
                    hasattr(m, "cpe") or hasattr(m, "conv")
                    or hasattr(m, "norm0")
                ):
                    to_time[f"block/{name}"] = m
                    for cname, cm in m.named_children():
                        to_time[f"module/{name}.{cname}"] = cm
                        # split `attn` into its children (qkv, rope, proj)
                        if fine_regions and cname == "attn":
                            for gname, gm in cm.named_children():
                                to_time[f"module/{name}.attn.{gname}"] = gm
            # the down / up (un)pooling module that opens each stage
            for name, m in model.named_modules():
                base = name.rsplit(".", 1)[-1]
                if base in ("down", "up") and hasattr(m, "forward"):
                    to_time[f"pool/{name}"] = m

        # record_function scopes add host work per label, so they are only
        # opened for torch.profiler runs
        self._rf = None
        self._rf_stack = []
        self.rf_note = "regions off"
        if record_regions or os.environ.get("PROFILE_REGIONS", "0") != "0":
            try:
                from torch.profiler import record_function
                self._rf = record_function
                self.rf_note = f"record_function ON for {len(to_time)} labels"
            except Exception as exc:  # pragma: no cover
                self._rf = None
                self.rf_note = f"record_function unavailable: {exc!r}"

        for label, m in to_time.items():
            self.handles.append(
                m.register_forward_pre_hook(self._mk_pre(label)))
            self.handles.append(m.register_forward_hook(self._mk_post(label)))
        # frame delimiter + whole-forward total on the model itself
        self.handles.append(model.register_forward_pre_hook(self._frame_pre))
        self._model = model
        self._post_handle = model.register_forward_hook(self._frame_post)
        self.handles.append(self._post_handle)

    def _mk_pre(self, label):
        def hook(_m, _i):
            if self._active and self.is_cuda and self._cur is not None:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                self._cur.append([label, e, None])
            # the event pair measures wall time; the record_function scope
            # lets torch.profiler attribute kernel time to the same label
            if self._rf is not None:
                r = self._rf(label)
                r.__enter__()
                self._rf_stack.append((label, r))
        return hook

    def _mk_post(self, label):
        def hook(_m, _i, _o):
            if self._rf is not None:
                while self._rf_stack:
                    ll, r = self._rf_stack.pop()
                    r.__exit__(None, None, None)
                    if ll == label:
                        break
            if self._active and self.is_cuda and self._cur is not None:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                for rec in reversed(self._cur):
                    if rec[0] == label and rec[2] is None:
                        rec[2] = e
                        break
        return hook

    def _frame_pre(self, _m, _i):
        if not self._active:
            return
        self._cur = []
        if self.is_cuda:
            self._frame_start = torch.cuda.Event(enable_timing=True)
            self._frame_start.record()
        if self._tome is not None:
            self._tome[0]()                       # tome_prof_reset

    def _frame_post(self, _m, _i, output):
        if not self._active:
            return
        rec = {"frame": self._frame_idx}
        if self.is_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            # always sync: tome_prof_flush skips its sync when no region
            # was recorded, and `end` must be complete
            torch.cuda.synchronize()
            rec["total_ms"] = float(self._frame_start.elapsed_time(end))
            rec["modules_ms"] = {
                label: float(pre.elapsed_time(post))
                for label, pre, post in self._cur if post is not None
            }
        if self._tome is not None:
            self._tome[1]()                       # tome_prof_flush
            times, counts = self._tome[2]()       # tome_prof_results
            # Coarse steps become `attn/<step>`. The match/ merge/ unmerge/
            # sub-regions are nested inside them, so they never enter the
            # (summed) attn_breakdown_ms; with fine_regions they are kept
            # separately, with call counts to give the per-call cost.
            attn, fine = {}, {}
            for k, v in times.items():
                if k.startswith(("match/", "merge/", "unmerge/")):
                    if self.fine_regions:
                        fine[f"fine/{k}"] = round(v, 5)
                        fine[f"n/{k}"] = counts.get(k, 0)
                    continue
                suffix = k.split("/", 1)[1] if "/" in k else k
                attn[f"attn/{suffix}"] = round(v, 5)
            rec["attn_breakdown_ms"] = attn
            if fine:
                rec["attn_fine_ms"] = fine
        n = None
        if isinstance(output, dict) and "seg_logits" in output:
            n = int(output["seg_logits"].shape[0])
        elif getattr(output, "coord", None) is not None:
            n = int(output.coord.shape[0])
        rec["n_voxels"] = n
        self.records.append(rec)
        # policy callbacks resolve their own events after the sync above
        for cb in self._frame_flush_cbs:
            cb()
        self._frame_idx += 1
        self._cur = None

    def end_frame_last(self):
        """Re-register the frame-end hook behind every model forward hook
        registered so far (the SpaceCache Commit Gate), so that their work
        is inside `total_ms`."""
        self._post_handle.remove()
        self._post_handle = self._model.register_forward_hook(
            self._frame_post)
        self.handles.append(self._post_handle)

    def add_frame_flush(self, cb):
        """Register a no-argument callback run after each frame's sync,
        e.g. to resolve CUDA events a policy recorded during the forward."""
        self._frame_flush_cbs.append(cb)

    def start(self):
        self._active = True

    def stop(self):
        self._active = False

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
