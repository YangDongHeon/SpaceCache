"""Per-component CUDA-event timer used by the block forwards.

The block scaffolds wrap every step in `timer.section(category)`;
`_resolve_timer` substitutes a no-op timer when none is given.
"""

from __future__ import annotations

import torch


class BlockTimer:
    """Per-stage, per-component CUDA-event timer.

    `section(name)` is a context manager that records one CUDA-event pair
    under the current stage (`set_stage`; "all" until set). Categories:

      Pool      GridPooling between stages
      CPE       position encoder (`block.cpe`; LitePT `block.conv`)
      LN        `block.norm*`
      LS        LayerScale `block.ls1` / `block.ls2`
      Attn      attention forward and its drop_path
      MLP       `block.mlp` and its drop_path
      Ordering  gathers and scatters (serialized reorders, merge maps)
      Others    other policy-specific overhead (e.g. merge selection)
    """

    CATEGORIES = ("Pool", "CPE", "LN", "LS", "Attn", "MLP",
                  "Ordering", "Others")
    STAGE_NAMES = ("embedding", "enc0", "enc1", "enc2", "enc3", "enc4")
    DEFAULT_STAGE = "all"

    def __init__(self, device, enabled=True):
        self.enabled = enabled and (
            getattr(device, "type", str(device)) == "cuda"
        )
        # list[(stage, category, start_evt, end_evt)]
        self._records = []
        self.current_stage = self.DEFAULT_STAGE

    def set_stage(self, name):
        """Tag subsequent sections with stage `name`."""
        self.current_stage = str(name)

    def section(self, name):
        return _BlockTimerSection(self, name)

    def collect(self):
        """Return {stage: {category: ms}}; stages without records are
        omitted."""
        if not self.enabled or not self._records:
            return {}
        torch.cuda.synchronize()
        out = {}
        for stg, cat, s, e in self._records:
            if stg not in out:
                out[stg] = {c: 0.0 for c in self.CATEGORIES}
            out[stg][cat] = out[stg].get(cat, 0.0) + s.elapsed_time(e)
        return out

    def collect_totals(self):
        """Return {category: ms} summed over all stages."""
        nested = self.collect()
        totals = {c: 0.0 for c in self.CATEGORIES}
        for stg in nested.values():
            for cat, ms in stg.items():
                totals[cat] += ms
        return totals


class _BlockTimerSection:
    __slots__ = ("timer", "name", "_start_evt", "_stage")

    def __init__(self, timer, name):
        self.timer = timer
        self.name = name
        self._start_evt = None
        self._stage = None

    def __enter__(self):
        if self.timer.enabled:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._start_evt.record()
            # tag with the stage at entry, even if set_stage runs inside
            self._stage = self.timer.current_stage
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.timer.enabled and self._start_evt is not None:
            end_evt = torch.cuda.Event(enable_timing=True)
            end_evt.record()
            self.timer._records.append(
                (self._stage, self.name, self._start_evt, end_evt)
            )
        self._start_evt = None
        return False


class _NullTimer:
    """No-op timer used when profiling is disabled."""
    enabled = False

    def set_stage(self, name):  # noqa: ARG002
        pass

    def section(self, name):  # noqa: ARG002
        return _NullSection()

    def collect(self):
        return {}

    def collect_totals(self):
        return {c: 0.0 for c in BlockTimer.CATEGORIES}


class _NullSection:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _resolve_timer(timer):
    """Return `timer`, or a no-op timer when it is None."""
    return timer if timer is not None else _NullTimer()
