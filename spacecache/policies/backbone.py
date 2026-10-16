"""The unmodified backbone (policy `backbone`)."""

from __future__ import annotations

from spacecache.common.timed_block import timed_block_forward


class BackbonePolicy:
    """Stock attention, run through the same timed block scaffold as the
    token-merging policies so that the instrumentation overhead matches."""

    name = "backbone"

    def run_block(self, block, point, ctx):
        return timed_block_forward(block, point, timer=ctx.get("timer"))
