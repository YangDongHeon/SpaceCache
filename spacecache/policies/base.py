"""Shared interface of the token-merging policies."""

from __future__ import annotations


class MergePolicy:
    """A policy that runs each transformer block through `run_block`
    (`run.py` routes every encoder and decoder block to it), unless it
    installs its own intercept via `install_override`."""

    name: str = ""

    def run_block(self, block, point, ctx):
        """Run one transformer block with this policy's merging and return
        the post-block Point. `ctx` carries `merge_ratio` and `timer` (a
        `BlockTimer` or None)."""
        raise NotImplementedError
