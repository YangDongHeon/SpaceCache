"""GitMerge3D (policies `gitmerge3d` and `gitmerge3d_adaptive`).

The attention step of the shared `timed_block_forward` scaffold runs inside
`tome_attention.tome_attn`, so only `block.attn` merges tokens.
`--merge_ratio` (`ctx["merge_ratio"]`) is the fraction of each patch's
tokens merged. `gitmerge3d` uses a fixed patch partition (the version of the
main tables, GitMerge3D with a dagger in the paper); `gitmerge3d_adaptive`
adds the official globally informed adaptive partition (Appendix Table 4).
"""

from __future__ import annotations

import math

from spacecache.common.timed_block import timed_block_forward
from spacecache.common.tome_attention import tome_attn
from spacecache.policies.base import MergePolicy


class GitMerge3DPolicy(MergePolicy):
    """GitMerge3D patch-based matching. `stride` is set per r to
    `ceil(1/(1-r))`, so the per-bin cap (stride-1)/stride never binds
    before the target ratio and exactly int(K * r) tokens are removed."""

    name = "gitmerge3d"

    no_rand: bool = False
    mean_head: bool = False

    @staticmethod
    def stride_for(r):
        """stride = ceil(1/(1-r)): a bin of `stride` tokens keeps one dst
        and can merge the other stride-1, so (stride-1)/stride >= r (e.g.
        r=0.5 -> 2, 0.75 -> 4, 0.9 -> 10). The -1e-9 absorbs float error
        (1/(1-0.8) = 5.000...1 must not ceil to 6)."""
        r = min(max(float(r), 0.0), 0.999)
        return max(2, math.ceil(1.0 / (1.0 - r) - 1e-9))

    def _additional_info(self, ctx):
        r = float(ctx["merge_ratio"])
        return dict(
            tome="patch",
            tome_attention=True,
            r=r,
            stride=self.stride_for(r),
            no_rand=self.no_rand,
            mean_head=self.mean_head,
        )

    def run_block(self, block, point, ctx):
        with tome_attn(block, self._additional_info(ctx)):
            return timed_block_forward(block, point, timer=ctx.get("timer"))


class GitMerge3DAdaptivePolicy(GitMerge3DPolicy):
    """GitMerge3D with its globally informed adaptive partition.

    Patches are split by global energy into a distinctive bucket, merged
    like `gitmerge3d` (removes int(K * r) tokens), and a redundant bucket
    that keeps only `low_r` tokens; attention runs per bucket. Both buckets
    are rectangular, so flash works (without the additive biases).
    `low_r` = 32 follows the GitMerge3D configs.
    """

    name = "gitmerge3d_adaptive"
    low_r: int = 32

    def _additional_info(self, ctx):
        r = float(ctx["merge_ratio"])
        return dict(
            tome="weighted_patch",
            tome_attention=True,
            r=r,
            stride=self.stride_for(r),
            low_r=self.low_r,
            no_rand=self.no_rand,
            mean_head=self.mean_head,
            equal_bucket_ratio=False,
        )
