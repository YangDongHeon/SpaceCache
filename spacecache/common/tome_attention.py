"""GitMerge3D token-merge attention for PT-v3m3, without editing the model.

`tome_attn(block, info)` reparents `block.attn` to `(_ToMeForwardMixin,
<stock class>)` for a context; the mixin holds the GitMerge3D
SerializedAttention methods, adapted from GitMerge3D
(https://github.com/anhtuanhsgs/GitMerge3D, Copyright (c) 2025 GitMerge3d,
MIT License). `patch`/`weighted_patch` keep flash (dense
`flash_attn_func` on the rectangular merged batch, without the additive
biases) unless `--dense_attn` is set.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn

try:
    import flash_attn
except ImportError:  # LitePT-only envs may not have it installed
    flash_attn = None

from pointcept.models.utils.misc import offset2bincount

from spacecache.common.token_merging_algos import *  # noqa: F401,F403
from spacecache.common.cost_meter import note_attention

# Per-call diagnostics of the weighted_patch path, reported by run.py as
# results.json["gitmerge3d_adaptive_stats"]. WPATCH_THRESH is the energy
# threshold that selects the uniform or the two-bucket path (GitMerge3D
# default: 0.2).
WPATCH_STATS = []
WPATCH_THRESH = 0.2


# Merge modes handled by the mixin; any other `tome` value runs the
# attention unmerged.
VALID_TOME_MODES = [
    "patch",
    "weighted_patch",
]


class _ToMeForwardMixin:
    def _serialized_out(self):
        """True when `additional_info["serialized_out"]` is set: `forward`
        then returns feat in serialized (padded patch) order and stashes
        {order, inverse, patch_size, metric} on `self._ser_ctx`, leaving
        the inverse permutation to the caller."""
        ai = getattr(self, "additional_info", None)
        return bool(ai) and bool(ai.get("serialized_out"))

    def cal_score(self, q, k, threshold=0.9):
        # score = -energy: high means globally redundant (aggressive
        # bucket). `q` is unused.
        score = -globally_informed_energy(k)  # [B]; high -> redundant
        p = torch.where(score < threshold, 1.0, 0.0).sum(-1) / score.shape[0]
        return score, p

    def token_merge_method(self, metric, r):
        if self.additional_info["tome"] in ["patch", "weighted_patch"]:
            merge, unmerge = patch_based_matching(
                metric,
                r=r,
                stride=self.additional_info["stride"],
                no_rand=self.additional_info["no_rand"],
                mean_head=self.additional_info.get("mean_head", False),
            )
        return merge, unmerge

    def process_merging(self, q, k, v, metric, order, inverse):
        B, H, K, _ = q.shape
        with tome_prof_region("nonflash/matching"):
            merge, unmerge = self.token_merge_method(
                metric, r=int(K * self.additional_info["r"])
            )
        if self.additional_info.get("trace_back", False):
            self.additional_info["cache_token_merge_unmerge"] = (merge, unmerge)
            self.additional_info["cache_point_order_inverse"] = (order, inverse)
            self.additional_info["patchify_info"] = {
                "n_segments": q.shape[0],
                "K": self.patch_size,
                "H": self.num_heads,
                "C": self.channels,
            }

        size = None
        if self.additional_info["tome_attention"]:
            with tome_prof_region("nonflash/merge"):
                if not getattr(self, "enable_flash", True):
                    # proportional attention (+log(size)) is dense-only
                    size = torch.ones(
                        (B, H, K, 1), device=v.device, dtype=v.dtype)
                    size = merge(size, mode="sum")
                v = merge(v)
                q = merge(q)
                k = merge(k)
        return q, k, v, size, merge, unmerge

    def process_weighted_merging(self, q, k, v, rpe_feats, entropy, p, low_r):
        # GitMerge3D two-bucket merge: the `high` (informative) patches
        # merge at the system ratio r, the `low` (redundant) patches keep
        # only `low_r` tokens.
        B, H, K, D = q.shape
        with tome_prof_region("nonflash/weighted/partition"):
            num_l_patch = math.ceil(B - B * p) - 1
            e_order = torch.argsort(entropy.squeeze(), descending=False)
            high = e_order[:-num_l_patch]
            low = e_order[-num_l_patch:]
            h_q, h_k, h_v = q[high], k[high], v[high]
            l_q, l_k, l_v = q[low], k[low], v[low]
            e_inverse = torch.argsort(torch.cat([high, low]))

        with tome_prof_region("nonflash/matching/high"):
            h_merge, h_unmerge = patch_based_matching(
                h_v,
                r=int(K * self.additional_info["r"]),
                stride=self.additional_info["stride"],
                no_rand=self.additional_info["no_rand"],
            )
        if K <= low_r:
            # patch too small to keep low_r tokens: keep the whole bucket
            l_merge = l_unmerge = do_nothing
        else:
            if self.additional_info.get("equal_bucket_ratio", False):
                # ablation: merge the low bucket at the high bucket's ratio
                l_r = int(K * self.additional_info["r"])
                l_stride = self.additional_info["stride"]
            else:
                l_r = K - low_r
                l_stride = K // low_r
            with tome_prof_region("nonflash/matching/low"):
                l_merge, l_unmerge = patch_based_matching(
                    l_v,
                    r=l_r,
                    stride=l_stride,
                    no_rand=self.additional_info["no_rand"],
                )

        with tome_prof_region("nonflash/merge"):
            if not getattr(self, "enable_flash", True):
                # fp32 sizes (as in GitMerge3D) promote the logits to fp32
                h_size = torch.ones((h_v.shape[0], H, K, 1), device=v.device)
                l_size = torch.ones((l_v.shape[0], H, K, 1), device=v.device)
                h_size = h_merge(h_size, mode="sum")
                l_size = l_merge(l_size, mode="sum")
            else:
                # the +log(size) bias cannot be expressed in flash
                h_size = l_size = None
            h_q = h_merge(h_q)
            h_k = h_merge(h_k)
            h_v = h_merge(h_v)
            l_q = l_merge(l_q)
            l_k = l_merge(l_k)
            l_v = l_merge(l_v)

            if rpe_feats is not None:
                h_rpe_feats = rpe_feats[high]
                l_rpe_feats = rpe_feats[low]

                l_rpe_feats = l_merge(l_rpe_feats)
                l_rpe_feats = l_rpe_feats.transpose(-2, -1)
                l_rpe_feats = l_merge(l_rpe_feats)
                l_rpe_feats = l_rpe_feats.transpose(-2, -1)

                h_rpe_feats = h_merge(h_rpe_feats)
                h_rpe_feats = h_rpe_feats.transpose(-2, -1)
                h_rpe_feats = h_merge(h_rpe_feats)
                h_rpe_feats = h_rpe_feats.transpose(-2, -1)
            else:
                l_rpe_feats = None
                h_rpe_feats = None

        return {
            "high_q": h_q,
            "high_k": h_k,
            "high_v": h_v,
            "low_q": l_q,
            "low_k": l_k,
            "low_v": l_v,
            "high_size": h_size,
            "high_merge": h_merge,
            "high_unmerge": h_unmerge,
            "low_size": l_size,
            "low_merge": l_merge,
            "low_unmerge": l_unmerge,
            "e_inverse": e_inverse,
            "low_rpe_feats": l_rpe_feats,
            "high_rpe_feats": h_rpe_feats,
        }

    def _flash_cast(self, t):
        # PT-v3m3 narrows to bf16 at the flash call; the LitePT subclass
        # overrides this with its fp16 convention
        return t.to(torch.bfloat16)

    def self_attn(self, q, k, v, unmerge=None, size=None, rpe_feats=None, out_dtype=None):
        # stock LitePT modules have no `enable_flash` (flash-only)
        if getattr(self, "enable_flash", True):
            # merged patches form a rectangular (B, K-r) batch: dense
            # flash_attn_func, without the size/rpe biases
            with tome_prof_region("flash/attn"):
                # post-merge shapes for the FLOPs counter (per bucket)
                note_attention(q, k)
                feat = flash_attn.flash_attn_func(
                    self._flash_cast(q.transpose(1, 2)),
                    self._flash_cast(k.transpose(1, 2)),
                    self._flash_cast(v.transpose(1, 2)),
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                ).transpose(1, 2)
        else:
            with tome_prof_region("nonflash/attn"):
                # post-merge shapes for the FLOPs counter (per bucket)
                note_attention(q, k)
                attn = (q * self.scale) @ k.transpose(-2, -1)
                if size is not None:
                    attn = attn + size.transpose(-2, -1).log()
                if rpe_feats is not None:
                    attn = attn + rpe_feats
            with tome_prof_region("nonflash/softmax"):
                if self.upcast_softmax:
                    attn = attn.float()
                attn = self.softmax(attn)
                if isinstance(self.attn_drop, nn.Dropout):
                    attn = self.attn_drop(attn)
                attn = attn.to(v.dtype)
            with tome_prof_region("nonflash/av"):
                feat = attn @ v
        if unmerge is not None:
            with tome_prof_region("nonflash/unmerge"):
                feat = unmerge(feat)
        if out_dtype is not None:
            feat = feat.to(out_dtype)
        return feat

    def _weighted_patch_attn(self, q, k, v, point, order, inverse, channels, out_dtype):
        """GitMerge3D weighted_forward: uniform `process_merging` when there
        are at most 10 patches or `cal_score`'s p >= 0.5, otherwise the
        two-bucket `process_weighted_merging`. Returns feat as (-1, C); the
        caller applies [inverse] and proj."""
        # rpe is an additive bias: dense path only
        if getattr(self, "enable_rpe", False) and not getattr(
                self, "enable_flash", True):
            rpe_feats = self.rpe(self.get_rel_pos(point, order))
        else:
            rpe_feats = None

        if q.shape[0] > 10:
            with tome_prof_region("nonflash/weighted/energy"):
                entropy, p = self.cal_score(v, v, WPATCH_THRESH)
            B = q.shape[0]
            # per-call score quantiles (diagnostics); skipped under
            # CUDA-graph capture, where float() would be a device->host sync
            if not torch.cuda.is_current_stream_capturing():
                qs = torch.quantile(
                    entropy.detach().float(),
                    torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9],
                                 device=entropy.device))
                WPATCH_STATS.append(dict(
                    p=float(p), B=B, thresh=WPATCH_THRESH,
                    path=("uniform" if p >= 0.5 else "two_bucket"),
                    num_l=int(math.ceil(B - B * float(p)) - 1)
                    if p < 0.5 else 0,
                    score_q=[round(float(x), 4) for x in qs]))
            if p >= 0.5:
                q, k, v, size, merge, unmerge = self.process_merging(
                    q, k, v, v, order, inverse
                )
                if rpe_feats is not None:
                    rpe_feats = merge(rpe_feats)
                    rpe_feats = rpe_feats.transpose(-2, -1)
                    rpe_feats = merge(rpe_feats)
                    rpe_feats = rpe_feats.transpose(-2, -1)
                feat = self.self_attn(
                    q, k, v, unmerge=unmerge, size=size, rpe_feats=rpe_feats
                )
            else:
                o = self.process_weighted_merging(
                    q, k, v, rpe_feats, entropy, p,
                    self.additional_info["low_r"],
                )
                h_feat = self.self_attn(
                    o["high_q"], o["high_k"], o["high_v"],
                    unmerge=o["high_unmerge"], size=o["high_size"],
                    rpe_feats=o["high_rpe_feats"],
                )
                l_feat = self.self_attn(
                    o["low_q"], o["low_k"], o["low_v"],
                    unmerge=o["low_unmerge"], size=o["low_size"],
                    rpe_feats=o["low_rpe_feats"],
                )
                with tome_prof_region("nonflash/weighted/reindex"):
                    feat = torch.cat([h_feat, l_feat], dim=0)
                    feat = feat[o["e_inverse"]]
        else:
            q, k, v, size, merge, unmerge = self.process_merging(
                q, k, v, v, order, inverse
            )
            if rpe_feats is not None:
                rpe_feats = merge(rpe_feats)
                rpe_feats = rpe_feats.transpose(-2, -1)
                rpe_feats = merge(rpe_feats)
                rpe_feats = rpe_feats.transpose(-2, -1)
            feat = self.self_attn(
                q, k, v, unmerge=unmerge, size=size, rpe_feats=rpe_feats
            )

        return feat.transpose(1, 2).reshape(-1, channels).to(out_dtype)

    def forward(self, point):
        # `min_bincount` is cached on the Point so the .min() device->host
        # sync happens once per stage, not per block.
        if "min_bincount" not in point.keys():
            point["min_bincount"] = int(
                offset2bincount(point.offset).min())
        self.patch_size = min(point["min_bincount"], self.patch_size_max)

        H = self.num_heads
        K = self.patch_size
        C = self.channels
        merging = (self.additional_info is not None
                   and self.additional_info["tome"] in VALID_TOME_MODES)

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        _ser_metric = None  # matching metric, stashed under serialized_out

        qkv = self.qkv(point.feat[order])

        if self.rope_base:
            rope_coord = point.coord[order].clone()
            if self.training:
                dd = {"device": rope_coord.device, "dtype": rope_coord.dtype}
                if self.shift_coords is not None and self.shift_coords > 0:
                    shift_xyz = torch.empty(3, **dd).uniform_(
                        -self.shift_coords, self.shift_coords
                    )
                    rope_coord += shift_xyz.unsqueeze(0)

                if self.jitter_coords is not None and self.jitter_coords > 1.0:
                    jitter_max = math.log(self.jitter_coords)
                    jitter_min = -jitter_max
                    jitter_xyz = (
                        torch.empty(3, **dd).uniform_(jitter_min, jitter_max).exp()
                    )
                    rope_coord *= jitter_xyz.unsqueeze(0)

                if self.rescale_coords is not None and self.rescale_coords > 1.0:
                    rescale_max = math.log(self.rescale_coords)
                    rescale_min = -rescale_max
                    rescale_val = (
                        torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
                    )
                    rope_coord *= rescale_val

            qkv = qkv.reshape(-1, 3, H, C // H)
            q, k, v = qkv.unbind(dim=1)
            q, k = self.rope(q, k, rope_coord)

            if tome_compute_bf16():
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)

            with tome_prof_region("nonflash/repack"):
                q = q.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
                k = k.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
                v = v.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
                _ser_metric = k  # ToMe matches on the keys
                if self.upcast_attention:
                    q = q.float()
                    k = k.float()

            if (
                self.additional_info is not None and self.additional_info["tome"] == "weighted_patch"
            ):
                feat = self._weighted_patch_attn(
                    q, k, v, point, order, inverse, C, qkv.dtype
                )
            else:
                merge = unmerge = None
                size = None
                if merging:
                    q, k, v, size, merge, unmerge = self.process_merging(
                        q, k, v, v, order, inverse
                    )

                if merging and self.enable_flash:
                    # rectangular (B, K-r) merged batch: dense flash in
                    # self_attn, without the size/rpe biases
                    feat = self.self_attn(
                        q, k, v,
                        unmerge=(unmerge if self.additional_info[
                            "tome_attention"] else None),
                    )
                else:
                    with tome_prof_region("nonflash/attn"):
                        note_attention(q, k)  # post-merge shapes for the FLOPs counter
                        attn = (q * self.scale) @ k.transpose(-2, -1)
                        if merging and size is not None:
                            attn = attn + size.transpose(-2, -1).log()
                    if self.enable_rpe:
                        rpe_feats = self.rpe(self.get_rel_pos(point, order))
                        if merging:
                            rpe_feats = merge(rpe_feats)
                            rpe_feats = rpe_feats.transpose(-2, -1)
                            rpe_feats = merge(rpe_feats)
                            rpe_feats = rpe_feats.transpose(-2, -1)
                        attn = attn + rpe_feats
                    with tome_prof_region("nonflash/softmax"):
                        if self.upcast_softmax:
                            attn = attn.float()
                        attn = self.softmax(attn)
                        attn = self.attn_drop(attn).to(v.dtype)
                    with tome_prof_region("nonflash/av"):
                        feat = attn @ v
                    if merging and self.additional_info["tome_attention"]:
                        with tome_prof_region("nonflash/unmerge"):
                            feat = unmerge(feat)
                feat = feat.transpose(1, 2).reshape(-1, C).to(qkv.dtype)
        else:
            # (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H)
                .permute(2, 0, 3, 1, 4)
                .unbind(dim=0)
            )
            _ser_metric = k  # ToMe matches on the keys
            if self.upcast_attention:
                q = q.float()
                k = k.float()

            merge = unmerge = None
            size = None
            if merging:
                q, k, v, size, merge, unmerge = self.process_merging(
                    q, k, v, v, order, inverse
                )

            if merging and self.enable_flash:
                # rectangular merged batch: dense flash, no size/rpe biases
                feat = self.self_attn(
                    q, k, v,
                    unmerge=(unmerge if self.additional_info[
                        "tome_attention"] else None),
                )
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                if merging and size is not None:
                    attn = attn + size.transpose(-2, -1).log()
                if self.enable_rpe:
                    rpe_feats = self.rpe(self.get_rel_pos(point, order))
                    if merging:
                        rpe_feats = merge(rpe_feats)
                        rpe_feats = rpe_feats.transpose(-2, -1)
                        rpe_feats = merge(rpe_feats)
                        rpe_feats = rpe_feats.transpose(-2, -1)
                    attn = attn + rpe_feats
                if self.upcast_softmax:
                    attn = attn.float()
                attn = self.softmax(attn)
                attn = self.attn_drop(attn).to(qkv.dtype)
                feat = attn @ v
                if merging and self.additional_info["tome_attention"]:
                    feat = unmerge(feat)
            feat = feat.transpose(1, 2).reshape(-1, C)

        if self._serialized_out():
            self._ser_ctx = dict(order=order, inverse=inverse,
                                 patch_size=K, metric=_ser_metric)
        else:
            feat = feat[inverse]

        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


# ---------------------------------------------------------------------------
#  Interception
# ---------------------------------------------------------------------------


@contextmanager
def tome_attn(block, additional_info):
    """Reparent `block.attn` to `(_ToMeForwardMixin, <orig class>)` for the
    context and attach `additional_info`; restores on exit. LitePT blocks
    dispatch to `litept_tome_attention.litept_tome_attn`; blocks with no
    `attn` (conv-only LitePT stages) are a no-op.
    """
    # local import: litept_tome_attention imports _ToMeForwardMixin from here
    from spacecache.common.litept_block import is_litept_block
    from spacecache.common.litept_tome_attention import litept_tome_attn

    if not hasattr(block, "attn"):
        yield
        return
    if is_litept_block(block):
        with litept_tome_attn(block, additional_info):
            yield
        return

    attn = block.attn
    orig_cls = attn.__class__
    tome_cls = type(
        f"{orig_cls.__name__}ToMe", (_ToMeForwardMixin, orig_cls), {}
    )
    attn.__class__ = tome_cls
    attn.additional_info = additional_info
    # the mixin forward re-derives patch_size from patch_size_max each call
    if not hasattr(attn, "patch_size_max"):
        attn.patch_size_max = int(attn.patch_size)
    # patch/weighted_patch keep flash (rectangular merged batch); other
    # modes run dense so that proportional attention applies, clearing
    # enable_flash as run.py --dense_attn does
    had_flash = getattr(attn, "enable_flash", False)
    if had_flash:
        flash_ok = (additional_info is not None
                    and additional_info.get("tome")
                    in ("patch", "weighted_patch"))
        if not flash_ok:
            attn.enable_flash = False
            if isinstance(getattr(attn, "attn_drop", None), float):
                attn.attn_drop = nn.Dropout(attn.attn_drop)
    try:
        yield
    finally:
        attn.__class__ = orig_cls
        if had_flash:
            attn.enable_flash = True
        if hasattr(attn, "additional_info"):
            del attn.additional_info
