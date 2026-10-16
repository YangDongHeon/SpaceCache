"""GitMerge3D token-merge attention for the LitePT-v1 backbone.

LitePT's `PointROPEAttention` is flash-only, applies PointROPE to q and k
over `grid_coord` and narrows qkv to fp16 itself, so
`_LitePTToMeForwardMixin` overrides only `forward` and inherits the merge
methods of `tome_attention._ToMeForwardMixin`. `install_litept_dense`
implements run.py --dense_attn for LitePT.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.utils.misc import offset2bincount

from spacecache.common.token_merging_algos import tome_prof_region
from spacecache.common.tome_attention import VALID_TOME_MODES, _ToMeForwardMixin


# ---------------------------------------------------------------------------
#  Token-mixing dtype (LitePT's `.half()`, made switchable)
# ---------------------------------------------------------------------------

_COMPUTE_DTYPE = {"dt": torch.float16}


def set_litept_compute_dtype(dtype):
    """Set the dtype q/k/v are narrowed to right after `qkv` (None = stay
    fp32, run.py --fp32). Default `torch.float16` == LitePT's own
    `qkv.half()`, i.e. before rope; PT-v3m3 instead narrows to bf16 after
    rope (`token_merging_algos.set_tome_compute_bf16`)."""
    _COMPUTE_DTYPE["dt"] = dtype


def litept_compute_dtype():
    return _COMPUTE_DTYPE["dt"]


def _flash_ready(t):
    """flash-attn only accepts fp16/bf16: halve fp32 inputs (--fp32) at the
    kernel boundary."""
    return t if t.dtype in (torch.float16, torch.bfloat16) else t.half()


# ---------------------------------------------------------------------------
#  LitePT forward (the only method that differs from the PT-v3m3 mixin)
# ---------------------------------------------------------------------------


class _LitePTToMeForwardMixin(_ToMeForwardMixin):
    """LitePT-shaped `forward`; every GitMerge3D method is inherited."""

    def _flash_cast(self, t):
        # LitePT convention: fp16/bf16 pass through, wider dtypes halve
        return _flash_ready(t)

    def forward(self, point):
        H = self.num_heads
        C = self.channels
        D = C // H
        info = getattr(self, "additional_info", None)
        merging = info is not None and info.get("tome") in VALID_TOME_MODES

        if merging:
            # clamp K to the smallest batch (as PT-v3m3's dense path does)
            # so the padded sequence reshapes into whole patches. K is
            # deterministic in `point`, so every block of the stage uses the
            # same K and the cached pad/cu_seqlens stay valid;
            # `min_bincount` is cached on the Point, so the device->host
            # `.min()` runs once per Point (model_graph injects it during
            # graph capture).
            if "min_bincount" not in point.keys():
                point["min_bincount"] = int(
                    offset2bincount(point.offset).min())
            self.patch_size = min(
                point["min_bincount"],
                int(getattr(self, "patch_size_max", self.patch_size)),
            )
        K = self.patch_size

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # reorder, then project (as PT-v3m3); stock LitePT projects first
        # and gathers the 3C-wide qkv, which is mathematically identical
        qkv = self.qkv(point.feat[order])
        out_dtype = qkv.dtype

        ## apply pointrope
        pos = point.grid_coord[order]
        pos = pos.reshape(-1, 3).unsqueeze(0)

        cdt = litept_compute_dtype()
        q, k, v = (qkv if cdt is None else qkv.to(cdt)).chunk(3, dim=-1)
        q = q.reshape(-1, H, D).transpose(0, 1)[None]   # [1, H, N, D]
        k = k.reshape(-1, H, D).transpose(0, 1)[None]

        # workround to make pointrope cuda float32 happy
        q = self.rope(q.float(), pos).to(q.dtype)
        k = self.rope(k.float(), pos).to(k.dtype)

        q = q.squeeze(0).transpose(0, 1)                # [N, H, D]
        k = k.squeeze(0).transpose(0, 1)
        v = v.reshape(-1, H, D)

        # merge: same repack -> match -> self_attn path as PT-v3m3;
        # `self_attn` picks dense flash or matmul attention on `enable_flash`
        if merging:
            with tome_prof_region("nonflash/repack"):
                q = q.reshape(-1, K, H, D).permute(0, 2, 1, 3)   # [B, H, K, D]
                k = k.reshape(-1, K, H, D).permute(0, 2, 1, 3)
                v = v.reshape(-1, K, H, D).permute(0, 2, 1, 3)

            if info.get("tome") == "weighted_patch":
                # GitMerge3D weighted_forward; returns (-1, C) in out_dtype
                feat = self._weighted_patch_attn(
                    q, k, v, point, order, inverse, C, out_dtype
                )
            else:
                q, k, v, size, _merge, unmerge = self.process_merging(
                    q, k, v, v, order, inverse
                )
                if not info.get("tome_attention", True):
                    unmerge = None
                # self_attn reports the post-merge shapes to the FLOPs counter
                feat = self.self_attn(q, k, v, unmerge=unmerge, size=size)
                feat = feat.transpose(1, 2).reshape(-1, C).to(out_dtype)
        elif getattr(self, "enable_flash", True):
            # stock LitePT: packed [N, 3, H, D] varlen flash
            with tome_prof_region("flash_tome/stack"):
                qkv_rotated = torch.stack([q, k, v], dim=1)
            with tome_prof_region("flash_tome/flash"):
                feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                    _flash_ready(qkv_rotated),
                    cu_seqlens,
                    max_seqlen=self.patch_size,
                    dropout_p=self.attn_drop if self.training else 0,
                    softmax_scale=self.scale,
                )
            feat = feat.reshape(-1, C).to(out_dtype)
        else:
            # dense fallback (install_litept_dense): per-patch matmul
            # attention via `self_attn`, as PT-v3m3's non-flash path (pad
            # slots duplicate real tokens, no mask). A batch shorter than
            # one patch keeps its own length (no padding to K), so the
            # layout can be ragged; then loop over cu_seqlens.
            if q.shape[0] % K == 0:
                with tome_prof_region("nonflash/repack"):
                    q = q.reshape(-1, K, H, D).permute(0, 2, 1, 3)
                    k = k.reshape(-1, K, H, D).permute(0, 2, 1, 3)
                    v = v.reshape(-1, K, H, D).permute(0, 2, 1, 3)
                feat = self.self_attn(q, k, v)
                feat = feat.transpose(1, 2).reshape(-1, C)
            else:
                bounds = cu_seqlens.tolist()
                feat = torch.cat([
                    self.self_attn(
                        q[s:e].transpose(0, 1)[None],
                        k[s:e].transpose(0, 1)[None],
                        v[s:e].transpose(0, 1)[None],
                    ).transpose(1, 2).reshape(e - s, C)
                    for s, e in zip(bounds, bounds[1:])
                ], dim=0)
            feat = feat.to(out_dtype)

        if getattr(self, "serialized_out", False):
            # set by the tome/algm policies: return feat in serialized
            # (padded patch) order and stash the layout; the caller
            # applies `inverse`
            self._ser_ctx = dict(order=order, inverse=inverse,
                                 patch_size=K)
        else:
            feat = feat[inverse]

        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


# ---------------------------------------------------------------------------
#  Installation
# ---------------------------------------------------------------------------


def is_litept_attention(mod):
    """LitePT `PointROPEAttention` (or a reparented subclass of it): has the
    attention trio plus a `rope` that is called as `rope(tokens, positions)`.
    PT-v3m3's RoPE lives behind `rope_base`, which LitePT never defines."""
    return (hasattr(mod, "qkv") and hasattr(mod, "proj")
            and hasattr(mod, "num_heads") and hasattr(mod, "rope")
            and not hasattr(mod, "rope_base"))


def _tome_class(orig_cls):
    return type(f"{orig_cls.__name__}ToMe",
                (_LitePTToMeForwardMixin, orig_cls), {})


@contextmanager
def litept_tome_attn(block, additional_info):
    """Attach `additional_info` to `block.attn` for the context,
    reparenting it to the LitePT ToMe class first if it is still the stock
    class. Restores both on exit."""
    attn = getattr(block, "attn", None)
    if attn is None:                      # conv-only LitePT block: nothing to merge
        yield
        return

    orig_cls = attn.__class__
    reparented = not isinstance(attn, _LitePTToMeForwardMixin)
    if reparented:
        attn.__class__ = _tome_class(orig_cls)
        _set_tome_defaults(attn)
    attn.additional_info = additional_info
    try:
        yield
    finally:
        if reparented:
            attn.__class__ = orig_cls
        if hasattr(attn, "additional_info"):
            del attn.additional_info


def _set_tome_defaults(attn):
    """Set the mixin attributes LitePT's stock class does not define. Only
    missing ones are set, so a second reparent (install_litept_dense, then
    litept_tome_attn) keeps the first values; `patch_size_max` is thus
    taken before any forward shrinks `patch_size`."""
    for name, default in (("patch_size_max", int(attn.patch_size)),
                          ("enable_rpe", False),
                          ("upcast_softmax", False),
                          ("upcast_attention", False)):
        if not hasattr(attn, name):
            setattr(attn, name, default)


def install_litept_dense(model, upcast_softmax=False):
    """`--dense_attn` for LitePT: reparent every `PointROPEAttention` to the
    ToMe mixin and clear `enable_flash`. Unmerged blocks then run the
    per-patch matmul fallback and merged blocks apply the `+log(size)` bias
    in `self_attn`'s dense branch. Returns the number of patched modules."""
    n = 0
    for m in model.modules():
        if not is_litept_attention(m):
            continue
        if not isinstance(m, _LitePTToMeForwardMixin):
            m.__class__ = _tome_class(m.__class__)
        _set_tome_defaults(m)
        m.enable_flash = False
        m.upcast_softmax = bool(upcast_softmax)
        n += 1
    return n
