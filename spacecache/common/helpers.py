"""Shared helpers: model loading, input preparation and world-cell geometry."""

import inspect
import random

import numpy as np
import torch

from pointcept.models import build_model
from pointcept.models.builder import MODELS

from spacecache.common import region_key


def _accepts_kwarg(type_name, kwarg):
    """Whether the registered backbone's `__init__` takes `kwarg`
    (PT-v3 takes `enable_flash`, LitePT does not)."""
    try:
        cls = MODELS.get(type_name)
        return kwarg in inspect.signature(cls.__init__).parameters
    except Exception:
        return False


def has_flash_attn():
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _strip_prefixes(key, prefixes=("module.", "backbone.")):
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def build_segmentor(model_cfg, weight, device):
    """Build a Pointcept segmentor from `cfg.model` and load a training
    checkpoint (`state_dict` with `module.`-prefixed keys) non-strictly.

    Without flash-attn, PT-v3 backbones fall back to the dense attention
    path with a uniform patch size of 1024.
    """
    model_cfg = dict(model_cfg)
    if not has_flash_attn() and "backbone" in model_cfg:
        bb = dict(model_cfg["backbone"])
        if _accepts_kwarg(bb.get("type"), "enable_flash"):
            bb["enable_flash"] = False
            if "enc_patch_size" in bb:
                n = len(bb["enc_patch_size"])
                bb["enc_patch_size"] = (1024,) * n
            model_cfg["backbone"] = bb

    model = build_model(model_cfg)
    ckpt = torch.load(str(weight), map_location="cpu", weights_only=False)
    state = (ckpt["state_dict"]
             if isinstance(ckpt, dict) and "state_dict" in ckpt
             else ckpt)
    state = {_strip_prefixes(k, prefixes=("module.",)): v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[segmentor] type={model_cfg.get('type')} loaded {len(state)} "
          f"tensors from {weight}")
    print(f"[segmentor] flash-attn={has_flash_attn()}, "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[segmentor]   sample missing: {sorted(missing)[:3]}")
    if unexpected:
        print(f"[segmentor]   sample unexpected: {sorted(unexpected)[:3]}")
    return model.to(device).eval()


def prepare_input(data_dict, device):
    """Move tensors to `device` and add `offset` for a single-frame batch."""
    inp = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            inp[k] = v.to(device, non_blocking=True)
        else:
            inp[k] = v
    if "offset" not in inp:
        inp["offset"] = torch.tensor([inp["coord"].shape[0]], device=device)
    return inp


def to_global(coord, pose, scale=1.0):
    """Map per-frame `coord` into the shared world frame: R @ c + scale * t.

    `pose` is the sensor-to-world 4x4 in metres. If the input pipeline
    applied a deterministic `RandomScale(scale=[s, s])`, the translation is
    scaled by `s` so every frame lands in the same s-scaled world frame.
    Computed in float64 (world translations reach ~1e3 m), with the
    translation added outside the matmul so TF32 cannot truncate it.
    """
    R = pose[:3, :3].to(torch.float64)
    t = pose[:3, 3].to(torch.float64)
    if scale != 1.0:
        t = t * float(scale)
    return coord.to(torch.float64) @ R.T + t


def extract_random_scale(transform_list, default=1.0):
    """Scale `s` of a deterministic `RandomScale(scale=[s, s])` entry."""
    for t in transform_list:
        if not (isinstance(t, dict) and t.get("type") == "RandomScale"):
            continue
        s = t.get("scale")
        if (
            s is not None
            and len(s) == 2
            and float(s[0]) == float(s[1])
            and not t.get("anisotropic", False)
        ):
            return float(s[0])
    return default


def extract_grid_size(transform_list, default=0.05):
    """`grid_size` of the GridSample entry in the transform pipeline."""
    for t in transform_list:
        if isinstance(t, dict) and t.get("type") == "GridSample":
            return float(t["grid_size"])
    return default


def voxel_hash(coord, voxel_size):
    """Region key of floor(coord / voxel_size) (the region index phi(v));
    see `region_key` for the key layout.

    The division is done in the precision of `coord` (float64 for
    `to_global` output): in float32 a large world translation shifts points
    across cell boundaries.
    """
    voxel = torch.floor(coord / voxel_size).to(torch.long)
    return region_key.encode(voxel)


def aggregate_cell_features(coord_g, feat, voxel_size):
    """Mean-pool `feat` per world cell -> (sorted cell hashes, cell features)."""
    h = voxel_hash(coord_g, voxel_size)
    uniq, inv = torch.unique(h, sorted=True, return_inverse=True)
    n_cell = int(uniq.numel())
    if n_cell == 0:
        return uniq, feat.new_zeros((0, feat.shape[1]))
    agg = feat.new_zeros((n_cell, feat.shape[1]))
    counts = feat.new_zeros((n_cell,))
    agg.scatter_add_(0, inv.unsqueeze(1).expand(-1, feat.shape[1]), feat)
    counts.scatter_add_(0, inv, torch.ones_like(inv, dtype=feat.dtype))
    agg = agg / counts.clamp(min=1).unsqueeze(1)
    return uniq, agg


def match_b_to_cells(coord_b_g, cell_hash_sorted, voxel_size):
    """For each point of B, whether its cell is in `cell_hash_sorted`.

    Returns (matched (N,) bool, insert (N,) long), where
    `cell_hash_sorted[insert] == hash_b` on matched rows.
    """
    hash_b = voxel_hash(coord_b_g, voxel_size)
    n_cells = cell_hash_sorted.shape[0]
    if n_cells == 0:
        matched = torch.zeros_like(hash_b, dtype=torch.bool)
        return matched, torch.zeros_like(hash_b)
    insert = torch.searchsorted(
        cell_hash_sorted, hash_b,
    ).clamp(max=n_cells - 1)
    matched = cell_hash_sorted[insert] == hash_b
    return matched, insert
