"""Method names, run names and their settings, in one place.

`run.py` writes every run to `<output_root>/<run name>/`. The run name is
the method, the attention path, and each setting that differs from its
default, joined by underscores:

    backbone_dense
    spacecache_s34_dense_L8_conf0.85_triton
    tome_dense_r0.025
    gitmerge3d_dense_r0.6
    algm_dense_tau0.995

`run_name` builds the name and `parse_run_name` reads it back. The table
and figure scripts use only these two functions, so the format is defined
here and nowhere else.
"""

from __future__ import annotations

import math

BACKBONE = "backbone"
SPACECACHE_S3 = "spacecache_s3"
SPACECACHE_S34 = "spacecache_s34"
# stage-selection ablation (Table 3(b)): one cache at another stage
SPACECACHE_SINGLE = ("spacecache_s0", "spacecache_s1", "spacecache_s2",
                     "spacecache_s4")
SPACECACHE = (SPACECACHE_S3, SPACECACHE_S34) + SPACECACHE_SINGLE
TOME = "tome"
ALGM = "algm"
GITMERGE3D = "gitmerge3d"
GITMERGE3D_ADAPTIVE = "gitmerge3d_adaptive"
MERGING = (TOME, ALGM, GITMERGE3D, GITMERGE3D_ADAPTIVE)
POLICIES = (BACKBONE,) + SPACECACHE + MERGING

# names used in the paper
LABELS = {
    BACKBONE: "Backbone",
    SPACECACHE_S3: "SpaceCache (Stage-3)",
    SPACECACHE_S34: "SpaceCache (Stage-3,4)",
    "spacecache_s0": "SpaceCache (Stage-0)",
    "spacecache_s1": "SpaceCache (Stage-1)",
    "spacecache_s2": "SpaceCache (Stage-2)",
    "spacecache_s4": "SpaceCache (Stage-4)",
    TOME: "ToMe",
    ALGM: "ALGM",
    GITMERGE3D: "GitMerge3D",
    GITMERGE3D_ADAPTIVE: "GitMerge3D (adaptive)",
}

# the setting each method is swept over (the x of its Pareto curve)
SWEPT = {TOME: "merge_ratio", GITMERGE3D: "merge_ratio",
         GITMERGE3D_ADAPTIVE: "merge_ratio", ALGM: "tau"}
SWEPT.update({p: "confidence" for p in SPACECACHE})

MERGE_REDUCE_TAG = {"segment_csr": "csr", "segment_csr_fused": "csrfused",
                    "cumsum": "cumsum"}


def fmt(x):
    """0.6 -> '0.6', 0.025 -> '0.025', 8.0 -> '8', inf -> 'inf'."""
    x = float(x)
    if math.isinf(x):
        return "inf"
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def run_name(policy, attention="dense", *, merge_ratio=None, tau=None,
             algm_cell=None, algm_cell_offset=None, algm_window=None,
             region_scale=1.0, lifetime=None, confidence=0.0,
             expiry="staggered", triton=False, region_key="packed",
             merge_stages=None, proportional_attn=False, clamp_patch=False,
             merge_reduce="scatter", seed=None, shard=None):
    """The directory name of one run (see the module docstring).

    Pass only what applies to `policy`: `merge_ratio` for ToMe and
    GitMerge3D; `tau` (and non-default ALGM windows) for ALGM; `lifetime`,
    `confidence`, `region_scale`, `expiry`, `triton`, `region_key` for
    SpaceCache. `shard` is (K, N) for a scene shard."""
    parts = [policy]
    if merge_stages:
        parts.append(merge_stages if isinstance(merge_stages, str)
                     else "".join(sorted(merge_stages)))
    parts.append(attention)
    if merge_ratio is not None:
        parts.append(f"r{fmt(merge_ratio)}")
    if tau is not None:
        parts.append(f"tau{fmt(tau)}")
    if algm_cell is not None:
        parts.append(f"cell{int(algm_cell)}")
    if algm_cell_offset is not None:
        parts.append(f"off{int(algm_cell_offset)}")
    if algm_window is not None:
        parts.append(f"w{int(algm_window)}")
    if region_scale is not None and float(region_scale) != 1.0:
        parts.append(f"rs{fmt(region_scale)}")
    if lifetime is not None:
        parts.append(f"L{fmt(lifetime)}")
    if confidence:
        parts.append(f"conf{fmt(confidence)}")
    if expiry != "staggered":
        parts.append(expiry)
    if triton:
        parts.append("triton")
    if region_key != "packed":
        parts.append(f"key{region_key}")
    if proportional_attn:
        parts.append("propattn")
    if clamp_patch:
        parts.append("clamp")
    if merge_reduce != "scatter":
        parts.append(MERGE_REDUCE_TAG[merge_reduce])
    if seed is not None:
        parts.append(f"seed{int(seed)}")
    if shard is not None:
        parts.append(f"shard{int(shard[0])}of{int(shard[1])}")
    return "_".join(parts)


def policy_of(name):
    """The method a run name starts with (longest match), else None."""
    best = None
    for p in POLICIES:
        if (name == p or name.startswith(p + "_")) and (
                best is None or len(p) > len(best)):
            best = p
    return best


def parse_run_name(name):
    """Inverse of `run_name` -> dict, or None for a foreign directory.

    Keys: policy, attention, merge_ratio, tau, algm_cell, algm_cell_offset,
    algm_window, region_scale, lifetime, confidence, expiry, triton,
    region_key, merge_stages, proportional_attn, clamp_patch, merge_reduce,
    seed, shard, and `swept`, the value the method is swept over
    (confidence 0.0 when the confidence rule is off)."""
    policy = policy_of(name)
    if policy is None:
        return None
    d = dict(policy=policy, attention=None, merge_ratio=None, tau=None,
             algm_cell=None, algm_cell_offset=None, algm_window=None,
             region_scale=1.0, lifetime=None, confidence=0.0,
             expiry="staggered", triton=False, region_key="packed",
             merge_stages=None, proportional_attn=False, clamp_patch=False,
             merge_reduce="scatter", seed=None, shard=None)
    reduce_of = {v: k for k, v in MERGE_REDUCE_TAG.items()}
    rest = name[len(policy) + 1:].split("_") if name != policy else []
    for tok in rest:
        if tok in ("dense", "flash"):
            d["attention"] = tok
        elif d["attention"] is None:
            d["merge_stages"] = tok          # e.g. enc3dec3, before the path
        elif tok.startswith("rs"):
            d["region_scale"] = float(tok[2:])
        elif tok.startswith("r"):
            d["merge_ratio"] = float(tok[1:])
        elif tok.startswith("tau"):
            d["tau"] = float(tok[3:])
        elif tok.startswith("cell"):
            d["algm_cell"] = int(tok[4:])
        elif tok.startswith("off"):
            d["algm_cell_offset"] = int(tok[3:])
        elif tok.startswith("w") and tok[1:].isdigit():
            d["algm_window"] = int(tok[1:])
        elif tok.startswith("L"):
            d["lifetime"] = float(tok[1:])
        elif tok.startswith("conf"):
            d["confidence"] = float(tok[4:])
        elif tok == "fixed":
            d["expiry"] = tok
        elif tok == "triton":
            d["triton"] = True
        elif tok.startswith("key"):
            d["region_key"] = tok[3:]
        elif tok == "propattn":
            d["proportional_attn"] = True
        elif tok == "clamp":
            d["clamp_patch"] = True
        elif tok in reduce_of:
            d["merge_reduce"] = reduce_of[tok]
        elif tok.startswith("seed"):
            d["seed"] = int(tok[4:])
        elif tok.startswith("shard"):
            k, n = tok[5:].split("of")
            d["shard"] = (int(k), int(n))
        else:
            return None
    swept = SWEPT.get(policy)
    d["swept"] = d.get(swept) if swept else None
    return d


def rename(name, **changes):
    """`name` with some settings changed, e.g. the FlashAttention twin of a
    dense run: rename(name, attention="flash")."""
    d = parse_run_name(name)
    if d is None:
        raise ValueError(f"not a run name: {name}")
    d.update(changes)
    d.pop("swept")
    return run_name(d.pop("policy"), d.pop("attention"), **d)


def strip_shard(name):
    """`<run>_shardKofN` -> `<run>`; other names are returned unchanged."""
    d = parse_run_name(name)
    if d is None or d["shard"] is None:
        return name
    return name[:name.rindex("_shard")]
