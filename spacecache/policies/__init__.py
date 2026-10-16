"""Inference policies evaluated by `spacecache/experiments/run.py`.

  backbone.py     the unmodified backbone (`backbone`)
  spacecache.py   SpaceCache (`spacecache_s3`, `spacecache_s34`) and the
                  single-stage placements of Table 3(b) (`spacecache_s{0,1,2,4}`)
  tome.py         ToMe on point tokens (`tome`)
  algm.py         ALGM on point tokens (`algm`)
  gitmerge3d.py   GitMerge3D (`gitmerge3d`, `gitmerge3d_adaptive`)

`build_policy(name)` returns a fresh policy; the names are those of
`spacecache.common.naming`.
"""

from __future__ import annotations


def _registry():
    from spacecache.policies.algm import AlgmPolicy
    from spacecache.policies.backbone import BackbonePolicy
    from spacecache.policies.gitmerge3d import (
        GitMerge3DAdaptivePolicy, GitMerge3DPolicy,
    )
    from spacecache.policies.spacecache import (
        SpaceCacheS0Policy, SpaceCacheS1Policy,
        SpaceCacheS2Policy, SpaceCacheS34Policy,
        SpaceCachePolicy, SpaceCacheS4Policy,
    )
    from spacecache.policies.tome import TomePolicy
    classes = (BackbonePolicy, SpaceCachePolicy,
               SpaceCacheS34Policy, SpaceCacheS0Policy,
               SpaceCacheS1Policy, SpaceCacheS2Policy,
               SpaceCacheS4Policy, TomePolicy, AlgmPolicy,
               GitMerge3DPolicy, GitMerge3DAdaptivePolicy)
    return {c.name: c for c in classes}


def build_policy(name):
    """A new policy instance for `--policy name`."""
    reg = _registry()
    if name not in reg:
        raise SystemExit(f"unknown --policy {name!r}; choose from "
                         f"{', '.join(reg)}")
    return reg[name]()
