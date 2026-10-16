"""Helpers for feeding a dataset sample directly to a segmentor."""

from __future__ import annotations


def drop_keys(inp: dict, keys) -> dict:
    """Pop non-model keys from a data dict in place."""
    for k in keys:
        inp.pop(k, None)
    return inp


NON_MODEL_KEYS = ("segment", "origin_segment", "inverse", "name", "split",
                  "pose")
