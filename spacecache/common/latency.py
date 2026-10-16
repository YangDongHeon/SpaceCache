"""Frame latency of a `run.py` result, including the Commit Gate.

run.py closes the frame timer after the SpaceCache commit and marks such
results with `commit_in_frame`. Results without the mark timed the frame
without the commit; for those the per-frame commit time (stage 3 + stage 4,
`spacecache_ms`) is added back, so every table reports the same
quantity.
"""

from __future__ import annotations


def commit_ms(r, stat="mean"):
    """Per-frame commit time of a SpaceCache result (0 for other methods)."""
    bd = r.get("spacecache_ms") or {}
    if stat == "median":
        s3, s4 = bd.get("median") or {}, bd.get("s4_median") or {}
    else:
        s3, s4 = bd, bd.get("s4") or {}
    return float(s3.get("commit") or 0.0) + float(s4.get("commit") or 0.0)


def with_commit(r, ms, stat="mean"):
    """`ms` plus the commit time when the run timed it outside the frame."""
    if ms is None or r.get("commit_in_frame"):
        return ms
    return float(ms) + commit_ms(r, stat)


def frame_ms(r, stat="mean"):
    """Clean frame latency: the mean over passes 2..N of a multi-pass run
    (pass 1 carries the FLOPs / memory hooks), else `timing.total_ms`."""
    t = r.get("timing") or {}
    n = r.get("frames_per_pass")
    pfm = t.get("per_frame_ms") or []
    if n and len(pfm) > n:
        ms = sum(pfm[n:]) / (len(pfm) - n)
    else:
        ms = t.get("total_ms")
    return with_commit(r, ms, stat)
