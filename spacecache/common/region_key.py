"""Region key: the int64 id of a world cell v = floor(p / cell).

packed (default)
    key = (vx + 2^23) << 39 | (vy + 2^23) << 15 | (vz + 2^14)
    One-to-one for |vx|, |vy| < 2^23 and |vz| < 2^14 cells; keys are
    non-negative and sort in (vx, vy, vz) order.
prime
    key = 73856093 vx + 19349663 vy + 83492791 vz (int64), the linear hash
    of earlier runs. Distinct cells can share a key.

Modular expiry staggers cells by `phase(key) mod L`. The phase is the prime
dot product of v in both modes, so the refresh schedule does not depend on
the key layout.
"""

from __future__ import annotations

import torch

MODES = ("packed", "prime")
PRIMES = (73856093, 19349663, 83492791)
XY_OFF = 1 << 23
Z_OFF = 1 << 14
XY_MASK = (1 << 24) - 1
Z_MASK = (1 << 15) - 1
SHIFT_X = 39
SHIFT_Y = 15

_mode = "packed"


def set_mode(mode):
    global _mode
    if mode not in MODES:
        raise ValueError(f"region key must be one of {MODES}, got {mode!r}")
    _mode = mode


def get_mode():
    return _mode


def is_packed():
    return _mode == "packed"


def _prime_dot(vx, vy, vz):
    return vx * PRIMES[0] + vy * PRIMES[1] + vz * PRIMES[2]


def encode(voxel):
    """(N, 3) int64 cell indices -> (N,) int64 keys."""
    vx, vy, vz = voxel.unbind(-1)
    if is_packed():
        return (((vx + XY_OFF) << SHIFT_X) | ((vy + XY_OFF) << SHIFT_Y)
                | (vz + Z_OFF))
    return _prime_dot(vx, vy, vz)


def decode(key):
    """Packed key -> (N, 3) int64 cell indices."""
    vx = (key >> SHIFT_X) - XY_OFF
    vy = ((key >> SHIFT_Y) & XY_MASK) - XY_OFF
    vz = (key & Z_MASK) - Z_OFF
    return torch.stack((vx, vy, vz), dim=-1)


_PHASE_OFF = PRIMES[0] * XY_OFF + PRIMES[1] * XY_OFF + PRIMES[2] * Z_OFF


def phase(key):
    """Expiry phase of a key (the prime dot product of its cell).

    Works on the shifted bit fields directly and folds the offsets into one
    constant, so the packed key costs 8 elementwise kernels here instead of
    a full decode.
    """
    if not is_packed():
        return key
    ph = (key >> SHIFT_X).mul_(PRIMES[0])
    ph.add_((key >> SHIFT_Y).bitwise_and_(XY_MASK), alpha=PRIMES[1])
    ph.add_(key & Z_MASK, alpha=PRIMES[2])
    return ph.sub_(_PHASE_OFF)
