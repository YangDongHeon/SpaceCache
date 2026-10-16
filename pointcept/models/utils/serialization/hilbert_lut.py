"""Table-driven 3D Hilbert encoding with the same output as `hilbert.encode`.

`hilbert.encode` loops over (bit, dimension) in Python, so its number of
tensor ops grows with the serialization depth. Here the curve is encoded as
a finite-state machine over sub-cube orientations, MSB level first:

    key = 0; state = 0
    for level in MSB..LSB:
        octant = (x_b << 2) | (y_b << 1) | z_b
        key    = (key << 3) | DIGIT[state, octant]
        state  = NEXT[state, octant]

and k levels are folded into one table over 8**k chunk values, so a depth-d
code takes ceil(d / k) table lookups.

The tables are derived from `hilbert.encode` on first use (a BFS over
prefixes, since Hilbert codes are prefix-determined) and checked against it
once per (device, depth). Environment variables: HILBERT_LUT=0 selects the
reference encoder (see default.py), HILBERT_LUT_VALIDATE=0 skips the check,
HILBERT_LUT_CHUNK sets k (default 4).
"""

from __future__ import annotations

import os

import torch

from .hilbert import encode as _ref_encode

# levels folded into one table (n_states * 8**k entries each for DIGIT/NEXT)
CHUNK_BITS = int(os.environ.get("HILBERT_LUT_CHUNK", "4"))

_base = None                 # (DIGIT, NEXT) for one level, on CPU
_tables: dict = {}           # (device, k) -> (DIGIT, NEXT)
_spread: dict = {}           # (device, k) -> spread table
_checked: set = set()        # (device, num_bits) already validated


# --- deriving the state machine from the reference implementation ----------


def _key(octs):
    """Reference Hilbert index of the prefix `octs` (a tuple of octants, MSB
    first), i.e. `encode` at depth len(octs)."""
    if not octs:
        return 0
    x = y = z = 0
    for o in octs:
        x = (x << 1) | ((o >> 2) & 1)
        y = (y << 1) | ((o >> 1) & 1)
        z = (z << 1) | (o & 1)
    locs = torch.tensor([[x, y, z]], dtype=torch.long)
    return int(_ref_encode(locs, 3, len(octs)).reshape(-1)[0])


def _sig(octs):
    """Signature of the state at prefix `octs`: the digit each next octant
    produces. Asserts that extending a prefix keeps the digits already
    emitted (prefix determinism)."""
    base = _key(octs)
    out = []
    for o in range(8):
        k = _key(octs + (o,))
        assert k >> 3 == base, (
            f"hilbert.encode is not prefix-determined at {octs}+{o}: "
            f"{k >> 3} != {base}. The LUT construction assumes the first b "
            f"levels of the code depend only on the first b bits of each "
            f"coordinate; if that changed, this module must be rewritten.")
        out.append(k & 7)
    return tuple(out)


def _build_base():
    """BFS over prefixes until the state set closes. Returns (DIGIT, NEXT),
    both (n_states, 8) int64 on CPU."""
    reps = [()]                       # one representative prefix per state
    index = {_sig(()): 0}
    digit, nxt = [], []
    i = 0
    while i < len(reps):
        p = reps[i]
        s = _sig(p)
        row = []
        for o in range(8):
            child = _sig(p + (o,))
            if child not in index:
                index[child] = len(reps)
                reps.append(p + (o,))
            row.append(index[child])
        digit.append(list(s))
        nxt.append(row)
        i += 1
        assert len(reps) <= 48, "state set did not close; unexpected curve"

    # a signature is a valid state only if its transitions do not depend on
    # the prefix that produced it
    for p in reps:
        s = index[_sig(p)]
        for o in range(8):
            assert index[_sig(p + (o,))] == nxt[s][o], (
                f"state {s} is not well defined: prefix {p} disagrees on "
                f"octant {o}. The signature is too coarse to be the state.")
    return (torch.tensor(digit, dtype=torch.long),
            torch.tensor(nxt, dtype=torch.long))


def _compose(d1, n1, d2, n2, k2):
    """Fold a table and then a k2-level table into one wider table.

    Entry (s, c1, c2) runs chunk c1 from state s, then c2 from the state
    reached; (c1, c2) is flattened to c1 * 8**k2 + c2 (MSB level first), as
    `encode` builds the chunk index.
    """
    d_mid, n_mid = d2[n1], n2[n1]                     # (S, 8**k1, 8**k2)
    d = (d1[:, :, None] << (3 * k2)) | d_mid
    return (d.reshape(d.shape[0], -1),
            n_mid.reshape(n_mid.shape[0], -1))


def _table(device, k):
    """(DIGIT, NEXT) covering k levels, on `device`."""
    hit = _tables.get((device, k))
    if hit is not None:
        return hit
    global _base
    if _base is None:
        _base = _build_base()
    d, n = _base
    for _ in range(k - 1):                            # 1 -> 2 -> ... -> k
        d, n = _compose(d, n, _base[0], _base[1], 1)
    out = (d.to(device), n.to(device))
    _tables[(device, k)] = out
    return out


def _spread_table(device, k):
    """v -> sum_j bit_j(v) * 8**j, so a k-bit field becomes one axis of the
    chunk index. Level 0 of the chunk is the MSB and lands at 8**(k-1)."""
    hit = _spread.get((device, k))
    if hit is not None:
        return hit
    v = torch.arange(1 << k, dtype=torch.long, device=device)
    out = torch.zeros_like(v)
    for j in range(k):
        out |= ((v >> j) & 1) << (3 * j)
    _spread[(device, k)] = out
    return out


# --- the encoder -----------------------------------------------------------


def _encode_impl(locs, num_bits, k_max):
    dev = locs.device
    x = locs[:, 0].long()
    y = locs[:, 1].long()
    z = locs[:, 2].long()
    key = torch.zeros_like(x)
    state = torch.zeros_like(x)

    lvl = 0
    while lvl < num_bits:
        k = min(k_max, num_bits - lvl)  # the last chunk may be shorter
        sh = num_bits - lvl - k
        m = (1 << k) - 1
        sp = _spread_table(dev, k)
        c = (sp[(x >> sh) & m] << 2) | (sp[(y >> sh) & m] << 1) | sp[(z >> sh) & m]
        d, n = _table(dev, k)
        idx = state * (1 << (3 * k)) + c
        key = (key << (3 * k)) | d.reshape(-1)[idx]
        state = n.reshape(-1)[idx]
        lvl += k
    return key


def encode(locs, num_dims, num_bits, chunk_bits=None):
    """Drop-in for `hilbert.encode` when num_dims == 3."""
    assert num_dims == 3, "the LUT is 3D-only; use hilbert.encode otherwise"
    k_max = max(1, int(chunk_bits or CHUNK_BITS))
    _validate(locs.device, num_bits, k_max)
    return _encode_impl(locs, num_bits, k_max)


def _validate(device, num_bits, k_max):
    """Check the LUT against `hilbert.encode` once per (device, depth), on up
    to 4096 points plus the cube corners (exhaustive for small depths)."""
    if (device, num_bits) in _checked:
        return
    _checked.add((device, num_bits))
    if os.environ.get("HILBERT_LUT_VALIDATE", "1") != "1":
        return
    lim = 1 << num_bits
    if lim ** 3 <= 4096:                              # small depth: exhaustive
        a = torch.arange(lim ** 3)
        locs = torch.stack([a // (lim * lim), (a // lim) % lim, a % lim], -1)
    else:
        g = torch.Generator().manual_seed(0)
        locs = torch.randint(0, lim, (4096, 3), generator=g)
        # cube corners, where orientation errors show up first
        edge = torch.tensor([[0, 0, 0], [lim - 1, lim - 1, lim - 1],
                             [lim - 1, 0, 0], [0, lim - 1, 0], [0, 0, lim - 1],
                             [0, lim - 1, lim - 1], [lim - 1, 0, lim - 1],
                             [lim - 1, lim - 1, 0]])
        locs = torch.cat([edge, locs])
    locs = locs.to(device)
    ref = _ref_encode(locs, 3, num_bits).reshape(-1)
    got = _encode_impl(locs, num_bits, k_max)
    bad = int((ref != got).sum())
    assert bad == 0, (
        f"hilbert LUT disagrees with hilbert.encode on {bad}/{len(ref)} points "
        f"at depth {num_bits}. Set HILBERT_LUT=0 to fall back while this is "
        f"investigated.")
