"""Latency of the packed vs the prime region key on the cache operations.

Every op that touches the key is timed in both modes on the same frame:

  key (torch)        to_global + voxel_hash
  lookup (fused)     triton lookup_coords (transform + key + table search)
  lookup (torch)     searchsorted + compare on precomputed keys
  lookup_keys        triton table search on precomputed keys
  sort               torch.sort of the keys (commit sorts the staged cells)
  due (torch)        SpaceCache._next_due, modular expiry
  due (fused)        triton stage_classify, modular expiry

The two modes are interleaved round by round and each entry is the median
over all timed reps (CUDA events, host launch included). Before timing, the
script checks that both keys give the same partition of the points, the
same hits and the same due frames, and that Triton matches torch.

    PYTHONPATH=. python spacecache/experiments/region_key_bench.py
    PYTHONPATH=. python spacecache/experiments/region_key_bench.py \
        --sizes 2000,8000,32000,128000 --iters 300 --out region_key_bench.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import torch

sys.path.insert(0, ".")

from spacecache.common import region_key  # noqa: E402
from spacecache.common import triton_cache as _tc  # noqa: E402
from spacecache.common.helpers import to_global, voxel_hash  # noqa: E402
from spacecache.common.space_cache import SpaceCache  # noqa: E402

MODES = ("prime", "packed")


def _time(fn, iters, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _pose(yaw, t, dev):
    c, s = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[c, -s, 0.0, t[0]], [s, c, 0.0, t[1]],
                         [0.0, 0.0, 1.0, t[2]], [0.0, 0.0, 0.0, 1.0]],
                        device=dev)


def _frame(n, extent, scale, dev, gen):
    """LiDAR-like points in the sensor frame (scaled coordinate units)."""
    r = torch.rand(n, device=dev, generator=gen) ** 0.5 * extent
    a = torch.rand(n, device=dev, generator=gen) * 6.2831853
    z = torch.rand(n, device=dev, generator=gen) * 6.0 - 2.0
    return torch.stack((r * torch.cos(a), r * torch.sin(a), z), 1) * scale


def check_equivalence(coord, pose, prev_pose, scale, cell, t, life, report):
    """Both modes: same partition, hits and due frames; Triton == torch."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        report.append(f"  [{'ok' if cond else 'FAIL'}] {name}")

    g = to_global(coord, pose, scale=scale)
    g_prev = to_global(coord, prev_pose, scale=scale)
    vox = torch.floor(g / cell).to(torch.long)
    keys, hits, dues = {}, {}, {}
    for mode in MODES:
        region_key.set_mode(mode)
        k = voxel_hash(g, cell)
        tab = torch.unique(voxel_hash(g_prev, cell))
        slot = torch.searchsorted(tab, k).clamp(max=tab.numel() - 1)
        hit = tab[slot] == k
        due = SpaceCache._next_due(k, life, t)
        keys[mode], hits[mode], dues[mode] = k, hit, due
        if _tc.is_available():
            k_t, _, hit_t = _tc.lookup_coords(coord, pose, scale, cell, tab)
            idx = torch.arange(k.numel(), device=k.device)
            _, due_t, _ = _tc.stage_classify(k, idx, life, None, t)
            check(f"{mode}: triton keys == torch "
                  f"(diffs={int((k_t != k).sum())})", (k_t == k).all())
            check(f"{mode}: triton hits == torch", (hit_t == hit).all())
            check(f"{mode}: triton due == torch", (due_t == due).all())

    region_key.set_mode("packed")
    n_cell = torch.unique(vox, dim=0).shape[0]
    n_pk = torch.unique(keys["packed"]).numel()
    n_pr = torch.unique(keys["prime"]).numel()
    n_pair = torch.unique(torch.stack((keys["packed"], keys["prime"]), 1),
                          dim=0).shape[0]
    check(f"packed one-to-one ({n_pk} keys / {n_cell} cells)", n_pk == n_cell)
    report.append(f"  [info] prime collisions: {n_cell - n_pr} cells")
    if n_pr == n_cell:
        check("same partition", n_pair == n_pk)
    check("decode(encode(v)) == v",
          (region_key.decode(keys["packed"]) == vox).all())
    check("phase(packed) == prime key",
          (region_key.phase(keys["packed"]) == keys["prime"]).all())
    if n_pr == n_cell:
        check("same hits", (hits["packed"] == hits["prime"]).all())
    check("same due frames", (dues["packed"] == dues["prime"]).all())
    return ok, float(hits["packed"].float().mean())


def bench_size(n, args, dev):
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    scale, cell = args.coord_scale, args.cell * args.coord_scale
    coord = _frame(n, args.extent, scale, dev, gen)
    pose = _pose(0.7, [2513.4, -1188.9, 31.2], dev)
    prev_pose = _pose(0.69, [2512.2, -1188.1, 31.2], dev)
    t = 17
    life = torch.full((n,), args.life, dtype=torch.long, device=dev)
    idx = torch.arange(n, device=dev)

    report = []
    ok, hit_frac = check_equivalence(coord, pose, prev_pose, scale, cell, t,
                                     life, report)

    # per-mode inputs of the ops that start from keys
    state = {}
    for mode in MODES:
        region_key.set_mode(mode)
        g_prev = to_global(coord, prev_pose, scale=scale)
        tab = torch.unique(voxel_hash(g_prev, cell))
        k = voxel_hash(to_global(coord, pose, scale=scale), cell)
        state[mode] = (tab, k)

    def ops(mode):
        tab, k = state[mode]
        o = {
            "key (torch)": lambda: voxel_hash(
                to_global(coord, pose, scale=scale), cell),
            "lookup (torch)": lambda: tab[torch.searchsorted(tab, k).clamp(
                max=tab.numel() - 1)] == k,
            "sort": lambda: torch.sort(k),
            "due (torch)": lambda: SpaceCache._next_due(k, life, t),
        }
        if _tc.is_available():
            o["lookup (fused)"] = lambda: _tc.lookup_coords(
                coord, pose, scale, cell, tab)
            o["lookup_keys (fused)"] = lambda: _tc.lookup_keys(k, tab)
            o["due (fused)"] = lambda: _tc.stage_classify(
                k, idx, life, None, t)
        return o

    names = list(ops("packed"))
    times = {m: {name: [] for name in names} for m in MODES}
    per_round = max(1, args.iters // args.rounds)
    for r in range(args.rounds):
        order = MODES if r % 2 == 0 else MODES[::-1]
        for mode in order:
            region_key.set_mode(mode)
            for name, fn in ops(mode).items():
                times[mode][name] += _time(fn, per_round,
                                           warm=10 if r == 0 else 3)
    region_key.set_mode("packed")

    rows = []
    for name in names:
        pr = _median(times["prime"][name])
        pk = _median(times["packed"][name])
        rows.append({"op": name, "prime_ms": pr, "packed_ms": pk,
                     "delta_pct": 100.0 * (pk - pr) / pr if pr else 0.0})
    return {"n": n, "table": int(state["packed"][0].numel()),
            "hit_frac": hit_frac, "checks_ok": ok, "checks": report,
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,4000,16000,64000",
                    help="token counts per frame (comma-separated)")
    ap.add_argument("--cell", type=float, default=0.4,
                    help="cell size in meters (stage 3 at 5 cm = 0.4)")
    ap.add_argument("--coord_scale", type=float, default=1.0,
                    help="scale of the input coordinates (RandomScale)")
    ap.add_argument("--extent", type=float, default=75.0,
                    help="sensor range in meters")
    ap.add_argument("--life", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200,
                    help="timed reps per op and mode")
    ap.add_argument("--rounds", type=int, default=4,
                    help="interleaved prime/packed rounds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional JSON output")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 1
    dev = torch.device("cuda")
    if not _tc.is_available():
        print("triton unavailable: timing the torch ops only")
    print(f"GPU: {torch.cuda.get_device_name(dev)}, cell {args.cell} m, "
          f"L={args.life}")

    results, all_ok = [], True
    for n in (int(x) for x in args.sizes.split(",")):
        res = bench_size(n, args, dev)
        results.append(res)
        all_ok &= res["checks_ok"]
        print(f"\n== N={n} tokens, table {res['table']} cells, "
              f"hit {res['hit_frac'] * 100:.1f}%")
        print("\n".join(res["checks"]))
        print(f"  {'op':<22}{'prime ms':>10}{'packed ms':>11}{'delta':>9}")
        for r in res["rows"]:
            print(f"  {r['op']:<22}{r['prime_ms']:>10.4f}"
                  f"{r['packed_ms']:>11.4f}{r['delta_pct']:>+8.1f}%")

    print(f"\nequivalence checks: {'all ok' if all_ok else 'FAILED'}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1)
        print(f"saved {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
