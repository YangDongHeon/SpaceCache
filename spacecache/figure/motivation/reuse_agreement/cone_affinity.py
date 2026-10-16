"""Prototype agreement of cached encoder features, per stage (Fig. 2c).

Both frames of a pair (A, B) are forwarded. At every encoder stage, B's
tokens are matched to A's world cells, and each class prototype (cone) is
the mean feature of B's single-class tokens of that ground-truth class.
Agreement asks whether A's cached cell feature and B's recomputed feature
pick the same prototype; per-class in-cone rates and cone margins are
saved as well.

    PYTHONPATH=. python \\
        spacecache/figure/motivation/reuse_agreement/cone_affinity.py \\
        --config spacecache/configs/waymo/utonia_dec/reuse_agreement.py \\
        --name waymo_utonia

The ScanNet runs use `--max_pairs_per_scene 3` (default 1). Writes
`<save_path>/<name>.{json,npz}` for `plot_cone_trend_multi.py`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import types
from collections import defaultdict

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _ROOT)

from pointcept.datasets import build_dataset                      # noqa: E402
from pointcept.utils.config import Config                         # noqa: E402

from spacecache.common.helpers import (                             # noqa: E402
    aggregate_cell_features, build_segmentor, extract_grid_size,
    extract_random_scale, match_b_to_cells, prepare_input, set_seed,
    to_global,
)
from spacecache.common.runtime import NON_MODEL_KEYS, drop_keys     # noqa: E402
from spacecache.figure.motivation.temporal_overlap.stage_overlap import (  # noqa: E402
    PoseSource, _quantiles, build_pairs,
)

log = logging.getLogger("cone_affinity")

EPS = 1e-9
VARIANTS = ("fresh", "fresh_overlap", "reused")
RULES = ("pure", "dominant")


# ---------------------------------------------------------------------------
#  Stage capture and world-cell matching
# ---------------------------------------------------------------------------


def _seed_forward(seed, device):
    """Reset the torch RNGs before a forward, so both frames of a pair see
    the same random draws (e.g. the shuffled serialization orders)."""
    torch.manual_seed(int(seed))
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


@contextlib.contextmanager
def capture_depths(model, holder, *, want_pi=False):
    """Store the output of every encoder stage in `holder`.

    Keys: `feat_enc{d}`, `coord_enc{d}` and, with `want_pi`, the pooling
    inverse `pi{d}` (level d-1 -> d) for d >= 1. Tensors are detached
    clones, so later in-place ops cannot alter them.
    """
    handles = []
    enc = model.backbone.enc._modules

    def _mk_enc(d):
        def hook(_m, _i, out):
            holder[f"feat_enc{d}"] = out.feat.detach().clone()
            if want_pi and d >= 1:
                holder[f"pi{d}"] = out.pooling_inverse.detach().clone()
            holder[f"coord_enc{d}"] = out.coord.detach().clone()
        return hook

    for d in range(5):
        if f"enc{d}" in enc:
            handles.append(enc[f"enc{d}"].register_forward_hook(_mk_enc(d)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def precompute_a_stage_cache(point_a_stage, pose_a, base_voxel, coord_scale,
                             depth, *, match_factor=1.0):
    """Mean stage feature of frame A per world cell of edge
    base_voxel * 2^depth * match_factor.

    Returns (sorted cell hashes, (n_cells, C) cell features).
    """
    voxel_size = float(base_voxel) * (2 ** int(depth)) * float(match_factor)
    coord_a_g = to_global(point_a_stage.coord, pose_a, scale=coord_scale)
    return aggregate_cell_features(coord_a_g, point_a_stage.feat, voxel_size)


def match_b_to_a_stage(point_b_stage, pose_b, base_voxel, coord_scale,
                       hash_a, depth, *, match_factor=1.0):
    """For every stage token of frame B, whether its world cell is in
    `hash_a`.

    Returns (matched (N,) bool, insert (N,) long), where `insert` indexes
    `hash_a` on matched tokens.
    """
    voxel_size = float(base_voxel) * (2 ** int(depth)) * float(match_factor)
    coord_b_g = to_global(point_b_stage.coord, pose_b, scale=coord_scale)
    return match_b_to_cells(coord_b_g, hash_a, voxel_size)


# ---------------------------------------------------------------------------
#  Per-token ground truth at a depth
# ---------------------------------------------------------------------------


def token_gt_counts(holder, depth, inverse_b, gt_raw, nc, n_tok, device):
    """(n_tok, nc) count of raw points of each GT class inside each stage-d
    token.

    Composing the GridPooling inverses pi1..pi{d} (captured with
    `capture_depths(want_pi=True)`) and indexing by GridSample's `inverse`
    maps a raw point to its stage-d token; stage 0 is the input voxel set.
    """
    if depth == 0:
        tok_of_voxel = torch.arange(n_tok, device=device)
    else:
        tok_of_voxel = holder["pi1"]
        for d in range(2, depth + 1):
            tok_of_voxel = holder[f"pi{d}"][tok_of_voxel]
    tok_of_raw = tok_of_voxel[inverse_b]
    valid = (gt_raw >= 0) & (gt_raw < nc)
    counts = torch.zeros(nc * n_tok, device=device)
    flat = gt_raw[valid] * n_tok + tok_of_raw[valid]
    counts.scatter_add_(0, flat, torch.ones(int(valid.sum()), device=device))
    return counts.reshape(nc, n_tok).t().to(torch.int64).cpu().numpy()


# ---------------------------------------------------------------------------
#  Class cones and the statistics read off them
# ---------------------------------------------------------------------------


def _centroids(featB, lab, pure_mask, nc, min_tok, spherical):
    """Per-class cone centre: mean of `featB` over the `pure_mask` tokens
    of the class, for classes with at least `min_tok` such tokens.

    Returns (mu (nc, C) float64, present (nc,) bool).
    """
    C = featB.shape[1]
    mu = np.zeros((nc, C), dtype=np.float64)
    present = np.zeros(nc, dtype=bool)
    for c in range(nc):
        sel = pure_mask & (lab == c)
        n = int(sel.sum())
        if n >= min_tok:
            f = featB[sel].astype(np.float64)
            if spherical:                       # vMF-style mean direction
                f = f / (np.linalg.norm(f, axis=1, keepdims=True) + EPS)
            mu[c] = f.mean(axis=0)
            present[c] = True
    return mu, present


def _classify(feat, mu, present, metric):
    """Nearest cone and similarities over the present classes.

    Returns (pred (M,) global class index, sims (M, K), pres_idx (K,)).
    """
    pres_idx = np.where(present)[0]
    M = mu[pres_idx].astype(np.float64)                    # (K, C)
    f = feat.astype(np.float64)
    if metric == "cosine":
        fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + EPS)
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + EPS)
        sims = fn @ Mn.T                                   # higher = closer
    else:                                                  # euclidean -> -dist
        sims = -np.linalg.norm(f[:, None, :] - M[None, :, :], axis=2)
    pred = pres_idx[np.argmax(sims, axis=1)]
    return pred, sims, pres_idx


def cone_stats(mu, present, feat_test, lab_test, nc, temp, metric):
    """Class-cone affinity of `feat_test` against the cones `mu`.

    Returns None when nothing can be scored. Otherwise a dict with the hard
    confusion (nearest-cone counts, rows = GT class), the soft affinity
    rows (softmax(sim / T) summed per GT class) and the per-token margin
    terms (similarity to the own cone vs. the nearest other cone). Only
    tokens whose own class has a cone are scored.
    """
    if feat_test.shape[0] == 0 or not present.any():
        return None
    pred, sims, pres_idx = _classify(feat_test, mu, present, metric)

    # column of each token's own class among the present cones (-1: none)
    col_of = np.full(nc, -1, dtype=np.int64)
    col_of[pres_idx] = np.arange(pres_idx.size)
    j = col_of[lab_test]
    keep = j >= 0
    if not keep.any():
        return None
    rows = np.arange(int(keep.sum()))
    own_j = j[keep]
    sims_k = sims[keep]

    hard = np.zeros((nc, nc), dtype=np.int64)
    np.add.at(hard, (lab_test[keep], pred[keep]), 1)

    z = sims_k / temp
    z = z - z.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=1, keepdims=True)
    soft = np.zeros((nc, nc), dtype=np.float64)
    row_n = np.zeros(nc, dtype=np.int64)
    lab_k = lab_test[keep]
    for c in np.unique(lab_k):
        m = lab_k == c
        row_n[c] = int(m.sum())
        soft[c, pres_idx] = probs[m].sum(axis=0)  # normalised in summarize
    own = sims_k[rows, own_j]
    other = sims_k.copy()
    other[rows, own_j] = -np.inf
    wrong = other.max(axis=1)
    margin = own - wrong
    return dict(
        hard=hard, soft=soft, row_n=row_n,
        n=int(keep.sum()),
        own_sum=float(own.sum()), wrong_sum=float(wrong.sum()),
        margin_pos=int((margin > 0).sum()),
        soft_own_sum=float(probs[rows, own_j].sum()),
        n_cones=int(pres_idx.size),
    )


def agreement_sims(mu, present, feat_recomputed, feat_cached, metric):
    """Similarity profiles of the recomputed and the cached features against
    the cones, or None with fewer than two cones."""
    if feat_recomputed.shape[0] == 0 or int(present.sum()) < 2:
        return None
    _, sims_f, pres_idx = _classify(feat_recomputed, mu, present, metric)
    _, sims_r, _ = _classify(feat_cached, mu, present, metric)
    return sims_f, sims_r, pres_idx


def agreement_stats(mu, present, feat_recomputed, feat_cached, masks, temp,
                    metric, eps=1e-6, _sims=None, _nc=None):
    """Prototype agreement: does the cached feature pick the cone the
    recomputed one picks?

    No per-token label is needed, so every reused token is scored. Per
    boolean mask in `masks`:

      top1     same nearest cone.
      top2     same set of the two nearest cones.
      soft     sum_c min(q_recomputed, q_cached) over q = softmax(sim / T),
               i.e. 1 - total variation.
      kl_*     KL(recomputed || cached): mean, median and p90 over tokens.
      js_mean  Jensen-Shannon divergence.

    `_cm` is the nearest-cone confusion (rows: recomputed, columns:
    cached); its diagonal gives top1. The softmax-based readings depend on
    T and are compared within a stage; top1 is comparable across stages.
    """
    sims = (_sims if _sims is not None
            else agreement_sims(mu, present, feat_recomputed, feat_cached,
                                metric))
    if sims is None:
        return None
    sims_f, sims_r, pres_idx = sims
    k = sims_f.shape[1]
    nc = int(mu.shape[0]) if mu is not None else int(_nc)

    def _soft(sims):
        z = sims / temp
        z = z - z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=1, keepdims=True)
        # floor q before the logs so a near-zero class cannot dominate the
        # KL; the argmax is unchanged
        return (1.0 - eps) * q + eps / k

    qf, qr = _soft(sims_f), _soft(sims_r)
    arg_f, arg_r = sims_f.argmax(axis=1), sims_r.argmax(axis=1)
    agree = arg_f == arg_r
    if k >= 2:
        t2f = np.sort(np.argpartition(-sims_f, 1, axis=1)[:, :2], axis=1)
        t2r = np.sort(np.argpartition(-sims_r, 1, axis=1)[:, :2], axis=1)
        agree2 = (t2f == t2r).all(axis=1)
    else:
        agree2 = agree
    mass = np.minimum(qf, qr).sum(axis=1)
    kl = (qf * np.log(qf / qr)).sum(axis=1)
    m = 0.5 * (qf + qr)
    js = 0.5 * ((qf * np.log(qf / m)).sum(axis=1)
                + (qr * np.log(qr / m)).sum(axis=1))

    # nearest cones as global class indices (the columns of `sims` are the
    # cones present in this frame only)
    glob_f, glob_r = pres_idx[arg_f], pres_idx[arg_r]

    out = {}
    for name, sel in masks.items():
        if sel is None or not sel.any():
            out[name] = None
            continue
        cm = np.zeros((nc, nc), dtype=np.int64)
        np.add.at(cm, (glob_f[sel], glob_r[sel]), 1)
        out[name] = dict(
            _cm=cm,
            n=int(sel.sum()),
            top1=float(agree[sel].mean()),
            top2=float(agree2[sel].mean()),
            soft=float(mass[sel].mean()),
            kl_mean=float(kl[sel].mean()),
            kl_median=float(np.median(kl[sel])),
            kl_p90=float(np.percentile(kl[sel], 90)),
            js_mean=float(js[sel].mean()),
        )
    return out


AGREE_KEYS = ("top1", "top2", "soft", "kl_mean", "kl_median", "kl_p90",
              "js_mean")


def calibrate_temperature(sims_list, target, lo=1e-3, hi=10.0, iters=32,
                          max_rows=20000, seed=0):
    """T at which softmax(sims / T) has mean normalised entropy `target`.

    Solving T per stage reads every stage at the same sharpness, as the
    similarity spread changes with depth. `sims_list` holds one (M, K)
    matrix per frame (K, the number of cones, varies); entropies are
    normalised by log K and token-weighted. The mean entropy is monotone
    in T, so T is found by bisection in log space.
    """
    rng = np.random.default_rng(seed)
    arrays = []
    for a in sims_list:
        if a.shape[0] == 0 or a.shape[1] < 2:
            continue
        if a.shape[0] > max_rows:            # a random sample of rows
            idx = rng.choice(a.shape[0], max_rows, replace=False)
            a = a[idx]
        arrays.append(a)
    if not arrays:
        return float("nan")

    def norm_entropy(t):
        total, n = 0.0, 0
        for a in arrays:
            z = a / t
            z = z - z.max(axis=1, keepdims=True)
            q = np.exp(z)
            q /= q.sum(axis=1, keepdims=True)
            h = -(q * np.log(np.clip(q, 1e-12, None))).sum(axis=1)
            total += float(h.sum()) / np.log(a.shape[1])
            n += a.shape[0]
        return total / max(n, 1)

    for _ in range(iters):
        mid = np.sqrt(lo * hi)                    # bisect in log T
        if norm_entropy(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


AGREE_SUBSETS = ("all", "pure", "mixed")


def agreement_from_sims(sims, masks, temp, nc, eps=1e-6):
    """`agreement_stats` on precomputed similarity profiles."""
    return agreement_stats(None, None, None, None, masks, temp, None,
                           eps=eps, _sims=sims, _nc=nc)


def _take_cm(agr, depth, store):
    """Move the per-pair confusion matrices of `agr` into `store` (summed
    per (depth, subset)), leaving only scalars in the record."""
    if agr is None:
        return agr
    for subset, block in agr.items():
        if block is None:
            continue
        cm = block.pop("_cm", None)
        if cm is not None:
            store[(depth, subset)] += cm
    return agr


def _new_acc(nc):
    return dict(hard=np.zeros((nc, nc), dtype=np.int64),
                soft=np.zeros((nc, nc), dtype=np.float64),
                row_n=np.zeros(nc, dtype=np.int64),
                n=0, own_sum=0.0, wrong_sum=0.0, margin_pos=0,
                soft_own_sum=0.0, n_cones=0, n_calls=0)


def _merge(acc, st):
    acc["hard"] += st["hard"]
    acc["soft"] += st["soft"]
    acc["row_n"] += st["row_n"]
    for k in ("n", "own_sum", "wrong_sum", "margin_pos", "soft_own_sum",
              "n_cones"):
        acc[k] += st[k]
    acc["n_calls"] += 1


def summarize(acc, class_mask=None):
    """Per-depth statistics from an accumulator.

    `in_cone_macro` is the mean per-class in-cone rate. `class_mask`
    restricts it to one class set for all variants of a depth, since a
    class can lack a cone or overlapped tokens in some variants.
    """
    n = max(acc["n"], 1)
    hard, row = acc["hard"], acc["hard"].sum(axis=1)
    nz = row > 0
    if class_mask is not None:
        nz = nz & class_mask
    macro = (float(np.mean(np.diag(hard)[nz] / row[nz])) if nz.any()
             else float("nan"))
    soft = acc["soft"] / np.maximum(acc["row_n"], 1)[:, None]
    return dict(
        n_tokens=int(acc["n"]),
        in_cone_micro=float(np.trace(hard) / max(hard.sum(), 1)),
        in_cone_macro=macro,
        in_cone_soft=float(acc["soft_own_sum"] / n),
        cos_own=float(acc["own_sum"] / n),
        cos_other=float(acc["wrong_sum"] / n),
        cone_margin=float((acc["own_sum"] - acc["wrong_sum"]) / n),
        margin_positive=float(acc["margin_pos"] / n),
        n_cones_mean=float(acc["n_cones"] / max(acc["n_calls"], 1)),
        chance=float(max(acc["n_calls"], 1) / max(acc["n_cones"], 1)),
        soft_rows=soft,
        hard_rows=hard,
    )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Config with a trained `weight`, e.g. "
                         "spacecache/configs/waymo/utonia_dec/reuse_agreement.py.")
    ap.add_argument("--depths", default="0,1,2,3,4",
                    help="Comma-separated encoder stages.")
    ap.add_argument("--stride", type=int, default=None,
                    help="Frames between A and B (default: cfg.stride).")
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="Total pairs (default: cfg.max_pairs).")
    ap.add_argument("--max_pairs_per_scene", type=int, default=1,
                    help="Pairs per sequence, spread evenly over it.")
    ap.add_argument("--mf", type=float, default=1.0,
                    help="Match factor for the world cell (1.0 = the "
                         "stage's own cell).")
    ap.add_argument("--min_cone_tokens", type=int, default=20,
                    help="Tokens a class needs in the current frame to get "
                         "a cone.")
    ap.add_argument("--soft_temp", type=float, default=0.15,
                    help="Softmax temperature of the in-cone soft rates, "
                         "and of agreement under --temp_mode fixed.")
    ap.add_argument("--temp_mode", default="entropy",
                    choices=["entropy", "fixed"],
                    help="entropy: solve one agreement temperature per "
                         "stage for a common mean entropy; fixed: use "
                         "--soft_temp at every stage.")
    ap.add_argument("--temp_target", type=float, default=0.5,
                    help="Target mean entropy / log K under --temp_mode "
                         "entropy.")
    ap.add_argument("--temp_cal_pairs", type=int, default=5,
                    help="Pairs used to solve the temperatures; they are "
                         "scored like the others afterwards.")
    ap.add_argument("--metric", default="cosine",
                    choices=["cosine", "euclidean"])
    ap.add_argument("--spherical", action="store_true",
                    help="Cone centre = mean of the normalised tokens (vMF "
                         "mean direction) instead of the plain mean.")
    ap.add_argument("--decommon", default="none",
                    choices=["none", "centroid_mean"],
                    help="Subtract the class-balanced mean of the class "
                         "centroids from features and cones before the "
                         "similarity.")
    ap.add_argument("--out", default=None, help="Override cfg.save_path.")
    ap.add_argument("--name", default=None,
                    help="Output basename (default: the config's stem).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    cfg = Config.fromfile(args.config)
    requested = str(args.device or cfg.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.info("cuda unavailable — falling back to cpu")
        requested = "cpu"
    device = torch.device(requested)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 37))
    set_seed(seed)
    weight = cfg.get("weight")
    assert weight, "config must specify `weight` (a trained checkpoint)"
    stride = (args.stride if args.stride is not None
              else int(cfg.get("stride", 1)))
    depths = [int(d) for d in args.depths.split(",") if d.strip() != ""]
    out_dir = args.out or cfg.get("save_path",
                                  "output/motivation/reuse_agreement")
    name = args.name or os.path.splitext(os.path.basename(args.config))[0]
    os.makedirs(out_dir, exist_ok=True)

    model = build_segmentor(cfg.model, weight, device)
    dataset = build_dataset(cfg.data.val)
    base_voxel = extract_grid_size(cfg.data.val.transform)
    coord_scale = extract_random_scale(cfg.data.val.transform)
    nc = int(cfg.model.num_classes)
    names = list(cfg.data.get("names", [f"class_{i}" for i in range(nc)]))
    cells_m = [base_voxel * (2 ** d) / coord_scale for d in depths]
    log.info("dataset %s: %d frames | depths=%s | cells=%s m",
             cfg.data.val.type, len(dataset), depths,
             [f"{c:.2f}" for c in cells_m])

    pairs = build_pairs(dataset, stride, args.max_pairs_per_scene,
                        args.max_pairs if args.max_pairs is not None
                        else cfg.get("max_pairs", None))
    log.info("pairs: %d over %d drives (stride=%d)", len(pairs),
             len({p[2] for p in pairs}), stride)
    if not pairs:
        raise SystemExit("no pairs — check stride / split / scene_list")

    pose_of = PoseSource(dataset)
    acc = {(d, r, v): _new_acc(nc)
           for d in depths for r in RULES for v in VARIANTS}
    purity = defaultdict(list)          # depth -> pure fraction per pair
    overlap = defaultdict(list)         # depth -> matched fraction per pair
    agree_pairs = defaultdict(list)     # depth -> per-pair agreement dicts
    agree_cm = defaultdict(lambda: np.zeros((nc, nc), dtype=np.int64))
    temps = {d: float(args.soft_temp) for d in depths}
    calibrating = args.temp_mode == "entropy"
    cal_sims = defaultdict(list)        # depth -> recomputed sims to solve on
    cal_stash = []                      # (rec entry, depth, sims, masks)
    records = []

    if device.type == "cuda":
        warm = dataset[0]
        warm.pop("pose", None)
        winp = prepare_input(warm, device)
        drop_keys(winp, NON_MODEL_KEYS)
        with torch.inference_mode():
            _ = model(winp)
        torch.cuda.synchronize()
        del warm, winp

    for k, (idx_a, idx_b, seg) in enumerate(pairs):
        set_seed(seed + int(idx_a))
        data_a = dataset[idx_a]
        set_seed(seed + int(idx_b))
        data_b = dataset[idx_b]
        pose_a = pose_of(idx_a, data_a).to(device)
        pose_b = pose_of(idx_b, data_b).to(device)
        inverse_b = torch.as_tensor(data_b["inverse"],
                                    dtype=torch.long).to(device)
        gt_raw = torch.as_tensor(data_b["origin_segment"],
                                 dtype=torch.long).to(device)
        inp_a = prepare_input(data_a, device)
        inp_b = prepare_input(data_b, device)
        for inp in (inp_a, inp_b):
            drop_keys(inp, NON_MODEL_KEYS)
        pair_seed = (seed + int(idx_a) * 31 + int(idx_b)) % (2 ** 31 - 1)

        ha, hb = {}, {}
        _seed_forward(pair_seed, device)
        with capture_depths(model, ha, want_pi=False):
            with torch.inference_mode():
                _ = model(inp_a)
        _seed_forward(pair_seed, device)
        with capture_depths(model, hb, want_pi=True):
            with torch.inference_mode():
                _ = model(inp_b)

        rec = dict(idx_a=int(idx_a), idx_b=int(idx_b), segment=str(seg),
                   depths={})
        for d in depths:
            a_shim = types.SimpleNamespace(coord=ha[f"coord_enc{d}"],
                                           feat=ha[f"feat_enc{d}"].float())
            b_shim = types.SimpleNamespace(coord=hb[f"coord_enc{d}"],
                                           feat=hb[f"feat_enc{d}"].float())
            hash_a, feat_a_cells = precompute_a_stage_cache(
                a_shim, pose_a, base_voxel, coord_scale, d,
                match_factor=args.mf)
            matched, insert = match_b_to_a_stage(
                b_shim, pose_b, base_voxel, coord_scale, hash_a, d,
                match_factor=args.mf)
            n_tok = int(b_shim.feat.shape[0])
            served = feat_a_cells[insert]

            counts = token_gt_counts(hb, d, inverse_b, gt_raw, nc, n_tok,
                                     device)
            featB = b_shim.feat.detach().cpu().numpy()
            featA = served.detach().float().cpu().numpy()
            match = matched.detach().cpu().numpy().astype(bool)

            tot = counts.sum(axis=1)
            valid = tot > 0
            pure = valid & ((counts > 0).sum(axis=1) == 1)
            lab = counts.argmax(axis=1)
            purity[d].append(float(pure.sum()) / max(int(valid.sum()), 1))
            overlap[d].append(float(match.sum()) / max(n_tok, 1))

            rec["depths"][str(d)] = dict(
                n_tokens=n_tok, n_valid=int(valid.sum()),
                n_pure=int(pure.sum()), n_matched=int(match.sum()),
                rules={},
            )

            # agreement: cones from pure tokens, scored on every reused
            # token (pure and mixed)
            mu_p, present_p = _centroids(featB, lab, pure, nc,
                                         args.min_cone_tokens,
                                         args.spherical)
            masks = dict(all=np.ones(int(match.sum()), dtype=bool),
                         pure=pure[match], mixed=(valid & ~pure)[match])
            sims = agreement_sims(mu_p, present_p, featB[match], featA[match],
                                  args.metric)
            entry = rec["depths"][str(d)]
            if sims is None:
                entry["agreement"] = None
            elif calibrating:
                # keep the profiles and score them once T is solved
                sims32 = (sims[0].astype(np.float32),
                          sims[1].astype(np.float32), sims[2])
                cal_sims[d].append(sims32[0])
                cal_stash.append((entry, d, sims32, masks))
            else:
                agr = _take_cm(agreement_from_sims(sims, masks, temps[d], nc),
                               d, agree_cm)
                entry["agreement"] = agr
                if agr is not None:
                    agree_pairs[d].append(agr)
            for rule in RULES:
                pool = pure if rule == "pure" else valid
                fB, fA = featB, featA
                if args.decommon == "centroid_mean":
                    # shared cone vertex: class-balanced mean of the class
                    # means, subtracted from both frames' features
                    present_c = [c for c in range(nc)
                                 if (pool & (lab == c)).sum() >=
                                 args.min_cone_tokens]
                    if present_c:
                        vertex = np.mean(
                            [featB[pool & (lab == c)].mean(axis=0)
                             for c in present_c], axis=0)
                        fB, fA = featB - vertex, featA - vertex
                mu, present = _centroids(
                    fB, lab, pool, nc, args.min_cone_tokens, args.spherical)
                tests = dict(
                    fresh=(fB, pool),
                    fresh_overlap=(fB, pool & match),
                    reused=(fA, pool & match),
                )
                rec_rule = {}
                for variant, (feat, mask) in tests.items():
                    st = cone_stats(mu, present, feat[mask], lab[mask], nc,
                                    args.soft_temp, args.metric)
                    if st is None:
                        continue
                    _merge(acc[(d, rule, variant)], st)
                    rec_rule[variant] = dict(
                        n=st["n"],
                        in_cone=float(np.trace(st["hard"]) /
                                      max(st["hard"].sum(), 1)),
                        cone_margin=float((st["own_sum"] - st["wrong_sum"]) /
                                          max(st["n"], 1)))
                rec["depths"][str(d)]["rules"][rule] = rec_rule
        records.append(rec)
        if calibrating and len(records) >= args.temp_cal_pairs:
            for d in depths:
                if cal_sims[d]:
                    temps[d] = calibrate_temperature(
                        cal_sims[d], args.temp_target)
            for entry, d, sims, masks in cal_stash:
                agr = _take_cm(agreement_from_sims(sims, masks, temps[d], nc),
                               d, agree_cm)
                entry["agreement"] = agr
                if agr is not None:
                    agree_pairs[d].append(agr)
            cal_stash.clear()
            cal_sims.clear()
            calibrating = False
            log.info("temperatures (entropy/logK = %.2f): %s",
                     args.temp_target,
                     {f"d{d}": round(temps[d], 4) for d in depths})
        if (k + 1) % 5 == 0 or k + 1 == len(pairs):
            d_last = depths[-1]
            r = records[-1]["depths"][str(d_last)]["rules"]["pure"]
            log.info("pair %d/%d (%d->%d) d%d pure: fresh %.3f | reused %.3f",
                     k + 1, len(pairs), idx_a, idx_b, d_last,
                     r.get("fresh", {}).get("in_cone", float("nan")),
                     r.get("reused", {}).get("in_cone", float("nan")))

    # aggregate over pairs
    stages, matrices = [], {}
    for d in depths:
        entry = dict(depth=d, cell_m=base_voxel * (2 ** d) / coord_scale,
                     purity=_quantiles(purity[d]),
                     overlap=_quantiles(overlap[d]), rules={})
        # agreement: quantiles of the per-pair values (one vote per pair)
        agree = {}
        for subset in AGREE_SUBSETS:
            vals = [a[subset] for a in agree_pairs[d]
                    if a.get(subset) is not None]
            if not vals:
                continue
            agree[subset] = dict(
                n_pairs=len(vals),
                n_tokens=int(sum(v["n"] for v in vals)),
                **{k: _quantiles([v[k] for v in vals]) for k in AGREE_KEYS},
            )
        entry["agreement"] = agree
        entry["soft_temp"] = temps[d]
        for rule in RULES:
            present = {v: acc[(d, rule, v)]["hard"].sum(axis=1) > 0
                       for v in VARIANTS}
            common = np.logical_and.reduce([present[v] for v in VARIANTS])
            entry.setdefault("class_set", {})[rule] = dict(
                n_common=int(common.sum()),
                per_variant={v: int(present[v].sum()) for v in VARIANTS},
                dropped={v: [names[i]
                             for i in np.where(present[v] & ~common)[0]]
                         for v in VARIANTS},
                absent=[names[i] for i in np.where(
                    ~np.logical_or.reduce([present[v] for v in VARIANTS]))[0]],
            )
            per_variant = {}
            for variant in VARIANTS:
                s = summarize(acc[(d, rule, variant)], common)
                matrices[f"soft_d{d}_{rule}_{variant}"] = s.pop("soft_rows")
                matrices[f"hard_d{d}_{rule}_{variant}"] = s.pop("hard_rows")
                per_variant[variant] = s
            entry["rules"][rule] = per_variant
        stages.append(entry)

    result = dict(
        config=args.config,
        dataset=dict(type=cfg.data.val.type,
                     label=cfg.get("figure_label", name),
                     n_frames=len(dataset), num_classes=nc,
                     class_names=names),
        model=dict(weight=weight, type=cfg.model.get("type"),
                   label=cfg.get("model_label", "")),
        sampling=dict(stride=stride, n_pairs=len(records),
                      n_drives=len({p[2] for p in pairs}),
                      max_pairs_per_scene=args.max_pairs_per_scene,
                      seed=seed),
        settings=dict(depths=depths, match_factor=args.mf,
                      soft_temp=args.soft_temp, temp_mode=args.temp_mode,
                      temp_target=args.temp_target,
                      temp_per_stage={str(d): temps[d] for d in depths},
                      metric=args.metric,
                      spherical=bool(args.spherical), decommon=args.decommon,
                      min_cone_tokens=args.min_cone_tokens,
                      base_voxel_m=base_voxel / coord_scale),
        stages=stages,
        pairs=records,
    )
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    for (d, subset), cm in agree_cm.items():
        matrices[f"agree_d{d}_{subset}"] = cm
    np.savez(os.path.join(out_dir, f"{name}.npz"),
             class_names=np.array(names), depths=np.array(depths), **matrices)

    log.info("=" * 78)
    log.info("%-5s %8s %8s %10s %10s %10s %9s", "depth", "cell(m)", "purity",
             "fresh", "fresh|ovl", "reused", "margin")
    for e in stages:
        p = e["rules"]["pure"]
        log.info("d%-4d %8.2f %7.1f%% %9.3f %10.3f %10.3f %9.3f",
                 e["depth"], e["cell_m"], 100 * e["purity"]["mean"],
                 p["fresh"]["in_cone_macro"],
                 p["fresh_overlap"]["in_cone_macro"],
                 p["reused"]["in_cone_macro"],
                 p["reused"]["cone_margin"])
    for e in stages:
        cs = e.get("class_set", {}).get("pure", {})
        if cs and cs["n_common"] != max(cs["per_variant"].values()):
            log.info("d%d: macro over %d classes shared by all variants; "
                     "scored in some but not all: %s", e["depth"],
                     cs["n_common"],
                     sorted({c for v in cs["dropped"].values() for c in v}))
    log.info("-" * 78)
    log.info("%-5s %8s %8s %9s %9s %9s %9s", "depth", "cell(m)", "T",
             "Top-1%", "Top-2%", "Soft%", "KL med")
    for e in stages:
        a = e.get("agreement", {}).get("all")
        if not a:
            continue
        log.info("d%-4d %8.2f %8.4f %8.1f%% %8.1f%% %8.1f%% %9.3f",
                 e["depth"], e["cell_m"], e.get("soft_temp", float("nan")),
                 100 * a["top1"]["mean"], 100 * a["top2"]["mean"],
                 100 * a["soft"]["mean"], a["kl_median"]["mean"])
    log.info("(agreement = recomputed vs cached on ALL reused tokens, cones "
             "from pure tokens; Top-1 is the one that is safe across stages. "
             "in-cone = macro over classes, label rule = pure; chance ~ "
             "1/%.0f cones)",
             max(stages[-1]["rules"]["pure"]["reused"]["n_cones_mean"], 1.0))
    log.info("wrote %s(.json/.npz)", os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
