"""Reuse Agreement probe (Fig. 5, Sec. 4.3): is the feature SpaceCache
serves the feature the backbone would have computed?

Reuse Agreement (RA) follows Appendix A.2. Prototypes mu_c are the mean
features of the pure (single-class) tokens of class c in the current frame,
with at least `min_cone_tokens` tokens per class. A feature's class profile
is q(f) = softmax([cos(f, mu_c)]_c / T_l), and two features agree by

    RA = sum_c min(q_c(f_a), q_c(f_b))  in [0, 1]

    f_served  what the Reuse Gate passes on at frame t: the cached entry for
              a hit, the feature computed on the miss subset for a miss
    f_oracle  the stage output of a second forward of the same frame with
              the cache disarmed (all tokens, same RNG, so the serialization
              matches)
    T_l       the per-stage temperatures of the motivation run (Fig. 2c),
              read from its json

A shadow table keeps, for every cached region, the oracle feature of the
frame the entry was written (f_oracle(v, t_w)), which splits a hit into

    stale  RA(f_oracle(v, t_w), f_oracle(v, t))   scene / view change only
    ctx    RA(f_served(v, t),  f_oracle(v, t_w))  reduced context only
    total  RA(f_served(v, t),  f_oracle(v, t))

For a miss (age 0), stale = 1 and ctx = total.

Per cached stage, two histograms of per-token sums are kept,

    H_t[age, conf_t, cat, bnd, stat]   conf_t: region confidence c(v) of
                                       this frame's output
    H_w[age, conf_w, cat, bnd, stat]   conf_w: confidence when the entry
                                       was written

with age 0 = computed this frame and a >= 1 = frames since written (the last
bin absorbs older entries), CONF_BINS equal confidence bins on [0, 1], cat =
static / movable / label changed since write, bnd = the region holds two or
more ground-truth classes, and the STATS below. Only frames on which the
stage has at least one hit are binned; per-frame means go to
`reuse_agreement_frames.jsonl`.

Attach with `spacecache/experiments/run.py --reuse_agreement` (accuracy runs:
--no_timing --no_metrics, region scale 1, one pass). The predictions of the
measured run are unchanged: the oracle runs with the policy disarmed and
restores the RNG state.
"""

from __future__ import annotations

import contextlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from spacecache.policies.spacecache import _MOVABLE_CLASSES

AGE_BINS = 64
CONF_BINS = 100
CATS = ("static", "movable", "changed")
STATS = ("n", "ra", "ra_stale", "ra_ctx", "top1", "cos",
         "cos_stale", "cos_ctx",
         "pts", "pts_agree", "pts_valid", "pts_ok", "pts_ok_oracle")
FRAME_FIELDS = ("n", "n_hit", "hit_ra", "hit_ra_stale", "hit_ra_ctx",
                "hit_top1", "hit_cos", "hit_age", "n_miss", "miss_ra",
                "miss_cos", "hit_pts", "hit_pts_agree")
# cone_affinity floors each profile before comparing (eps / K)
_EPS = 1e-6
# per-stage T of the motivation run (same Utonia encoder as the decoder model)
DEFAULT_TEMP_JSON = "output/motivation/reuse_agreement/waymo_utonia.json"


def load_temps(spec=None, json_path=DEFAULT_TEMP_JSON):
    """{depth: T}. `spec` "3:0.1035,4:0.0525" overrides the json's
    settings.temp_per_stage (the entropy-calibrated T of cone_affinity)."""
    if spec:
        return {int(d): float(t) for d, t in
                (x.split(":") for x in spec.split(",") if x.strip())}
    with open(json_path) as f:
        st = json.load(f)["settings"]
    if st.get("temp_mode") != "entropy":
        raise ValueError(f"{json_path}: temp_mode={st.get('temp_mode')}, "
                         "expected the entropy-calibrated temperatures")
    return {int(d): float(t) for d, t in st["temp_per_stage"].items()}


_SI = {s: i for i, s in enumerate(STATS)}


def _upsert(tab, new_key, new_feat, new_meta):
    """Insert-or-overwrite rows of a sorted (key, feat, meta) table; the
    newest row wins (stable sort keeps old rows first inside a key run)."""
    if new_key.numel() == 0:
        return tab
    if tab is None:
        k, f, m = new_key, new_feat, new_meta
    else:
        k = torch.cat([tab[0], new_key])
        f = torch.cat([tab[1], new_feat.to(tab[1].dtype)])
        m = torch.cat([tab[2], new_meta])
    order = torch.argsort(k, stable=True)
    k = k[order]
    last = torch.ones_like(k, dtype=torch.bool)
    last[:-1] = k[1:] != k[:-1]
    idx = order[last]
    return k[last], f[idx], m[idx]


class ReuseAgreementProbe:
    """Hooks around a SpaceCache policy; see module doc."""

    def __init__(self, policy, model, cfg, device, temps,
                 min_cone_tokens=20):
        if abs(float(policy.mf) - 1.0) > 1e-9:
            raise ValueError("--reuse_agreement needs --region_scale 1 (one token per "
                             "region, so a cached row maps to one token)")
        self.policy = policy
        self.model = model
        self.device = device
        self.min_cone = int(min_cone_tokens)
        # (depth, frame-stash attr, level0-map key, stage desc)
        self.depths = [(int(policy.DEPTH), "_frame", "c3", policy.stage)]
        if getattr(policy, "cache4", None) is not None:
            self.depths.append((int(policy.D4_DEPTH), "_frame4", "c4",
                                policy.stage4))
        missing = [d for d, *_ in self.depths if d not in temps]
        if missing:
            raise ValueError(f"no temperature for stage(s) {missing}")
        self.temps = {d: float(temps[d]) for d, *_ in self.depths}
        self.names = [str(n) for n in (cfg.data.get("names", None) or [])]
        self._mov = None
        self._nc = len(self.names)
        self._in_oracle = False
        self.reset()

    # -- state --------------------------------------------------------------

    def reset(self):
        """Drop tables, histograms and frame rows (call with the policy's
        reset_state, e.g. after the armed warmup)."""
        nd = len(self.depths)
        self._tab = [None] * nd
        self._H_t = [None] * nd
        self._H_w = [None] * nd
        self._frames = []
        self._o_feat = [None] * nd
        self._o_coord = [None] * nd
        self._o_pred = None
        self._st = [None] * nd
        self.counts = dict(frames=0, misaligned=0, no_cones=0,
                           hit_not_found=0)

    def _movable(self, nc, dev):
        if self._mov is None or int(self._mov.numel()) != nc:
            low = [n.lower() for n in self.names]
            self._mov = torch.tensor(
                [i < len(low) and low[i] in _MOVABLE_CLASSES
                 for i in range(nc)], dtype=torch.bool, device=dev)
        return self._mov

    # -- hooks --------------------------------------------------------------

    @contextlib.contextmanager
    def context(self):
        """Register AFTER policy.streaming_context: the block hooks must see
        the Reuse Gate's restored output and the model hook must run after
        the Commit Gate(s)."""
        enc = self.model.backbone.enc._modules
        handles = [self.model.register_forward_pre_hook(self._pre)]
        for k, (d, _, _, desc) in enumerate(self.depths):
            stage = enc[f"enc{d}"]
            last = (stage._modules.get(f"block{desc.n_blocks - 1}")
                    or stage._modules["fused_blocks"])
            handles.append(last.register_forward_hook(self._block_hook(k)))
        handles.append(self.model.register_forward_hook(self._post))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _pre(self, module, args):
        """Oracle forward of this frame, cache disarmed, same RNG."""
        pol = self.policy
        if self._in_oracle or not pol.armed or not args:
            return None
        if pol._fi == 0:                       # reset_gate ran first: new scene
            self._tab = [None] * len(self.depths)
        nd = len(self.depths)
        self._o_feat, self._o_coord = [None] * nd, [None] * nd
        self._st = [None] * nd
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state()
                    if self.device.type == "cuda" else None)
        pol.armed = False
        self._in_oracle = True
        try:
            out = module(dict(args[0]))
        finally:
            pol.armed = True
            self._in_oracle = False
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
        logits = out.get("seg_logits") if isinstance(out, dict) else None
        self._o_pred = None if logits is None else logits.argmax(dim=1)
        if logits is not None:
            self._nc = int(logits.shape[1])
        return None

    def _block_hook(self, k):
        _, fattr, ckey, _ = self.depths[k]

        def hook(_m, _i, out):
            if self._in_oracle:
                self._o_feat[k] = out.feat.detach().clone()
                self._o_coord[k] = self._tok_id(out)
                return None
            if not self.policy.armed or self._o_feat[k] is None:
                return None
            fm = getattr(self.policy, fattr, None)
            if fm is not None:
                self._st[k] = self._measure(k, fm, fm.get(ckey), out)
            return None
        return hook

    # -- per stage ------------------------------------------------------------

    @staticmethod
    def _tok_id(point):
        """Token identity for the oracle/served alignment check: the integer
        grid coord when the Point has one (exact), else the pooled coord."""
        g = point.get("grid_coord") if hasattr(point, "get") else None
        return (g if g is not None else point.coord).detach()

    def _same_tokens(self, ref, point):
        cur = self._tok_id(point)
        if cur.shape != ref.shape:
            return False
        if cur.dtype.is_floating_point:
            return bool(torch.allclose(cur, ref, rtol=0, atol=1e-5))
        return bool(torch.equal(cur, ref))

    def _measure(self, k, fm, cmap, out):
        pol = self.policy
        fo, served = self._o_feat[k], out.feat
        n = int(served.shape[0])
        if (int(fo.shape[0]) != n
                or not self._same_tokens(self._o_coord[k], out)):
            self.counts["misaligned"] += 1
            return None
        seg0 = pol._seg0
        if (cmap is None or seg0 is None or self._o_pred is None
                or int(cmap.numel()) != int(seg0.numel())):
            return None
        dev = served.device
        hit, keys = fm["view"].hit, fm["keys"]
        seg0 = seg0.reshape(-1).long()
        nc = int(self._nc)
        valid = (seg0 >= 0) & (seg0 < nc)
        cnt = torch.bincount(cmap[valid] * nc + seg0[valid],
                             minlength=n * nc).view(n, nc)
        has = cnt.sum(dim=1) > 0
        lab = torch.where(has, cnt.argmax(dim=1),
                          torch.full_like(has, -1, dtype=torch.long))
        ncls = (cnt > 0).sum(dim=1)
        pure, bnd = ncls == 1, ncls >= 2

        # prototypes: oracle features of pure tokens, ground-truth label
        fo32, fs32 = fo.float(), served.float()
        idx = lab[pure]
        ccount = torch.bincount(idx, minlength=nc)
        present = ccount >= self.min_cone
        if int(present.sum()) < 2:
            self.counts["no_cones"] += 1
            return None
        musum = torch.zeros(nc, fo32.shape[1], device=dev,
                            dtype=torch.float32)
        musum.index_add_(0, idx, fo32[pure])
        mun = F.normalize(musum[present]
                          / ccount[present].unsqueeze(1).float(), dim=1)

        temp = self.temps[self.depths[k][0]]
        n_cones = int(mun.shape[0])

        def prof(f):
            """cosine profile -> floored softmax(cos / T), as cone_affinity"""
            sims = F.normalize(f, dim=1) @ mun.T
            q = torch.softmax(sims / temp, dim=1)
            return sims, (1.0 - _EPS) * q + _EPS / n_cones

        def overlap(qa, qb):
            return torch.minimum(qa, qb).sum(dim=1)

        so, qo = prof(fo32)
        ss, qs = prof(fs32)
        ra = overlap(qo, qs)
        top1 = so.argmax(dim=1) == ss.argmax(dim=1)
        cos = F.cosine_similarity(fs32, fo32, dim=1)

        # misses: written this frame -> oracle-at-write == oracle now
        ra_stale, ra_ctx = torch.ones_like(ra), ra.clone()
        cos_stale, cos_ctx = torch.ones_like(cos), cos.clone()
        age = torch.zeros(n, dtype=torch.long, device=dev)
        lab_w = lab.clone()
        hit_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        confw = None
        hidx = torch.nonzero(hit, as_tuple=True)[0]
        n_hit = int(hidx.numel())
        tab = self._tab[k]
        if n_hit and tab is not None:
            tk, tf, tm = tab
            q = keys[hidx]
            pos = torch.searchsorted(tk, q).clamp(max=int(tk.numel()) - 1)
            found = tk[pos] == q
            self.counts["hit_not_found"] += int((~found).sum())
            hf, pf = hidx[found], pos[found]
            fw = tf[pf].float()
            _, qw = prof(fw)
            ra_stale[hf] = overlap(qw, qo[hf])
            ra_ctx[hf] = overlap(qs[hf], qw)
            cos_stale[hf] = F.cosine_similarity(fw, fo32[hf], dim=1)
            cos_ctx[hf] = F.cosine_similarity(fs32[hf], fw, dim=1)
            meta = tm[pf]
            age[hf] = (int(pol._fi) - meta[:, 0]).round().long().clamp(
                1, AGE_BINS - 1)
            lab_w[hf] = meta[:, 2].long()
            hit_ok[hf] = True
            confw = (hf, meta[:, 1].float())
        elif n_hit:
            self.counts["hit_not_found"] += n_hit
        return dict(n=n, n_hit=n_hit, hit=hit, hit_ok=hit_ok, keys=keys,
                    cmap=cmap, has=has, lab=lab, lab_w=lab_w, bnd=bnd,
                    ra=ra, ra_stale=ra_stale, ra_ctx=ra_ctx, top1=top1,
                    cos=cos, cos_stale=cos_stale, cos_ctx=cos_ctx,
                    age=age, confw=confw, fo=fo)

    # -- per frame ------------------------------------------------------------

    def _post(self, _module, _args, output):
        pol = self.policy
        if self._in_oracle or not pol.armed:
            return
        logits = output.get("seg_logits") if isinstance(output, dict) else None
        if logits is None or self._o_pred is None or pol._seg0 is None:
            return               # e.g. the warmup input carries no GT
        seg0 = pol._seg0.reshape(-1).long()
        nc = int(logits.shape[1])
        dev = logits.device
        pred = logits.argmax(dim=1)
        valid = (seg0 >= 0) & (seg0 < nc)
        agree_pt = pred == self._o_pred
        ok_s = (pred == seg0) & valid
        ok_o = (self._o_pred == seg0) & valid
        mov = self._movable(nc, dev)
        fi = int(pol._fi)
        ncell = AGE_BINS * CONF_BINS * len(CATS) * 2
        rows = []
        for k, st in enumerate(self._st):
            if st is None:
                rows.append(None)
                continue
            n, cmap = st["n"], st["cmap"]
            conf_t = pol._cell_confidence(output, cmap, n)

            def csum(w):
                return torch.bincount(cmap, weights=w.double(), minlength=n)

            pts = torch.bincount(cmap, minlength=n).double()
            hit, has, lab, lab_w = st["hit"], st["has"], st["lab"], st["lab_w"]
            conf_w = conf_t.clone()
            if st["confw"] is not None:
                conf_w[st["confw"][0]] = st["confw"][1]
            changed = hit & (lab_w >= 0) & (lab_w != lab)
            movable = mov[lab.clamp(min=0)] | (
                hit & (lab_w >= 0) & mov[lab_w.clamp(min=0)])
            cat = movable.long()
            cat[changed] = 2
            scored = has & (st["hit_ok"] | ~hit)
            S = torch.stack([
                torch.ones(n, dtype=torch.float64, device=dev),
                st["ra"].double(), st["ra_stale"].double(),
                st["ra_ctx"].double(), st["top1"].double(),
                st["cos"].double(), st["cos_stale"].double(),
                st["cos_ctx"].double(),
                pts, csum(agree_pt), csum(valid), csum(ok_s), csum(ok_o),
            ], dim=1)
            if st["n_hit"] > 0:
                if self._H_t[k] is None:
                    self._H_t[k] = torch.zeros(ncell, len(STATS),
                                               dtype=torch.float64, device=dev)
                    self._H_w[k] = torch.zeros_like(self._H_t[k])
                for H, c in ((self._H_t[k], conf_t), (self._H_w[k], conf_w)):
                    b = (c * CONF_BINS).long().clamp(0, CONF_BINS - 1)
                    flat = (((st["age"] * CONF_BINS + b) * len(CATS) + cat)
                            * 2 + st["bnd"].long())
                    H.index_add_(0, flat[scored], S[scored])
            hs, ms = scored & hit, scored & ~hit
            rows.append(torch.stack([
                scored.sum().double(), hs.sum().double(),
                S[hs, _SI["ra"]].sum(), S[hs, _SI["ra_stale"]].sum(),
                S[hs, _SI["ra_ctx"]].sum(), S[hs, _SI["top1"]].sum(),
                S[hs, _SI["cos"]].sum(), st["age"][hs].sum().double(),
                ms.sum().double(), S[ms, _SI["ra"]].sum(),
                S[ms, _SI["cos"]].sum(),
                S[hs, _SI["pts"]].sum(), S[hs, _SI["pts_agree"]].sum()]))
            # shadow table: every recomputed region gets its oracle feature
            # at write (a hit always refers to its region's latest miss)
            midx = torch.nonzero(~hit, as_tuple=True)[0]
            if midx.numel():
                meta = torch.stack([
                    torch.full((int(midx.numel()),), float(fi),
                               dtype=torch.float64, device=dev),
                    conf_t[midx].double(), lab[midx].double(), pts[midx],
                ], dim=1)
                self._tab[k] = _upsert(self._tab[k], st["keys"][midx],
                                       st["fo"][midx], meta)
        seg = pol._cur_seg
        self._frames.append((dict(
            frame_ptr=int(pol._frame_ptr) - 1,
            segment=int(seg) if isinstance(seg, int) else None,
            fi=fi), rows))
        self.counts["frames"] += 1
        nd = len(self.depths)
        self._st, self._o_feat, self._o_coord = [None] * nd, [None] * nd, \
            [None] * nd
        self._o_pred = None

    # -- output ---------------------------------------------------------------

    def save(self, out_dir):
        """Write reuse_agreement.npz + reuse_agreement_frames.jsonl; return
        the summary block for results.json."""
        shape = (AGE_BINS, CONF_BINS, len(CATS), 2, len(STATS))
        arrays = dict(stats=np.asarray(STATS), cats=np.asarray(CATS),
                      depths=np.asarray([d for d, *_ in self.depths]),
                      age_bins=np.int64(AGE_BINS),
                      conf_bins=np.int64(CONF_BINS),
                      names=np.asarray(self.names),
                      temps=np.asarray([self.temps[d]
                                        for d, *_ in self.depths]))
        summary = dict(counts=dict(self.counts), depths={},
                       temps={f"d{d}": t for d, t in self.temps.items()})
        for k, (d, *_) in enumerate(self.depths):
            if self._H_t[k] is None:
                continue
            Ht = self._H_t[k].view(*shape).cpu().numpy()
            arrays[f"H_t_d{d}"] = Ht
            arrays[f"H_w_d{d}"] = self._H_w[k].view(*shape).cpu().numpy()
            hit = Ht[1:].sum(axis=(0, 1, 2, 3))
            miss = Ht[0].sum(axis=(0, 1, 2))
            summary["depths"][f"d{d}"] = dict(
                n_hit=int(hit[_SI["n"]]), n_miss=int(miss[_SI["n"]]),
                **{f"hit_{s}": float(hit[_SI[s]] / max(hit[0], 1))
                   for s in ("ra", "ra_stale", "ra_ctx", "top1", "cos")},
                hit_pred_agree=float(hit[_SI["pts_agree"]]
                                     / max(hit[_SI["pts"]], 1)),
                miss_ra=float(miss[_SI["ra"]] / max(miss[0], 1)),
                miss_cos=float(miss[_SI["cos"]] / max(miss[0], 1)))
        np.savez_compressed(os.path.join(out_dir, "reuse_agreement.npz"),
                            **arrays)
        vecs = [v for _, rows in self._frames for v in rows if v is not None]
        vals = (torch.stack(vecs).cpu().tolist() if vecs else [])
        it = iter(vals)
        with open(os.path.join(out_dir, "reuse_agreement_frames.jsonl"),
                  "w") as f:
            for meta, rows in self._frames:
                rec = dict(meta)
                for (d, *_), v in zip(self.depths, rows):
                    if v is not None:
                        rec.update({f"{nm}_d{d}": x for nm, x
                                    in zip(FRAME_FIELDS, next(it))})
                f.write(json.dumps(rec) + "\n")
        return summary
