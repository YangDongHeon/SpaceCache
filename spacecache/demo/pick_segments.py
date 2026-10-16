"""Rank demo segments by how clearly they show the two claims.

Speed: the backbone and ToMe fall behind the stream (their output delay
grows) while SpaceCache keeps up. Accuracy (labelled presets only):
SpaceCache tracks the backbone's error rate on every prefix of the segment
while ToMe stays above it. Both are computed exactly as render_demo.py
draws them.

    PYTHONPATH=. python spacecache/demo/pick_segments.py
    PYTHONPATH=. python spacecache/demo/pick_segments.py \\
        --preset waymo_litept_5hz
"""

import argparse
import json
import os

import numpy as np

from spacecache.demo import render_demo as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="waymo_utonia", choices=D.PRESETS)
    ap.add_argument("--root", default=None,
                    help="default: the preset's")
    ap.add_argument("--skip", type=int, default=3,
                    help="frames ignored at the start of the error curves")
    args = ap.parse_args()
    D.PRESET = D.PRESETS[args.preset]
    D.METHODS, D.PERIOD = D.PRESET["methods"], D.PRESET["period"]
    root = args.root or D.PRESET["root"]
    gt = D.PRESET["gt"]
    P = D.PERIOD

    meta = json.load(open(os.path.join(root, "dump",
                                       D.METHODS["ours"][0], "frames.json")))
    names = meta["frames"]
    n_all = len(names)
    lat_all = {k: D.load_latency(root, v[1], n_all)
               for k, v in D.METHODS.items()}
    rows = []
    for seg in dict.fromkeys(os.path.dirname(f) for f in names):
        idx = [i for i, f in enumerate(names) if os.path.dirname(f) == seg]
        try:
            arr = D.arrival_times([names[i] for i in idx])
        except SystemExit:
            continue
        row = dict(seg=os.path.basename(seg)[8:28], n=len(idx))
        for k in D.METHODS:
            lat = lat_all[k][idx]
            a, _, dn = D.simulate(lat, arr)
            delay = dn - a
            row[f"{k}_late"] = float((delay > P).mean())
            row[f"{k}_delay_end"] = float(delay[-1])
            row[f"{k}_delay_max"] = float(delay.max())
        # positive on every term = the segment shows the claims; the delay
        # margins scale with the frame period
        terms = [row["backbone_delay_end"] - 1.6 * P,
                 row["tome_delay_end"] - 1.4 * P,
                 1.3 * P - row["ours_delay_max"]]
        if gt:
            counts = {k: D.load_counts(root, v[1], n_all, idx)
                      for k, v in D.METHODS.items()}
            err = {k: np.array([100 - D.cum_acc(c, j)
                                for j in range(len(idx))])
                   for k, c in counts.items()}
            s = slice(args.skip, None)
            row["ours_gap_max"] = float(
                (err["ours"] - err["backbone"])[s].max())
            row["tome_gap_min"] = float(
                (err["tome"] - err["backbone"])[s].min())
            row["err_end"] = {k: round(float(e[-1]), 2)
                              for k, e in err.items()}
            terms += [0.3 - row["ours_gap_max"], row["tome_gap_min"] - 0.5]
        row["score"] = min(terms)
        rows.append(row)
    rows.sort(key=lambda r: -r["score"])
    print(f"{'segment':22s}   n score | end delay s  base/tome/ours  "
          f"(ours max) | late% base/tome/ours"
          + (" | err% end base/tome/ours | ours-base max  tome-base min"
             if gt else ""))
    for r in rows:
        line = (f"{r['seg']:22s} {r['n']:3d} {r['score']:+.2f} | "
                f"{r['backbone_delay_end']:4.2f} {r['tome_delay_end']:4.2f} "
                f"{r['ours_delay_end']:4.2f} ({r['ours_delay_max']:4.2f}) | "
                f"{r['backbone_late'] * 100:3.0f} {r['tome_late'] * 100:3.0f} "
                f"{r['ours_late'] * 100:3.0f}")
        if gt:
            e = r["err_end"]
            line += (f" | {e['backbone']:5.2f} {e['tome']:5.2f} "
                     f"{e['ours']:5.2f} | {r['ours_gap_max']:+5.2f}  "
                     f"{r['tome_gap_min']:+5.2f}")
        print(line)


if __name__ == "__main__":
    main()
