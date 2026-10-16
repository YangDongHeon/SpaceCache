"""Add `scene_token` / `scene_name` to existing nuScenes info pkls.

`preprocess_nuscenes_info.py` writes both keys; this script adds them to
pkls created without them, using only the raw metadata tables
(`sample.json` maps the sample token to a scene token, `scene.json` maps
that to a name such as "scene-0003"). `NuScenesSceneMixin` uses them to
group frames by scene (`sequence_index`) and to select scenes
(`scene_list`).

    python pointcept/datasets/preprocessing/nuscenes/add_scene_id.py \
        --data_root data/nuscenes --splits val
    python pointcept/datasets/preprocessing/nuscenes/add_scene_id.py \
        --data_root data/nuscenes --splits train,val --dry_run

Each pkl is rewritten in place after copying the original to `<name>.bak`.
`--dry_run` only reports; a pkl that already has the keys is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from collections import Counter


def load_tables(data_root, version):
    """sample_token -> (scene_token, scene_name), from the raw JSON tables."""
    meta = os.path.join(data_root, "raw", version)
    for name in ("sample.json", "scene.json"):
        p = os.path.join(meta, name)
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p} — point --version at the metadata "
                             f"directory that holds sample.json/scene.json")
    with open(os.path.join(meta, "scene.json")) as f:
        scene_name = {s["token"]: s["name"] for s in json.load(f)}
    with open(os.path.join(meta, "sample.json")) as f:
        samples = json.load(f)
    out = {}
    for s in samples:
        st = s["scene_token"]
        out[s["token"]] = (st, scene_name.get(st, st))
    print(f"[tables] {len(scene_name)} scenes, {len(out)} samples "
          f"from {meta}")
    return out


def patch_pkl(path, lut, dry_run):
    if not os.path.isfile(path):
        print(f"[skip] {path} (not present)")
        return
    with open(path, "rb") as f:
        infos = pickle.load(f)
    if all(i.get("scene_token") for i in infos):
        print(f"[skip] {os.path.basename(path)}: already has scene_token "
              f"for all {len(infos)} frames")
        return

    missed = [i for i in infos if i["token"] not in lut]
    if missed:
        # e.g. a test-split pkl with trainval metadata, or a wrong --version
        raise SystemExit(
            f"{os.path.basename(path)}: {len(missed)} of {len(infos)} sample "
            f"tokens are not in the metadata tables — wrong --version?")

    for i in infos:
        i["scene_token"], i["scene_name"] = lut[i["token"]]

    per_scene = Counter(i["scene_token"] for i in infos)
    lens = sorted(per_scene.values())
    print(f"[{os.path.basename(path)}] {len(infos)} frames in "
          f"{len(per_scene)} scenes | frames/scene "
          f"min {lens[0]} / median {lens[len(lens) // 2]} / max {lens[-1]}")

    if dry_run:
        print("  (--dry_run: not written)")
        return
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  backup -> {bak}")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(infos, f)
    os.replace(tmp, path)          # atomic: never leaves a half-written pkl
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval",
                    help="metadata dir under <data_root>/raw holding "
                         "sample.json and scene.json")
    ap.add_argument("--splits", default="val",
                    help="comma list of train,val,test")
    ap.add_argument("--sweeps", type=int, default=10,
                    help="the N in nuscenes_infos_Nsweeps_<split>.pkl")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    lut = load_tables(a.data_root, a.version)
    for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
        patch_pkl(os.path.join(a.data_root, "info",
                               f"nuscenes_infos_{a.sweeps}sweeps_{split}.pkl"),
                  lut, a.dry_run)


if __name__ == "__main__":
    main()
