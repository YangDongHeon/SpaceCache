"""Run spacecache/experiments/run.py and dump every frame for the demo video.

Takes the run.py flags plus `--dump_dir`. A forward hook on the model saves,
per frame, the voxel coordinates (metres, sensor frame), the predicted and
ground-truth labels, the ego pose and, for SpaceCache, which voxels were
served from the stage-3 / stage-4 cache. The hook is registered before the
SpaceCache gates, so it runs before the Commit Gate clears the frame state.

    PYTHONPATH=. python spacecache/demo/dump_frames.py \\
        --config spacecache/demo/waymo_utonia_demo.py \\
        --policy spacecache_s34 --lifetime 8 --confidence 0.95 \\
        --triton_cache --no_timing --no_metrics \\
        --dump_dir output/demo/waymo_utonia_demo/dump/spacecache_s34_c095
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import spacecache.experiments.run as R  # noqa: E402
from spacecache.common.helpers import extract_random_scale  # noqa: E402


def _pop_arg(argv, name):
    for i, a in enumerate(argv):
        if a == name:
            val = argv[i + 1]
            del argv[i:i + 2]
            return val
        if a.startswith(name + "="):
            del argv[i]
            return a.split("=", 1)[1]
    return None


def main():
    dump_dir = _pop_arg(sys.argv, "--dump_dir")
    if dump_dir is None:
        raise SystemExit("--dump_dir is required")
    # warmup forwards would be dumped as frames
    _pop_arg(sys.argv, "--warmup")
    sys.argv += ["--warmup", "0"]
    os.makedirs(dump_dir, exist_ok=True)

    # the input coordinates are scaled by the val pipeline's RandomScale
    # (0.2 for Utonia, none for LitePT); read it from the config, as every
    # policy sees the same pipeline
    cfg = R.Config.fromfile(_pop_arg(list(sys.argv), "--config"))
    state = dict(policy=None, frame=0,
                 scale=float(extract_random_scale(cfg.data.val.transform)))

    def _hit(frame, key):
        if frame is None:
            return None
        view, cmap = frame.get("view"), frame.get(key)
        if view is None or cmap is None:
            return None
        return view.hit[cmap].cpu().numpy()

    def dump_hook(_module, args, output):
        inp = args[0]
        p = state["policy"]
        rec = dict(
            coord=(inp["coord"].float().cpu().numpy()
                   / state["scale"]).astype(np.float32),
            pred=output["seg_logits"].argmax(1).to(torch.uint8).cpu().numpy(),
            gt=inp["segment"].reshape(-1).to(torch.int16).cpu().numpy(),
        )
        pose = inp.get("pose")
        if pose is not None:
            rec["pose"] = pose.reshape(-1, 4, 4)[0].double().cpu().numpy()
        if p is not None and getattr(p, "armed", False):
            h3 = _hit(getattr(p, "_frame", None), "c3")
            h4 = _hit(getattr(p, "_frame4", None), "c4")
            if h3 is not None:
                rec["hit3"] = h3
            if h4 is not None:
                rec["hit4"] = h4
        np.savez_compressed(
            os.path.join(dump_dir, f"{state['frame']:05d}.npz"), **rec)
        state["frame"] += 1

    orig_build_policy = R.build_policy
    orig_build_segmentor = R.build_segmentor

    def build_policy(name):
        state["policy"] = orig_build_policy(name)
        return state["policy"]

    def build_segmentor(*a, **k):
        model = orig_build_segmentor(*a, **k)
        model.register_forward_hook(dump_hook)
        return model

    R.build_policy = build_policy
    R.build_segmentor = build_segmentor
    R.main()

    # frame -> (sequence directory, frame name), for picking segments
    ds = R.build_dataset(cfg.data.val)
    names = [os.path.relpath(d, ds.data_root) if isinstance(d, str) else str(d)
             for d in ds.data_list]
    with open(os.path.join(dump_dir, "frames.json"), "w") as f:
        json.dump(dict(n_frames=state["frame"], frames=names,
                       class_names=list(cfg.data.names)), f, indent=1)


if __name__ == "__main__":
    main()
