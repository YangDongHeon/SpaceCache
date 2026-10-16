"""nuScenes + Utonia (decoder head, config 4b), full lidarseg val split.

The training-time val pipeline (NuScenesImagePointDataset, RandomScale 0.2,
GridSample 0.01) with `pose` added to Collect (lidar_to_global_pose @
ref_from_car, matching the returned coords). Val has 6019 keyframes at 2 Hz
in 150 scenes of 20 s; `NuScenesSceneMixin` provides per-scene grouping
(`sequence_index`) and selection (`scene_list`), which need `scene_token` in
the info pkl (see preprocessing/nuscenes/add_scene_id.py). The loader
projects six camera images per frame, so use several workers (e.g.
--num_workers 8). Slices in `utonia_dec/`.
"""

_base_ = ["../../../configs/utonia/semseg-utonia-v1m1-4b-nuscenes-dec.py"]

data = dict(
    val=dict(
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment",
                      "inverse", "pose"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
    ),
)

weight = "exp/utonia/semseg-utonia-v1m1-4b-nuscenes-dec/model/model_best.pth"

stride = 1
lifetime = 8  # L: SpaceCache lifetime in frames (2 Hz, i.e. 4 s)
confidence = 0.85  # theta: regions with mean top-1 confidence <= theta are not reused

output_root = "output/runs/nuscenes/utonia_dec"

device = "cuda"
seed = 37
