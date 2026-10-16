"""nuScenes + LitePT-B, full lidarseg val split.

LitePT's own val pipeline (grid 0.05, coord + strength features, no camera
projection, no RandomScale): each backbone is evaluated with the transform
it was trained with. The only change from the training config is `pose` in
Collect, for streaming evaluation. `NuScenesDataset` keeps `coord` in the
LiDAR frame, so its `pose` is the plain `lidar_to_global_pose`. Slices in
`litept_base/`.
"""

_base_ = ["../../../configs/nuscenes/semseg-litept-v1m1-0-base.py"]

data = dict(
    val=dict(
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment",
                      "inverse", "pose"),
                feat_keys=("coord", "strength"),
            ),
        ],
    ),
)

weight = "exp/litept/nuscenes/semseg-litept-v1m1-0-base/model/model_best.pth"

stride = 1
lifetime = 8  # L: SpaceCache lifetime in frames (2 Hz, i.e. 4 s)
confidence = 0.85  # theta: regions with mean top-1 confidence <= theta are not reused

output_root = "output/runs/nuscenes/litept_base"

device = "cuda"
seed = 37
