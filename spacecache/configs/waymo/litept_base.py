"""Waymo + LitePT-B, full validation split.

LitePT's own val pipeline (grid 0.05, coord + strength features, no
RandomScale / PointClip): each backbone is evaluated with the transform it
was trained with. `pose` is added to Collect for streaming evaluation.
Slices in `litept_base/`.
"""

_base_ = ["../../../configs/waymo/semseg-litept-v1m1-0-base.py"]

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

weight = "exp/litept/waymo/semseg-litept-v1m1-0-base/model/model_best.pth"

stride = 1
lifetime = 8  # L: SpaceCache lifetime in frames (2 Hz, i.e. 4 s)
confidence = 0.85  # theta: regions with mean top-1 confidence <= theta are not reused

output_root = "output/runs/waymo/litept_base"

device = "cuda"
seed = 37
