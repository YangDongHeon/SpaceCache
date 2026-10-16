"""Waymo + Utonia (decoder head, config 5b), full validation split.

The training-time val transform, with `pose`, `origin_segment`, `inverse`
and `grid_coord` collected for evaluation. Slices in `utonia_dec/` override
`data.val.scene_list` (substring match against the frame directory names).
"""

_base_ = ["../../../configs/utonia/semseg-utonia-v1m1-5b-waymo-dec.py"]

data = dict(
    val=dict(
        transform=[
            dict(type="RandomScale", scale=[0.2, 0.2]),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="PointClip",
                point_cloud_range=(
                    -75.2 * 0.2, -75.2 * 0.2, -4 * 0.2,
                    75.2 * 0.2, 75.2 * 0.2, 2 * 0.2,
                ),
            ),
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

weight = "exp/utonia/semseg-utonia-v1m1-5b-waymo-dec/model/model_best.pth"

stride = 1
lifetime = 8  # L: SpaceCache lifetime in frames (2 Hz, i.e. 4 s)
confidence = 0.85  # theta: regions with mean top-1 confidence <= theta are not reused

output_root = "output/runs/waymo/utonia_dec"

device = "cuda"
seed = 37
