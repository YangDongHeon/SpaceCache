"""ScanNet video + LitePT-B, all val scenes.

LitePT's own pipeline (grid 0.02, no RandomScale, color + normal features),
with ApplyPose and the world-anchored lattice as in `utonia_dec.py`. Slices
in `litept_base/`.
"""

_base_ = ["../../../configs/scannet/semseg-litept-v1m1-0-base.py"]

data = dict(
    val=dict(
        type="ScanNetVideoDataset",
        data_root="data/scannetvideo/val",
        split="scene*",
        transform=[
            dict(
                type="ApplyPose",
                rotate_keys=("normal",),
                shift_from_root="data/scannet/val",
            ),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
                world_anchor=True,
                lattice_stride=16,
                pose_scale=1.0,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment",
                      "inverse", "pose"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
)

weight = "exp/litept/scannet/semseg-litept-v1m1-0-base/model/model_best.pth"

lifetime = 4  # L: SpaceCache lifetime in frames (3 fps, i.e. 1.3 s)
confidence = 0.85  # theta: regions with mean top-1 confidence <= theta are not reused
exclude_unlabeled = False

output_root = "output/runs/scannet_video/litept_base"

device = "cuda"
seed = 37
