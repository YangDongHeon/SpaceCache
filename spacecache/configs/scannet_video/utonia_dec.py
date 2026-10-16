"""ScanNet video (per-frame RGB-D back-projections) + Utonia (decoder head,
config 0b), all val scenes under data/scannetvideo/val.

`split` is a glob over scene directories ("scene*" takes all of them);
frames are sorted, and ApplyPose(shift_from_root) resolves each frame's scene
offset from `data_dict["split"]`. Frames are sampled at 3 fps (every 10th
frame of the 30 Hz capture), and lifetime counts frames.

`exclude_unlabeled=False`: regions with ignore-labelled points stay
cacheable. Indoor frames contain many such points, and ignore points do not
count towards mIoU. Slices in `utonia_dec/`.
"""

_base_ = ["../../../configs/utonia/semseg-utonia-v1m1-0b-scannet-dec.py"]

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
            dict(type="RandomScale", scale=[0.5, 0.5]),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
                world_anchor=True,
                lattice_stride=16,
                pose_scale=0.5,
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
        test_mode=False,
    ),
)

weight = "exp/utonia/semseg-utonia-v1m1-0b-scannet-dec/model/model_best.pth"

lifetime = 4  # L: SpaceCache lifetime in frames (3 fps, i.e. 1.3 s)
confidence = 0.85  # theta: regions with mean top-1 confidence <= theta are not reused
exclude_unlabeled = False

output_root = "output/runs/scannet_video/utonia_dec"

device = "cuda"
seed = 37
