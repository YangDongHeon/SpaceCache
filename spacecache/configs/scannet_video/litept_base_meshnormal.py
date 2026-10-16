"""ScanNet video + LitePT-B with oracle normals from the fused mesh.

Each frame point takes the normal of the nearest fused-mesh vertex
(`MeshNormal`), i.e. the input LitePT-B was trained on. This is not causal:
the mesh is reconstructed from the whole sequence. LitePT-B cannot use the
normals estimated from single depth frames, so its indoor evaluation uses
this config; the backbone and every method see the same input.
"""

_base_ = ["./litept_base.py"]

data = dict(
    val=dict(
        transform=[
            dict(type="ApplyPose", rotate_keys=("normal",),
                 shift_from_root="data/scannet/val"),
            # after ApplyPose with the same root (shared coordinates)
            dict(type="MeshNormal", root="data/scannet/val", max_dist=0.05),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv",
                 mode="train", return_grid_coord=True, return_inverse=True,
                 world_anchor=True, lattice_stride=16, pose_scale=1.0),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment", "origin_segment",
                       "inverse", "pose"),
                 feat_keys=("color", "normal")),
        ],
    ),
)

output_root = "output/runs/scannet_video/litept_base_meshnormal"
