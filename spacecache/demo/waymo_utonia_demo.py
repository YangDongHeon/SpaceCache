"""Demo config: Waymo + Utonia on 8 candidate segments for the demo video.

Every segment has 30 labelled frames exactly 0.5 s apart (about half of the
validation segments have gaps or 0.1-0.4 s spacing), a moving ego vehicle
and many pedestrians or cyclists. The latency runs and the frame dumps of
the demo both use this config, so they line up frame by frame.
"""

_base_ = ["../configs/waymo/utonia_dec.py"]

output_root = "output/demo/waymo_utonia_demo"

data = dict(
    val=dict(
        scene_list=[
            "segment-10289507859301986274_4200_000_4220_000_with_camera_labels",
            "segment-14739149465358076158_4740_000_4760_000_with_camera_labels",
            "segment-17694030326265859208_2340_000_2360_000_with_camera_labels",
            "segment-17860546506509760757_6040_000_6060_000_with_camera_labels",
            "segment-8079607115087394458_1240_000_1260_000_with_camera_labels",
            "segment-8506432817378693815_4860_000_4880_000_with_camera_labels",
            "segment-8679184381783013073_7740_000_7760_000_with_camera_labels",
            "segment-9231652062943496183_1740_000_1760_000_with_camera_labels",
        ],
    ),
)
