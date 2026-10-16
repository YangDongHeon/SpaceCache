"""Stage-overlap figure, ScanNet video: every consecutive frame pair of the
10 evaluation scenes (3 fps, i.e. 0.33 s apart). `ApplyPose` consumes the
pose, so the per-pair sensor displacement is not reported.
"""

_base_ = ["../utonia_dec.py"]

data = dict(
    val=dict(
        split=[
            "scene0025_00",
            "scene0050_00",
            "scene0169_00",
            "scene0217_00",
            "scene0426_00",
            "scene0488_00",
            "scene0565_00",
            "scene0568_00",
            "scene0580_00",
            "scene0583_00",
        ],
    ),
)

max_frames = 10000

figure_label = "ScanNet"
save_path = "output/motivation/stage_overlap"

stride = 1
frame_gap_s = 1.0 / 3.0
max_pairs_per_scene = None
max_pairs = None
