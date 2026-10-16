"""Indoor evaluation split with oracle mesh normals: 10 ScanNet val scenes
(2221 frames, the scenes of `utonia_dec/stage_overlap.py`)."""

_base_ = ["../litept_base_meshnormal.py"]

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
