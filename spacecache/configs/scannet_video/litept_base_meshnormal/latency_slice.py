"""Indoor latency slice (latency / FLOPs / memory): 2 scenes, 395 frames."""

_base_ = ["../litept_base_meshnormal.py"]

data = dict(
    val=dict(
        split=[
            "scene0426_00",
            "scene0568_00",
        ],
    ),
)
