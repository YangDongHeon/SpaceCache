"""Demo config: Waymo + LitePT-B on a 5 Hz stream (every other 10 Hz sweep).

Data from spacecache/demo/preprocess_waymo_stream.py. Most sweeps are
unlabelled, so the accuracy numbers of these runs are meaningless; the demo
uses their latency and predictions only. `segment` marks the short-range
LiDAR points as ignored, so SpaceCache keeps them out of the cache as in
the paper.
"""

_base_ = ["../configs/waymo/litept_base.py"]

output_root = "output/demo/waymo_litept_5hz"

# in frames, as in the paper; at 5 Hz an entry lives at most 1.6 s
lifetime = 8

data = dict(
    val=dict(
        data_root="data/waymo_stream",
        scene_list=[
            "segment-4490196167747784364_616_569_636_569_with_camera_labels",
            "segment-5574146396199253121_6759_360_6779_360_with_camera_labels",
            "segment-8888517708810165484_1549_770_1569_770_with_camera_labels",
        ],
    ),
)
