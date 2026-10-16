"""Latency slice (latency / FLOPs / memory): 10 whole segments, 299 frames,
chosen with spacecache/experiments/pick_latency_slice.py."""

_base_ = ["../utonia_dec.py"]

# separate output root from the full-split runs
output_root = "output/runs/waymo/utonia_dec_latency_slice"

data = dict(
    val=dict(
        scene_list=[
            "segment-12306251798468767010_560_000_580_000_with_camera_labels",
            "segment-13356997604177841771_3360_000_3380_000_with_camera_labels",
            "segment-14244512075981557183_1226_840_1246_840_with_camera_labels",
            "segment-17135518413411879545_1480_000_1500_000_with_camera_labels",
            "segment-17136314889476348164_979_560_999_560_with_camera_labels",
            "segment-3039251927598134881_1240_610_1260_610_with_camera_labels",
            "segment-4490196167747784364_616_569_636_569_with_camera_labels",
            "segment-5574146396199253121_6759_360_6779_360_with_camera_labels",
            "segment-8888517708810165484_1549_770_1569_770_with_camera_labels",
            "segment-9243656068381062947_1297_428_1317_428_with_camera_labels",
        ],
    ),
)
