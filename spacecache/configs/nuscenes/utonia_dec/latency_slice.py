"""Latency slice (latency / FLOPs / memory), nuScenes: 8 whole scenes, 321
keyframes, chosen with spacecache/experiments/pick_latency_slice.py (the same scenes
for both backbones).

Whole scenes (20 s, 39-41 keyframes) are used because streaming evaluation
resets its cache at every scene boundary (`sequence_index` from
`NuScenesSceneMixin`); the scenes were selected so that the per-frame
voxel-count distribution matches the full 150-scene val split.
"""

_base_ = ["../utonia_dec.py"]

data = dict(
    val=dict(
        scene_list=[
            "scene-0101",
            "scene-0103",
            "scene-0276",
            "scene-0522",
            "scene-0557",
            "scene-0783",
            "scene-0800",
            "scene-0907",
        ],
    ),
)
