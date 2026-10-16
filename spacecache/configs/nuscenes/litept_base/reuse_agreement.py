"""Prototype-agreement figure, nuScenes + LitePT-B: 60 frame pairs, one per
scene (`cone_affinity.py --max_pairs_per_scene 1`, the default).

Pairs are grouped by scene through `sequence_index` (`NuScenesSceneMixin`),
which needs scene ids in the info pkl. The figure reads the encoder stages
only, so the segmentation checkpoint is used as is.
"""

_base_ = ["../litept_base.py"]

figure_label = "nuScenes"
model_label = "LitePT-B"
save_path = "output/motivation/reuse_agreement"

stride = 1
max_pairs = 60
