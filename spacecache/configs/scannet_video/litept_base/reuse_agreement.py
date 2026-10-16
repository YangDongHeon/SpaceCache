"""Prototype-agreement figure, ScanNet video + LitePT-B: 60 frame pairs.

Only 20 val scenes are preprocessed, so run `cone_affinity.py` with
`--max_pairs_per_scene 3` (pairs spread evenly over each scene) to obtain 60
pairs, as in the outdoor settings. The figure reads the encoder stages only,
so the segmentation checkpoint is used as is.
"""

_base_ = ["../litept_base.py"]

figure_label = "ScanNet-video"
model_label = "LitePT-B"
save_path = "output/motivation/reuse_agreement"

stride = 1
max_pairs = 60
