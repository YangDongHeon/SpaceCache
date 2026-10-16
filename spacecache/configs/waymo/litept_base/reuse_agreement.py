"""Prototype-agreement figure, Waymo + LitePT-B: 60 frame pairs, one per
segment (`cone_affinity.py --max_pairs_per_scene 1`, the default)."""

_base_ = ["../litept_base.py"]

figure_label = "Waymo"
model_label = "LitePT-B"
save_path = "output/motivation/reuse_agreement"

stride = 1
max_pairs = 60
