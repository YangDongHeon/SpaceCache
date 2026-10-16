"""Prototype-agreement figure, nuScenes + Utonia: 60 frame pairs, one per
scene. Uses the decoder-head checkpoint; the figure reads the encoder stages
only."""

_base_ = ["../utonia_dec.py"]

figure_label = "nuScenes"
model_label = "Utonia"
save_path = "output/motivation/reuse_agreement"

stride = 1
max_pairs = 60
