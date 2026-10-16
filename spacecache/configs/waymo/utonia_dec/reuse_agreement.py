"""Prototype-agreement figure, Waymo + Utonia: 60 frame pairs, one per
segment (`cone_affinity.py --max_pairs_per_scene 1`, the default). Uses the
decoder-head checkpoint; the figure reads the encoder stages only, which are
frozen in this model."""

_base_ = ["../utonia_dec.py"]

figure_label = "Waymo"
model_label = "Utonia"
save_path = "output/motivation/reuse_agreement"

stride = 1
max_pairs = 60
