"""Prototype-agreement figure, ScanNet video + Utonia: 60 frame pairs.

Only 20 val scenes are preprocessed, so run `cone_affinity.py` with
`--max_pairs_per_scene 3` (pairs spread evenly over each scene) to obtain 60
pairs, as in the outdoor settings. Uses the decoder-head checkpoint; the
figure reads the encoder stages only, which are frozen in this model.
"""

_base_ = ["../utonia_dec.py"]

figure_label = "ScanNet-video"
model_label = "Utonia"
save_path = "output/motivation/reuse_agreement"

stride = 1
max_pairs = 60
