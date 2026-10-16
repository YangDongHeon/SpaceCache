"""Stage-overlap figure, nuScenes: every consecutive keyframe pair (2 Hz) on
the default per-frame voxel lattice, the one every outdoor evaluation uses
(Fig. 2(b))."""

_base_ = ["../utonia_dec.py"]

figure_label = "NuScenes"
save_path = "output/motivation/stage_overlap"

stride = 1
frame_gap_s = 0.5
max_pairs_per_scene = None
max_pairs = None
