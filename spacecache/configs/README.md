# Evaluation configs

One config per (dataset, backbone) pair, plus small slice configs that
inherit from it. A config selects the data, the val transform and the
checkpoint; the method and its settings (`--policy`, `--lifetime`,
`--confidence`, `--merge_ratio`, ...) are command-line arguments of the runners
(`spacecache/experiments/run.py` and the figure scripts in
`spacecache/figure/motivation/`).

```
<dataset>/<model>.py            full split: val transform + checkpoint
<dataset>/<model>/<slice>.py    _base_ = ["../<model>.py"] + what differs
```

A slice states only what differs from its parent:

```python
"""<what this slice selects>."""
_base_ = ["../utonia_dec.py"]
data = dict(val=dict(scene_list=["<segment>"]))  # ScanNet video: split=[...]
max_frames = 20
```

Dicts are merged recursively with the parent and lists are replaced, so a
config that changes `transform` restates the whole list (see
`scannet_video/litept_base_meshnormal.py`).

## Full splits

The val transform of the training config with `pose` (and `origin_segment`,
`inverse`) collected for evaluation. The ScanNet video configs also map each
frame to the world frame (`ApplyPose`) and use the world-anchored voxel
lattice (`GridSample(world_anchor=True)`).

| config | data | backbone |
|---|---|---|
| `waymo/utonia_dec.py` | Waymo val | Utonia, decoder head (5b) |
| `waymo/litept_base.py` | Waymo val | LitePT-B |
| `nuscenes/utonia_dec.py` | nuScenes lidarseg val | Utonia, decoder head (4b) |
| `nuscenes/litept_base.py` | nuScenes lidarseg val | LitePT-B |
| `scannet_video/utonia_dec.py` | ScanNet video (`data/scannetvideo/val`) | Utonia, decoder head (0b) |
| `scannet_video/litept_base.py` | ScanNet video | LitePT-B |
| `scannet_video/litept_base_meshnormal.py` | ScanNet video, oracle mesh normals | LitePT-B |

## Slices

| slice | selects |
|---|---|
| `waymo/{utonia_dec,litept_base}/latency_slice.py` | latency slice (latency / FLOPs / memory): 10 whole segments, 299 frames |
| `nuscenes/{utonia_dec,litept_base}/latency_slice.py` | latency slice: 8 whole scenes, 321 keyframes |
| `scannet_video/litept_base_meshnormal/latency_slice.py` | indoor latency slice: 2 scenes, 395 frames |
| `scannet_video/litept_base_meshnormal/eval10.py` | indoor evaluation split: 10 scenes, 2221 frames |
| `waymo/{utonia_dec,litept_base}/quick20.py` | smoke test: first 20 frames |
| `{waymo,nuscenes}/utonia_dec/stage_overlap.py` | temporal-overlap figure (Fig. 2(b)): all consecutive 2 Hz frame pairs on the default lattice of the evaluation configs |
| `scannet_video/utonia_dec/stage_overlap.py` | temporal-overlap figure, indoor: consecutive frame pairs of the 10 evaluation scenes (3 fps) |
| `*/reuse_agreement.py` | Reuse Agreement figure (Fig. 2(c)): 60 frame pairs; the Utonia leaves use the decoder-head checkpoint, whose encoder is the frozen Utonia encoder |

The outdoor latency slices consist of whole sequences, because streaming
evaluation resets its cache at every sequence boundary. They were chosen
with `spacecache/experiments/pick_latency_slice.py` so that the per-frame voxel-count
distribution matches the full split; both backbones use the same frames.

## Keys read by the runners

| key | used for |
|---|---|
| `weight` | checkpoint, relative to the repository root |
| `output_root` | default output root of `run.py` (`--output_dir` overrides it) |
| `lifetime`, `confidence` | SpaceCache defaults for L and θ (overridden by `--lifetime`, `--confidence`) |
| `exclude_unlabeled` | keep regions with ignore-labelled points out of the cache (default True) |
| `max_frames` | evaluate only the first N frames |
| `stride`, `max_pairs`, `max_pairs_per_scene`, `frame_gap_s` | frame-pair sampling of the figure scripts |
| `save_path`, `figure_label`, `model_label` | output directory and labels of the figure scripts |
| `seed`, `device` | defaults for `--seed` and the device |

## Examples

```bash
# backbone on the full Waymo split
PYTHONPATH=. python spacecache/experiments/run.py \
    --config spacecache/configs/waymo/utonia_dec.py --policy backbone --dense_attn

# SpaceCache (Stage-3) on the Waymo latency slice
PYTHONPATH=. python spacecache/experiments/run.py \
    --config spacecache/configs/waymo/utonia_dec/latency_slice.py \
    --policy spacecache_s3 --lifetime 8 --confidence 0.85 --dense_attn

# smoke test
PYTHONPATH=. python spacecache/experiments/run.py \
    --config spacecache/configs/waymo/utonia_dec/quick20.py \
    --policy gitmerge3d --merge_ratio 0.5 --dense_attn
```
