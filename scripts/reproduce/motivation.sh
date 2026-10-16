#!/usr/bin/env bash
# Motivation study (Fig. 2).
#   (b) temporal overlap: share of the stage-l tokens of frame t+1 whose region
#       (stage cell size, global frame) is occupied at frame t, for every
#       consecutive frame pair of the validation streams. No model is needed.
#   (c) semantic consistency: agreement of the class-prototype similarity
#       profiles of overlapped features, per encoder stage, for Utonia and
#       LitePT-B (60 frame pairs per dataset).
#
#   bash scripts/reproduce/motivation.sh              # first GPU
#   bash scripts/reproduce/motivation.sh --gpus 2     # on GPU 2
#
# Results: output/motivation/{stage_overlap,reuse_agreement}/, figures in
# figure/motivation/.

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/reproduce/common.sh
parse_gpus "$@"
export CUDA_VISIBLE_DEVICES=${GPUS[0]}

OVERLAP=spacecache/figure/motivation/temporal_overlap
AGREE=spacecache/figure/motivation/reuse_agreement
CFG=spacecache/configs

# (b) temporal overlap, on the voxel lattice of the evaluation configs
# (outdoor: the default per-frame lattice; ScanNet: world-anchored)
OV=output/motivation/stage_overlap
for job in "waymo/utonia_dec/stage_overlap.py waymo" \
           "nuscenes/utonia_dec/stage_overlap.py nuscenes" \
           "scannet_video/utonia_dec/stage_overlap.py scannet"; do
    set -- $job
    "$PY" $OVERLAP/stage_overlap.py --config "$CFG/$1" --name "$2" \
        --out "$OV" --num_workers 8
done
"$PY" $OVERLAP/plot_stage_overlap.py --json "$OV/waymo.json" \
    --json "$OV/nuscenes.json" --json "$OV/scannet.json"

# (c) semantic consistency; the 20 ScanNet scenes give 3 pairs each
for job in "waymo/utonia_dec/reuse_agreement.py waymo_utonia 1" \
           "waymo/litept_base/reuse_agreement.py waymo_litept 1" \
           "nuscenes/utonia_dec/reuse_agreement.py nuscenes_utonia 1" \
           "nuscenes/litept_base/reuse_agreement.py nuscenes_litept 1" \
           "scannet_video/utonia_dec/reuse_agreement.py scannet_utonia 3" \
           "scannet_video/litept_base/reuse_agreement.py scannet_litept 3"; do
    set -- $job
    "$PY" $AGREE/cone_affinity.py --config "$CFG/$1" --name "$2" \
        --out output/motivation/reuse_agreement --max_pairs_per_scene "$3" \
        --depths 0,1,2,3,4
done
"$PY" $AGREE/plot_cone_trend_multi.py --runs output/motivation/reuse_agreement \
    --by_model
