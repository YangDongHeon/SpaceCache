#!/usr/bin/env bash
# Preprocess a downloaded dataset into data/<dataset>.
#
#   bash scripts/prepare_data.sh waymo    /path/to/waymo      # {training,validation}/*.tfrecord
#   bash scripts/prepare_data.sh nuscenes /path/to/nuscenes   # v1.0-trainval, v1.0-test, samples, sweeps, lidarseg
#   bash scripts/prepare_data.sh scannet  /path/to/scannet    # scans/<scene>/<scene>.sens, _2d-label-filt.zip
#
# scannet converts the RGB-D videos of the evaluation scenes into
# data/scannetvideo (scripts/reproduce/prepare_scannet_video.sh); the fused
# scenes of data/scannet come preprocessed from Pointcept. Run it in the
# environment of environment-data.yml (conda activate spacecache-data).
# Env: SPLITS (Waymo splits, default "validation"; add "training" to train
# the checkpoints), WORKERS (default: all cores), PY.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
WORKERS="${WORKERS:-$(nproc)}"
DS=${1:?dataset: waymo | nuscenes | scannet}
RAW=$(realpath "${2:?path to the downloaded dataset}")
PRE=pointcept/datasets/preprocessing

case "$DS" in
waymo)
    # shellcheck disable=SC2086
    "$PY" $PRE/waymo/preprocess_waymo.py --dataset_root "$RAW" \
        --output_root data/waymo --splits ${SPLITS:-validation} \
        --num_workers "$WORKERS" ;;
nuscenes)
    # the info files hold paths relative to data/nuscenes/raw
    mkdir -p data/nuscenes
    if [ ! -e data/nuscenes/raw ]; then
        ln -s "$RAW" data/nuscenes/raw
    elif [ "$(realpath data/nuscenes/raw)" != "$RAW" ]; then
        echo "data/nuscenes/raw already points to $(realpath data/nuscenes/raw)" >&2
        exit 1
    fi
    "$PY" $PRE/nuscenes/preprocess_nuscenes_info.py \
        --dataset_root data/nuscenes/raw --output_root data/nuscenes ;;
scannet)
    SCANNET_SCANS="$RAW/scans" WORKERS="$WORKERS" PY="$PY" \
        bash scripts/reproduce/prepare_scannet_video.sh
    DS=scannetvideo ;;
*)
    echo "unknown dataset $DS (waymo | nuscenes | scannet)" >&2
    exit 1 ;;
esac
echo "=== done: data/$DS"
