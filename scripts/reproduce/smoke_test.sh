#!/usr/bin/env bash
# Quick end-to-end check on the first 20 Waymo validation frames: the backbone,
# SpaceCache (stages 3+4) and the three token-merging baselines.
#
#   bash scripts/reproduce/smoke_test.sh                  # Utonia (default)
#   BB=litept_base bash scripts/reproduce/smoke_test.sh
#   bash scripts/reproduce/smoke_test.sh --gpus 2         # on GPU 2
#
# Runs on one GPU, the first of --gpus (default: the first GPU).
#
# Results: output/smoke/waymo/<backbone>/<run>/results.json

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/reproduce/common.sh
parse_gpus "$@"
export CUDA_VISIBLE_DEVICES=${GPUS[0]}

BB="${BB:-utonia_dec}"
CFG=spacecache/configs/waymo/$BB/quick20.py
OUT=output/smoke/waymo/$BB
COMMON=(--dense_attn --timing_level stage --output_dir "$OUT")

for args in "--policy backbone" \
            "--policy spacecache_s34 --lifetime 8 --confidence 0.85" \
            "--policy tome --merge_ratio 0.02" \
            "--policy algm --algm_tau 0.98" \
            "--policy gitmerge3d --merge_ratio 0.5"; do
    echo "=== $args"
    # shellcheck disable=SC2086
    "$PY" spacecache/experiments/run.py --config "$CFG" "${COMMON[@]}" $args
done
grep -h '"mIoU"' "$OUT"/*/results.json
