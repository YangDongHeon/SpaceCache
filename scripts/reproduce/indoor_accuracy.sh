#!/usr/bin/env bash
# Indoor accuracy (Table 2): LitePT-B on the ScanNet video stream,
# 10 validation scenes / 2,221 frames at 3 Hz, surface normals taken from the
# reconstructed mesh. Dense attention (the path the Jetson latency runs use),
# FLOPs recorded, each run split into frame-balanced scene shards over GPUs.
#
#   bash scripts/reproduce/indoor_accuracy.sh [--gpus 0,1]
#
# Runs the backbone and the operating points of Table 2, i.e. the swept
# values of Appendix Table 5 (ScanNet column) listed in table2_points.txt.
# GitMerge3D uses the official scatter_reduce merge. Afterwards
# spacecache/experiments/indoor_table.py writes the mIoU and GFLOPs columns of
# Table 2 (the kernel-time columns follow from indoor_latency_orin.sh).
# Finished runs are skipped.
# Results: output/indoor/accuracy/scannet_video/litept_base_meshnormal/,
# table: output/indoor/table2.md

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/reproduce/common.sh
parse_gpus "$@"

CFG=spacecache/configs/scannet_video/litept_base_meshnormal/eval10.py
OUT=output/indoor/accuracy/scannet_video/litept_base_meshnormal
mapfile -t POINTS < <(grep -v '^#' scripts/reproduce/table2_points.txt \
                      | grep -v '^[[:space:]]*$')

run() {
    run_sharded "$CFG" "$OUT" --dense_attn --no_timing --shard_balance "$@"
}

run --policy backbone
for p in "${POINTS[@]}"; do
    IFS='|' read -r _budget _cell _run args <<< "$p"
    # shellcheck disable=SC2086
    run $args
done

"$PY" spacecache/experiments/indoor_table.py --acc_root "$OUT"
echo "=== done: $OUT"
