#!/usr/bin/env bash
# Outdoor latency / FLOPs / memory (Table 1, Fig. 6) of every run found in
# the accuracy sweep, measured on the representative latency slices (Waymo: 10
# segments / 299 frames; nuScenes: 8 scenes / 321 frames) with dense
# attention on an NVIDIA RTX A6000. Each job owns one GPU.
#
# Five passes over the slice: pass 1 records FLOPs and peak memory, the
# latency is the mean over passes 2-5 (no metric hooks). SpaceCache uses the
# fused Triton cache kernels.
#
#   bash scripts/reproduce/outdoor_latency.sh [--gpus 0,1]
#
# DS=waymo or DS=nuscenes restricts the datasets. Run outdoor_accuracy.sh
# first: the job list is derived from its result directories. ACC_ROOT and
# OUT_ROOT override the accuracy root and the output root.
# Results: output/outdoor/latency/<dataset>/<backbone>/

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/reproduce/common.sh
parse_gpus "$@"

ACC_ROOT="${ACC_ROOT:-output/outdoor/accuracy}"
OUT_ROOT="${OUT_ROOT:-output/outdoor/latency}"
COMMON=(--dense_attn --timing_level frame --triton_cache --passes 5)
DS="${DS:-}"

JOBS=$(mktemp)
"$PY" spacecache/experiments/gen_latency_jobs.py --root "$ACC_ROOT" > "$JOBS"
while IFS='|' read -r cfg args; do
    [ -n "$cfg" ] || continue
    ds=${cfg%%/*}
    [ -z "$DS" ] || [ "$ds" = "$DS" ] || continue
    bb=$(basename "$(dirname "spacecache/configs/$cfg")")
    # shellcheck disable=SC2086
    queue "spacecache/configs/$cfg" "$OUT_ROOT/$ds/$bb" "${COMMON[@]}" $args
done < "$JOBS"
wait
rm -f "$JOBS"
echo "=== done: $OUT_ROOT"
