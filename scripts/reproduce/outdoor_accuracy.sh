#!/usr/bin/env bash
# Outdoor accuracy sweep (Table 1, Fig. 6): full validation split, dense
# attention (the same path as the latency runs), every method over its knob.
# Every run is split into one scene shard per GPU and merged.
#
#   bash scripts/reproduce/outdoor_accuracy.sh                     # all four cells
#   bash scripts/reproduce/outdoor_accuracy.sh waymo utonia_dec    # one cell
#   bash scripts/reproduce/outdoor_accuracy.sh --gpus 0,1          # only GPUs 0 and 1
#
# ONLY="backbone spacecache_s34" restricts the methods.
# Results: output/outdoor/accuracy/<dataset>/<backbone>/

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/reproduce/common.sh
parse_gpus "$@"
set -- ${ARGS[@]+"${ARGS[@]}"}

if [ $# -eq 0 ]; then
    for cell in "waymo utonia_dec" "waymo litept_base" \
                "nuscenes utonia_dec" "nuscenes litept_base"; do
        # shellcheck disable=SC2086
        bash scripts/reproduce/outdoor_accuracy.sh $cell
    done
    exit 0
fi
DS=${1:?dataset: waymo | nuscenes}
BB=${2:?backbone: utonia_dec | litept_base}

CFG=spacecache/configs/$DS/$BB.py
OUT=output/outdoor/accuracy/$DS/$BB
ONLY="${ONLY:-}"
want() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]; }

# Swept values of the points reported in the paper (Fig. 6; Table 1 picks
# from the same points): ALGM tau, ToMe / GitMerge3D merge ratio r,
# SpaceCache confidence threshold theta at lifetime L = 8.
case "$DS/$BB" in
waymo/utonia_dec)
    ALGM_TAU="0.96 0.97 0.98 0.995 1.0"
    TOME_R="0.01 0.015 0.025 0.04 0.045"
    GM3D_R="0.1 0.2 0.3 0.4 0.5 0.6 0.65 0.72"
    S3_THETA="0 0.55 0.7 0.8 0.95 0.99"
    S34_THETA="0 0.5 0.55 0.8 0.95 0.99" ;;
waymo/litept_base)
    ALGM_TAU="0.6 0.7 0.75 0.85 0.95 0.97"
    TOME_R="0.01 0.05 0.095 0.12 0.16 0.2"
    GM3D_R="0.5 0.645 0.7 0.72"
    S3_THETA="0 0.7 0.8 0.95 0.99"
    S34_THETA="0 0.6 0.8 0.95 0.99" ;;
nuscenes/utonia_dec)
    ALGM_TAU="0.7 0.95 0.97 0.995"
    TOME_R="0.01 0.02 0.025 0.035 0.045 0.05"
    GM3D_R="0.1 0.2 0.3 0.35 0.45 0.5 0.6"
    S3_THETA="0.8 0.9 0.97 0.99"
    S34_THETA="0.7 0.85 0.95 0.985 0.99" ;;
nuscenes/litept_base)
    ALGM_TAU="0.9 0.925 0.95 0.98 1.0"
    TOME_R="0.02 0.04 0.06 0.09 0.12"
    GM3D_R="0.1 0.2 0.33 0.45 0.6"
    S3_THETA="0 0.5 0.65 0.8 0.95 0.99"
    S34_THETA="0.5 0.7 0.825 0.95 0.97 0.99" ;;
*) echo "unknown cell $DS/$BB" >&2; exit 1 ;;
esac

ACC=(--dense_attn --no_timing --no_metrics)
SC=(--lifetime 8)

want backbone && run_sharded "$CFG" "$OUT" "${ACC[@]}" --policy backbone
for t in $ALGM_TAU; do
    want algm && run_sharded "$CFG" "$OUT" "${ACC[@]}" \
        --policy algm --algm_tau "$t"
done
for r in $TOME_R; do
    want tome && run_sharded "$CFG" "$OUT" "${ACC[@]}" \
        --policy tome --merge_ratio "$r"
done
for r in $GM3D_R; do
    want gitmerge3d && run_sharded "$CFG" "$OUT" "${ACC[@]}" \
        --policy gitmerge3d --merge_ratio "$r"
done
for t in $S3_THETA; do
    want spacecache_s3 && run_sharded "$CFG" "$OUT" "${ACC[@]}" \
        --policy spacecache_s3 "${SC[@]}" --confidence "$t"
done
for t in $S34_THETA; do
    want spacecache_s34 && run_sharded "$CFG" "$OUT" "${ACC[@]}" \
        --policy spacecache_s34 "${SC[@]}" --confidence "$t"
done
echo "=== done: $OUT"
