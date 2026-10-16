#!/usr/bin/env bash
# Ablations (Table 3, Appendix C.3) on Waymo with Utonia.
#   (a) lifetime L in {1, 2, 4, 8, 16, 32} without / with the confidence rule
#       (theta = 0.85), SpaceCache at stages 3+4; L = 1 never reuses a feature
#       and reproduces the backbone. Accuracy runs log the region statistics
#       (dynamic / static regions, entry age, evicted regions).
#   (b) stage selection: a single cache at stage 0-4, or stages 3+4, native
#       region size, L = 8, theta in {0, 0.85, 0.99} (the table uses 0.85).
# Accuracy: full validation split, dense attention (as the main tables), one
# scene shard per GPU.
# Latency: 10-segment latency slice, dense attention, 2 passes (pass 1
# records FLOPs / memory, pass 2 the latency), fused Triton cache operations.
#
#   bash scripts/reproduce/ablation.sh            # all steps: acc, lat, tables
#   bash scripts/reproduce/ablation.sh acc|lat|tables
#   bash scripts/reproduce/ablation.sh --gpus 0,1 # only GPUs 0 and 1
#
# EXPIRY=fixed reuses every cache entry for exactly L-1 frames instead of
# the default expiry staggered by the region index (run names tagged _fixed).
# Results: output/ablation/{accuracy,latency}/waymo/utonia_dec/, tables in
# output/ablation/

set -euo pipefail
cd "$(dirname "$0")/../.."

source scripts/reproduce/common.sh
parse_gpus "$@"
STEP=${ARGS[0]:-all}

CFG_FULL=spacecache/configs/waymo/utonia_dec.py
CFG_LAT=spacecache/configs/waymo/utonia_dec/latency_slice.py
ROOT=output/ablation
ACC=$ROOT/accuracy/waymo/utonia_dec
LAT=$ROOT/latency/waymo/utonia_dec
THETA=0.85
LIFETIMES="2 4 8 16 32"
PLACEMENT_THETAS="0 0.85 0.99"
EXPIRY="${EXPIRY:-staggered}"
S34=(--policy spacecache_s34 --expiry "$EXPIRY")

case "$STEP" in
all)
    run_steps ablation.sh acc lat tables ;;
acc)
    A=(--dense_attn --no_timing --no_metrics)
    run_sharded "$CFG_FULL" "$ACC" "${A[@]}" --policy backbone
    run_sharded "$CFG_FULL" "$ACC" "${A[@]}" "${S34[@]}" \
        --lifetime 1 --confidence 0 --region_stats
    for L in $LIFETIMES; do
        for t in 0 "$THETA"; do
            run_sharded "$CFG_FULL" "$ACC" "${A[@]}" "${S34[@]}" \
                --lifetime "$L" --confidence "$t" --region_stats
        done
    done
    for t in $PLACEMENT_THETAS; do
        for s in s0 s1 s2 s3 s4 s34; do
            run_sharded "$CFG_FULL" "$ACC" "${A[@]}" \
                --policy "spacecache_${s}" --lifetime 8 \
                --confidence "$t" --expiry "$EXPIRY"
        done
    done ;;
lat)
    T=(--dense_attn --timing_level frame --passes 2)
    queue "$CFG_LAT" "$LAT" "${T[@]}" --policy backbone
    queue "$CFG_LAT" "$LAT" "${T[@]}" --triton_cache "${S34[@]}" \
        --lifetime 1 --confidence 0
    for L in $LIFETIMES; do
        for t in 0 "$THETA"; do
            queue "$CFG_LAT" "$LAT" "${T[@]}" --triton_cache "${S34[@]}" \
                --lifetime "$L" --confidence "$t"
        done
    done
    for t in $PLACEMENT_THETAS; do
        for s in s0 s1 s2 s3 s4 s34; do
            queue "$CFG_LAT" "$LAT" "${T[@]}" --triton_cache \
                --policy "spacecache_${s}" --lifetime 8 \
                --confidence "$t" --expiry "$EXPIRY"
        done
    done
    wait ;;
tables)
    # (a): hit fraction, hit accuracy, speedup, mIoU with / without the rule
    "$PY" spacecache/experiments/confidence_analysis.py --abl "$ROOT" \
        --lat_name latency --lifetimes "${LIFETIMES// /,}" --theta "$THETA" \
        --expiry "$EXPIRY"
    # (a) dynamic / static region accuracy and the Appendix C.3 statistics
    "$PY" spacecache/experiments/ablation_table.py --abl "$ROOT" \
        --lat_name latency --lifetimes "1,${LIFETIMES// /,}" \
        --theta "$THETA" --place_mf 1,1,1 --expiry "$EXPIRY"
    "$PY" spacecache/experiments/lifetime_analysis.py --acc_root "$ACC" \
        --lifetimes "${LIFETIMES// /,}" --theta "$THETA" --theta0 \
        --out "$ROOT/lifetime" --expiry "$EXPIRY"
    # (b) stage selection
    "$PY" spacecache/experiments/placement_table.py \
        --acc_root "$ACC" --lat_root "$LAT" --thetas 0.85 \
        --expiry "$EXPIRY" ;;
*) echo "unknown step $STEP (all | acc | lat | tables)" >&2; exit 1 ;;
esac
