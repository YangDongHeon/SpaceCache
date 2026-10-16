#!/usr/bin/env bash
# Appendix Table 4: GitMerge3D with and without its adaptive patch partition
# at the same merge ratio r = 0.5. `gitmerge3d` is the version of the main
# tables (GitMerge3D, dagger), `gitmerge3d_adaptive` the official adaptive
# partition.
#   acc    full-val mIoU, dense attention, scene shards over the GPUs:
#          Waymo / nuScenes x Utonia / LitePT-B and the ScanNet video stream
#          (LitePT-B, 10 scenes, mesh normals)
#   lat    outdoor latency on the latency slices (RTX A6000, 5 passes, mean of
#          passes 2-5 as in Table 1), backbone + both versions, one job per GPU
#   orin   ScanNet kernel time on the Jetson AGX Orin, the protocol of
#          indoor_latency_orin.sh (every 8th frame of the 2-scene slice)
#   table  the Table 4 markdown (spacecache/experiments/gitmerge3d_adaptive_table.py)
#
#   bash scripts/reproduce/gitmerge3d_adaptive.sh        # acc, lat, table
#   bash scripts/reproduce/gitmerge3d_adaptive.sh acc --gpus 0,1
#   bash scripts/reproduce/gitmerge3d_adaptive.sh orin   # on the Orin
#   ORIN_CSV=<orin OUT>/graph_fair.csv bash scripts/reproduce/gitmerge3d_adaptive.sh table
#
# Results: output/gitmerge3d_adaptive/{accuracy,latency,indoor_accuracy,
# kernel_time}, table: output/gitmerge3d_adaptive/table4.md. They are kept
# apart from the main sweeps, so the r = 0.5 points do not enter Table 1,
# Fig. 6 or the Table 2 points. The backbone accuracy is read from the main
# sweeps (outdoor_accuracy.sh, indoor_accuracy.sh) when it exists there.
# Env (orin): STRIDE (8), REPLAYS (20), TAG (short host name), OUT.

set -euo pipefail
cd "$(dirname "$0")/../.."

source scripts/reproduce/common.sh
parse_gpus "$@"
STEP=${ARGS[0]:-all}

R=0.5
ROOT=output/gitmerge3d_adaptive
MAIN_ACC=output/outdoor/accuracy
MAIN_IN=output/indoor/accuracy/scannet_video/litept_base_meshnormal
IN_ACC_CFG=spacecache/configs/scannet_video/litept_base_meshnormal/eval10.py
IN_LAT_CFG=spacecache/configs/scannet_video/litept_base_meshnormal/latency_slice.py
CELLS="waymo/utonia_dec waymo/litept_base nuscenes/utonia_dec nuscenes/litept_base"
METHODS="gitmerge3d gitmerge3d_adaptive"

# latency slice of a dataset (Table 1 latency runs)
slice_of() {
    case "$1" in
    waymo) echo latency_slice.py ;;
    nuscenes) echo latency_slice.py ;;
    esac
}

case "$STEP" in
all)
    run_steps gitmerge3d_adaptive.sh acc lat table ;;
acc)
    ACC=(--dense_attn --no_timing --no_metrics)
    for cell in $CELLS; do
        cfg=spacecache/configs/$cell.py
        out=$ROOT/accuracy/$cell
        [ -f "$MAIN_ACC/$cell/backbone_dense/results.json" ] ||
            run_sharded "$cfg" "$out" "${ACC[@]}" --policy backbone
        for m in $METHODS; do
            run_sharded "$cfg" "$out" "${ACC[@]}" --policy "$m" \
                --merge_ratio "$R"
        done
    done
    out=$ROOT/indoor_accuracy
    [ -f "$MAIN_IN/backbone_dense/results.json" ] ||
        run_sharded "$IN_ACC_CFG" "$out" "${ACC[@]}" --shard_balance \
            --policy backbone
    for m in $METHODS; do
        run_sharded "$IN_ACC_CFG" "$out" "${ACC[@]}" --shard_balance \
            --policy "$m" --merge_ratio "$R"
    done ;;
lat)
    T=(--dense_attn --timing_level frame --passes 5)
    for cell in $CELLS; do
        cfg=spacecache/configs/$cell/$(slice_of "${cell%%/*}")
        out=$ROOT/latency/$cell
        queue "$cfg" "$out" "${T[@]}" --policy backbone
        for m in $METHODS; do
            queue "$cfg" "$out" "${T[@]}" --policy "$m" --merge_ratio "$R"
        done
    done
    wait ;;
orin)
    STRIDE="${STRIDE:-8}"
    REPLAYS="${REPLAYS:-20}"
    TAG="${TAG:-$(hostname -s)}"
    OUT="${OUT:-$ROOT/kernel_time/$TAG/piecewise}"
    COMMON=(--dense_attn --timing_level frame --no_metrics --passes 1
            --graph_capture --graph_mode piecewise
            --graph_frames "stride:${STRIDE}:0" --graph_replays "$REPLAYS"
            --num_workers 0 --skip_existing)
    mkdir -p "$OUT"
    {
        printf 'budget\tmethod\tcell\trun\targs\n'
        printf 'r0.5\t+GitMerge3D\tgitmerge3d_r0.5\tgitmerge3d_dense_r0.5\t--policy gitmerge3d --merge_ratio %s\n' "$R"
        printf 'r0.5\t+GitMerge3D (adaptive)\tgitmerge3d_adaptive_r0.5\tgitmerge3d_adaptive_dense_r0.5\t--policy gitmerge3d_adaptive --merge_ratio %s\n' "$R"
    } > "$OUT/cells.tsv"
    # backbone first and last to expose clock drift
    for c in "backbone|--policy backbone" \
             "gitmerge3d_r0.5|--policy gitmerge3d --merge_ratio $R" \
             "gitmerge3d_adaptive_r0.5|--policy gitmerge3d_adaptive --merge_ratio $R" \
             "backbone_end|--policy backbone"; do
        name="${c%%|*}"
        args="${c#*|}"
        mkdir -p "$OUT/$name"
        echo ">>> $name: $args"
        # shellcheck disable=SC2086
        "$PY" spacecache/experiments/run.py --config "$IN_LAT_CFG" $args \
            "${COMMON[@]}" --output_dir "$OUT/$name" \
            > "$OUT/$name/launch.log" 2>&1 \
            || echo "[FAIL] $name (see $OUT/$name/launch.log)"
    done
    "$PY" spacecache/experiments/graph_fair_table.py --root "$OUT" --with_adaptive ;;
table)
    "$PY" spacecache/experiments/gitmerge3d_adaptive_table.py --root "$ROOT" \
        --main_acc "$MAIN_ACC" --main_indoor "$MAIN_IN" \
        ${ORIN_CSV:+--orin "$ORIN_CSV"} ;;
*) echo "unknown step $STEP (all | acc | lat | orin | table)" >&2; exit 1 ;;
esac
