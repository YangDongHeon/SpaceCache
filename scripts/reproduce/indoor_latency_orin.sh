#!/usr/bin/env bash
# Indoor latency (Table 2) on a Jetson AGX Orin. An indoor frame is small, so
# eager latency is dominated by host-side kernel launches; Table 2 therefore
# reports the GPU kernel time of each frame (the summed kernel durations of
# its forward and the Commit Gate). Every captured frame is also replayed as
# a chain of CUDA graphs (piecewise mode: host steps such as syncs, dynamic
# shapes and spconv rulebooks run for real between the graph segments), which
# checks that the chain runs the same kernels.
#
# Protocol: the 2-scene latency slice (395 frames) is streamed in full so
# SpaceCache reaches every captured frame with its true cache state; every
# 8th frame is captured (50 frames), each replayed 20 times. The backbone
# runs first and last to expose clock drift. Methods run at the Table 2
# operating points of table2_points.txt; GitMerge3D uses the official
# scatter_reduce merge.
#
# Before starting on the Orin: sudo nvpmodel -m 0 && sudo jetson_clocks,
# and keep the GPU otherwise idle for the whole sweep.
#
#   bash scripts/reproduce/indoor_latency_orin.sh
#
# Then, with the accuracy runs of indoor_accuracy.sh present:
#   PYTHONPATH=. python spacecache/experiments/indoor_table.py --orin <OUT>/graph_fair.csv
#
# Env: STRIDE (8), REPLAYS (20), OUT, TAG (default: short host name)

set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

CFG=spacecache/configs/scannet_video/litept_base_meshnormal/latency_slice.py
STRIDE="${STRIDE:-8}"
REPLAYS="${REPLAYS:-20}"
TAG="${TAG:-$(hostname -s)}"
OUT="${OUT:-output/indoor/kernel_time/$TAG/piecewise}"
mapfile -t POINTS < <(grep -v '^#' scripts/reproduce/table2_points.txt \
                      | grep -v '^[[:space:]]*$')

COMMON=(--dense_attn --timing_level frame --no_metrics --passes 1
        --graph_capture --graph_mode piecewise
        --graph_frames "stride:${STRIDE}:0" --graph_replays "$REPLAYS"
        --num_workers 0 --skip_existing)

# cells: backbone, every operating point, backbone again. cells.tsv gives
# graph_fair_table.py the mIoU-drop column of each cell.
mkdir -p "$OUT"
CELLS=("backbone|--policy backbone")
printf 'budget\tmethod\tcell\n' > "$OUT/cells.tsv"
for p in "${POINTS[@]}"; do
    IFS='|' read -r budget cell _run args <<< "$p"
    CELLS+=("${cell}|${args}")
    printf -- '-%s\t-\t%s\n' "$budget" "$cell" >> "$OUT/cells.tsv"
done
CELLS+=("backbone_end|--policy backbone")

fails=()
for c in "${CELLS[@]}"; do
    name="${c%%|*}"
    args="${c#*|}"
    mkdir -p "$OUT/$name"
    echo ">>> $name: $args"
    # shellcheck disable=SC2086
    if ! "$PY" spacecache/experiments/run.py --config "$CFG" $args \
            "${COMMON[@]}" --output_dir "$OUT/$name" \
            > "$OUT/$name/launch.log" 2>&1; then
        echo "[FAIL] $name (see $OUT/$name/launch.log)"
        fails+=("$name")
    fi
done

"$PY" spacecache/experiments/graph_fair_table.py --root "$OUT"
echo "Table 2: copy $OUT/graph_fair.csv to the machine with the"
echo "indoor_accuracy.sh results and run there:"
echo "  PYTHONPATH=. python spacecache/experiments/indoor_table.py --orin <path>/graph_fair.csv"
if (( ${#fails[@]} )); then
    echo "failed cells: ${fails[*]}"
    exit 1
fi
