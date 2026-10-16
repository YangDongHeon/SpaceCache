#!/usr/bin/env bash
# Efficiency analysis (Fig. 7, Sec. 4.3, Appendix D, Table 6) on Waymo with
# Utonia: every method at its operating point within a 2-point mIoU drop.
#
#   bash scripts/reproduce/efficiency.sh          # all steps below, in order
#   bash scripts/reproduce/efficiency.sh picks    # operating points (CPU)
#   bash scripts/reproduce/efficiency.sh block    # per-module timing
#   bash scripts/reproduce/efficiency.sh fine     # merge sub-steps (appendix)
#   bash scripts/reproduce/efficiency.sh flash    # peak memory with FlashAttention
#   bash scripts/reproduce/efficiency.sh report   # Fig. 7 + Table 6 (CPU)
#   bash scripts/reproduce/efficiency.sh --gpus 0,1   # only GPUs 0 and 1
#
# The picks are each method's lowest-encoder-FLOPs setting within the budget,
# read from the outdoor accuracy and latency sweeps, so run
# outdoor_accuracy.sh and outdoor_latency.sh first.
# Latency is measured without FlashAttention; memory with it. The block
# and fine runs make two passes over the scenes, as the ablation runs, and
# every time is the mean over the second pass.
# Figures: figure/experiment/breakdown/, tables: output/efficiency/

set -euo pipefail
cd "$(dirname "$0")/../.."

source scripts/reproduce/common.sh
parse_gpus "$@"
STEP=${ARGS[0]:-all}

CFG=spacecache/configs/waymo/utonia_dec/latency_slice.py
OUT=output/efficiency
BLOCK=$OUT/block/waymo/utonia_dec
FINE=$OUT/fine/waymo/utonia_dec
FLASH=$OUT/flash/waymo/utonia_dec
PICKS=$OUT/picks.tsv
BUDGET=0.02

pick_args() {   # pick_args [merge_only] -> one run.py argument string per line
    [ -f "$PICKS" ] || { echo "run '$0 picks' first" >&2; exit 1; }
    if [ "${1:-}" = merge_only ]; then
        awk -F'\t' '$2 !~ /^backbone/ && $2 !~ /^spacecache/ {print $3}' "$PICKS"
    else
        awk -F'\t' '{print $3}' "$PICKS"
    fi
}

case "$STEP" in
all)
    run_steps efficiency.sh picks block fine flash report ;;
picks)
    mkdir -p "$OUT"
    "$PY" spacecache/experiments/efficiency_breakdown.py --budget "$BUDGET" \
        --pick_by flops --list_picks > "$PICKS"
    cat "$PICKS" ;;
block)
    while IFS= read -r args; do
        # shellcheck disable=SC2086
        queue "$CFG" "$BLOCK" --dense_attn --timing_level block \
            --warmup_frames 1 --passes 2 --no_metrics $args
    done < <(pick_args)
    wait ;;
fine)
    while IFS= read -r args; do
        # shellcheck disable=SC2086
        queue "$CFG" "$FINE" --dense_attn --timing_level block --fine_regions \
            --warmup_frames 1 --passes 2 --no_metrics $args
    done < <(pick_args merge_only)
    wait ;;
flash)
    while IFS= read -r args; do
        # shellcheck disable=SC2086
        queue "$CFG" "$FLASH" --timing_level frame --passes 1 $args
    done < <(pick_args)
    wait ;;
report)
    "$PY" spacecache/experiments/efficiency_breakdown.py --budget "$BUDGET" \
        --pick_by flops --block_dir "$BLOCK" --fine_dir "$FINE" \
        --mem_flash_dir "$FLASH" --rescale none \
        --paper_dir figure/experiment/breakdown
    "$PY" spacecache/experiments/flops_table.py --budget "$BUDGET" ;;
*) echo "unknown step $STEP" >&2; exit 1 ;;
esac
