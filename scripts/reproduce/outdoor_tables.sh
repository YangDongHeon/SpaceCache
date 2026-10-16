#!/usr/bin/env bash
# Build Table 1 and Fig. 6 from the outdoor accuracy and latency runs.
#
#   bash scripts/reproduce/outdoor_tables.sh
#   LAT_ROOT=<other latency root> bash scripts/reproduce/outdoor_tables.sh
#
# Table 1: each method's fewest-FLOPs setting (total FLOPs) within 1 (and 2)
# mIoU points of the backbone (the ringed point in Fig. 6), with its latency
# (Commit Gate included) and total FLOPs -> output/outdoor/table1.md
# Fig. 6:  figure/experiment/outdoor/pareto.{png,pdf,csv} by default
#
# Env: ACC_ROOT, LAT_ROOT, TABLE (markdown path), FIG_DIR

set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

MAIN=output/outdoor
ACC_ROOT="${ACC_ROOT:-$MAIN/accuracy}"
LAT_ROOT="${LAT_ROOT:-$MAIN/latency}"
TABLE="${TABLE:-$MAIN/table1.md}"
FIG_DIR="${FIG_DIR:-figure/experiment/outdoor}"
"$PY" spacecache/experiments/main_table_outdoor.py \
    --root "$ACC_ROOT" --lat_root "$LAT_ROOT" \
    --budgets 0.01,0.02 --flops total --pick_by total --out "$TABLE"
"$PY" spacecache/figure/experiment/outdoor/plot_pareto.py \
    --acc_root "$ACC_ROOT" --lat_root "$LAT_ROOT" --out "$FIG_DIR"
