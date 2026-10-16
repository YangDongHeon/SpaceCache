#!/usr/bin/env bash
# Download only the 10 Waymo validation segments of the latency slice
# (spacecache/configs/waymo/utonia_dec/latency_slice.py, 299 frames) instead of
# the whole dataset. Needs a Waymo Open Dataset account (register at
# waymo.com/open and accept the license with the Google account that gsutil
# uses) and the Google Cloud SDK (`gsutil`).
#
#   bash scripts/reproduce/download_waymo_slice.sh /path/to/waymo
#   bash scripts/prepare_data.sh waymo /path/to/waymo
#
# Env: WAYMO_BUCKET (default: the Perception v1.4.3 per-segment files);
# check it with `gsutil ls "$WAYMO_BUCKET" | head` if the copy fails.

set -euo pipefail
cd "$(dirname "$0")/../.."

RAW="${1:?usage: $0 /path/to/waymo_raw}"
WAYMO_BUCKET="${WAYMO_BUCKET:-gs://waymo_open_dataset_v_1_4_3/individual_files/validation}"
CFG=spacecache/configs/waymo/utonia_dec/latency_slice.py

mapfile -t SEGMENTS < <(grep -o 'segment-[0-9_]*_with_camera_labels' "$CFG")
[ "${#SEGMENTS[@]}" -gt 0 ] || { echo "no segments found in $CFG" >&2; exit 1; }

mkdir -p "$RAW/validation"
for seg in "${SEGMENTS[@]}"; do
    dst="$RAW/validation/$seg.tfrecord"
    if [ -f "$dst" ]; then
        echo "skip $seg (already downloaded)"
        continue
    fi
    echo "=== $seg"
    gsutil cp "$WAYMO_BUCKET/$seg.tfrecord" "$dst"
done
echo "=== ${#SEGMENTS[@]} segments in $RAW/validation"
