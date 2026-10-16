#!/usr/bin/env bash
# Convert ScanNet v2 validation RGB-D videos into per-frame point clouds
# (data/scannetvideo/val/<scene>/frame_XXXXXX/{coord,color,normal,segment20,
# segment200,pose}.npy). Each depth frame is back-projected in the camera
# frame; `ApplyPose` maps it to the scene's world frame at load time.
#
# Needs, for every scene, <scene>.sens and <scene>_2d-label-filt.zip from the
# official ScanNet release (obtained after accepting the ScanNet terms of use).
#
#   SCANNET_SCANS=/path/to/scannet/scans bash scripts/reproduce/prepare_scannet_video.sh
#
# Settings: every 10th frame of the 30 Hz capture (3 Hz), every depth pixel,
# depth truncated at 8 m, camera-frame coordinates.
# The first 10 scenes are the evaluation set of the paper (2,221 frames);
# the motivation study (Fig. 2) uses all 20.

set -u
cd "$(dirname "$0")/../.."

ORIG="${SCANNET_SCANS:?set SCANNET_SCANS to the directory holding <scene>/<scene>.sens}"
OUT="${OUT:-data/scannetvideo/val}"
PY="${PY:-python}"
WORKERS="${WORKERS:-16}"
SCENES="${SCENES:-scene0025_00 scene0050_00 scene0169_00 scene0217_00 \
scene0426_00 scene0488_00 scene0565_00 scene0568_00 scene0580_00 scene0583_00 \
scene0019_00 scene0257_00 scene0304_00 scene0412_00 scene0414_00 scene0549_00 \
scene0575_00 scene0578_00 scene0665_00 scene0701_00}"

mkdir -p "$OUT"
for s in $SCENES; do
    sens="$ORIG/$s/$s.sens"
    zip="$ORIG/$s/${s}_2d-label-filt.zip"
    if [ ! -f "$sens" ] || [ ! -f "$zip" ]; then
        echo "skip $s: missing $sens or $zip"
        continue
    fi
    if ls -d "$OUT/$s"/frame_* > /dev/null 2>&1; then
        echo "skip $s: already converted"
        continue
    fi
    echo "=== $s"
    "$PY" scripts/preprocess_scannet_video.py \
        --sens "$sens" --label-zip "$zip" --out "$OUT/$s" \
        --frame-skip 10 --pixel-stride 1 --depth-trunc 8.0 \
        --frame local --num-workers "$WORKERS"
done
