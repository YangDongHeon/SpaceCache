#!/usr/bin/env bash
# Fig. 5 (Sec. 3.3): Reuse Agreement of SpaceCache entries by region
# confidence and by entry age. Waymo / Utonia, SpaceCache at stages 3 and 4,
# representative scenes (10 segments, 299 frames), one scene shard per GPU.
# Every run adds --reuse_agreement: a second forward per frame with the cache
# disarmed gives the backbone feature each served token is scored against
# (Appendix A.2), so a run takes about twice an accuracy run.
#
# Both runs of the figure use theta=0 (no confidence rule), so the curves are
# not shaped by the control they motivate:
#   L=8    (a) RA by region confidence, at the default lifetime
#   L=inf  (b) RA by entry age, every region stays cached
#
#   bash scripts/reproduce/reuse_agreement.sh           # run, then figure
#   bash scripts/reproduce/reuse_agreement.sh smoke     # alignment check, ~5 min
#   bash scripts/reproduce/reuse_agreement.sh run       # the two runs
#   bash scripts/reproduce/reuse_agreement.sh figure    # Fig. 5 panels (CPU)
#   bash scripts/reproduce/reuse_agreement.sh analyze   # further diagnostics (CPU)
#   bash scripts/reproduce/reuse_agreement.sh --gpus 0,1   # only GPUs 0 and 1
#
# RA uses the per-stage temperatures of the motivation run
# (output/motivation/reuse_agreement/waymo_utonia.json, written by
# scripts/reproduce/motivation.sh); RA_TEMPS="3:0.1035,4:0.0525" sets them
# directly instead. Keep the <run>_shardKofN/ directories: the figure
# sums their reuse_agreement.npz.
# Results: output/reuse_agreement/waymo/utonia_dec/,
# figure: figure/method/reuse_agreement/{a_ra_conf,b_ra_age}.*

set -euo pipefail
cd "$(dirname "$0")/../.."

source scripts/reproduce/common.sh
parse_gpus "$@"
STEP="${ARGS[0]:-all}"

CFG=spacecache/configs/waymo/utonia_dec/latency_slice.py
ROOT=output/reuse_agreement/waymo/utonia_dec
SMOKE=output/reuse_agreement/smoke
THETA="${THETA:-0.85}"
SC=(--policy spacecache_s34 --confidence 0)
PROBE=(--reuse_agreement --no_timing --no_metrics)
[ -n "${RA_TEMPS:-}" ] && PROBE+=(--ra_temps "$RA_TEMPS")

case "$STEP" in
all)
    run_steps reuse_agreement.sh run figure ;;
smoke)
    # L=1 reuses nothing, so every served token must equal the backbone
    # feature (same tokens, same RNG); L=inf must produce hits of every age.
    rm -rf "$SMOKE"
    for L in 1 inf; do
        CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" spacecache/experiments/run.py \
            --config "$CFG" --output_dir "$SMOKE" --max_frames 12 \
            --worker_ctx "$WORKER_CTX" "${PROBE[@]}" "${SC[@]}" \
            --lifetime "$L"
    done
    "$PY" - "$SMOKE" <<'EOF'
import glob, json, os, sys
for p in sorted(glob.glob(sys.argv[1] + "/*/results.json")):
    r = json.load(open(p))
    ra = r.get("reuse_agreement") or {}
    c = ra.get("counts") or {}
    print(" ", r["run_name"], "| T", ra.get("temps"), "| counts", c)
    assert c.get("frames", 0) > 0, "the probe recorded no frame"
    assert c.get("misaligned", 0) == 0, "backbone tokens != served tokens"
    rows = [json.loads(x) for x in open(os.path.join(
        os.path.dirname(p), "reuse_agreement_frames.jsonl"))]
    if r["run_name"].endswith("_L1"):
        n = sum(f.get("n_miss_d3", 0) for f in rows)
        t = sum(f.get("miss_ra_d3", 0) for f in rows)
        print(f"    L=1 stage 3: RA {t / max(n, 1):.4f}")
        assert n > 0 and t / n > 0.995, "L=1 served != backbone feature"
    else:
        assert c.get("hit_not_found", 0) == 0, "hit without a shadow row"
        assert (ra.get("depths") or {}).get("d3", {}).get("n_hit", 0) > 0, \
            "L=inf has no hits"
EOF
    echo "=== smoke OK" ;;
run)
    run_sharded "$CFG" "$ROOT" "${PROBE[@]}" "${SC[@]}" --lifetime 8
    run_sharded "$CFG" "$ROOT" "${PROBE[@]}" "${SC[@]}" --lifetime inf ;;
figure)
    "$PY" spacecache/figure/method/reuse_agreement/plot_reuse_agreement.py \
        --root "$ROOT" --theta "$THETA" ;;
analyze)
    "$PY" spacecache/experiments/reuse_agreement/analyze.py \
        --root "$ROOT" --theta "$THETA" ;;
*)
    echo "unknown step $STEP (all | smoke | run | figure | analyze)" >&2
    exit 1 ;;
esac
