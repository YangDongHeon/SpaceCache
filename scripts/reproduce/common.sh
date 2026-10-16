# Helpers shared by the reproduction scripts (sourced from the repo root).
#
#   --gpus 0,1              GPUs to use (see parse_gpus)
#   PY=python               interpreter (override if needed)
#   WORKER_CTX=spawn        DataLoader multiprocessing context

# parse_gpus ARGS...
#   Sets GPUS from `--gpus 0,1` (or `--gpus=0,1`) anywhere in ARGS, else from
#   CUDA_VISIBLE_DEVICES, else every GPU on the machine; ARGS keeps the other
#   arguments. Accuracy runs split their scenes over these GPUs; timing runs
#   take one GPU each, one job at a time per GPU.
parse_gpus() {
    ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
        --gpus) export CUDA_VISIBLE_DEVICES="${2:?--gpus needs a list, e.g. --gpus 0,1}"
                shift 2 ;;
        --gpus=*) export CUDA_VISIBLE_DEVICES="${1#--gpus=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
        esac
    done
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
    else
        mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
    fi
    [ "${#GPUS[@]}" -gt 0 ] || GPUS=(0)
    echo "=== GPUs: ${GPUS[*]}"
}

PY="${PY:-python}"
# DataLoader workers as fresh processes: forked workers can segfault once the
# SpaceCache Triton kernels are compiled (timing is unaffected)
WORKER_CTX="${WORKER_CTX:-spawn}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

# run_sharded CFG OUT ARGS...
#   Accuracy run split into one scene shard per GPU (whole scenes per shard,
#   so the SpaceCache timeline is unchanged), then merged exactly.
run_sharded() {
    local cfg=$1 out=$2
    shift 2
    local n=${#GPUS[@]} k rc=0
    local pids=()
    echo "=== [${n} shards] ${cfg} $*"
    for ((k = 0; k < n; k++)); do
        CUDA_VISIBLE_DEVICES=${GPUS[$k]} "$PY" spacecache/experiments/run.py \
            --config "$cfg" --output_dir "$out" --skip_existing \
            --worker_ctx "$WORKER_CTX" \
            --scene_shard "$k/$n" "$@" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    if [ "$rc" -ne 0 ]; then
        echo "!!! shard run failed: $*" >&2
        return 1
    fi
    "$PY" spacecache/experiments/merge_shards.py --cell "$out"
}

# queue CFG OUT ARGS...
#   Whole-slice run on the next GPU, at most one job per GPU at a time
#   (timing runs must not share a GPU). Call `wait` after the last job.
JOB=0
queue() {
    local cfg=$1 out=$2
    shift 2
    local gpu=${GPUS[$((JOB % ${#GPUS[@]}))]}
    echo "=== [gpu ${gpu}] ${cfg} $*"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" spacecache/experiments/run.py \
        --config "$cfg" --output_dir "$out" --skip_existing \
            --worker_ctx "$WORKER_CTX" "$@" &
    JOB=$((JOB + 1))
    if ((JOB % ${#GPUS[@]} == 0)); then wait; fi
}

# run_steps SCRIPT STEP...
#   Run a step-based script once per step, in order (its default "all").
run_steps() {
    local script=$1 s
    shift
    for s in "$@"; do
        echo "##### $script $s"
        bash "scripts/reproduce/$script" "$s"
    done
}
