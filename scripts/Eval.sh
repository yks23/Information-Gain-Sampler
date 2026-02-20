#!/bin/bash
################################################################################
# Unified evaluation script for Masked Diffusion Models
#
# Usage:
#   bash Eval.sh --task humaneval --model /path/to/model --mode info-gain
#   bash Eval.sh --task math500 --model GSAI-ML/LLaDA-8B-Instruct --mode pc_sampler
#   bash Eval.sh --task sudoku --model /path/to/model --mode info-gain --candidate_number 8
#   bash Eval.sh --task countdown --model /path/to/dream --mode info-gain --no_shot
#   bash Eval.sh --task humaneval --model /path/to/model --mode info-gain --use_kv_cache
#
# Built-in task defaults (gen_length, steps, block_length, data_path) are
# applied automatically. Any parameter can be overridden via CLI flags.
################################################################################

set -euo pipefail
cd "$(dirname "$0")"   # ensure we run from scripts/

# ========================= Defaults ==========================
TASK=""
MODEL=""
MODE="info-gain"
DEVICE="cuda"
TEMPERATURE=0.7
ALPHA=10
LAMBD=0.0
BASELINE_NAME="../data/baseline/reference_corpus.json"
POSITION_TEMPERATURE=0
CANDIDATE_NUMBER=1
HEURISTIC="confidence"
TOKENS_PER_STEP=""
GAMMA=0.01
THREAD=0.9
NO_SHOT=""
USE_KV_CACHE=""
RESULT_DIR=""

# User-overridable task params (empty = use built-in default)
GEN_LENGTH=""
STEPS=""
BLOCK_LENGTH=""
DATA_PATH=""

# ========================= CLI Parsing ==========================
usage() {
    cat <<EOF
Usage: bash Eval.sh --task TASK --model MODEL [OPTIONS]

Required:
  --task TASK             Task name: humaneval|mbpp|math500|gsm8k|gpqa|sudoku|countdown|creativity_writing
  --model MODEL           Model name or local path

Generation mode:
  --mode MODE             Generation mode (default: info-gain)
                          Choices: original|pc_sampler|eb_sampler|fast_dllm|entropy|margin|info-gain

Common options:
  --device DEV            CUDA device (default: cuda)
  --temperature T         Sampling temperature (default: 0.7)
  --alpha A               Alpha for PC-Sampler (default: 10)
  --lambd L               Lambda for PC-Sampler (default: 0.0)
  --baseline_name PATH    Baseline frequency file (default: ../data/baseline/reference_corpus.json)
  --gamma G               Gamma for EB-Sampler (default: 0.01)
  --thread T              Threshold for fast_dllm (default: 0.9)
  --no_shot               Disable few-shot examples
  --use_kv_cache          Enable KV cache (completed blocks are cached)

Task param overrides (auto-detected if omitted):
  --gen_length N          Generated answer length
  --steps N               Number of sampling steps
  --block_length N        Block length for semi-autoregressive
  --tokens_per_step N     Tokens decoded per step (K)
  --data_path PATH        Dataset file path

Info-Gain specific:
  --position_temperature T  Position sampling temperature (default: 0)
  --candidate_number N      Number of candidate actions (default: 1)
  --heuristic H             Heuristic: pc|confidence|neg_entropy|margin|uniform (default: confidence)

Output:
  --result_dir DIR        Result output directory (default: auto-generated)
  --result_path PATH      Full result file path (overrides result_dir)

EOF
    exit 1
}

# Parse CLI arguments
EXTRA_ARGS=()
RESULT_PATH_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)                  TASK="$2";                  shift 2 ;;
        --model)                 MODEL="$2";                 shift 2 ;;
        --mode)                  MODE="$2";                  shift 2 ;;
        --device)                DEVICE="$2";                shift 2 ;;
        --temperature)           TEMPERATURE="$2";           shift 2 ;;
        --alpha)                 ALPHA="$2";                 shift 2 ;;
        --lambd)                 LAMBD="$2";                 shift 2 ;;
        --baseline_name)         BASELINE_NAME="$2";         shift 2 ;;
        --gamma)                 GAMMA="$2";                 shift 2 ;;
        --thread)                THREAD="$2";                shift 2 ;;
        --gen_length)            GEN_LENGTH="$2";            shift 2 ;;
        --steps)                 STEPS="$2";                 shift 2 ;;
        --block_length)          BLOCK_LENGTH="$2";          shift 2 ;;
        --tokens_per_step)       TOKENS_PER_STEP="$2";       shift 2 ;;
        --data_path)             DATA_PATH="$2";             shift 2 ;;
        --position_temperature)  POSITION_TEMPERATURE="$2";  shift 2 ;;
        --candidate_number)      CANDIDATE_NUMBER="$2";      shift 2 ;;
        --heuristic)             HEURISTIC="$2";             shift 2 ;;
        --result_dir)            RESULT_DIR="$2";            shift 2 ;;
        --result_path)           RESULT_PATH_OVERRIDE="$2";  shift 2 ;;
        --no_shot)               NO_SHOT="--no_shot";        shift   ;;
        --use_kv_cache)          USE_KV_CACHE="--use_kv_cache"; shift  ;;
        -h|--help)               usage                                ;;
        *)
            echo "Unknown option: $1"; usage ;;
    esac
done

# Validate required args
if [[ -z "$TASK" ]]; then echo "Error: --task is required."; usage; fi
if [[ -z "$MODEL" ]]; then echo "Error: --model is required."; usage; fi

# ========================= Task Defaults ==========================
# Built-in per-task defaults. Override any with CLI flags above.
apply_task_defaults() {
    case "$TASK" in
        humaneval)
            : "${GEN_LENGTH:=256}";  : "${STEPS:=256}";  : "${BLOCK_LENGTH:=32}"
            : "${DATA_PATH:=../data/humaneval.jsonl}"
            : "${TOKENS_PER_STEP:=1}"
            ;;
        mbpp)
            : "${GEN_LENGTH:=256}";  : "${STEPS:=256}";  : "${BLOCK_LENGTH:=32}"
            # Try sanitized-mbpp.json first, fallback to mbpp.jsonl
            if [ -z "${DATA_PATH:-}" ]; then
                if [ -f "../data/sanitized-mbpp.json" ]; then
                    DATA_PATH="../data/sanitized-mbpp.json"
                else
                    DATA_PATH="../data/mbpp.jsonl"
                fi
            fi
            : "${TOKENS_PER_STEP:=1}"
            ;;
        math500)
            : "${GEN_LENGTH:=256}";  : "${STEPS:=256}";  : "${BLOCK_LENGTH:=32}"
            : "${DATA_PATH:=../data/math500.jsonl}"
            : "${TOKENS_PER_STEP:=1}"
            ;;
        gsm8k)
            : "${GEN_LENGTH:=512}";  : "${STEPS:=512}";  : "${BLOCK_LENGTH:=16}"
            : "${DATA_PATH:=../data/gsm8k.jsonl}"
            : "${TOKENS_PER_STEP:=1}"
            ;;
        gpqa)
            : "${GEN_LENGTH:=256}";  : "${STEPS:=256}";  : "${BLOCK_LENGTH:=32}"
            : "${DATA_PATH:=../data/gpqa.jsonl}"
            : "${TOKENS_PER_STEP:=1}"
            ;;
        sudoku)
            : "${GEN_LENGTH:=0}";    : "${STEPS:=16}";   : "${BLOCK_LENGTH:=16}"
            : "${DATA_PATH:=../data/sudoku.csv}"
            : "${TOKENS_PER_STEP:=1}"
            ;;
        countdown)
            : "${GEN_LENGTH:=48}";   : "${STEPS:=48}";   : "${BLOCK_LENGTH:=48}"
            : "${DATA_PATH:=../data/countdown.jsonl}"
            : "${TOKENS_PER_STEP:=1}"
            ;;
        creativity_writing)
            : "${GEN_LENGTH:=512}";  : "${STEPS:=512}";  : "${BLOCK_LENGTH:=32}"
            : "${DATA_PATH:=../src/benchmarks/text_tasks/creativity_writing/data/creativity_writing.jsonl}"
            : "${TOKENS_PER_STEP:=1}"
            ;;
        *)
            echo "Error: Unknown task '$TASK'."
            echo "Supported tasks: humaneval, mbpp, math500, gsm8k, gpqa, sudoku, countdown, creativity_writing"
            exit 1
            ;;
    esac
}

apply_task_defaults

# ========================= Result Path ==========================
# Auto-generate model short name from model path
model_short=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')

if [[ -n "$RESULT_PATH_OVERRIDE" ]]; then
    RESULT_PATH="$RESULT_PATH_OVERRIDE"
else
    : "${RESULT_DIR:=../results/${model_short}_eval}"
    mkdir -p "$RESULT_DIR"
    RESULT_PATH="${RESULT_DIR}/${TASK}_${MODE}_T${TEMPERATURE}_K${TOKENS_PER_STEP}.txt"
fi

# ========================= Run Evaluation ==========================
echo "======================================================="
echo " Task           : $TASK"
echo " Model          : $MODEL"
echo " Mode           : $MODE"
echo " Device         : $DEVICE"
echo " gen_length     : $GEN_LENGTH"
echo " steps          : $STEPS"
echo " block_length   : $BLOCK_LENGTH"
echo " tokens_per_step: $TOKENS_PER_STEP"
echo " temperature    : $TEMPERATURE"
echo " heuristic      : $HEURISTIC"
echo " candidate_num  : $CANDIDATE_NUMBER"
echo " pos_temperature: $POSITION_TEMPERATURE"
echo " use_kv_cache   : ${USE_KV_CACHE:-off}"
echo " data_path      : $DATA_PATH"
echo " result_path    : $RESULT_PATH"
echo "======================================================="

CMD=(
    python eval.py
    --task "$TASK"
    --model_name "$MODEL"
    --device "$DEVICE"
    --gen_length "$GEN_LENGTH"
    --steps "$STEPS"
    --block_length "$BLOCK_LENGTH"
    --temperature "$TEMPERATURE"
    --mode "$MODE"
    --lambd "$LAMBD"
    --alpha "$ALPHA"
    --baseline_name "$BASELINE_NAME"
    --thread "$THREAD"
    --gamma "$GAMMA"
    --position_temperature "$POSITION_TEMPERATURE"
    --candidate_number "$CANDIDATE_NUMBER"
    --heuristic "$HEURISTIC"
    --data_path "$DATA_PATH"
    --result_path "$RESULT_PATH"
)

# Optional flags
if [[ -n "$TOKENS_PER_STEP" ]]; then
    CMD+=(--tokens_per_step "$TOKENS_PER_STEP")
fi

if [[ -n "$NO_SHOT" ]]; then
    CMD+=($NO_SHOT)
fi

if [[ -n "$USE_KV_CACHE" ]]; then
    CMD+=($USE_KV_CACHE)
fi

echo ""
echo "Running: ${CMD[*]}"
echo ""

"${CMD[@]}"
