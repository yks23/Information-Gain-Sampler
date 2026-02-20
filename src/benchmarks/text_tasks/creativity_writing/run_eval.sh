#!/bin/bash
################################################################################
# Creativity Writing evaluation script
#
# This script wraps the main eval pipeline for the creativity_writing task
# and optionally runs LLM-as-judge evaluation afterwards.
#
# Usage:
#   # Step 1: Generate outputs
#   bash run_eval.sh --model /path/to/model --mode info-gain
#
#   # Step 2: Run judge (separate step, requires OPENAI_API_KEY)
#   bash run_eval.sh --judge --model_outputs outputs/model_a.json \
#       --reference_outputs outputs/model_b.json
#
#   # Full pipeline: generate + single-score judge
#   bash run_eval.sh --model /path/to/model --mode info-gain --run_judge --judge_mode single
################################################################################

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ========================= Defaults ==========================
MODEL=""
MODE="info-gain"
DEVICE="cuda"
TEMPERATURE=0.7
GEN_LENGTH=512
STEPS=512
BLOCK_LENGTH=32
TOKENS_PER_STEP=1
HEURISTIC="confidence"
CANDIDATE_NUMBER=1
POSITION_TEMPERATURE=0

# Judge options
JUDGE_ONLY=false
RUN_JUDGE=false
JUDGE_MODEL="gpt-4o"
JUDGE_MODE="pairwise"
MODEL_OUTPUTS=""
REFERENCE_OUTPUTS=""

# ========================= CLI Parsing ==========================
usage() {
    cat <<EOF
Usage: bash run_eval.sh [OPTIONS]

Generation options:
  --model MODEL           Model name or path (required for generation)
  --mode MODE             Generation mode (default: info-gain)
  --device DEV            CUDA device (default: cuda)
  --temperature T         Sampling temperature (default: 0.7)
  --gen_length N          Generated answer length (default: 512)
  --steps N               Number of sampling steps (default: 512)
  --block_length N        Block length (default: 32)
  --tokens_per_step N     Tokens per step (default: 1)
  --heuristic H           Heuristic function (default: confidence)
  --candidate_number N    Candidate actions (default: 1)
  --position_temperature T  Position temperature (default: 0)

Judge options:
  --judge                 Run judge only (skip generation)
  --run_judge             Run judge after generation
  --judge_model MODEL     Judge model name (default: gpt-4o)
  --judge_mode MODE       Judge mode: pairwise|single (default: pairwise)
  --model_outputs PATH    Path to model outputs JSON
  --reference_outputs PATH  Path to reference outputs JSON (pairwise mode)

EOF
    exit 1
}

EXTRA_EVAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                 MODEL="$2";                 shift 2 ;;
        --mode)                  MODE="$2";                  shift 2 ;;
        --device)                DEVICE="$2";                shift 2 ;;
        --temperature)           TEMPERATURE="$2";           shift 2 ;;
        --gen_length)            GEN_LENGTH="$2";            shift 2 ;;
        --steps)                 STEPS="$2";                 shift 2 ;;
        --block_length)          BLOCK_LENGTH="$2";          shift 2 ;;
        --tokens_per_step)       TOKENS_PER_STEP="$2";       shift 2 ;;
        --heuristic)             HEURISTIC="$2";             shift 2 ;;
        --candidate_number)      CANDIDATE_NUMBER="$2";      shift 2 ;;
        --position_temperature)  POSITION_TEMPERATURE="$2";  shift 2 ;;
        --judge)                 JUDGE_ONLY=true;             shift   ;;
        --run_judge)             RUN_JUDGE=true;              shift   ;;
        --judge_model)           JUDGE_MODEL="$2";           shift 2 ;;
        --judge_mode)            JUDGE_MODE="$2";            shift 2 ;;
        --model_outputs)         MODEL_OUTPUTS="$2";         shift 2 ;;
        --reference_outputs)     REFERENCE_OUTPUTS="$2";     shift 2 ;;
        --no_shot)               EXTRA_EVAL_ARGS+=(--no_shot); shift ;;
        --use_kv_cache)          EXTRA_EVAL_ARGS+=(--use_kv_cache); shift ;;
        -h|--help)               usage ;;
        *)                       echo "Unknown option: $1"; usage ;;
    esac
done

# ========================= Generation Phase ==========================
if [[ "$JUDGE_ONLY" == false ]]; then
    if [[ -z "$MODEL" ]]; then
        echo "Error: --model is required for generation."
        usage
    fi

    model_short=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')
    OUTPUT_FILE="${SCRIPT_DIR}/outputs/${model_short}_${MODE}_T${TEMPERATURE}_K${TOKENS_PER_STEP}.json"
    MODEL_OUTPUTS="$OUTPUT_FILE"

    echo "======================================================="
    echo " Task           : creativity_writing"
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
    echo " Output         : $OUTPUT_FILE"
    echo "======================================================="

    cd "$PROJECT_ROOT/scripts"
    python eval.py \
        --task creativity_writing \
        --model_name "$MODEL" \
        --device "$DEVICE" \
        --gen_length "$GEN_LENGTH" \
        --steps "$STEPS" \
        --block_length "$BLOCK_LENGTH" \
        --temperature "$TEMPERATURE" \
        --mode "$MODE" \
        --heuristic "$HEURISTIC" \
        --candidate_number "$CANDIDATE_NUMBER" \
        --position_temperature "$POSITION_TEMPERATURE" \
        --tokens_per_step "$TOKENS_PER_STEP" \
        --data_path "$SCRIPT_DIR/data/creativity_writing.jsonl" \
        --result_path "$OUTPUT_FILE" \
        "${EXTRA_EVAL_ARGS[@]}"

    echo ""
    echo "Generation complete. Outputs saved to: $OUTPUT_FILE"
fi

# ========================= Judge Phase ==========================
if [[ "$JUDGE_ONLY" == true ]] || [[ "$RUN_JUDGE" == true ]]; then
    if [[ -z "$MODEL_OUTPUTS" ]]; then
        echo "Error: --model_outputs is required for judge evaluation."
        exit 1
    fi

    echo ""
    echo "======================================================="
    echo " Running LLM-as-Judge Evaluation"
    echo " Judge model    : $JUDGE_MODEL"
    echo " Judge mode     : $JUDGE_MODE"
    echo " Model outputs  : $MODEL_OUTPUTS"
    echo " Reference      : ${REFERENCE_OUTPUTS:-N/A}"
    echo "======================================================="

    JUDGE_ARGS=(
        python "$SCRIPT_DIR/judge.py"
        --model_outputs "$MODEL_OUTPUTS"
        --mode "$JUDGE_MODE"
        --judge_model "$JUDGE_MODEL"
    )

    if [[ -n "$REFERENCE_OUTPUTS" ]]; then
        JUDGE_ARGS+=(--reference_outputs "$REFERENCE_OUTPUTS")
    fi

    "${JUDGE_ARGS[@]}"
fi

echo ""
echo "Done."

