#!/usr/bin/env bash
# Info-Gain / LookUM evaluation for LLaDA-architecture models (LLaDA, SDAR, TraDo)
#
# Prerequisites:
#   git clone --branch dllm https://github.com/ZHZisZZ/lm-evaluation-harness.git lm-evaluation-harness
#   pip install -e "lm-evaluation-harness[ifeval,math]"
#   pip install -e .
#
# Usage:
#   bash examples/info-gain/llada/eval.sh                                                    # LLaDA Info-Gain
#   bash examples/info-gain/llada/eval.sh --variant lookum                                   # LLaDA LookUM
#   bash examples/info-gain/llada/eval.sh --model_name_or_path JetLM/SDAR-8B-Chat --model_type sdar
#   bash examples/info-gain/llada/eval.sh --model_name_or_path Gen-Verse/TraDo-8B-Instruct --model_type trado

set -e
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

model_name_or_path="GSAI-ML/LLaDA-8B-Instruct"
model_type="llada"  # llada, sdar, trado
num_gpu=1
max_new_tokens=256
steps=256
block_size=32
threshold="0.9"
candidate_number=8
position_temperature="0.1"
variant="info_gain"
use_cache="prefix"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --model_type) model_type="$2"; shift 2 ;;
    --num_gpu) num_gpu="$2"; shift 2 ;;
    --max_new_tokens) max_new_tokens="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --block_size) block_size="$2"; shift 2 ;;
    --threshold) threshold="$2"; shift 2 ;;
    --candidate_number) candidate_number="$2"; shift 2 ;;
    --position_temperature) position_temperature="$2"; shift 2 ;;
    --variant) variant="$2"; shift 2 ;;
    --use_cache) use_cache="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

lm_eval_model="info_gain_${model_type}"
common_args="pretrained=${model_name_or_path},use_cache=${use_cache},threshold=${threshold},candidate_number=${candidate_number},position_temperature=${position_temperature},variant=${variant},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},suppress_tokens=[],begin_suppress_tokens=[]"

echo ">>> model_type=${model_type} variant=${variant} use_cache=${use_cache}"
echo ">>> model=${model_name_or_path}"
echo ">>> lm_eval --model ${lm_eval_model}"
echo

for task_args in \
    "--tasks gsm8k --num_fewshot 5" \
    "--tasks minerva_math --num_fewshot 4" \
    "--tasks humaneval_instruct_llada --num_fewshot 0 --confirm_run_unsafe_code" \
    "--tasks mbpp_instruct_llada --num_fewshot 3 --confirm_run_unsafe_code"; do

    echo "======== ${task_args} ========"
    accelerate launch --num_processes "${num_gpu}" \
        dllm/pipelines/info_gain/llada/eval.py \
        ${task_args} \
        --model "${lm_eval_model}" \
        --apply_chat_template \
        --model_args "${common_args}"
done
