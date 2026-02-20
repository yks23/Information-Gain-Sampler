#!/bin/bash
################################################################################
# Paper Experiments Reproduction Script
# 
# This script reproduces all experiments from the paper:
# - Fully-Attention MDM Reasoning (Dream-7B-Instruct)
# - Semi-Autoregressive MDM Reasoning (SDAR-8B-Chat, TraDo-8B-Instruct)
# - Multimodal Text-to-Image Generation (MMaDa)
# - Creative Writing (SDAR-8B-Chat)
#
# Usage:
#   bash scripts/reproduce_paper.sh [--device DEVICE] [--skip SKIP_EXP]
#
# Examples:
#   # Run all experiments
#   bash scripts/reproduce_paper.sh
#
#   # Run on specific device
#   bash scripts/reproduce_paper.sh --device cuda:1
#
#   # Skip specific experiments (comma-separated: exp1,exp2,exp3,exp4)
#   bash scripts/reproduce_paper.sh --skip exp2,exp4
################################################################################

set -euo pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
DEVICE="cuda:0"
SKIP_EXP=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip)
            SKIP_EXP="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/reproduce_paper.sh [--device DEVICE] [--skip SKIP_EXP]"
            exit 1
            ;;
    esac
done

# Check if experiment should be skipped
should_skip() {
    local exp="$1"
    if [[ -n "$SKIP_EXP" ]] && [[ "$SKIP_EXP" == *"$exp"* ]]; then
        return 0
    fi
    return 1
}

# Results directory
RESULTS_DIR="$PROJECT_ROOT/results/paper_reproduction"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Paper Experiments Reproduction"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Device: $DEVICE"
echo "Results Dir: $RESULTS_DIR"
echo ""

# ============================================================================
# Experiment 1: Fully-Attention MDM Reasoning (Dream-7B-Instruct)
# ============================================================================
if ! should_skip "exp1"; then
    echo "=========================================="
    echo "Experiment 1: Fully-Attention MDM Reasoning"
    echo "Model: Dream-7B-Instruct"
    echo "=========================================="
    echo ""
    
    MODEL="dream"
    POSITION_TEMP=0.1
    CANDIDATE_NUM=8
    HEURISTIC="confidence"
    NUM_RUNS=5
    
    # Tasks: Math (GSM8K, MATH-500), Code (HumanEval, MBPP), Planning (Sudoku, Countdown)
    TASKS=("gsm8k" "math500" "humaneval" "mbpp" "sudoku" "countdown")
    
    for task in "${TASKS[@]}"; do
        for k in 1 2; do
            echo "  Running: $task (K=$k)"
            TASK_RESULTS_DIR="$RESULTS_DIR/exp1_${task}_K${k}"
            mkdir -p "$TASK_RESULTS_DIR"
            
            for run in $(seq 1 $NUM_RUNS); do
                echo "    Run $run/$NUM_RUNS..."
                RESULT_FILE="$TASK_RESULTS_DIR/run_${run}.txt"
                
                bash scripts/Eval.sh \
                    --task "$task" \
                    --model "$MODEL" \
                    --device "$DEVICE" \
                    --mode info-gain \
                    --position_temperature "$POSITION_TEMP" \
                    --candidate_number "$CANDIDATE_NUM" \
                    --heuristic "$HEURISTIC" \
                    --tokens_per_step "$k" \
                    --temperature 0.0 \
                    --result_path "$RESULT_FILE" \
                    2>&1 | tee "$TASK_RESULTS_DIR/run_${run}.log" | tail -3 || {
                    echo -e "${RED}    ❌ Run $run failed${NC}"
                }
            done
            
            echo -e "${GREEN}  ✅ $task (K=$k) completed${NC}"
        done
    done
    
    echo ""
    echo -e "${GREEN}Experiment 1 completed!${NC}"
    echo ""
fi

# ============================================================================
# Experiment 2: Semi-Autoregressive MDM Reasoning
# ============================================================================
if ! should_skip "exp2"; then
    echo "=========================================="
    echo "Experiment 2: Semi-Autoregressive MDM Reasoning"
    echo "Models: SDAR-8B-Chat, TraDo-8B-Instruct"
    echo "=========================================="
    echo ""
    
    MODELS=("sdar" "trado")
    TOKEN_TEMP=0.7
    BLOCK_LENGTH=16
    NUM_RUNS=5
    
    # Tasks: Math (GSM8K, MATH-500), Code (HumanEval, MBPP)
    TASKS=("gsm8k" "math500" "humaneval" "mbpp")
    
    for model in "${MODELS[@]}"; do
        echo "  Model: $model"
        for task in "${TASKS[@]}"; do
            for k in 1 2; do
                echo "    Running: $task (K=$k)"
                TASK_RESULTS_DIR="$RESULTS_DIR/exp2_${model}_${task}_K${k}"
                mkdir -p "$TASK_RESULTS_DIR"
                
                for run in $(seq 1 $NUM_RUNS); do
                    echo "      Run $run/$NUM_RUNS..."
                    RESULT_FILE="$TASK_RESULTS_DIR/run_${run}.txt"
                    
                    bash scripts/Eval.sh \
                        --task "$task" \
                        --model "$model" \
                        --device "$DEVICE" \
                        --mode info-gain \
                        --position_temperature 0.1 \
                        --candidate_number 8 \
                        --heuristic confidence \
                        --tokens_per_step "$k" \
                        --temperature "$TOKEN_TEMP" \
                        --block_length "$BLOCK_LENGTH" \
                        --use_kv_cache \
                        --result_path "$RESULT_FILE" \
                        2>&1 | tee "$TASK_RESULTS_DIR/run_${run}.log" | tail -3 || {
                        echo -e "${RED}      ❌ Run $run failed${NC}"
                    }
                done
                
                echo -e "${GREEN}    ✅ $task (K=$k) completed${NC}"
            done
        done
    done
    
    echo ""
    echo -e "${GREEN}Experiment 2 completed!${NC}"
    echo ""
fi

# ============================================================================
# Experiment 3: Multimodal Text-to-Image Generation (MMaDa)
# ============================================================================
if ! should_skip "exp3"; then
    echo "=========================================="
    echo "Experiment 3: Multimodal Text-to-Image Generation"
    echo "Model: MMaDa"
    echo "=========================================="
    echo ""
    
    MULTIMODAL_DIR="$PROJECT_ROOT/src/benchmarks/multimodal_tasks/multimodal_eval"
    EXP3_RESULTS_DIR="$RESULTS_DIR/exp3_multimodal"
    mkdir -p "$EXP3_RESULTS_DIR"
    
    if [ -d "$MULTIMODAL_DIR" ] && [ -f "$MULTIMODAL_DIR/t2i_generate.py" ]; then
        cd "$MULTIMODAL_DIR"
        
        # GenEval evaluation
        echo "  Running GenEval evaluation..."
        python eval_multimodal.py \
            --pipeline all \
            2>&1 | tee "$EXP3_RESULTS_DIR/geneval.log" || {
            echo -e "${RED}  ❌ GenEval evaluation failed${NC}"
        }
        
        # ImageNet-512 FID evaluation
        echo "  Running ImageNet-512 FID evaluation..."
        if [ -f "scripts/run_imagenet_fid.sh" ]; then
            bash scripts/run_imagenet_fid.sh \
                2>&1 | tee "$EXP3_RESULTS_DIR/imagenet_fid.log" || {
                echo -e "${RED}  ❌ ImageNet-512 FID evaluation failed${NC}"
            }
        else
            echo -e "${YELLOW}  ⚠️  ImageNet-512 script not found, skipping${NC}"
        fi
        
        cd "$PROJECT_ROOT"
        echo -e "${GREEN}  ✅ Multimodal evaluation completed${NC}"
    else
        echo -e "${YELLOW}  ⚠️  Multimodal evaluation directory not found, skipping${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}Experiment 3 completed!${NC}"
    echo ""
fi

# ============================================================================
# Experiment 4: Creative Writing (SDAR-8B-Chat)
# ============================================================================
if ! should_skip "exp4"; then
    echo "=========================================="
    echo "Experiment 4: Creative Writing"
    echo "Model: SDAR-8B-Chat"
    echo "=========================================="
    echo ""
    
    MODEL="sdar"
    TEMPERATURES=(0.5 1.0 1.5)
    NUM_RUNS=5
    
    for temp in "${TEMPERATURES[@]}"; do
        for k in 1 2; do
            echo "  Running: Temperature=$temp, K=$k"
            TASK_RESULTS_DIR="$RESULTS_DIR/exp4_creative_writing_T${temp}_K${k}"
            mkdir -p "$TASK_RESULTS_DIR"
            
            for run in $(seq 1 $NUM_RUNS); do
                echo "    Run $run/$NUM_RUNS..."
                RESULT_FILE="$TASK_RESULTS_DIR/run_${run}.txt"
                
                bash scripts/Eval.sh \
                    --task creativity_writing \
                    --model "$MODEL" \
                    --device "$DEVICE" \
                    --mode info-gain \
                    --position_temperature 0.1 \
                    --candidate_number 8 \
                    --heuristic confidence \
                    --tokens_per_step "$k" \
                    --temperature "$temp" \
                    --result_path "$RESULT_FILE" \
                    2>&1 | tee "$TASK_RESULTS_DIR/run_${run}.log" | tail -3 || {
                    echo -e "${RED}    ❌ Run $run failed${NC}"
                }
            done
            
            echo -e "${GREEN}  ✅ Temperature=$temp, K=$k completed${NC}"
        done
    done
    
    echo ""
    echo -e "${GREEN}Experiment 4 completed!${NC}"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Experiment directories:"
echo "  - exp1_*: Fully-Attention MDM Reasoning (Dream-7B-Instruct)"
echo "  - exp2_*: Semi-Autoregressive MDM Reasoning (SDAR, TraDo)"
echo "  - exp3_*: Multimodal Text-to-Image Generation (MMaDa)"
echo "  - exp4_*: Creative Writing (SDAR-8B-Chat)"
echo ""
echo "To view results, check individual result files in each experiment directory."
echo ""

