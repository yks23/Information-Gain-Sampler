#!/bin/bash
#SBATCH --job-name=lookum_dream_eval
#SBATCH --output=./logs/lookum_eval_%j.out
#SBATCH --error=./logs/lookum_eval_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=a01


# 设置工作目录
PROJECT_ROOT="/home/fit/alex/WORK/Kaisen.Yang/opensource/evaluation/Uncode-new"
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"
echo "Project root: $PROJECT_ROOT"

# 创建必要的目录
mkdir -p logs
mkdir -p results
mkdir -p outputs/lookum

# 激活conda环境
source ~/WORK/miniconda3/etc/profile.d/conda.sh
conda activate Uɴᴄᴏᴅᴇ

# 设置模型路径（使用本地Dream模型）
# 可以使用相对路径 "dream" 或 "model/dream"，或者绝对路径
MODEL_NAME="${MODEL_NAME:-dream}"
DEVICE="${DEVICE:-cuda:0}"

# 多GPU设置
NUM_GPUS="${NUM_GPUS:-4}"  # 默认使用4个GPU，可以通过环境变量覆盖

# LookUM参数（与Info-Gain对齐）
VARIANT="lookum"
CANDIDATE_NUMBER=8
POSITION_TEMPERATURE=0.1
THRESHOLD=0.8
TEMPERATURE=0.7

# 测试不同的 tokens_per_step 配置
TOKENS_PER_STEP_VALUES=(1 2)  # 每步解码1个和2个token

# 任务配置
TASKS=("math500" "humaneval" "gsm8k" "mbpp" "sudoku" "countdown")
TASK_CONFIGS=(
    "math500:512:512:16"
    "humaneval:256:256:16"
    "gsm8k:256:256:16"
    "mbpp:256:256:16"
    "sudoku:0:16:16"
    "countdown:48:48:48"
)

# 结果目录（使用dream模型标识，包含时间戳）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_BASE_DIR="results/lookum_dream_eval_${TIMESTAMP}"
mkdir -p "$RESULT_BASE_DIR"

echo "======================================================="
echo "LookUM Evaluation Script (Dream Model) - Multi-GPU"
echo "======================================================="
echo "Model: $MODEL_NAME"
echo "Device: $DEVICE"
echo "Number of GPUs: $NUM_GPUS"
echo "Variant: $VARIANT"
echo "Result Directory: $RESULT_DIR"
echo "======================================================="
echo ""

# 对每个 tokens_per_step 值运行测试
for tokens_per_step in "${TOKENS_PER_STEP_VALUES[@]}"; do
    echo ""
    echo "======================================================="
    echo "Testing with tokens_per_step = $tokens_per_step"
    echo "======================================================="
    
    # 为每个 K 值创建单独的结果目录
    RESULT_DIR="${RESULT_BASE_DIR}/K${tokens_per_step}"
    mkdir -p "$RESULT_DIR"
    
    # 运行每个任务
    for task_config in "${TASK_CONFIGS[@]}"; do
        IFS=':' read -r task gen_length steps block_length <<< "$task_config"
    
    echo ""
    echo "======================================================="
    echo "Running task: $task"
    echo "  gen_length: $gen_length"
    echo "  steps: $steps"
    echo "  block_length: $block_length"
    echo "======================================================="
    
    # 设置任务特定的数据路径（使用绝对路径）
    case "$task" in
        math500)
            DATA_PATH="$PROJECT_ROOT/data/math500.jsonl"
            ;;
        humaneval)
            DATA_PATH="$PROJECT_ROOT/data/humaneval.jsonl"
            ;;
        gsm8k)
            DATA_PATH="$PROJECT_ROOT/data/gsm8k.jsonl"
            ;;
        mbpp)
            DATA_PATH="$PROJECT_ROOT/data/sanitized-mbpp.json"
            ;;
        sudoku)
            DATA_PATH="$PROJECT_ROOT/data/sudoku.csv"
            ;;
        countdown)
            DATA_PATH="$PROJECT_ROOT/data/countdown.jsonl"
            ;;
        *)
            echo "Unknown task: $task"
            continue
            ;;
    esac
    
    # 检查数据文件是否存在
    if [ ! -f "$DATA_PATH" ]; then
        echo "Warning: Data file not found: $DATA_PATH"
        echo "Current directory: $(pwd)"
        echo "Available data files:"
        ls -la "$PROJECT_ROOT/data/"*.jsonl 2>/dev/null | head -5 || echo "Cannot list data directory"
        echo "Skipping task: $task"
        continue
    fi
    
    echo "Using data file: $DATA_PATH"
    
    # 生成结果文件路径（包含 K 值）
    RESULT_FILE="$RESULT_DIR/${task}_lookum_dream_T${TEMPERATURE}_K${tokens_per_step}.txt"
    OUTPUT_FILE="outputs/lookum/${task}_lookum_dream_T${TEMPERATURE}_K${tokens_per_step}_outputs.jsonl"
    LOG_FILE="$RESULT_DIR/${task}_lookum_dream_K${tokens_per_step}.log"
    READABLE_DIR="$RESULT_DIR/${task}_readable"
    
    # 确保输出目录存在
    mkdir -p "outputs/lookum"
    mkdir -p "$READABLE_DIR"
    
    echo "  Result file: $RESULT_FILE"
    echo "  Readable results directory: $READABLE_DIR"
    echo "  Log file: $LOG_FILE"
    echo "  Number of GPUs: $NUM_GPUS"
    echo "  Note: Results are saved in real-time to worker_results/ and readable files"
    echo ""
    
    # 根据任务类型决定是否使用few-shot
    # 数学和代码任务不使用few-shot，其他任务可以使用
    NO_SHOT_FLAG=""
    case "$task" in
        math500|gsm8k|humaneval|mbpp)
            NO_SHOT_FLAG="--no_shot"
            echo "  Using zero-shot for $task (no few-shot examples)"
            ;;
        *)
            echo "  Using default shot settings for $task"
            ;;
    esac
    
    # 运行多GPU评估
    python scripts/eval_multigpu.py \
        --task "$task" \
        --model_name "$MODEL_NAME" \
        --num_gpus "$NUM_GPUS" \
        --mode info-gain \
        --variant "$VARIANT" \
        --gen_length "$gen_length" \
        --steps "$steps" \
        --block_length "$block_length" \
        --temperature "$TEMPERATURE" \
        --candidate_number "$CANDIDATE_NUMBER" \
        --position_temperature "$POSITION_TEMPERATURE" \
        --threshold "$THRESHOLD" \
        --tokens_per_step "$tokens_per_step" \
        --data_path "$DATA_PATH" \
        --result_path "$RESULT_FILE" \
        $NO_SHOT_FLAG \
        2>&1 | tee "$LOG_FILE"
    
    # 检查运行状态
    if [ $? -eq 0 ]; then
        echo "Task $task completed successfully."
        echo "Results saved to: $RESULT_FILE"
    else
        echo "Error: Task $task failed. Check log: $LOG_FILE"
    fi
    
        echo ""
    done  # 结束任务循环
    
    echo ""
    echo "======================================================="
    echo "Completed testing with tokens_per_step = $tokens_per_step"
    echo "Results directory: $RESULT_DIR"
    echo "======================================================="
done  # 结束 tokens_per_step 循环

echo ""
echo "======================================================="
echo "All tests completed!"
echo "Base results directory: $RESULT_BASE_DIR"
echo "Outputs directory: outputs/lookum"
echo "Logs directory: logs"
echo "======================================================="

# 汇总结果
echo ""
echo "Summary of results:"
for tokens_per_step in "${TOKENS_PER_STEP_VALUES[@]}"; do
    RESULT_DIR="${RESULT_BASE_DIR}/K${tokens_per_step}"
    echo ""
    echo "--- tokens_per_step = $tokens_per_step ---"
    for task_config in "${TASK_CONFIGS[@]}"; do
        IFS=':' read -r task gen_length steps block_length <<< "$task_config"
        RESULT_FILE="$RESULT_DIR/${task}_lookum_dream_T${TEMPERATURE}_K${tokens_per_step}.txt"
        if [ -f "$RESULT_FILE" ]; then
            echo "  $task: $RESULT_FILE"
            if grep -q "accuracy\|pass@1\|exact_match" "$RESULT_FILE" 2>/dev/null; then
                echo "    Metrics:"
                grep -E "accuracy|pass@1|exact_match|score" "$RESULT_FILE" | head -5
            fi
        else
            echo "  $task: Not found"
        fi
    done
done

