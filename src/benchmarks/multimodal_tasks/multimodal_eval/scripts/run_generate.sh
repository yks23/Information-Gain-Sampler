#!/bin/bash
# ============================================================
# MMaDA 文生图 - 生成脚本 (Info-Gain 版)
# ============================================================
# 用法: bash scripts/run_generate.sh
# 请先根据实际环境修改下面的路径和参数
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# =================== 模型路径 ===================
# 优先使用环境变量，否则使用默认相对路径
MMADA_MODEL_PATH="${MMADA_MODEL_PATH:-../mmada-mix}"          # MMaDA 模型路径
VQ_MODEL_PATH="${VQ_MODEL_PATH:-../magvitv2}"              # VQ 模型路径
CONFIG_FILE="configs/mmada_t2i.yaml"     # 配置文件

# 如果是相对路径，转换为绝对路径（相对于脚本所在目录的上级目录）
if [[ ! "$MMADA_MODEL_PATH" =~ ^/ ]]; then
    # 相对路径：相对于 SCRIPT_DIR 的上级目录
    PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    MMADA_MODEL_PATH="$PARENT_DIR/${MMADA_MODEL_PATH#../}"
    # 如果路径不存在，尝试从项目根目录查找
    if [[ ! -d "$MMADA_MODEL_PATH" ]]; then
        PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
        ALT_PATH="$PROJECT_ROOT/model/${MMADA_MODEL_PATH##*/}"
        if [[ -d "$ALT_PATH" ]]; then
            MMADA_MODEL_PATH="$ALT_PATH"
        fi
    fi
fi
if [[ ! "$VQ_MODEL_PATH" =~ ^/ ]]; then
    # 相对路径：相对于 SCRIPT_DIR 的上级目录
    PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    VQ_MODEL_PATH="$PARENT_DIR/${VQ_MODEL_PATH#../}"
    # 如果路径不存在，尝试从项目根目录查找
    if [[ ! -d "$VQ_MODEL_PATH" ]]; then
        PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
        ALT_PATH="$PROJECT_ROOT/model/${VQ_MODEL_PATH##*/}"
        if [[ -d "$ALT_PATH" ]]; then
            VQ_MODEL_PATH="$ALT_PATH"
        fi
    fi
fi

# 验证路径是否存在
if [[ ! -d "$MMADA_MODEL_PATH" ]]; then
    echo "错误: MMaDA 模型路径不存在: $MMADA_MODEL_PATH"
    exit 1
fi
if [[ ! -d "$VQ_MODEL_PATH" ]]; then
    echo "错误: VQ 模型路径不存在: $VQ_MODEL_PATH"
    exit 1
fi

# =================== 输入 ===================
PROMPTS_FILE="prompts/generation_prompts.txt"
METADATA_FILE="prompts/evaluation_metadata.jsonl"

# =================== 输出 ===================
OUTPUT_DIR="./output_geneval"

# =================== 生成参数 ===================
BATCH_SIZE=1
SAMPLES_PER_PROMPT=4
GUIDANCE_SCALE=3.5
GENERATION_TIMESTEPS=50
GENERATION_TEMPERATURE=1.0

# =================== Info-Gain 参数 ===================
CANDIDATE_NUMBER=4    # 候选动作数量（越大越精确但越慢）

echo "=========================================="
echo "MMaDA 文生图 - Info-Gain 生成"
echo "=========================================="
echo "模型: $MMADA_MODEL_PATH"
echo "输出: $OUTPUT_DIR"
echo "提示: $PROMPTS_FILE"
echo "参数: scale=$GUIDANCE_SCALE, steps=$GENERATION_TIMESTEPS, temp=$GENERATION_TEMPERATURE"
echo "Info-Gain 候选数: $CANDIDATE_NUMBER"
echo "=========================================="

python t2i_generate.py \
    config="$CONFIG_FILE" \
    mmada_model_path="$MMADA_MODEL_PATH" \
    vq_model_path="$VQ_MODEL_PATH" \
    validation_prompts_file="$PROMPTS_FILE" \
    output_dir="$OUTPUT_DIR" \
    batch_size=$BATCH_SIZE \
    text_to_image.samples_per_prompt=$SAMPLES_PER_PROMPT \
    text_to_image.guidance_scale=$GUIDANCE_SCALE \
    text_to_image.generation_timesteps=$GENERATION_TIMESTEPS \
    text_to_image.generation_temperature=$GENERATION_TEMPERATURE \
    candidate_number=$CANDIDATE_NUMBER \
    geneval_metadata_file="$METADATA_FILE" \
    use_geneval_format=True

echo ""
echo "✅ 生成完成！输出目录: $OUTPUT_DIR"
