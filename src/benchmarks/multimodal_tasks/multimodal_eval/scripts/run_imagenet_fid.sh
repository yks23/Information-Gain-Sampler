#!/bin/bash
# ============================================================
# ImageNet FID 端到端评测脚本
#
# 流程:
#   1. 使用 ImageNet 1000 类标签提示生成图像 (50 张/类 = 50K 张)
#   2. 计算 FID / sFID / IS / Precision / Recall
#
# 用法:
#   # 完整流程（生成 + 评测）
#   bash scripts/run_imagenet_fid.sh
#
#   # 仅评测（已有生成图像）
#   bash scripts/run_imagenet_fid.sh --eval-only
#
#   # 小规模测试（10 类 × 2 张 = 20 张，快速验证流程）
#   bash scripts/run_imagenet_fid.sh --test
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# =================== 默认参数 ===================
# 模型路径
MMADA_MODEL_PATH="${MMADA_MODEL_PATH:-../mmada-mix}"
VQ_MODEL_PATH="${VQ_MODEL_PATH:-../magvitv2}"
CONFIG_FILE="configs/mmada_t2i.yaml"

# 输入
PROMPTS_FILE="prompts/labels_prompts.txt"
METADATA_FILE="prompts/labels_metadata.jsonl"

# 输出
OUTPUT_DIR="${OUTPUT_DIR:-./output_imagenet_fid}"

# 参考批次 (npz)
REF_BATCH="${REF_BATCH:-./VIRTUAL_imagenet512.npz}"

# 生成参数
BATCH_SIZE="${BATCH_SIZE:-1}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-50}"   # 每类 50 张，共 50K
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
GENERATION_TIMESTEPS="${GENERATION_TIMESTEPS:-50}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-1.0}"
CANDIDATE_NUMBER="${CANDIDATE_NUMBER:-4}"

# FID 评测参数
FID_BATCH_SIZE="${FID_BATCH_SIZE:-64}"
RESIZE="${RESIZE:-}"   # 为空则不 resize；设为 256 表示 resize 到 256x256

# 模式
EVAL_ONLY=false
TEST_MODE=false

# =================== 解析参数 ===================
for arg in "$@"; do
    case "$arg" in
        --eval-only) EVAL_ONLY=true ;;
        --test)      TEST_MODE=true ;;
        *)           echo "未知参数: $arg"; exit 1 ;;
    esac
done

# 测试模式：缩小规模
if $TEST_MODE; then
    SAMPLES_PER_PROMPT=2
    GENERATION_TIMESTEPS=10
    # 创建临时提示文件（取前 10 类）
    TEST_PROMPTS="/tmp/test_labels_10.txt"
    TEST_METADATA="/tmp/test_labels_meta_10.jsonl"
    head -10 "$PROMPTS_FILE" > "$TEST_PROMPTS"
    head -10 "$METADATA_FILE" > "$TEST_METADATA"
    PROMPTS_FILE="$TEST_PROMPTS"
    METADATA_FILE="$TEST_METADATA"
    OUTPUT_DIR="./output_imagenet_fid_test"
    echo "⚡ 测试模式: 10 类 × 2 张 = 20 张, steps=$GENERATION_TIMESTEPS"
fi

echo "=========================================="
echo "ImageNet FID 端到端评测"
echo "=========================================="
echo "模型:       $MMADA_MODEL_PATH"
echo "VQ:         $VQ_MODEL_PATH"
echo "输出:       $OUTPUT_DIR"
echo "参考批次:   $REF_BATCH"
echo "生成参数:   scale=$GUIDANCE_SCALE, steps=$GENERATION_TIMESTEPS, temp=$GENERATION_TEMPERATURE"
echo "每类图像数: $SAMPLES_PER_PROMPT"
echo "Info-Gain:  candidates=$CANDIDATE_NUMBER"
if [ -n "$RESIZE" ]; then
    echo "Resize:     ${RESIZE}x${RESIZE}"
fi
echo "=========================================="

# =================== Step 1: 生成图像 ===================
if ! $EVAL_ONLY; then
    echo ""
    echo ">>> [Step 1/2] 生成 ImageNet 类条件图像..."
    echo ""

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
    echo "✅ 图像生成完成: $OUTPUT_DIR"
else
    echo ""
    echo ">>> 跳过生成，使用已有图像: $OUTPUT_DIR"
fi

# =================== Step 2: FID 评测 ===================
echo ""
echo ">>> [Step 2/2] 计算 FID / sFID / IS / Precision / Recall..."
echo ""

if [ ! -f "$REF_BATCH" ]; then
    echo "错误: 参考批次文件不存在: $REF_BATCH"
    echo "请确保 VIRTUAL_imagenet512.npz 在 MultiModal_eval/ 目录下"
    exit 1
fi

RESIZE_FLAG=""
if [ -n "$RESIZE" ]; then
    RESIZE_FLAG="--resize $RESIZE"
fi

python eval_fid.py "$REF_BATCH" "$OUTPUT_DIR" \
    --batch-size $FID_BATCH_SIZE \
    $RESIZE_FLAG

# 清理测试临时文件
if $TEST_MODE; then
    rm -f "$TEST_PROMPTS" "$TEST_METADATA" 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "✅ ImageNet FID 评测完成！"
echo "=========================================="

