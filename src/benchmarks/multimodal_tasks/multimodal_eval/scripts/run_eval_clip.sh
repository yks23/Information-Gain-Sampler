#!/bin/bash
# ============================================================
# CLIP Score 评测脚本
# ============================================================
# 用法: bash scripts/run_eval_clip.sh [IMAGE_DIR]
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

IMAGE_DIR="${1:-./output_geneval}"
OUTPUT_FILE="${2:-results/clip_scores.json}"

echo "=========================================="
echo "CLIP Score 评测"
echo "=========================================="
echo "图像目录: $IMAGE_DIR"
echo "输出文件: $OUTPUT_FILE"
echo "=========================================="

python eval_clip_score.py \
    --image-dir "$IMAGE_DIR" \
    --output "$OUTPUT_FILE" \
    --clip-arch ViT-L-14 \
    --clip-pretrained openai \
    --batch-size 32

