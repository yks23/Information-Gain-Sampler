#!/bin/bash
# ============================================================
# GenEval 评测脚本
# ============================================================
# 用法: bash scripts/run_eval_geneval.sh
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

IMAGE_DIR="${1:-./output_geneval}"
RESULT_FILE="${2:-results/geneval_results.jsonl}"

echo "=========================================="
echo "GenEval 评测"
echo "=========================================="
echo "图像目录: $IMAGE_DIR"
echo "结果文件: $RESULT_FILE"
echo "=========================================="

bash eval_geneval.sh "$IMAGE_DIR" "$RESULT_FILE"

echo ""
echo "查看详细分数..."
python view_scores.py "$RESULT_FILE"

