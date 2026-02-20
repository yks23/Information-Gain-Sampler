#!/bin/bash
# ============================================================
# FID / IS / Precision / Recall 评测脚本
# ============================================================
# 用法: bash scripts/run_eval_fid.sh <REF_BATCH> <SAMPLE_DIR> [BATCH_SIZE] [--resize N]
#
# 示例:
#   bash scripts/run_eval_fid.sh ./VIRTUAL_imagenet512.npz ./output_geneval
#   bash scripts/run_eval_fid.sh ./VIRTUAL_imagenet512.npz ./output_geneval 64 --resize 256
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

REF_BATCH="${1:?用法: bash scripts/run_eval_fid.sh <REF_BATCH> <SAMPLE_DIR> [BATCH_SIZE] [--resize N]}"
SAMPLE_DIR="${2:?用法: bash scripts/run_eval_fid.sh <REF_BATCH> <SAMPLE_DIR> [BATCH_SIZE] [--resize N]}"
BATCH_SIZE="${3:-64}"
shift 3 2>/dev/null || shift $# 2>/dev/null || true

# 收集额外参数 (如 --resize 256)
EXTRA_ARGS="$*"

echo "=========================================="
echo "FID / IS / Precision / Recall 评测"
echo "=========================================="
echo "参考批次: $REF_BATCH"
echo "样本目录: $SAMPLE_DIR"
echo "批次大小: $BATCH_SIZE"
if [ -n "$EXTRA_ARGS" ]; then
    echo "额外参数: $EXTRA_ARGS"
fi
echo "=========================================="

python eval_fid.py "$REF_BATCH" "$SAMPLE_DIR" --batch-size "$BATCH_SIZE" $EXTRA_ARGS
