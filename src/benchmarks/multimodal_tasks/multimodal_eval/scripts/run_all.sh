#!/bin/bash
# ============================================================
# 一键运行：生成 + 全部评测
# ============================================================
# 用法: bash scripts/run_all.sh
#
# 流程:
#   1. 生成图像（GenEval 格式）
#   2. GenEval 评测（对象检测 + 颜色 + 位置）
#   3. CLIP Score 评测（语义对齐）
#   4. FID 评测（需要 VIRTUAL_imagenet512.npz）
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="./output_geneval"
RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "MMaDA 文生图 完整评测流程"
echo "============================================================"

# Step 1: 生成图像
echo ""
echo ">>> [Step 1/4] 生成图像..."
bash scripts/run_generate.sh

# Step 2: GenEval 评测
echo ""
echo ">>> [Step 2/4] GenEval 评测..."
bash scripts/run_eval_geneval.sh "$OUTPUT_DIR" "$RESULTS_DIR/geneval_results.jsonl"

# Step 3: CLIP Score 评测
echo ""
echo ">>> [Step 3/4] CLIP Score 评测..."
bash scripts/run_eval_clip.sh "$OUTPUT_DIR" "$RESULTS_DIR/clip_scores.json"

# Step 4: FID 评测
REF_BATCH="./VIRTUAL_imagenet512.npz"
if [ -f "$REF_BATCH" ]; then
    echo ""
    echo ">>> [Step 4/4] FID 评测..."
    bash scripts/run_eval_fid.sh "$REF_BATCH" "$OUTPUT_DIR"
else
    echo ""
    echo ">>> [Step 4/4] 跳过 FID 评测: 未找到参考批次 $REF_BATCH"
    echo "    请将 VIRTUAL_imagenet512.npz 放到 MultiModal_eval/ 目录下"
fi

echo ""
echo "============================================================"
echo "✅ 全部评测完成！结果保存于: $RESULTS_DIR/"
echo "============================================================"
ls -la "$RESULTS_DIR/"

