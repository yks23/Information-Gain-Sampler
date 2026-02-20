#!/bin/bash
# ============================================================
# GenEval 评测流程脚本
#
# GenEval 通过目标检测 (Mask2Former) + CLIP 颜色分类来判断
# 生成图像是否满足 prompt 中描述的对象、数量、颜色、位置关系等。
#
# 前置条件：
#   1. 已安装 mmdet (mmdetection)、open_clip、clip_benchmark
#   2. 已克隆 mmdetection 仓库到项目根目录
#   3. 已下载 Mask2Former 模型权重
#
# 用法：
#   bash eval_geneval.sh <IMAGE_DIR> [RESULT_FILE]
#
# 示例：
#   bash eval_geneval.sh ./output_geneval ./results/geneval_results.jsonl
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 参数
IMAGE_DIR="${1:?用法: bash eval_geneval.sh <IMAGE_DIR> [RESULT_FILE]}"
RESULT_FILE="${2:-results/geneval_results.jsonl}"

# Mask2Former 模型路径（可按需修改）
MODEL_PATH="${MASK2FORMER_MODEL_PATH:-${PROJECT_ROOT}/models/mask2former}"

echo "=========================================="
echo "GenEval 评测"
echo "=========================================="
echo "图像目录:  $IMAGE_DIR"
echo "结果文件:  $RESULT_FILE"
echo "模型路径:  $MODEL_PATH"
echo "=========================================="

# 检查图像目录
if [ ! -d "$IMAGE_DIR" ]; then
    echo "错误: 图像目录不存在: $IMAGE_DIR"
    exit 1
fi

# 检查 Mask2Former 模型
CKPT_FILE="$MODEL_PATH/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
if [ ! -f "$CKPT_FILE" ]; then
    echo ""
    echo "未找到 Mask2Former 模型权重，正在下载..."
    mkdir -p "$MODEL_PATH"
    if [ -f "$PROJECT_ROOT/geneval/evaluation/download_models.sh" ]; then
        cd "$PROJECT_ROOT/geneval/evaluation"
        bash download_models.sh "$MODEL_PATH/"
        cd "$SCRIPT_DIR"
    else
        echo "错误: 找不到下载脚本。请手动下载 Mask2Former 模型到 $MODEL_PATH"
        exit 1
    fi
fi

# 创建结果目录
mkdir -p "$(dirname "$RESULT_FILE")"

# 步骤 1: 评测图像
echo ""
echo "[Step 1/2] 评测图像..."
cd "$PROJECT_ROOT/geneval"
python evaluation/evaluate_images.py \
    "$IMAGE_DIR" \
    --outfile "$SCRIPT_DIR/$RESULT_FILE" \
    --model-path "$MODEL_PATH/"
cd "$SCRIPT_DIR"

# 步骤 2: 汇总分数
echo ""
echo "[Step 2/2] 汇总分数..."
cd "$PROJECT_ROOT/geneval"
python evaluation/summary_scores.py "$SCRIPT_DIR/$RESULT_FILE"
cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "✅ GenEval 评测完成！"
echo "结果文件: $RESULT_FILE"
echo "=========================================="

