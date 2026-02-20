# MMaDA 文生图 & 评测工具

本目录整合了 MMaDA 的**图像生成**和**评测**完整流程，包括：

| 功能 | 脚本 | 说明 |
|------|------|------|
| 文生图生成 | `t2i_generate.py` | 从文本提示生成图像，支持多GPU、IGP采样、GenEval格式输出 |
| GenEval 评测 | `eval_geneval.sh` | 基于 Mask2Former 目标检测 + CLIP 颜色分类，评测对象/数量/颜色/位置 |
| FID/IS 评测 | `eval_fid.py` | 计算 FID、sFID、Inception Score、Precision、Recall（支持 `--resize`） |
| CLIP Score 评测 | `eval_clip_score.py` | 计算图像与文本的语义对齐分数 |
| ImageNet FID 端到端 | `scripts/run_imagenet_fid.sh` | 一键生成 50K 图 + FID 评测（支持 `--test` 快速验证） |
| 格式转换 | `convert_format.py` | 将普通图像目录转换为 GenEval 目录结构 |
| 结果查看 | `view_scores.py` | 查看 GenEval 评测结果的详细分数 |

---

## 目录结构

```
MultiModel_eval/
├── README.md                     # 本文件
├── t2i_generate.py               # 文生图脚本
├── eval_fid.py                   # FID / IS / Precision / Recall 评测
├── eval_clip_score.py            # CLIP Score 评测
├── eval_geneval.sh               # GenEval 评测（shell 脚本）
├── convert_format.py             # 普通图像 -> GenEval 格式
├── view_scores.py                # 查看 GenEval 结果
├── configs/
│   └── mmada_t2i.yaml            # 生成配置模板
├── prompts/
│   ├── generation_prompts.txt    # GenEval 提示文本
│   ├── evaluation_metadata.jsonl # GenEval metadata
│   ├── labels_prompts.txt        # ImageNet 类标签提示
│   └── labels_metadata.jsonl     # ImageNet 类标签 metadata
├── VIRTUAL_imagenet512.npz       # ImageNet 512 参考统计（FID 评测需要）
├── scripts/
│   ├── run_all.sh                # 一键运行：生成 + 全部评测
│   ├── run_generate.sh           # 仅生成
│   ├── run_eval_geneval.sh       # 仅 GenEval 评测
│   ├── run_eval_fid.sh           # 仅 FID 评测
│   ├── run_eval_clip.sh          # 仅 CLIP Score 评测
│   └── run_imagenet_fid.sh       # ImageNet FID 端到端（生成 + 评测）
└── results/                      # 评测结果输出目录（运行后生成）
```

---

## 环境准备

### 1. 基础依赖

确保已安装 MMaDA 项目的基础依赖：

```bash
cd /path/to/MMaDA
pip install -r requirements.txt
```

### 2. 评测专用依赖

```bash
# GenEval 评测需要
pip install mmdet open_clip_torch clip_benchmark

# FID 评测需要
pip install tensorflow scipy

# CLIP Score 评测需要
pip install open_clip_torch

# 结果查看需要
pip install pandas
```

### 3. mmdetection（GenEval 需要）

```bash
# 克隆 mmdetection 到项目根目录
cd /path/to/MMaDA
git clone https://github.com/open-mmlab/mmdetection.git
```

### 4. Mask2Former 模型权重（GenEval 需要）

```bash
# 下载到 models/mask2former/
mkdir -p models/mask2former
cd geneval/evaluation
bash download_models.sh ../../models/mask2former/
```

### 5. MMaDA 模型

确保以下模型可用（本地路径或 HuggingFace 路径）：
- **MMaDA 模型**: `./mmada-mix` 或 `Gen-Verse/MMaDA-8B-MixCoT`
- **VQ 模型**: `./magvitv2` 或 `showlab/magvitv2`

---

## 快速开始

### 一键运行（生成 + 全部评测）

```bash
cd MultiModel_eval

# 修改 scripts/run_generate.sh 中的模型路径，然后：
bash scripts/run_all.sh
```

### 分步运行

#### Step 1：生成图像

```bash
# 方式一：使用脚本（推荐，先编辑 scripts/run_generate.sh 中的路径）
bash scripts/run_generate.sh

# 方式二：直接调用 python
python t2i_generate.py \
    config=configs/mmada_t2i.yaml \
    mmada_model_path=../mmada-mix \
    vq_model_path=../magvitv2 \
    validation_prompts_file=prompts/generation_prompts.txt \
    output_dir=./output_geneval \
    batch_size=1 \
    text_to_image.samples_per_prompt=4 \
    text_to_image.guidance_scale=3.5 \
    text_to_image.generation_timesteps=50 \
    text_to_image.generation_temperature=1.0 \
    geneval_metadata_file=prompts/evaluation_metadata.jsonl \
    use_geneval_format=True
```

**IGP 采样（可选）**：

```bash
python t2i_generate.py \
    config=configs/mmada_t2i.yaml \
    mmada_model_path=../mmada-mix \
    vq_model_path=../magvitv2 \
    validation_prompts_file=prompts/labels_prompts.txt \
    output_dir=./output_igp \
    batch_size=1 \
    text_to_image.samples_per_prompt=10 \
    text_to_image.guidance_scale=3.5 \
    text_to_image.generation_timesteps=5 \
    text_to_image.generation_temperature=0.4 \
    igp.use_igp=True \
    igp.num_candidates=8 \
    igp.position_tau=0.4 \
    igp.heuristic=confidence \
    geneval_metadata_file=prompts/labels_metadata.jsonl \
    use_geneval_format=True
```

#### Step 2：GenEval 评测

```bash
# 使用脚本
bash scripts/run_eval_geneval.sh ./output_geneval

# 或直接调用
bash eval_geneval.sh ./output_geneval results/geneval_results.jsonl

# 查看详细结果
python view_scores.py results/geneval_results.jsonl
```

#### Step 3：CLIP Score 评测

```bash
# 使用脚本
bash scripts/run_eval_clip.sh ./output_geneval

# 或直接调用
python eval_clip_score.py \
    --image-dir ./output_geneval \
    --output results/clip_scores.json \
    --clip-arch ViT-L-14 \
    --clip-pretrained openai
```

#### Step 4：FID / IS 评测

```bash
# 需要参考批次文件（npz 格式），已内置 VIRTUAL_imagenet512.npz
bash scripts/run_eval_fid.sh ./VIRTUAL_imagenet512.npz ./output_geneval

# 或直接调用
python eval_fid.py ./VIRTUAL_imagenet512.npz ./output_geneval --batch-size 64

# 将生成图 resize 到 256x256 后再计算 FID
python eval_fid.py ./VIRTUAL_imagenet512.npz ./output_geneval --batch-size 64 --resize 256
```

#### ImageNet FID 端到端评测

```bash
# 完整流程：生成 1000 类 × 50 张 = 50K 张 + 计算 FID
bash scripts/run_imagenet_fid.sh

# 小规模测试（10 类 × 2 张 = 20 张，快速验证流程）
bash scripts/run_imagenet_fid.sh --test

# 仅评测（已有生成图像）
bash scripts/run_imagenet_fid.sh --eval-only

# 自定义参数
MMADA_MODEL_PATH=../mmada-mix \
SAMPLES_PER_PROMPT=50 \
RESIZE=256 \
bash scripts/run_imagenet_fid.sh
```

---

## 格式转换

如果你已经用普通格式（所有图像直接在一个目录中）生成了图像，可以转换为 GenEval 格式：

```bash
python convert_format.py \
    --input-dir ./plain_images \
    --metadata-file prompts/evaluation_metadata.jsonl \
    --output-dir ./output_geneval \
    --images-per-prompt 4
```

---

## GenEval 输出格式说明

GenEval 要求的目录结构：

```
output_geneval/
├── 00000/
│   ├── metadata.jsonl          # {"prompt": "...", "tag": "...", "include": [...]}
│   └── samples/
│       ├── 0000.png
│       ├── 0001.png
│       ├── 0002.png
│       └── 0003.png
├── 00001/
│   ├── metadata.jsonl
│   └── samples/
│       └── ...
└── ...
```

使用 `t2i_generate.py` 配合 `use_geneval_format=True` 可以直接生成此格式，无需额外转换。

---

## 评测指标说明

| 指标 | 说明 | 越高/低越好 |
|------|------|------------|
| **GenEval Overall** | 对象/数量/颜色/位置的综合准确率 | 越高越好 |
| **FID** | Fréchet Inception Distance，衡量分布距离 | 越低越好 |
| **sFID** | Spatial FID，考虑空间结构的 FID | 越低越好 |
| **IS** | Inception Score，衡量质量和多样性 | 越高越好 |
| **Precision** | 生成图像落在真实分布中的比例 | 越高越好 |
| **Recall** | 真实图像被生成分布覆盖的比例 | 越高越好 |
| **CLIP Score** | 图像-文本语义相似度 | 越高越好 |

---

## 常见参数说明

### 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `guidance_scale` | 3.5 | 无分类器引导尺度，0=无条件生成 |
| `generation_timesteps` | 50 | 生成步数，越多质量越好但更慢 |
| `generation_temperature` | 1.0 | 采样温度，<1 更确定，>1 更多样 |
| `samples_per_prompt` | 1 | 每个提示生成的图像数量 |
| `batch_size` | 1 | 批次大小 |

### IGP 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `igp.use_igp` | False | 是否启用 IGP 采样 |
| `igp.num_candidates` | 8 | 候选动作数量 |
| `igp.position_tau` | 1.0 | Position Sampler 的 Gumbel 温度 |
| `igp.heuristic` | confidence | 启发式方法: confidence / margin / neg_entropy |

