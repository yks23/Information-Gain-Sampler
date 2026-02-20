# Info-Gain 采样器：掩码扩散模型统一解码框架

[English README](README.md) | [中文版 README](README_zh.md)

一个用于掩码扩散模型（MDM）的统一解码框架，结合轨迹规划和信息增益最大化。本仓库提供了 **Info-Gain 采样器** 的实现，这是一种灵活的解码策略，支持多种启发式函数，可适应各种生成任务。

## 概述

Info-Gain 采样器扩展了 PC-Sampler 框架，采用信息论的动作选择。它支持：

- **多种启发式函数**：置信度、PC值、负熵、边际和均匀采样
- **灵活的轨迹控制**：位置感知加权和随机位置采样
- **统一接口**：所有基线方法（熵、边际、置信度等）都作为基础 `generate` 函数的特例实现
- **多种模型**：TraDo、LLaDA、Dream、SDAR、MMaDA 和自回归基线（Mistral、Qwen）
- **KV-Cache 优化**：所有 MDM 模型自动支持 KV-cache 以加速生成
- **多种评测任务**：HumanEval、MBPP、MATH-500、GSM8K、GPQA、Sudoku、Countdown、创意写作
- **多模态评测**：用于文本到图像生成的 GenEval（FID、CLIP Score 等）

## 研究动机

掩码扩散模型（MDMs）已成为自回归模型（ARMs）的强大替代方案，用于离散序列生成。通过利用双向注意力，MDMs 摆脱了严格的从左到右生成限制，提供了前所未有的解码路径灵活性。这种灵活性在需要双向注意力的任务中解锁了卓越性能，例如代码填充、生物序列设计和长时程规划任务。

然而，由于**训练-推理不匹配**，这种潜力在很大程度上仍未得到充分利用。虽然 MDMs 在随机掩码模式下训练，但推理需要一个多步骤、对顺序敏感的解码过程。导航大量可能的解码顺序需要采样器仔细选择接下来要揭示的 token。因此，生成质量在很大程度上取决于采样器的有效性。

### 贪婪采样的问题

现有的采样器主要依赖**局部确定性启发式**（如置信度、熵或边际）来贪婪地选择下一个解码目标。这些方法旨在通过优先考虑最确定的位置来最小化错误累积。然而，由于**局部启发式的短视性**，这些采样器通常不够鲁棒：它们忽略了当前解码决策对未来不确定性的长期影响。因此，它们经常优先考虑在语法上看起来*自信*但在语义上次优的 token，导致错误传播和生成质量受损。

**关键问题**：贪婪优化是否足以最小化跨步骤的累积不确定性？

为了量化整个生成过程中的不确定性，我们引入轨迹 $\tau = z_T \rightarrow z_{T-1} \rightarrow \ldots \rightarrow z_0$ 上的**累积熵** $\tilde{H}$：

$$\tilde{H}(\tau) := \sum_{t=T}^{1} C(a_t \mid z_t)$$

其中 $C(a_t \mid z_t) = \sum_{\ell \in A_t} H^{(\ell)}(z_t)$ 表示步骤 $t$ 选择的 token 的边际熵之和。

### 案例研究

**案例研究 1：单向乘法**

模型的任务是生成方程 $a \times b = c$，其中 $a$ 和 $b$ 是十进制因子，$c$ 是二进制乘积。这个任务本质上是单向的：从因子计算乘积是直接的，而因式分解在计算上是困难的。

出现两条解码路径：
- **(i) 乘积优先**：首先解码二进制乘积 $c$（低不确定性：从 $\{0,1\}$ 中选择）
- **(ii) 因子优先**：首先解码十进制因子 $a$ 和 $b$（高不确定性：每个有 10 个可能值）

贪婪采样器错误地偏向路径 (i)，因为二进制数字表现出较低的每 token 不确定性。然而，通过最小化即时不确定性，贪婪策略在未确定因子的情况下就承诺了乘积，导致错误的方程和高残差不确定性。经验上，基于确定性的贪婪采样器（Entropy）以 73.2% 的概率优先选择路径 (i)。

相反，最优策略会首先解决较高不确定性的因子 $a$ 和 $b$；一旦确定，$c$ 可以以几乎零不确定性确定。我们的 Info-Gain 采样器通过以 84% 的概率优先进行因子解码，有效解决了这一挑战。

**案例研究 2：二元判断**

模型使用模板 `[推理-掩码] 答案是（是/否）：[答案-掩码]` 判断算术陈述的真假。答案 token 通常表现出较低的局部不确定性（二元选择：是/否），而推理步骤涉及更高的不确定性。

贪婪采样器倾向于过早解码答案 token，在底层推理未解决之前就做出承诺。这导致错误的判断，并在推理位置留下高残差不确定性。如实验所示，贪婪采样器仅达到 67-73% 的准确率，累积熵较高（31.68-35.60），而自回归基线达到 90% 的准确率，累积熵较低（25.19）。

### 关键观察

**观察 1**：现有的基于确定性的贪婪采样器通常无法找到接近最优的解码路径。最优解码动作不仅应根据其自身的预测确定性进行评估，还应考虑它为生成过程的其余部分提供的*信息增益*。

**观察 2**：MDMs 的双向架构使得信息增益估计在一次前向传播中高效完成，绕过了昂贵的迭代计算。与具有下一个 token 瓶颈的 ARMs 不同，MDMs 可以在一次前向传播中同时评估任何解码动作对整个序列不确定性的影响。

这些观察激发了 Info-Gain 采样器，它平衡即时确定性和信息增益，以优先考虑全局信息决策，并产生更鲁棒的解码轨迹。

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0（推荐支持 CUDA）
- 支持 CUDA 的 GPU（推荐用于模型推理）

### 安装依赖

```bash
git clone <repository-url>
cd Uncode-new

# 安装核心依赖
pip install -r requirements.txt

# 可选：用于多模态评测（FID、GenEval）
pip install tensorflow scipy mmdet open_clip_torch clip_benchmark pandas
```

### 验证安装

```bash
python -c "import torch; import transformers; print('安装成功！')"
```

## 数据准备

### 文本任务数据集

所有文本任务数据集应放置在 `data/` 目录下：

```
data/
├── humaneval.jsonl          # HumanEval 数据集
├── mbpp.jsonl               # MBPP 数据集（或 sanitized-mbpp.json）
├── math500.jsonl            # MATH-500 数据集
├── gsm8k.jsonl              # GSM8K 数据集
├── gpqa.jsonl               # GPQA 数据集
├── countdown.jsonl          # Countdown 数据集
├── sudoku.csv               # Sudoku 数据集
└── baseline/                # Baseline 频率文件
    ├── reference_corpus.json
    ├── reference_corpus_dream.json
    └── reference_corpus_llada.json
```

**数据集来源和下载说明**：
- **HumanEval**：从 [OpenAI 仓库](https://github.com/openai/human-eval) 下载 `HumanEval.jsonl.gz`，解压后保存为 `data/humaneval.jsonl`
- **MBPP**：从 [Google 仓库](https://github.com/google-research/google-research/tree/master/mbpp) 下载或使用 HuggingFace Datasets：`datasets.load_dataset("mbpp")`，保存为 `data/mbpp.jsonl`
- **MATH-500**：从 [MATH 数据集](https://github.com/hendrycks/math) 提取 500 个问题，保存为 `data/math500.jsonl`
- **GSM8K**：从 [HuggingFace Datasets](https://huggingface.co/datasets/gsm8k) 下载：`datasets.load_dataset("gsm8k", "main")`，保存为 `data/gsm8k.jsonl`
- **GPQA**：从 [GPQA 仓库](https://github.com/idavidrein/gpqa) 下载，保存为 `data/gpqa.jsonl`
- **Countdown**：包含在仓库中，位于 `data/countdown.jsonl`（或从源下载）
- **Sudoku**：包含在仓库中，位于 `data/sudoku.csv`（或从源下载）
- **创意写作**：包含在仓库中，位于 `Creativity_writing/data/creativity_writing.jsonl`（200 个提示）

### Baseline 文件

Baseline 频率文件（`data/baseline/reference_corpus*.json`）用于 PC-Sampler 启发式。生成方法：

```bash
# 从参考语料库生成 baseline
python utils/calculate_p_baseline.py \
    --input_file /path/to/reference_corpus.jsonl \
    --output_file data/baseline/reference_corpus.json \
    --model_name GSAI-ML/LLaDA-8B-Instruct  # 或您的模型名称
```

**脚本功能**：
1. 加载参考语料库（JSONL 格式）
2. 使用指定模型的分词器对所有文本进行分词
3. 计算语料库中的 token 频率分布
4. 将 baseline 分布保存为 JSON 文件，格式：
   ```json
   {
     "num_token": <总token数>,
     "p_baseline_dict": {<token_id>: <频率>, ...}
   }
   ```

**何时生成单独的 baseline**：
- 不同的分词器产生不同的 token ID → 为 Dream 和 LLaDA 模型分别生成 baseline
- 不同的语料库领域 → 生成领域特定的 baseline（例如，代码任务的代码语料库）
- 不同的模型词汇表 → 每个模型系列需要自己的 baseline

**推荐的 baseline 语料库**：使用与任务领域匹配的大型、多样化文本语料库（例如，Wikipedia、Common Crawl 子集）。

### 多模态数据

对于多模态评测，您需要以下文件：

1. **GenEval 提示**：
   - 位置：`MultiModal_eval/prompts/generation_prompts.txt`
   - 内容：用于文本到图像生成评测的文本提示
   - 格式：每行一个提示
   - 元数据：`MultiModal_eval/prompts/evaluation_metadata.jsonl`（JSONL 格式，包含提示元数据）

2. **ImageNet 参考统计**（用于 FID 评测）：
   - 位置：`data/VIRTUAL_imagenet512.npz`
   - 内容：ImageNet 512×512 图像的预计算 InceptionV3 特征
   - 用途：FID 计算的参考分布
   - **下载**：
     ```bash
     # 下载到 data/ 目录
     wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz \
          -O data/VIRTUAL_imagenet512.npz
     ```
   - **直接 URL**：`https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz`
   - 文件大小：约 200MB（压缩后）

3. **Mask2Former 模型**（用于 GenEval 目标检测）：
   - 下载位置：`models/mask2former/`（相对于项目根目录）
   - 模型文件：`mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth`
   - **下载**：
     ```bash
     # 方式 1：使用 mmdetection 的模型库（推荐）
     mkdir -p models/mask2former
     wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
          -O models/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
     
     # 方式 2：使用下载脚本（如果在 MMaDA 项目中可用）
     # cd /path/to/MMaDA/geneval/evaluation
     # bash download_models.sh ../../models/mask2former/
     ```
   - **直接 URL**：`https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth`
   - 文件大小：约 200MB
   - 用途：用于 GenEval 评测的目标检测和分割

## 模型准备

### 支持的模型

框架支持以下模型：

| 模型 | 类型 | HuggingFace 路径 | 本地路径 |
|------|------|------------------|----------|
| **TraDo** | MDM | `Gen-Verse/TraDo-8B-Instruct`<br>`Gen-Verse/TraDo-4B-Instruct`<br>`Gen-Verse/TraDo-8B-Thinking` | `/model/trado` |
| **LLaDA** | MDM | `GSAI-ML/LLaDA-8B-Instruct` | `/model/llada` |
| **Dream** | MDM | `Dream-org/Dream-v0-Instruct-7B` | `/model/dream` |
| **SDAR** | MDM | `JetLM/SDAR-8B-Chat` | `/model/sdar` |
| **MMaDA** | MDM | `Gen-Verse/MMaDA-8B-MixCoT`（必需） | `/model/mmada` |
| Mistral | AR | `mistralai/Mistral-7B-Instruct-v0.2` | `/model/mistral` |
| Qwen | AR | `Qwen/Qwen-7B-Chat` | `/model/qwen` |

### 下载模型

模型可以从**本地路径**（推荐）或 **HuggingFace Hub** 加载。框架自动检测模型类型并处理两种情况。

**加载优先级**：
1. **本地路径**（推荐）：如果路径作为目录存在，将直接使用
2. **HuggingFace Hub**：如果不是本地路径，将从 HuggingFace Hub 加载

**模型类型检测**（不区分大小写）：
- **本地路径**：从目录名称和 `config.json`（如果可用）检测
- **HuggingFace Hub**：从模型名称子字符串检测

**检测规则**（按顺序检查）：
1. **TraDo**：包含 "trado" → `TraDoAdapter`
2. **Dream**：包含 "dream" → `DreamAdapter`
3. **SDAR**：包含 "sdar" → `SDARAdapter`
4. **LLaDA**：包含 "llada" → `LLaDAAdapter`
5. **MMaDA**：包含 "mmada" → `MMaDAAdapter`
6. **Mistral**：包含 "mistral" → `MistralAdapter`
7. **Qwen**：包含 "qwen" → `QwenAdapter`
8. **默认**：未知模型 → `LLaDAAdapter`（假设为 MDM 模型）

**使用方法**：

```python
from src.models import get_model_adapter

# 从 ./model/ 目录加载（推荐 - 更快，无需下载）
# 模型应放置在项目根目录的 ./model/ 目录中
adapter = get_model_adapter("trado", device="cuda:0")  # 查找 ./model/trado/
adapter = get_model_adapter("llada", device="cuda:0")  # 查找 ./model/llada/
adapter = get_model_adapter("dream", device="cuda:0")  # 查找 ./model/dream/
adapter = get_model_adapter("sdar", device="cuda:0")  # 查找 ./model/sdar/
adapter = get_model_adapter("mmada", device="cuda:0")  # 查找 ./model/mmada/

# 从绝对路径加载（也支持）
adapter = get_model_adapter("/model/llada", device="cuda:0")
adapter = get_model_adapter("/model/dream", device="cuda:0")
adapter = get_model_adapter("/model/sdar", device="cuda:0")

# 从 HuggingFace Hub 加载（如果未缓存则自动下载）
adapter = get_model_adapter("Gen-Verse/TraDo-8B-Instruct", device="cuda:0")
adapter = get_model_adapter("GSAI-ML/LLaDA-8B-Instruct", device="cuda:0")
adapter = get_model_adapter("Dream-org/Dream-v0-Instruct-7B", device="cuda:0")
adapter = get_model_adapter("JetLM/SDAR-8B-Chat", device="cuda:0")
adapter = get_model_adapter("Gen-Verse/MMaDA-8B-MixCoT", device="cuda:0")
```

**模型目录结构**：

模型应组织在项目根目录的 `./model/` 目录中：

```
Uncode-new/
├── model/                    # 模型目录（./model/，git-ignored）
│   ├── trado/                # TraDo 模型（TraDo-8B-Instruct, TraDo-4B-Instruct 等）
│   │   ├── config.json
│   │   ├── model-*.safetensors
│   │   └── tokenizer.json
│   ├── llada/                # LLaDA 模型
│   │   ├── config.json
│   │   ├── model-*.safetensors
│   │   └── tokenizer.json
│   ├── dream/                # Dream 模型
│   │   ├── config.json
│   │   └── ...
│   ├── sdar/                 # SDAR 模型
│   │   └── ...
│   ├── mmada/                # MMaDA 模型（用于多模态任务）
│   │   ├── config.json
│   │   └── ...
│   └── ...                   # 其他模型
```

**注意**：对于本地模型，确保模型目录包含：
- `config.json` 或 `config.yaml` - 模型配置
- 模型权重（`.safetensors` 或 `.bin` 文件） - 模型参数
- `tokenizer.json` 和相关分词器文件 - 分词器配置

**模型下载大小**（近似值）：
- TraDo-8B-Instruct：约 16GB
- TraDo-4B-Instruct：约 8GB
- TraDo-8B-Thinking：约 16GB
- LLaDA-8B-Instruct：约 16GB
- Dream-v0-Instruct-7B：约 14GB
- MMaDA-8B-MixCoT：约 16GB（文本到图像生成必需）
- Mistral-7B-Instruct-v0.2：约 13GB
- Qwen-7B-Chat：约 14GB

**从 HuggingFace 下载模型**：

```bash
# TraDo 模型
huggingface-cli download Gen-Verse/TraDo-8B-Instruct --local-dir ./model/trado
huggingface-cli download Gen-Verse/TraDo-4B-Instruct --local-dir ./model/trado-4b
huggingface-cli download Gen-Verse/TraDo-8B-Thinking --local-dir ./model/trado-thinking

# LLaDA
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada

# Dream
huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir ./model/dream

# MMaDA（用于多模态任务 - 必须使用 MixCoT 版本）
huggingface-cli download Gen-Verse/MMaDA-8B-MixCoT --local-dir ./model/mmada
```

### 多模态模型

对于使用 MMaDA 进行文本到图像评测，您需要两个模型：

1. **MMaDA 模型**（主文本到图像模型）：
   - **HuggingFace**：`Gen-Verse/MMaDA-8B-MixCoT`（**必需** - 文本到图像生成必须使用 MixCoT 版本）
   - **直接下载**：
     ```bash
     # 使用 huggingface-cli（推荐）
     huggingface-cli download Gen-Verse/MMaDA-8B-MixCoT --local-dir ./model/mmada
     
     # 或放置在 model/ 目录中
     huggingface-cli download Gen-Verse/MMaDA-8B-MixCoT --local-dir ./model/mmada
     
     # 或使用 Python
     from huggingface_hub import snapshot_download
     snapshot_download(repo_id="Gen-Verse/MMaDA-8B-MixCoT", local_dir="./model/mmada")
     ```
   - 本地路径：`./model/mmada/`（推荐）或 `./mmada-mix/`（备用）
   - 用途：从文本提示生成图像
   - 大小：约 8B 参数，约 16GB 磁盘空间
   - **注意**：MMaDA-8B-MixCoT 是文本到图像生成任务必需的版本。Base 版本不适用于此评估框架。

2. **VQ 模型**（向量量化模型 - MAGVITv2）：
   - **HuggingFace**：`showlab/magvitv2`
   - **直接下载**：
     ```bash
     # 使用 huggingface-cli（推荐）
     huggingface-cli download showlab/magvitv2 --local-dir ./model/magvitv2
     
     # 或放置在 model/ 目录中
     huggingface-cli download showlab/magvitv2 --local-dir ./model/magvitv2
     
     # 或使用 Python
     from huggingface_hub import snapshot_download
     snapshot_download(repo_id="showlab/magvitv2", local_dir="./model/magvitv2")
     ```
   - 本地路径：`./model/magvitv2/`（推荐）或 `./magvitv2/`（备用）
   - 用途：将图像编码/解码为离散 token
   - 大小：约 600M 参数，约 1.2GB 磁盘空间

**模型加载优先级**：
1. `model/mmada/` 和 `model/magvitv2/`（推荐）
2. 项目根目录：`mmada-mix/` 和 `magvitv2/`（备用）
3. 配置文件路径（最后选择）

**设置说明**：
- 使用 HuggingFace `from_pretrained()` 或 `huggingface-cli download` 下载模型
- 将模型放置在 `./model/` 目录（推荐）或项目根目录
- 确保有足够的磁盘空间：两个模型约 20GB
- **注意**：如果直接使用 HuggingFace 路径，模型会在首次使用时自动下载

详细设置和配置说明请参见 [MultiModal_eval/README.md](MultiModal_eval/README.md)。

## 复现论文实验

我们提供了一个统一的脚本来一键复现论文中的所有实验：

```bash
# 运行所有实验（可能需要数小时/天，取决于硬件）
bash scripts/reproduce_paper.sh

# 在指定设备上运行
bash scripts/reproduce_paper.sh --device cuda:1

# 跳过特定实验（逗号分隔：exp1,exp2,exp3,exp4）
bash scripts/reproduce_paper.sh --skip exp2,exp4
```

### 实验详情

脚本复现论文中的四个主要实验：

1. **实验1：全注意力 MDM 推理（Dream-7B-Instruct）**
   - 任务：GSM8K、MATH-500、HumanEval、MBPP、Sudoku、Countdown
   - 参数：`position_temp=0.1`，`candidate_number=8`，`K=1,2`
   - 每个任务运行 5 次以确保统计显著性
   - 结果：`results/paper_reproduction/exp1_*/`

2. **实验2：半自回归 MDM 推理**
   - 模型：SDAR-8B-Chat、TraDo-8B-Instruct
   - 任务：GSM8K、MATH-500、HumanEval、MBPP
   - 参数：`token_temp=0.7`，`block_length=16`，`K=1,2`
   - 每个配置运行 5 次
   - 结果：`results/paper_reproduction/exp2_*/`

3. **实验3：多模态文本到图像生成（MMaDa）**
   - 评估：GenEval 和 ImageNet-512 FID
   - 参数：`position_temp=0.4`，`candidate_number=8`，50 步余弦调度器
   - 结果：`results/paper_reproduction/exp3_multimodal/`

4. **实验4：创意写作（SDAR-8B-Chat）**
   - 温度：0.5、1.0、1.5
   - 参数：`K=1,2`，`position_temp=0.1`，`candidate_number=8`
   - 每个配置运行 5 次
   - 结果：`results/paper_reproduction/exp4_*/`

### 实验设置

**超参数**：
- 推理任务：位置温度 $\tau_{\text{pos}} = 0.1$，候选数量 $N = 8$，加速阈值 $\gamma = 0.8$
- 文本到图像：位置温度 $\tau_{\text{pos}} = 0.4$，候选数量 $N = 8$，50 步余弦调度器
- 评估指标：Pass@1 准确率、累积熵 $\tilde{H}$

**基线**：我们与 Uniform、Entropy、Confidence、Margin、KLASS 和 PC-Sampler 方法进行比较。

### 主要结果

#### 全注意力 MDM（Dream-7B-Instruct）

Info-Gain 采样器在 Dream-7B-Instruct 上始终优于所有基线，通过有效的加速技术，生成时间仅增加 24%，GPU 内存使用仅增加 20%。

- **平均准确率提升**：比最佳基线提升 3.6%（K=2）和 2.9%（K=1）
- **累积熵降低**：仅为最佳基线累积熵的 47.8%（K=2）和 50.8%（K=1）
- 这证实了 Info-Gain 找到更全局优化轨迹的能力

#### 半自回归 MDM

半自回归模型（SDAR-8B-Chat、TraDo-8B-Instruct）的结果进一步验证了 Info-Gain 采样器的鲁棒性。

- **平均准确率提升**：在 K=1 设置下，SDAR-8B-Chat 和 TraDo-8B-Instruct 分别提升超过 20.3% 和 20.8%
- **累积熵降低**：在不同架构上显著降低（例如，SDAR 在 K=2 时从 210.3 降至 74.1）
- 值得注意的是，虽然引入非零 token 温度（$\tau_{\text{token}} = 0.7$）会降低基线性能，但 Info-Gain 采样器保持显著领先

#### 文本到图像生成

在多模态设置中，Info-Gain 采样器在对齐度和保真度方面都表现出色。

- **GenEval**：最高平均分数 58.2（vs. Margin 基线的 56.3）
  - 显著改善"位置"（25.0 vs. 19.0）和"属性"（32.0 vs. 29.0）子分数
- **ImageNet-512**：显著改进
  - FID：从 43.3 降至 38.1
  - IS（Inception Score）：从 53.3 提升至 63.0

#### 创意写作

对于创意写作，Info-Gain 采样器在各种 token 温度下始终优于所有基线。

- **平均胜率**：在所有设置和基线上为 63.1%
- **峰值性能**：在高温（$\tau_{\text{token}} = 1.5$）下，对 Entropy 基线达到 80.3% 的胜率
- 通过其前瞻机制优先考虑信息性动作，Info-Gain 采样器对温度缩放表现出卓越的鲁棒性，有效平衡创造力和连贯性

### 消融研究

#### 累积不确定性的优化

Info-Gain 采样器在优化累积不确定性方面显著优于基线：

1. Info-Gain 启发式平衡即时成本和未来收益，产生比贪婪 Entropy 基线更早稳定的非线性熵增长
2. 累积熵与准确率显示出强烈的**负相关**（Pearson's $r = -0.70$），验证了其作为解码质量可靠代理的有效性

#### Info-Gain 变体的比较

我们在固定计算预算下比较 Info-Gain 采样器（$B=1$）、Info-Gain 束搜索（$B>1$）和 Best-of-N（BoN）：

1. **Info-Gain 采样器（$B=1$）** 接近帕累托前沿，在保持高度可并行化并避免复杂 KV-cache 管理的同时实现接近最优的结果
2. 两种 Info-Gain 变体都显著优于 **BoN**，证明通过信息增益进行全局规划优于简单地增加独立样本
3. 在给定扩展预算下增加**束大小**会产生边际不确定性降低，但会带来更高的内存开销

#### 与温度采样的兼容性

Info-Gain 采样器在各种温度尺度上保持稳定、低的轨迹不确定性，无需敏感调整。重要的是，低累积熵反映的是更优化的解码，而不是模式崩溃，这由创意写作中保留的多样性和竞争性胜率证明。相比之下，其他基线对温度变化高度敏感，导致解码不稳定。

### 输出结构

所有结果保存在 `results/paper_reproduction/` 中，结构如下：

```
results/paper_reproduction/
├── exp1_gsm8k_K1/
│   ├── run_1.txt
│   ├── run_1.log
│   ├── run_2.txt
│   └── ...
├── exp1_gsm8k_K2/
├── exp2_sdar_gsm8k_K1/
├── exp3_multimodal/
│   ├── geneval.log
│   └── imagenet_fid.log
└── exp4_creative_writing_T0.5_K1/
    └── ...
```

每个结果文件包含该次运行的评估指标（准确率、FID、IS 等）。脚本会自动聚合多次运行的结果。

### 要求

- 所有必需的模型必须下载并放置在 `model/` 目录中
- 所有数据集必须准备在 `data/` 目录中
- 足够的 GPU 内存（推荐：24GB+ 用于大型模型）
- 预计时间：1-3 天，取决于硬件和实验数量

## 快速开始

### 简单用法（Eval.sh）

运行评测最简单的方式是通过 CLI 参数化的 `Eval.sh` 脚本：

```bash
cd scripts

# LLaDA 在 HumanEval 上使用 Info-Gain 采样器
bash Eval.sh --task humaneval --model GSAI-ML/LLaDA-8B-Instruct --mode info-gain \
    --candidate_number 8 --position_temperature 0.2

# Dream 在 MATH-500 上使用 PC-Sampler
bash Eval.sh --task math500 --model /path/to/Dream-v0-Instruct-7B --mode pc_sampler

# Sudoku 使用 Info-Gain
bash Eval.sh --task sudoku --model /path/to/model --mode info-gain --candidate_number 8

# Countdown 不使用 few-shot
bash Eval.sh --task countdown --model /path/to/model --mode info-gain --no_shot
```

内置任务默认值（gen_length、steps、block_length、data_path）会自动应用。任何参数都可以通过 CLI 标志覆盖。运行 `bash Eval.sh --help` 查看完整用法。

### 编程用法

```python
from src.models import get_model_adapter
from src.generators import generate, generate_with_info_gain
from src.prompts import get_task_prompt
from src.prompts.model_templates import apply_model_template
import torch

# 加载模型（自动检测 Dream / LLaDA / SDAR / AR）
adapter = get_model_adapter("GSAI-ML/LLaDA-8B-Instruct", device="cuda:0")
tokenizer = adapter.tokenizer
model = adapter.model

# 构建提示
input_data = {"problem": "What is 2 + 2?"}
query = get_task_prompt("math500", input_data, use_shot=True)
prompt_str = apply_model_template(adapter, tokenizer, query, task="math500")
prompt = tokenizer(prompt_str)['input_ids']
prompt = torch.tensor(prompt).to("cuda:0").unsqueeze(0)

# 使用 Info-Gain 采样器生成
output = generate(
    model=model,
    prompt=prompt,
    steps=256,
    gen_length=256,
    block_length=32,
    baseline_name="../data/baseline/reference_corpus.json",
    temperature=0.0,
    candidate_number=8,          # >1 启用 Info-Gain 模式
    position_temperature=0.2,    # >0 启用位置采样
    heuristic='confidence',
    mask_id=adapter.mask_id,
    adapter=adapter,              # 模型适配器（自动检测模型特定行为）
    use_kv_cache=True,           # 启用 KV-cache 优化（可选）
)

# 解码结果
generated_text = tokenizer.batch_decode(output[:, prompt.shape[1]:], skip_special_tokens=True)[0]
print(generated_text)
```

## Info-Gain 采样器

Info-Gain 采样器利用 MDMs 的双向特性，平衡解码决策的即时不确定性成本与其对剩余掩码位置的预期信息增益。

### 目标函数

我们首先将**状态不确定性**定义为状态 $z_t$ 中掩码位置的平均边际熵：

$$\mathcal{H}(z_t) = \frac{1}{|\mathcal{M}_t|} \sum_{\ell \in \mathcal{M}_t} H^{(\ell)}(z_t)$$

状态不确定性量化了模型仍需解决的信息量，可以通过一次前向传播高效计算。

动作 $a_t$ 的**信息增益**定义为它引起的状态不确定性减少（等价地，剩余掩码位置的边际熵减少）：

$$\text{IG}(a_t; z_t) := \mathcal{H}(z_t) - \mathcal{H}(z_{t-1})$$

其中 $z_{t-1} = \text{Apply}(z_t, a_t)$ 表示从状态 $z_t$ 执行动作 $a_t$ 后获得的状态。

解码动作 $a_t$ 的总影响被分解为两个组成部分：

1. **即时成本**：当前步骤中正在解码的 token 的不确定性，通过所选位置的边际熵之和 $C(a_t \mid z_t)$ 来衡量。

2. **信息增益**：剩余掩码位置不确定性的减少，由 $\text{IG}(a_t; z_t)$ 量化。

为了平衡这两个组成部分，我们将 Info-Gain 采样器目标定义为：

$$J_{IG}(a_t \mid z_t) = \underbrace{\text{IG}(a_t; z_t)}_{\text{信息增益}} - \underbrace{C(a_t \mid z_t)}_{\text{即时成本}}$$

### 实现

在每个解码步骤中，Info-Gain 采样器遵循**三步循环**来确定并执行最具信息性的动作：

1. **采样**：我们使用*动作采样器*采样一个多样动作的候选集 $\mathcal{C} = \{a_t^{(1)}, \dots, a_t^{(N)}\}$。这通过提出多个潜在动作来探索组合巨大的动作空间。

2. **评估**：我们计算候选集 $\mathcal{C}$ 中所有候选 $a_t$ 的目标 $J_{IG}(a_t \mid z_t)$。关键的是，这种评估非常高效，因为它只需要一次批处理前向传播来同时估计所有候选的未来信息增益。

3. **转换**：选择最优动作为 $a_t^* = \arg\max_{a \in \mathcal{C}} J_{IG}(a \mid z_t)$。然后我们执行此动作以转换到下一个状态 $z_{t-1}^*$，重复循环直到所有掩码位置都被填充。

**动作采样器**：我们通过两阶段采样过程生成大小为 $N$ 的候选集 $\mathcal{C}$ 来探索巨大的动作空间：
- **Token 采样**：使用 token 温度 $\tau_{\text{token}}$ 从 $p_\theta$ 中抽取 token $v_\ell$
- **位置采样**：使用位置温度 $\tau_{\text{pos}}$ 在确定性分数 $\phi(\ell, z_t)$ 上的 softmax 来选择位置 $\ell \in \mathcal{M}_t$

每个候选动作 $a_t = \{(\ell, v_\ell)\}$ 通过配对这些样本形成，为评估提供多样且高质量的集合。

### 高效实现

为了确保效率，候选评估在单次批处理前向传播中并行执行。我们通过以下方式进一步优化采样器：

- **块级计算**：将信息增益计算限制在当前活动块 $\mathcal{B}$，这实现了有效的 KV 缓存
- **高置信度绕过**：如果最大 token 概率超过阈值 $\gamma$，相应的位置直接固定到动作集中。这种混合方法在保持规划质量的同时显著降低了推理延迟

由于 Info-Gain 在解码过程中有效降低不确定性，高置信度绕过更频繁地触发，使该机制异常高效。

### 为什么 Info-Gain 有效

**状态不确定性**是确定当前解码状态是否接近**训练数据流形**的有效指标。当解码内容在逻辑上连贯且流畅表达时，状态位于数据流形附近，导致集中的预测概率分布和低不确定性。相反，如果模型对某些文本形式缺乏足够的训练，一旦解码在推理过程中进入这些覆盖不足的区域，就会表现出分散的概率分布和高不确定性的特征。

现有的基于确定性的贪婪采样器无法识别由状态不确定性反映的数据流形偏离信号。它们在每个步骤中选择最确定的动作，但局部确定的动作不一定对应于使后续状态保持在数据流形上的动作。

相比之下，Info-Gain 采样器通过其信息增益项主动感知和利用状态不确定性。当候选动作会导致状态偏离数据流形时，结果状态表现出增加的不确定性，这会对信息增益项产生负面影响，并防止此类动作被优先考虑。这种通过状态不确定性感知来识别和维护数据流形的机制使 Info-Gain 采样器能够在整个解码路径中保持逻辑连贯性和流畅表达，即使在高温采样下也是如此。

### 核心概念

**Info-Gain 采样器** 是一种通过最大化信息增益来选择动作（要解码的 token 位置）的解码策略。它有两种工作模式：

1. **传统模式**（`candidate_number=1`）：
   - 基于启发式分数的贪心选择
   - 等价于传统的不确定性采样器

2. **Info-Gain 模式**（`candidate_number>1`）：
   - 采样多个候选动作
   - 通过计算信息增益评估每个候选
   - 选择最大化即时成本 − 信息增益的动作

### 启发式函数

| 启发式 | 描述 |
|--------|------|
| `confidence`（默认） | 模型置信度（预测 token 的概率） |
| `pc` | PC-Sampler 启发式，基于频率校准 |
| `neg_entropy` | 负熵（熵越高 = 分数越低） |
| `margin` | top-1 和 top-2 概率之间的边际 |
| `uniform` | 均匀随机采样 |

### 参数

| 参数 | 描述 | 默认值 | 说明 |
|------|------|--------|------|
| `candidate_number` | 候选动作数量 | 1 | >1 启用 Info-Gain 模式 |
| `position_temperature` | 位置采样温度 | 0.0 | >0 启用随机采样 |
| `heuristic` | 启发式函数类型 | `confidence` | 见上表 |
| `beam_size` | Beam search 队列大小 | 1 | >1 用于 Info-Gain Beam Search |
| `use_kv_cache` | 启用 KV-cache 优化 | False | 加速 MDM 模型生成 |
| `use_block_causal_mask` | 使用块因果注意力掩码 | False | 用于块扩散注意力模式 |

### 基线方法作为特例

所有基线解码方法都作为基础 `generate` 函数的特例实现：

- **`original`**：`candidate_number=1`，`heuristic='confidence'`
- **`pc_sampler`**：`candidate_number=1`，`heuristic='pc'`
- **`entropy`**：`candidate_number=1`，`heuristic='neg_entropy'`
- **`margin`**：`candidate_number=1`，`heuristic='margin'`

## 支持的模型

| 模型 | 类型 | 适配器 | KV-Cache | 说明 |
|------|------|--------|----------|------|
| LLaDA | MDM | `LLaDAAdapter` | ✅（截断） | 默认 mask_id=126336 |
| Dream | MDM | `DreamAdapter` | ✅（原生） | Logits 偏移一个位置 |
| SDAR | MDM | `SDARAdapter` | ✅（截断） | 接口已就绪；模型实现待完成 |
| Mistral | AR 基线 | `MistralAdapter` | ❌ | 标准聊天模板 |
| Qwen | AR 基线 | `QwenAdapter` | ❌ | 标准聊天模板 |

**模型检测**：模型类型通过分析模型名称或路径中的关键词自动检测。检测不区分大小写，适用于 HuggingFace Hub 路径和本地路径。使用 `get_model_adapter(model_name, device)` 进行自动检测和加载。

**KV-Cache 支持**：所有 MDM 模型都支持 KV-cache 优化以加速生成：
- **Dream 模型**：使用原生 `store_kv` 参数进行高效的缓存管理（无需截断）
- **LLaDA/SDAR 模型**：需要在前向传播后截断缓存（自动处理）
- **性能**：长序列速度提升 30-50%，特别是在使用 Info-Gain 采样时

## 评测任务

### 文本任务

| 任务 | 数据集 | 描述 |
|------|--------|------|
| `humaneval` | `humaneval.jsonl` | Python 代码补全 |
| `mbpp` | `mbpp.jsonl` | Python 代码生成 |
| `math500` | `math500.jsonl` | 数学推理 |
| `gsm8k` | `gsm8k.jsonl` | 小学数学 |
| `gpqa` | `gpqa.jsonl` | 研究生级别问答 |
| `sudoku` | `sudoku.csv` | 4×4 数独谜题求解 |
| `countdown` | `countdown.jsonl` | 算术运算游戏 |
| `creativity_writing` | `creativity_writing.jsonl` | 创意故事写作（200 个提示） |

### 多模态任务

| 任务 | 描述 | 指标 |
|------|------|------|
| `geneval` | 文本到图像生成评测 | FID、sFID、IS、Precision、Recall、CLIP Score |

详细创意写作评测说明请参见 [Creativity_writing/README.md](Creativity_writing/README.md)。
多模态评测说明请参见 [MultiModal_eval/README.md](MultiModal_eval/README.md)。

## KV-Cache 优化

KV-cache 优化通过缓存先前计算的键值状态显著加速生成。所有 MDM 模型（Dream、LLaDA、SDAR）都支持 KV-cache：

```python
output = generate(
    model=model,
    prompt=prompt,
    steps=256,
    gen_length=256,
    block_length=32,
    adapter=adapter,
    use_kv_cache=True,  # 启用 KV-cache
    # ... 其他参数
)
```

**工作原理**：
1. **预填充阶段**：处理提示 token 一次并缓存其键值状态
2. **逐块生成**：对于每个生成块，只处理新 token，同时重用缓存的前缀状态
3. **缓存更新**：完成每个块后，使用新块的键值状态更新缓存
4. **前瞻优化**：在 Info-Gain 模式中，候选评估重用已提交的前缀缓存

**实现细节**：
- **Dream 模型**：使用原生 `store_kv=False` 参数防止前瞻期间缓存增长
- **LLaDA/SDAR 模型**：在前向传播后自动截断缓存回到已提交长度
- **块完成**：缓存在每个块完成时更新一次，而不是每个去噪步骤

**性能**：KV-cache 可以将长序列的生成时间减少 30-50%，特别是在使用 Info-Gain 采样时，需要评估多个候选。

## 评测

所有评测任务都可以使用统一的 `Eval.sh` 脚本运行。每个任务都有内置的生成参数默认值，可以通过命令行标志覆盖。

### 文本任务评测

#### HumanEval（代码补全）

```bash
cd scripts

# Info-Gain 采样器（推荐）
bash Eval.sh \
    --task humaneval \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --heuristic confidence \
    --use_kv_cache

# PC-Sampler 基线
bash Eval.sh \
    --task humaneval \
    --model /path/to/model \
    --mode pc_sampler \
    --lambd 0.25 \
    --alpha 100

# Original（基于置信度）
bash Eval.sh \
    --task humaneval \
    --model /path/to/model \
    --mode original
```

**输出**：结果保存到 `results/humaneval_<mode>_<timestamp>/`

#### MBPP（代码生成）

```bash
bash Eval.sh \
    --task mbpp \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2
```

**输出**：结果保存到 `results/mbpp_<mode>_<timestamp>/`

#### MATH-500（数学推理）

```bash
bash Eval.sh \
    --task math500 \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --temperature 0.7
```

**输出**：结果保存到 `results/math500_<mode>_<timestamp>/`

#### GSM8K（小学数学）

```bash
bash Eval.sh \
    --task gsm8k \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --gen_length 512 \
    --steps 512
```

**输出**：结果保存到 `results/gsm8k_<mode>_<timestamp>/`

#### GPQA（研究生级别问答）

```bash
bash Eval.sh \
    --task gpqa \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8
```

**输出**：结果保存到 `results/gpqa_<mode>_<timestamp>/`

#### Sudoku（4×4 谜题求解）

```bash
bash Eval.sh \
    --task sudoku \
    --model /path/to/model \
    --mode info-gain \
    --candidate_number 8
```

**注意**：Sudoku 使用带有嵌入 mask token 的特殊提示格式。任务会根据谜题自动调整 `gen_length` 和 `steps`。

**输出**：结果保存到 `results/sudoku_<mode>_<timestamp>/`

#### Countdown（算术运算）

```bash
# 使用 few-shot 示例（默认）
bash Eval.sh \
    --task countdown \
    --model /path/to/model \
    --mode info-gain \
    --candidate_number 8

# 不使用 few-shot 示例
bash Eval.sh \
    --task countdown \
    --model /path/to/model \
    --mode info-gain \
    --candidate_number 8 \
    --no_shot
```

**输出**：结果保存到 `results/countdown_<mode>_<timestamp>/`

#### 创意写作

```bash
# 生成输出
bash Eval.sh \
    --task creativity_writing \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --gen_length 512

# 输出保存到 Creativity_writing/outputs/<model>_<mode>.json
```

**使用 LLM-as-Judge 评测**：

```bash
cd Creativity_writing

# 成对比较（比较两个模型）
python judge.py \
    --model_outputs outputs/model_a.json \
    --reference_outputs outputs/model_b.json \
    --judge_model gpt-4o \
    --mode pairwise

# 单分数评级
python judge.py \
    --model_outputs outputs/model_a.json \
    --judge_model gpt-4o \
    --mode single
```

**注意**：为 judge 脚本设置 `OPENAI_API_KEY` 环境变量。

**输出**：
- 生成：`Creativity_writing/outputs/<model>_<mode>.json`
- Judge 结果：`Creativity_writing/outputs/judge_<mode>_<timestamp>.json`

### 多模态任务评测

#### GenEval（文本到图像生成）

**步骤 1：生成图像**

```bash
cd MultiModal_eval

# 编辑 scripts/run_generate.sh 设置模型路径，然后：
bash scripts/run_generate.sh

# 或直接调用：
python t2i_generate.py \
    config=configs/mmada_t2i.yaml \
    mmada_model_path=../mmada-mix \
    vq_model_path=../magvitv2 \
    validation_prompts_file=prompts/generation_prompts.txt \
    output_dir=./output_geneval \
    batch_size=1 \
    text_to_image.samples_per_prompt=4 \
    use_geneval_format=True
```

**步骤 2：使用 GenEval 评测**

```bash
# GenEval 评测（目标检测 + 颜色分类）
bash scripts/run_eval_geneval.sh ./output_geneval

# 查看详细分数
python view_scores.py results/geneval_results.jsonl
```

**步骤 3：CLIP Score 评测**

```bash
bash scripts/run_eval_clip.sh ./output_geneval

# 输出：results/clip_scores.json
```

**步骤 4：FID / IS / Precision / Recall**

```bash
# FID 评测（需要 data/VIRTUAL_imagenet512.npz）
bash scripts/run_eval_fid.sh ./data/VIRTUAL_imagenet512.npz ./output_geneval

# 或直接调用：
python eval_fid.py \
    ./data/VIRTUAL_imagenet512.npz \
    ./output_geneval \
    --batch-size 64
```

**一键流程**（推荐用于完整评测）：

```bash
cd MultiModal_eval
bash scripts/run_all.sh
```

**流程步骤**：
1. **生成**：从 GenEval 提示生成图像（保存到 `output_geneval/`）
2. **GenEval**：评测目标检测、计数、颜色和空间关系
3. **CLIP Score**：计算图像和提示之间的语义对齐
4. **FID**：计算分布相似性指标（需要 `data/VIRTUAL_imagenet512.npz`）

**自动下载**：
- **InceptionV3 模型**（用于 FID 计算）：首次使用时自动从 OpenAI 下载
  - URL：`https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb`
  - 位置：`MultiModal_eval/classify_image_graph_def.pb`（自动下载）
  - 大小：约 100MB

**输出位置**：
- 生成的图像：`MultiModal_eval/output_geneval/`
- GenEval 结果：`MultiModal_eval/results/geneval_results.jsonl`
- CLIP 分数：`MultiModal_eval/results/clip_scores.json`
- FID 指标：打印到控制台（也保存到日志文件）

#### ImageNet FID 端到端

```bash
cd MultiModal_eval

# 完整运行（50K 图像）
bash scripts/run_imagenet_fid.sh

# 测试运行（100 图像）
bash scripts/run_imagenet_fid.sh --test

# 仅评测（跳过生成）
bash scripts/run_imagenet_fid.sh --eval-only
```

**输出**：FID 分数打印到控制台并保存到 `results/imagenet_fid_<timestamp>.txt`

### 通用评测选项

所有任务支持以下通用选项：

```bash
bash Eval.sh \
    --task <task_name> \
    --model <model_path> \
    --mode <generation_mode> \
    --device cuda:0 \                    # GPU 设备
    --temperature 0.7 \                  # 采样温度
    --gen_length 256 \                   # 覆盖默认 gen_length
    --steps 256 \                        # 覆盖默认 steps
    --block_length 32 \                  # 覆盖默认 block_length
    --use_kv_cache \                     # 启用 KV-cache 优化
    --result_dir results/custom_dir \    # 自定义输出目录
    --result_path results/custom.json   # 自定义输出文件路径
```

**生成模式**：
- `original`：基于置信度的贪心选择
- `pc_sampler`：PC-Sampler，带频率校准
- `eb_sampler`：基于熵的采样器
- `fast_dllm`：快速 dLLM，带动态阈值
- `entropy`：负熵启发式
- `margin`：边际启发式
- `info-gain`：Info-Gain 采样器（推荐）

**Info-Gain 特定选项**：
- `--candidate_number N`：要评估的候选动作数量（默认：1）
  - `N=1`：基于启发式分数的贪心选择（基线模式）
  - `N>1`：Info-Gain 模式 - 评估 N 个候选并选择信息增益最大的候选
- `--position_temperature T`：随机位置采样的温度（默认：0.0）
  - `T=0`：确定性选择（始终选择 top-k 位置）
  - `T>0`：使用 Gumbel 噪声的随机采样（增加探索）
- `--heuristic H`：用于评分位置的启发式函数（默认：`confidence`）
  - 选项：`confidence`、`pc`、`neg_entropy`、`margin`、`uniform`
  - 参见"启发式函数"部分了解详情
- `--tokens_per_step K`：每步解码的 token 数量（默认：1）
  - `K=1`：标准解码（每步一个 token）
  - `K>1`：K 步解码（同时解码 K 个 token）

### 直接使用 eval.py

要获得更多控制，直接使用 `eval.py`：

```bash
cd scripts
python eval.py \
    --task humaneval \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --device cuda:0 \
    --mode info-gain \
    --heuristic confidence \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --data_path ../data/humaneval.jsonl \
    --result_path ../results/humaneval_info_gain
```

### 使用 lm-evaluation-harness

与 lm-evaluation-harness 框架集成：

```bash
cd scripts
python eval_llada.py \
    --model llada_dist \
    --model_args model_path=GSAI-ML/LLaDA-8B-Base,mode=info-gain \
    --tasks lambada_openai \
    --batch_size 32
```

## 复现性

本节提供完整的端到端示例，用于复现评测结果。

### 示例 1：使用 Info-Gain 采样器的 HumanEval

**步骤 1：环境设置**

```bash
# 克隆仓库
git clone <repository-url>
cd Uncode-new

# 安装依赖
pip install -r requirements.txt
```

**步骤 2：数据准备**

```bash
# 下载 HumanEval 数据集
# 将其放置在 data/humaneval.jsonl

# 生成 baseline 文件（如需要）
python utils/calculate_p_baseline.py \
    --input_file /path/to/reference_corpus.jsonl \
    --output_file data/baseline/reference_corpus_llada.json \
    --model_name GSAI-ML/LLaDA-8B-Instruct
```

**步骤 3：运行评测**

```bash
cd scripts

bash Eval.sh \
    --task humaneval \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --heuristic confidence \
    --use_kv_cache \
    --device cuda:0
```

**步骤 4：查看结果**

```bash
# 结果保存到 results/humaneval_info-gain_<timestamp>/
# 检查结果文件中的准确率和通过率
cat results/humaneval_info-gain_*/results.txt
```

### 示例 2：使用 PC-Sampler 的 MATH-500

```bash
cd scripts

bash Eval.sh \
    --task math500 \
    --model /path/to/Dream-v0-Instruct-7B \
    --mode pc_sampler \
    --lambd 0.25 \
    --alpha 100 \
    --baseline_name ../data/baseline/reference_corpus_dream.json \
    --temperature 0.7
```

### 示例 3：使用 LLM-as-Judge 的创意写作

**步骤 1：生成输出**

```bash
cd scripts

bash Eval.sh \
    --task creativity_writing \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --gen_length 512
```

**步骤 2：使用 Judge 评测**

```bash
cd Creativity_writing

# 设置 API 密钥
export OPENAI_API_KEY="your-api-key"

# 运行 judge 评测
python judge.py \
    --model_outputs outputs/llada_8b_instruct_info-gain_confidence_K1.json \
    --judge_model gpt-4o \
    --mode single
```

### 示例 4：多模态评测（GenEval + FID）

**步骤 1：设置多模态环境**

```bash
cd MultiModal_eval

# 安装额外依赖
pip install tensorflow scipy mmdet open_clip_torch clip_benchmark pandas

# 克隆 mmdetection（如需要）
cd ..
git clone https://github.com/open-mmlab/mmdetection.git
cd MultiModal_eval
```

**步骤 2：准备模型**

```bash
# 确保 MMaDA 和 VQ 模型可用
# 编辑 scripts/run_generate.sh 设置路径：
# mmada_model_path=../mmada-mix
# vq_model_path=../magvitv2
```

**步骤 3：运行完整评测**

```bash
bash scripts/run_all.sh
```

这将：
1. 从 GenEval 提示生成图像
2. 运行 GenEval 评测（目标检测 + 颜色）
3. 计算 CLIP Score
4. 计算 FID、IS、Precision、Recall

**步骤 4：查看结果**

```bash
# GenEval 结果
python view_scores.py results/geneval_results.jsonl

# CLIP Score
cat results/clip_scores.json

# FID 结果（打印到控制台）
```

### 复现论文结果

要复现论文或先前实验的结果，请确保以下方面的一致性：

1. **模型版本**：使用完全相同的模型检查点（相同的提交哈希或版本标签）
   - 如果使用 HuggingFace 模型，检查模型提交哈希
   - 对于本地模型，确保权重完全相同

2. **超参数**：完全匹配所有生成参数
   - `temperature`、`steps`、`gen_length`、`block_length`
   - `candidate_number`、`position_temperature`、`heuristic`
   - `lambd`、`alpha`（用于 PC-Sampler）
   - `use_kv_cache` 标志（影响生成速度但不影响结果）

3. **数据分割**：使用相同的数据集版本和分割
   - 确保数据集文件完全相同（相同的样本数量、相同的顺序）
   - 对于有训练/测试分割的任务，使用相同的分割

4. **Baseline 文件**：使用从相同参考语料库生成的 baseline 文件
   - 相同的语料库来源和大小
   - 用于分词的相同分词器/模型
   - 验证 baseline 文件内容匹配

5. **随机种子**：如果复现性至关重要，设置随机种子
   - 注意：当前实现不暴露种子参数
   - 对于确定性结果，确保从外部控制 PyTorch/CUDA 随机状态

## 贡献

代码风格、提交规范和 Pull Request 流程指南请参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

[在此添加您的许可证信息]

---

**注意**：本仓库正在积极维护中。如有问题、疑问或贡献，请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

