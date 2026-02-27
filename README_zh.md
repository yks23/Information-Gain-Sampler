# 信息增益采样器：掩码扩散模型的解码框架

[中文版 README](README_zh.md) | [English README](README.md)

一个统一的解码框架，用于掩码扩散模型（MDMs），结合轨迹规划与信息增益最大化。本仓库提供了 **Info-Gain Sampler** 的实现，这是一个灵活的解码策略，支持多种启发式函数，可适应各种生成任务。

> **注意**：本仓库正在积极开发中，用于持续实验，尚未完全整理。我们还提供了 [dllm](https://github.com/ZHZisZZ/dllm) 框架的适配版本。`dllm/` 目录是一个 Git 子模块，包含我们集成到 dllm 框架中的 Info-Gain 采样器实现。

**初始化子模块：**

```bash
# 克隆仓库时包含子模块
git clone --recurse-submodules git@github.com:yks23/Information-Gain-Sampler.git

# 或者如果已经克隆了仓库，初始化子模块
git submodule update --init --recursive
```

然后可以进入 `dllm/` 目录，参照 [`dllm/README.md`](dllm/README.md) 使用 dllm 框架集成版本。

## 目录

- [动机](#动机)
  - [贪婪采样的问题](#贪婪采样的问题)
  - [案例研究](#案例研究)
  - [关键观察](#关键观察)
- [Info-Gain Sampler](#info-gain-sampler)
  - [目标函数](#目标函数)
  - [实现](#实现)
  - [高效实现](#高效实现)
- [安装](#安装)
- [模型准备](#模型准备)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
  - [任务特定脚本](#任务特定脚本推荐)
  - [统一脚本](#统一脚本evalsh)
  - [编程式使用](#编程式使用)
- [复现论文实验](#复现论文实验)
  - [实验详情](#实验详情)
  - [实验设置](#实验设置)
  - [主要结果](#主要结果)
  - [消融研究](#消融研究)
- [许可证](#许可证)
- [引用](#引用)

---

## 动机

掩码扩散模型（MDMs）已成为自回归模型（ARMs）在离散序列生成方面的强大替代方案。通过利用双向注意力，MDMs 摆脱了严格的从左到右生成模式，在解码路径上获得了前所未有的灵活性。这种灵活性在需要双向注意力的任务中释放出卓越性能，如代码填充、生物序列设计和长时程规划任务。

然而，由于**训练-推理不匹配**，这种潜力在很大程度上仍未得到充分利用。虽然 MDMs 在随机掩码模式下训练，但推理需要一个多步骤、对顺序敏感的解码过程。在大量可能的解码顺序中导航需要一个采样器，仔细选择接下来要揭示的 token。因此，生成质量在很大程度上取决于采样器的有效性。

### 贪婪采样的问题

现有的采样器主要依赖**局部确定性启发式**（如置信度、熵或边际）来贪婪地选择下一个解码目标。这些方法旨在通过优先考虑最确定的位置来最小化错误累积。然而，由于**局部启发式的短视性**：它们忽略了当前解码决策对未来不确定性的长期影响，因此这些采样器通常不够鲁棒。因此，它们经常优先考虑在语法上看起来*自信*但在语义上次优的 token，导致错误传播和生成质量下降。

**关键问题**：贪婪优化是否足以最小化跨步骤的累积不确定性？

为了量化整个生成过程中的不确定性，我们引入了轨迹 $\tau = z_T \rightarrow z_{T-1} \rightarrow \ldots \rightarrow z_0$ 上的**累积熵** $\tilde{H}$：

$$\tilde{H}(\tau) := \sum_{t=T}^{1} C(a_t \mid z_t)$$

其中 $C(a_t \mid z_t) = \sum_{\ell \in A_t} H^{(\ell)}(z_t)$ 表示在步骤 $t$ 选择的 token 的边际熵之和。

### 案例研究

**案例研究 1：单向乘法**

模型的任务是生成方程 $a \times b = c$，其中 $a$ 和 $b$ 是十进制因子，$c$ 是二进制乘积。这个任务本质上是单向的：从因子计算乘积是直接的，而因式分解在计算上是困难的。

出现了两条解码路径：
- **(i) 乘积优先**：首先解码二进制乘积 $c$（低不确定性：从 $\{0,1\}$ 中选择）
- **(ii) 因子优先**：首先解码十进制因子 $a$ 和 $b$（高不确定性：每个有 10 个可能值）

贪婪采样器错误地倾向于路径 (i)，因为二进制数字表现出较低的每 token 不确定性。然而，通过最小化即时不确定性，贪婪策略在未固定因子的情况下就承诺了乘积，导致错误的方程和高残差不确定性。经验上，基于贪婪确定性的采样器（Entropy）以 73.2% 的概率优先选择路径 (i)。

相反，最优策略会首先解决更高不确定性的因子 $a$ 和 $b$；一旦固定，$c$ 可以以几乎为零的不确定性确定。我们的 Info-Gain Sampler 通过以 84% 的概率优先考虑因子解码，有效地解决了这一挑战。

**案例研究 2：二元判断**

模型使用模板判断算术陈述的真假：`[推理-掩码] 答案是（是/否）：[答案-掩码]`。答案 token 通常表现出较低的局部不确定性（二元选择：是/否），而推理步骤涉及更高的不确定性。

贪婪采样器倾向于过早解码答案 token，在底层推理解决之前就做出承诺。这导致错误的判断，并在推理位置留下高残差不确定性。如实验所示，贪婪采样器仅达到 67-73% 的准确率，累积熵较高（31.68-35.60），而自回归基线达到 90% 的准确率，累积熵较低（25.19）。

### 关键观察

**观察 1**：现有的基于贪婪确定性的采样器通常无法找到接近最优的解码路径。最优解码动作不仅应该根据其自身的预测确定性来评估，还应该根据它为生成过程的其余部分提供的*信息增益*来评估。

**观察 2**：MDMs 的双向架构使得在一次前向传播中高效估计信息增益成为可能，绕过了昂贵的迭代计算。与具有下一个 token 瓶颈的 ARMs 不同，MDMs 可以在一次前向传播中同时评估任何解码动作对整个序列不确定性的影响。

这些观察激发了 Info-Gain Sampler，它平衡即时确定性和信息增益，优先考虑全局信息化的决策，产生更鲁棒的解码轨迹。

## Info-Gain Sampler

Info-Gain Sampler 利用 MDMs 的双向特性，平衡解码决策的即时不确定性成本与其在剩余掩码位置上的预期信息增益。

### 目标函数

我们首先将**状态不确定性**定义为状态 $z_t$ 中掩码位置上的平均边际熵：

$$\mathcal{H}(z_t) = \frac{1}{|\mathcal{M}_t|} \sum_{\ell \in \mathcal{M}_t} H^{(\ell)}(z_t)$$

状态不确定性量化了模型需要解决的剩余信息，可以通过一次前向传播高效计算。

动作 $a_t$ 的**信息增益**定义为它引起的状态不确定性的减少（等价地，剩余掩码位置上的边际熵的减少）：

$$\text{IG}(a_t; z_t) := \mathcal{H}(z_t) - \mathcal{H}(z_{t-1})$$

其中 $z_{t-1} = \text{Apply}(z_t, a_t)$ 表示从状态 $z_t$ 执行动作 $a_t$ 后获得的状态。

解码动作 $a_t$ 的总影响被分解为两个组成部分：

1. **即时成本**：当前步骤中正在解码的 token 的不确定性，通过所选位置 $C(a_t \mid z_t)$ 上的边际熵之和来衡量。

2. **信息增益**：剩余掩码位置不确定性的减少，由 $\text{IG}(a_t; z_t)$ 量化。

为了平衡这两个组成部分，我们将 Info-Gain Sampler 目标定义为：

$$J_{IG}(a_t \mid z_t) = \underbrace{\text{IG}(a_t; z_t)}_{\text{信息增益}} - \underbrace{C(a_t \mid z_t)}_{\text{即时成本}}$$

### 实现

在每个解码步骤中，Info-Gain Sampler 遵循**三步循环**来确定并执行最具信息量的动作：

1. **采样**：我们使用*动作采样器*采样一个多样动作的候选集 $\mathcal{C} = \{a_t^{(1)}, \dots, a_t^{(N)}\}$。这通过提出多个潜在动作来探索组合巨大的动作空间。

2. **评估**：我们计算集合 $\mathcal{C}$ 中所有候选 $a_t$ 的目标 $J_{IG}(a_t \mid z_t)$。关键是，这种评估非常高效，因为它只需要一次批量前向传播来同时估计所有候选的未来信息增益。

3. **转移**：最优动作被选择为 $a_t^* = \arg\max_{a \in \mathcal{C}} J_{IG}(a \mid z_t)$。然后我们执行此动作以转移到下一个状态 $z_{t-1}^*$，重复循环直到所有掩码位置都被填充。

**动作采样器**：我们通过两阶段采样过程生成大小为 $N$ 的候选集 $\mathcal{C}$ 来探索大型动作空间：
- **Token 采样**：使用 token 温度 $\tau_{\text{token}}$ 从 $p_\theta$ 中抽取 token $v_\ell$
- **位置采样**：使用位置温度 $\tau_{\text{pos}}$ 在确定性分数 $\phi(\ell, z_t)$ 上的 softmax 选择位置 $\ell \in \mathcal{M}_t$

每个候选动作 $a_t = \{(\ell, v_\ell)\}$ 通过配对这些样本形成，为评估提供多样且高质量的集合。

### 高效实现

为了确保效率，候选评估在单次批量前向传播中并行执行。我们通过以下方式进一步优化采样器：

- **块级计算**：将信息增益计算限制在当前活动块 $\mathcal{B}$ 上，这实现了有效的 KV 缓存
- **高置信度绕过**：如果最大 token 概率超过阈值 $\gamma$，相应的位置直接固定到动作集中。这种混合方法显著降低了推理延迟，同时保持了规划质量

由于 Info-Gain 在解码过程中有效减少不确定性，高置信度绕过更频繁地触发，使该机制异常高效。

---

## 安装

### 要求

- Python >= 3.8
- PyTorch >= 2.0.0（推荐支持 CUDA）
- 支持 CUDA 的 GPU（推荐用于模型推理）

### 安装依赖

```bash
git clone git@github.com:yks23/Information-Gain-Sampler.git
cd Information-Gain-Sampler

conda create -n info-gain python=3.10
conda activate info-gain

# 安装核心依赖
pip install -r requirements.txt

# 可选：用于多模态评估（FID, GenEval）
pip install tensorflow scipy mmdet open_clip_torch clip_benchmark pandas
```

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
Information-Gain-Sampler/
├── model/                    # 模型目录（./model/，git 忽略）
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
- 模型权重（`.safetensors` 或 `.bin` 文件）- 模型参数
- `tokenizer.json` 和相关 tokenizer 文件 - Tokenizer 配置

**模型下载大小**（近似）：
- TraDo-8B-Instruct: ~16GB
- LLaDA-8B-Instruct: ~16GB
- Dream-v0-Instruct-7B: ~14GB
- MMaDA-8B-MixCoT: ~16GB（文本到图像生成所需）

**从 HuggingFace 下载模型**：

```bash
# TraDo 模型
huggingface-cli download Gen-Verse/TraDo-8B-Instruct --local-dir ./model/trado

# LLaDA
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada

# Dream
huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir ./model/dream

# SDAR
huggingface-cli download JetLM/SDAR-8B-Chat --local-dir ./model/sdar

# MMaDA（用于多模态任务 - 需要 MixCoT 版本）
huggingface-cli download Gen-Verse/MMaDA-8B-MixCoT --local-dir ./model/mmada
```

### 多模态模型

对于使用 MMaDA 的文本到图像评估，您需要两个模型：

1. **MMaDA 模型**（主要文本到图像模型）：
   - **HuggingFace**：`Gen-Verse/MMaDA-8B-MixCoT`（**必需** - 使用 MixCoT 版本进行文本到图像生成）
   - **直接下载**：
     ```bash
     # 使用 huggingface-cli（推荐）
     huggingface-cli download Gen-Verse/MMaDA-8B-MixCoT --local-dir ./model/mmada
     ```
   - 本地路径：`./model/mmada/`（首选）或 `./mmada-mix/`（备用）
   - 用途：从文本提示生成图像
   - 大小：~8B 参数，~16GB 磁盘空间
   - **注意**：MMaDA-8B-MixCoT 是文本到图像生成任务所需的版本。Base 版本不适用于此评估框架。

2. **VQ 模型**（向量量化模型 - MAGVITv2）：
   - **HuggingFace**：`showlab/magvitv2`
   - **直接下载**：
     ```bash
     # 使用 huggingface-cli（推荐）
     huggingface-cli download showlab/magvitv2 --local-dir ./model/magvitv2
     ```
   - 本地路径：`./model/magvitv2/`（首选）或 `./magvitv2/`（备用）
   - 用途：将图像编码/解码为离散 token
   - 大小：~600M 参数，~1.2GB 磁盘空间

**模型加载优先级**：
1. `model/mmada/` 和 `model/magvitv2/`（首选）
2. 项目根目录：`mmada-mix/` 和 `magvitv2/`（备用）
3. 配置文件路径（最后手段）

**设置说明**：
- 使用 HuggingFace `from_pretrained()` 或 `huggingface-cli download` 下载模型
- 将模型放置在 `./model/` 目录（首选）或项目根目录
- 确保足够的磁盘空间：两个模型约 ~20GB
- **注意**：如果直接使用 HuggingFace 路径，模型将在首次使用时自动下载

详细设置和配置说明请参见 [src/benchmarks/multimodal_tasks/multimodal_eval/README.md](src/benchmarks/multimodal_tasks/multimodal_eval/README.md)。

## 数据准备

### 基线文件

基线频率文件（`data/baseline/reference_corpus*.json`）用于 PC-Sampler 启发式。生成它们：

```bash
# 从参考语料库生成基线
python utils/calculate_p_baseline.py \
    --input_file /path/to/reference_corpus.jsonl \
    --output_file data/baseline/reference_corpus.json \
    --model_name GSAI-ML/LLaDA-8B-Instruct  # 或您的模型名称
```

**脚本功能**：
1. 加载参考语料库（JSONL 格式）
2. 使用指定模型的 tokenizer 对所有文本进行标记化
3. 计算整个语料库的 token 频率分布
4. 将基线分布保存为 JSON 文件，格式为：
   ```json
   {
     "num_token": <总token数>,
     "p_baseline_dict": {<token_id>: <频率>, ...}
   }
   ```

**何时生成单独的基线**：
- 不同的 tokenizer 产生不同的 token ID → 为 Dream 和 LLaDA 模型生成单独的基线
- 不同的语料库领域 → 生成特定领域的基线（例如，代码任务的代码语料库）
- 不同的模型词汇表 → 每个模型系列都需要自己的基线

**推荐的基线语料库**：使用与您的任务领域匹配的大型、多样化的文本语料库（例如，Wikipedia、Common Crawl 子集）。

### 多模态数据

对于多模态评估，您需要以下文件：

1. **GenEval 提示**：
   - 位置：`src/benchmarks/multimodal_tasks/multimodal_eval/prompts/generation_prompts.txt`
   - 包含：用于文本到图像生成评估的文本提示
   - 格式：每行一个提示
   - 元数据：`src/benchmarks/multimodal_tasks/multimodal_eval/prompts/evaluation_metadata.jsonl`（JSONL 格式，包含提示元数据）

2. **ImageNet 参考统计**（用于 FID 评估）：
   - 位置：`data/VIRTUAL_imagenet512.npz`
   - 包含：ImageNet 512×512 图像的预计算 InceptionV3 特征
   - 用途：FID 计算的参考分布
   - **下载**：
     ```bash
     # 下载到 data/ 目录
     wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz \
          -O data/VIRTUAL_imagenet512.npz
     ```
   - **直接 URL**：`https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz`
   - 文件大小：~200MB（压缩）

3. **Mask2Former 模型**（用于 GenEval 目标检测）：
   - 下载位置：`models/mask2former/`（相对于项目根目录）
   - 模型文件：`mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth`
   - **下载**：
     ```bash
     # 选项 1：使用 mmdetection 的模型库（推荐）
     mkdir -p models/mask2former
     wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
          -O models/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
     ```
   - **直接 URL**：`https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth`
   - 文件大小：~200MB
   - 用途：用于 GenEval 评估的目标检测和分割

## 快速开始

### 任务特定脚本（推荐）

我们为不同的任务类型和算法提供专门的脚本：

**推理任务**（代码、数学、逻辑）：
```bash
cd scripts
python eval_reasoning.py --task humaneval --model_name GSAI-ML/LLaDA-8B-Instruct --mode info-gain
python eval_reasoning.py --task math500 --model_name /model/llada --mode pc_sampler
```

**创意写作任务**：
```bash
cd scripts
python eval_writing.py --model_name GSAI-ML/LLaDA-8B-Instruct --mode info-gain --gen_length 512
```

**Info-Gain 算法**（所有任务）：
```bash
cd scripts
python eval_info_gain.py --task humaneval --model_name GSAI-ML/LLaDA-8B-Instruct --candidate_number 8
```

**基线算法**（original, pc_sampler, eb_sampler 等）：
```bash
cd scripts
python eval_baselines.py --task humaneval --model_name /model/llada --mode pc_sampler
```

**多模态任务**（文本到图像）：
```bash
cd scripts
python eval_multimodal.py --pipeline all  # 完整流程
python eval_multimodal.py --pipeline generate  # 仅生成
python eval_multimodal.py --pipeline geneval --image_dir ./output_geneval  # 仅评估
```

### 统一脚本（Eval.sh）

或者，您可以使用统一的 CLI 参数化 `Eval.sh` 脚本：

```bash
cd scripts

# 使用 Info-Gain Sampler 在 HumanEval 上运行 LLaDA
bash Eval.sh --task humaneval --model GSAI-ML/LLaDA-8B-Instruct --mode info-gain \
    --candidate_number 8 --position_temperature 0.2

# 使用 PC-Sampler 在 MATH-500 上运行 Dream
bash Eval.sh --task math500 --model /model/dream --mode pc_sampler

# 使用 Info-Gain 运行 Sudoku
bash Eval.sh --task sudoku --model /model/llada --mode info-gain --candidate_number 8

# 无少样本运行 Countdown
bash Eval.sh --task countdown --model /model/llada --mode info-gain --no_shot
```

内置任务默认值（gen_length, steps, block_length, data_path）会自动应用。任何参数都可以通过 CLI 标志覆盖。运行 `bash Eval.sh --help` 查看完整用法。

### 编程式使用

```python
from src.models import get_model_adapter
from src.generators.base import generate
from src.prompts.model_templates import apply_model_template
import torch

# 加载模型（自动检测 Dream / LLaDA / SDAR / AR）
adapter = get_model_adapter("GSAI-ML/LLaDA-8B-Instruct", device="cuda:0")
tokenizer = adapter.tokenizer
model = adapter.model

# 构建提示
query = "2 + 2 等于多少？"
messages = [{"role": "user", "content": query}]
prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
prompt = tokenizer(prompt_str)['input_ids']
prompt = torch.tensor(prompt).to("cuda:0").unsqueeze(0)

# 使用 Info-Gain Sampler 生成
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
    use_kv_cache=True,           # 启用 KV 缓存优化（可选）
)

# 解码结果
generated_text = tokenizer.batch_decode(output[:, prompt.shape[1]:], skip_special_tokens=True)[0]
print(generated_text)
```

完整示例请参见 [src/example_usage.py](src/example_usage.py)。

---

## 复现论文实验

我们提供了一个统一的脚本来一键复现论文中的所有实验：

```bash
# 运行所有实验（可能需要数小时/数天，取决于硬件）
bash scripts/reproduce_paper.sh

# 在特定设备上运行
bash scripts/reproduce_paper.sh --device cuda:1

# 跳过特定实验（逗号分隔：exp1,exp2,exp3,exp4）
bash scripts/reproduce_paper.sh --skip exp2,exp4
```

### 实验详情

脚本复现论文中的四个主要实验：

1. **实验 1：全注意力 MDM 推理（Dream-7B-Instruct）**
   - 任务：GSM8K, MATH-500, HumanEval, MBPP, Sudoku, Countdown
   - 参数：`position_temp=0.1`, `candidate_number=8`, `K=1,2`
   - 每个任务运行 5 次以获得统计显著性
   - 结果：`results/paper_reproduction/exp1_*/`

2. **实验 2：半自回归 MDM 推理**
   - 模型：SDAR-8B-Chat, TraDo-8B-Instruct
   - 任务：GSM8K, MATH-500, HumanEval, MBPP
   - 参数：`token_temp=0.7`, `block_length=16`, `K=1,2`
   - 每个配置运行 5 次
   - 结果：`results/paper_reproduction/exp2_*/`

3. **实验 3：多模态文本到图像生成（MMaDa）**
   - 评估：GenEval 和 ImageNet-512 FID
   - 参数：`position_temp=0.4`, `candidate_number=8`，50 步余弦调度器
   - 结果：`results/paper_reproduction/exp3_multimodal/`

4. **实验 4：创意写作（SDAR-8B-Chat）**
   - 温度：0.5, 1.0, 1.5
   - 参数：`K=1,2`, `position_temp=0.1`, `candidate_number=8`
   - 每个配置运行 5 次
   - 结果：`results/paper_reproduction/exp4_*/`

### 实验设置

**超参数**：
- 推理任务：位置温度 $\tau_{\text{pos}} = 0.1$，候选数量 $N = 8$，加速阈值 $\gamma = 0.8$
- 文本到图像：位置温度 $\tau_{\text{pos}} = 0.4$，候选数量 $N = 8$，50 步余弦调度器
- 评估指标：Pass@1 准确率，累积熵 $\tilde{H}$

**基线**：我们与 Uniform、Entropy、Confidence、Margin、KLASS 和 PC-Sampler 方法进行比较。

### 主要结果

#### 全注意力 MDM（Dream-7B-Instruct）

Info-Gain Sampler 在 Dream-7B-Instruct 上始终优于所有基线，通过有效的加速技术，生成时间仅增加 24%，GPU 内存使用仅增加 20%。

- **平均准确率提升**：比最佳基线高 3.6%（K=2）和 2.9%（K=1）
- **累积熵降低**：仅为最佳基线累积熵的 47.8%（K=2）和 50.8%（K=1）
- 这证实了 Info-Gain 找到更全局优化轨迹的能力

#### 半自回归 MDM

半自回归模型（SDAR-8B-Chat, TraDo-8B-Instruct）的结果进一步验证了 Info-Gain Sampler 的鲁棒性。

- **平均准确率提升**：在 K=1 设置下，SDAR-8B-Chat 和 TraDo-8B-Instruct 分别超过 20.3% 和 20.8%
- **累积熵降低**：在不同架构上显著降低（例如，SDAR 在 K=2 时从 210.3 降至 74.1）
- 值得注意的是，虽然引入非零 token 温度（$\tau_{\text{token}} = 0.7$）会降低基线性能，但 Info-Gain Sampler 仍保持显著领先

#### 文本到图像生成

在多模态设置中，Info-Gain Sampler 在对齐和保真度方面都表现出色。

- **GenEval**：最高平均分数 58.2（vs. Margin 基线的 56.3）
  - 显著改善"位置"（25.0 vs. 19.0）和"属性"（32.0 vs. 29.0）子分数
- **ImageNet-512**：显著改进
  - FID：从 43.3 降至 38.1
  - IS（Inception Score）：从 53.3 提升至 63.0

#### 创意写作

对于创意写作，Info-Gain Sampler 在各种 token 温度下始终优于所有基线。

- **平均胜率**：在所有设置和基线中为 63.1%
- **峰值性能**：在高温（$\tau_{\text{token}} = 1.5$）下，对 Entropy 基线达到 80.3% 的胜率
- 通过其前瞻机制优先考虑信息性动作，Info-Gain Sampler 对温度缩放表现出卓越的鲁棒性，有效平衡创造力和连贯性

### 消融研究

#### 累积不确定性的优化

Info-Gain Sampler 在优化累积不确定性方面显著优于基线：

1. Info-Gain 启发式平衡即时成本和未来收益，产生非线性熵增长，比贪婪 Entropy 基线更早稳定
2. 累积熵与准确率显示出强烈的**负相关**（Pearson's $r = -0.70$），验证了它作为解码质量的可靠代理

#### Info-Gain 变体的比较

我们在固定计算预算下比较 Info-Gain Sampler（$B=1$）、Info-Gain Beam Search（$B>1$）和 Best-of-N（BoN）：

1. **Info-Gain Sampler（$B=1$）** 接近 Pareto 前沿，在保持高度可并行化和避免复杂 KV 缓存管理的同时实现接近最优的结果
2. 两种 Info-Gain 变体都显著优于 **BoN**，证明通过信息增益进行全局规划优于简单地增加独立样本
3. 在给定扩展预算下增加**束大小**会产生边际不确定性降低，但会带来更高的内存开销

#### 与温度采样的兼容性

Info-Gain Sampler 在各种温度尺度上保持稳定、低轨迹不确定性，无需敏感调整。重要的是，低累积熵反映了更优化的解码，而不是模式崩溃，这通过创意写作中保持的多样性和竞争性胜率得到证明。相比之下，其他基线对温度变化高度敏感，导致解码不稳定。

---

## 许可证

MIT 许可证。

## 引用

如果您在研究中使用此代码，请引用：

```bibtex
@article{info-gain-sampler,
  title={Improve Sampling for Masked Diffusion Models via Information Gain},
  author={Kaisen Yang, Jayden Teoh, Kaicheng Yang, Yitong Zhang, Alex Lamb},
  year={2026},
  journal={arXiv preprint},
}
```

