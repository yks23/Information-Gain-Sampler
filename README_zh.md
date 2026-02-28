# Improving Sampling for Masked Diffusion Models via Information Gain

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
- [Info-Gain Sampler](#info-gain-sampler)
- [安装](#安装)
- [模型准备](#模型准备)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [项目状态](#项目状态)
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

贪婪采样器存在短视问题：它们倾向于优先解码局部不确定性低的位置，但忽略了这些决策对全局不确定性的影响。例如，在生成方程 $a \times b = c$ 时，贪婪采样器会优先解码二进制乘积 $c$（低不确定性），而不是先解决更高不确定性的因子 $a$ 和 $b$，导致错误的方程。在判断任务中，贪婪采样器会过早解码答案 token，在推理完成前就做出承诺，准确率仅为 67-73%，而 Info-Gain Sampler 通过优先考虑信息增益，能够找到更优的解码路径。

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

在每个解码步骤中，Info-Gain Sampler 遵循**三步循环**：

1. **采样**：采样多样动作的候选集 $\mathcal{C} = \{a_t^{(1)}, \dots, a_t^{(N)}\}$
2. **评估**：计算所有候选的目标 $J_{IG}(a_t \mid z_t)$（通过一次批量前向传播高效完成）
3. **转移**：选择最优动作 $a_t^* = \arg\max_{a \in \mathcal{C}} J_{IG}(a \mid z_t)$ 并执行，重复直到所有掩码位置被填充

### 高效实现

通过块级计算和 KV 缓存优化，候选评估在单次批量前向传播中并行执行。高置信度绕过机制（阈值 $\gamma$）在不确定性降低时自动触发，显著减少推理延迟。

---

## 安装

**要求**：Python >= 3.8, PyTorch >= 2.0.0（推荐支持 CUDA），支持 CUDA 的 GPU

```bash
git clone --recurse-submodules git@github.com:yks23/Information-Gain-Sampler.git
cd Information-Gain-Sampler
git submodule update --init --recursive

conda create -n info-gain python=3.10
conda activate info-gain

# 安装核心依赖
pip install -r requirements.txt

# 可选：dllm 框架集成（参考 dllm/README.md）
cd dllm/ && pip install -e . && cd ..

# 可选：多模态评估
pip install tensorflow scipy mmdet open_clip_torch clip_benchmark pandas
```

## 模型准备

### 支持的模型

| 模型 | 类型 | HuggingFace 路径 | 本地路径 |
|------|------|------------------|----------|
| **TraDo** | MDM | `Gen-Verse/TraDo-8B-Instruct` | `./model/trado` |
| **LLaDA** | MDM | `GSAI-ML/LLaDA-8B-Instruct` | `./model/llada` |
| **Dream** | MDM | `Dream-org/Dream-v0-Instruct-7B` | `./model/dream` |
| **SDAR** | MDM | `JetLM/SDAR-8B-Chat` | `./model/sdar` |
| **MMaDA** | MDM | `Gen-Verse/MMaDA-8B-MixCoT` | `./model/mmada` |

**使用方法**：

```python
from src.models import get_model_adapter

# 从本地目录加载（推荐）
adapter = get_model_adapter("llada", device="cuda:0")  # 查找 ./model/llada/

# 从 HuggingFace Hub 加载（自动下载）
adapter = get_model_adapter("GSAI-ML/LLaDA-8B-Instruct", device="cuda:0")
```

**下载模型**：

```bash
# 示例：下载 LLaDA 模型
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada
```

**多模态模型**：文本到图像任务需要 MMaDA 模型（`Gen-Verse/MMaDA-8B-MixCoT`）和 VQ 模型（`showlab/magvitv2`）。详细说明请参见 [src/benchmarks/multimodal_tasks/multimodal_eval/README.md](src/benchmarks/multimodal_tasks/multimodal_eval/README.md)。

## 数据准备

### 基线文件

PC-Sampler 需要基线频率文件（`data/baseline/reference_corpus*.json`）：

```bash
python utils/calculate_p_baseline.py \
    --input_file /path/to/reference_corpus.jsonl \
    --output_file data/baseline/reference_corpus.json \
    --model_name GSAI-ML/LLaDA-8B-Instruct
```

### 多模态数据

多模态评估需要以下文件：

- **GenEval 提示**：`src/benchmarks/multimodal_tasks/multimodal_eval/prompts/generation_prompts.txt`
- **ImageNet 参考统计**（FID 评估）：
  ```bash
  wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz \
       -O data/VIRTUAL_imagenet512.npz
  ```
- **Mask2Former 模型**（GenEval 目标检测）：
  ```bash
  mkdir -p models/mask2former
  wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
       -O models/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
  ```

## 快速开始

### 任务特定脚本

```bash
cd scripts

# 推理任务
python eval_reasoning.py --task humaneval --model_name GSAI-ML/LLaDA-8B-Instruct --mode info-gain

# 创意写作
python eval_writing.py --model_name GSAI-ML/LLaDA-8B-Instruct --mode info-gain

# 多模态任务
python eval_multimodal.py --pipeline all
```

### 统一脚本

```bash
cd scripts

# Info-Gain Sampler
bash Eval.sh --task humaneval --model GSAI-ML/LLaDA-8B-Instruct --mode info-gain \
    --candidate_number 8 --position_temperature 0.2

# 运行 bash Eval.sh --help 查看完整用法
```

### 编程式使用

```python
from src.models import get_model_adapter
from src.generators.base import generate
import torch

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
    model=model, prompt=prompt, steps=256, gen_length=256, block_length=32,
    baseline_name="../data/baseline/reference_corpus.json", temperature=0.0,
    candidate_number=8, position_temperature=0.2, heuristic='confidence',
    mask_id=adapter.mask_id, adapter=adapter, use_kv_cache=True,
)

generated_text = tokenizer.batch_decode(output[:, prompt.shape[1]:], skip_special_tokens=True)[0]
print(generated_text)
```

完整示例请参见 [src/example_usage.py](src/example_usage.py)。

---

## 项目状态

### 进行中

- Evaluation Codes 整理
- Protein Generation Quality Test
- ...

### 已完成

- arXiv 论文发布
- dllm 适配

---

## 许可证

MIT 许可证。

## 引用

如果您在研究中使用此代码，请引用：

```bibtex
@misc{yang2026improvingsamplingmaskeddiffusion,
      title={Improving Sampling for Masked Diffusion Models via Information Gain}, 
      author={Kaisen Yang and Jayden Teoh and Kaicheng Yang and Yitong Zhang and Alex Lamb},
      year={2026},
      eprint={2602.18176},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.18176}, 
}
```

