# Improving Sampling for Masked Diffusion Models via Information Gain

[中文版 README](README_zh.md) | [English README](README.md) | [论文](https://arxiv.org/abs/2602.18176)

一个统一的解码框架，用于掩码扩散模型（MDMs），通过平衡即时确定性与长期信息增益，实现更鲁棒的生成质量。

---

## 快速上手

**第一步 — 安装并下载模型**

```bash
git clone --recurse-submodules git@github.com:yks23/Information-Gain-Sampler.git
cd Information-Gain-Sampler
conda create -n info-gain python=3.10 && conda activate info-gain
pip install -r requirements.txt

# 下载 LLaDA（或替换为 dream / sdar / trado，见下方模型列表）
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada
```

**第二步 — 使用预设 config 一键运行**

```bash
# GSM8K + Info-Gain 采样器
python run.py --config configs/gsm8k_info_gain.yaml

# 替换模型无需修改 config
python run.py --config configs/gsm8k_info_gain.yaml --model dream
python run.py --config configs/gsm8k_info_gain.yaml --model sdar

# 快速冒烟测试（仅 2 条样本）
python run.py --config configs/gsm8k_info_gain.yaml --max_samples 2
```

**可用 config**（位于 `configs/`）：

| Config | 任务 | 采样器 |
|--------|------|--------|
| `gsm8k_info_gain.yaml` | GSM8K | Info-Gain |
| `math500_info_gain.yaml` | MATH-500 | Info-Gain |
| `humaneval_info_gain.yaml` | HumanEval | Info-Gain |
| `mbpp_info_gain.yaml` | MBPP | Info-Gain |
| `writing_info_gain.yaml` | 创意写作 | Info-Gain |
| `gsm8k_original.yaml` | GSM8K | 贪婪基线 |

任何 config 键都可以在命令行覆盖：`python run.py --config X.yaml --key value`。

```bash
python run.py --list_configs   # 列出所有可用 config
```

---

## 目录

- [动机](#动机)
- [Info-Gain Sampler](#info-gain-sampler)
- [模型](#模型)
- [安装](#安装)
- [进阶用法](#进阶用法)
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

- **并行候选评估**：所有候选动作在单次批量前向传播中同时评估，充分利用 MDMs 的双向架构
- **KV 缓存支持**：可选的前缀缓存和双缓存模式，加速推理（多模态任务默认禁用）
- **动态阈值**：高置信度绕过机制（阈值 $\gamma$）在不确定性充分降低时自动触发，显著减少推理延迟
- **无外部依赖**：核心 Info-Gain 函数自包含，避免复杂的依赖链

---

## 模型

| 模型 | HuggingFace 路径 | 本地别名 |
|------|-----------------|----------|
| **LLaDA** | `GSAI-ML/LLaDA-8B-Instruct` | `llada` |
| **Dream** | `Dream-org/Dream-v0-Instruct-7B` | `dream` |
| **SDAR** | `JetLM/SDAR-8B-Chat` | `sdar` |
| **TraDo** | `Gen-Verse/TraDo-8B-Instruct` | `trado` |
| **MMaDA** | `Gen-Verse/MMaDA-8B-MixCoT` | `mmada` |

```bash
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada
huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir ./model/dream
huggingface-cli download JetLM/SDAR-8B-Chat --local-dir ./model/sdar
huggingface-cli download Gen-Verse/TraDo-8B-Instruct --local-dir ./model/trado
```

## 安装

**要求**：Python ≥ 3.10，PyTorch ≥ 2.0（需要 CUDA）

```bash
git clone --recurse-submodules git@github.com:yks23/Information-Gain-Sampler.git
cd Information-Gain-Sampler
conda create -n info-gain python=3.10 && conda activate info-gain
pip install -r requirements.txt

# 可选：dllm 框架集成（accelerate 多 GPU 评估）
cd dllm/ && pip install -e . && cd ..
```

## 进阶用法

<details>
<summary>所有 run.py 参数</summary>

Config 键 / 命令行参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task` | — | `gsm8k` `math500` `humaneval` `mbpp` `creativity_writing` `sudoku` `countdown` |
| `model` | — | 本地别名或 HuggingFace 路径 |
| `mode` | `info-gain` | `info-gain` `original` `pc_sampler` `eb_sampler` `fast_dllm` |
| `variant` | `info_gain` | `info_gain` 或 `lookum` |
| `candidate_number` | `8` | 每步评估的候选动作数 |
| `position_temperature` | `0.2` | 位置采样多样性温度 |
| `threshold` | `0.8` | 高置信度跳过阈值 |
| `use_cache` | `prefix` | `none` `prefix` `dual` |
| `temperature` | `0.0` | Token 采样温度 |
| `gen_length` | `256` | 生成 token 数 |
| `steps` | `256` | 去掩码步数 |
| `block_length` | `32` | 双向注意力块大小 |
| `max_samples` | `null` | 限制样本数（快速测试用） |

</details>

<details>
<summary>多 GPU 评估</summary>

```bash
# 多 GPU（eval_multigpu.py）
python scripts/eval_multigpu.py \
    --task gsm8k --model_name llada --num_gpus 4 \
    --mode info-gain --candidate_number 8 \
    --position_temperature 0.2 --threshold 0.8 \
    --use_cache prefix --gen_length 256 --steps 256

# 或通过 dllm/accelerate（推荐大规模评估）
cd dllm
accelerate launch --num_processes 4 \
    dllm/pipelines/info_gain/llada/eval.py \
    --tasks "gsm8k" --model "llada" --apply_chat_template \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,use_cache=prefix,threshold=0.8,candidate_number=8,position_temperature=0.2,max_new_tokens=256,steps=256,block_size=32"
```
</details>

<details>
<summary>dllm 框架（SDAR / TraDo）</summary>

```bash
cd dllm

# SDAR
accelerate launch --num_processes 1 \
    dllm/pipelines/info_gain/sdar/eval.py \
    --tasks "gsm8k" --model "sdar" --apply_chat_template \
    --model_args "pretrained=JetLM/SDAR-8B-Chat,use_cache=prefix,threshold=0.8,candidate_number=8,position_temperature=0.2,max_new_tokens=256,steps=256,block_size=32"

# TraDo
accelerate launch --num_processes 1 \
    dllm/pipelines/info_gain/sdar/eval.py \
    --tasks "gsm8k" --model "trado" --apply_chat_template \
    --model_args "pretrained=Gen-Verse/TraDo-8B-Instruct,use_cache=prefix,threshold=0.8,candidate_number=8,position_temperature=0.2,max_new_tokens=256,steps=256,block_size=32"
```
</details>

<details>
<summary>多模态（MMaDA 文本到图像）</summary>

需要 Python 3.11 和独立 conda 环境：
```bash
conda create -n mmada python=3.11
conda activate mmada
pip install einops diffusers jaxtyping tensorflow scipy mmdet open_clip_torch clip_benchmark pandas
pip install -r requirements.txt
```

```bash
cd scripts

# 完整流程：生成 + 评估
python eval_multimodal.py --pipeline all \
    --mmada_model_path ./model/mmada \
    --vq_model_path ./model/magvitv2 \
    --conda_env mmada

# 仅生成图像
python eval_multimodal.py --pipeline generate \
    --mmada_model_path ./model/mmada \
    --vq_model_path ./model/magvitv2 \
    --conda_env mmada

# 仅评估已有图像（无需 conda env）
python eval_multimodal.py --pipeline geneval --image_dir ./output_geneval
```
</details>

<details>
<summary>PC-Sampler 数据准备</summary>

```bash
python utils/calculate_p_baseline.py \
    --input_file /path/to/reference_corpus.jsonl \
    --output_file data/baseline/reference_corpus.json \
    --model_name GSAI-ML/LLaDA-8B-Instruct
```
</details>

---

## 项目状态

### 已完成

- 发布 arXiv 论文（[arXiv:2602.18176](https://arxiv.org/abs/2602.18176)）
- dllm 框架集成，支持完整缓存功能（LLaDA、Dream、SDAR、TraDo）
- 独立 `InfoGainSampler`，无需 dllm 依赖
- 预设实验 config，一条命令复现论文结果
- 统一 `run.py` 入口

### 进行中

- Beam search 功能整理
- 蛋白质生成质量测试

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

