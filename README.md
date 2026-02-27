# Info-Gain Sampler for Masked Diffusion Models

[中文版 README](README_zh.md) | [English README](README.md)

A unified decoding framework for Masked Diffusion Models (MDMs) that combines trajectory planning with information-gain maximization. This repository provides an implementation of the **Info-Gain Sampler**, a flexible decoding strategy that supports multiple heuristic functions and can adapt to various generation tasks.

> **Note**: This repository is under active development for ongoing experiments and has not been fully cleaned up. The `dllm/` directory is an integration sub-repository based on [dllm](https://github.com/ZHZisZZ/dllm), containing our Info-Gain / LookUM sampler pipelines that plug into the dllm framework.

## Sampling Variants

| Variant | Objective (maximise) | Description |
|---------|---------------------|-------------|
| **Info-Gain** | $J = \text{IG}(a) - C(a)$ | Balances information gain with immediate cost |
| **LookUM** | $J = \text{IG}(a)$ | Pure future uncertainty reduction (drops $C$) |

## Repository Structure

This repo provides **two evaluation paths**:

```
Information-Gain-Sampler/
│
├── src/                            # ── Path A: standalone evaluation framework ──
│   ├── generators/                 # Info-Gain, LookUM, PC-Sampler, EB-Sampler, Fast-dLLM
│   │   ├── info_gain.py            #   ★ Info-Gain & LookUM (variant param)
│   │   └── base.py                 #   Core generate() with KV-cache, block, bypass
│   ├── prompts/                    # Task prompt templates
│   ├── utils/                      # Evaluation utilities
│   └── benchmarks/                 # Benchmark tasks (text & multimodal)
├── scripts/                        # eval.py, Eval.sh (path A entry points)
│
├── dllm/                           # ── Path B: dllm framework integration ──
│   ├── dllm/pipelines/info_gain/   #   ★ Info-Gain & LookUM sampler pipelines
│   │   ├── core.py                 #     Shared: entropy, candidates, scoring
│   │   ├── llada/                  #     LLaDA (sampler.py + eval.py)
│   │   └── dream/                  #     Dream  (sampler.py + eval.py)
│   ├── examples/info-gain/         #   Inference + eval scripts
│   ├── scripts/                    #   prepare_eval_data.py, check_eval_env.py
│   └── README.md                   #   Full setup & usage guide
│
├── data/                           # Baseline frequency files (gitignored)
├── model/                          # Model weights (gitignored)
├── requirements.txt                # Dependencies for path A (src/)
└── requirements+.txt               # Combined dependencies (src/ + dllm/)
```

### Which path to use?

| | Path A (`src/`) | Path B (`dllm/`) |
|---|---|---|
| **Models** | Requires `src/models/` adapter package | Uses dllm's model loading (HuggingFace) |
| **Eval harness** | Custom `scripts/eval.py` | `lm-evaluation-harness` (standardised) |
| **Supported models** | LLaDA, Dream, SDAR, TraDo, MMaDA | LLaDA, Dream |
| **Cache modes** | KV-cache (prefix) | none / prefix / dual |
| **Install** | `pip install -r requirements.txt` | See `dllm/README.md` |

**Recommended**: Use **Path B** (`dllm/`) for LLaDA / Dream evaluation with standardised benchmarks (GSM8K, MATH, HumanEval, MBPP).

## dllm Evaluation Quick Start

```bash
cd dllm/

# 1. Install
pip install -e .
git clone --branch dllm https://github.com/ZHZisZZ/lm-evaluation-harness.git lm-evaluation-harness
pip install -e "lm-evaluation-harness[ifeval,math]"

# 2. Download data
python scripts/prepare_eval_data.py --local_dir ./eval_data

# 3. Verify
python scripts/check_eval_env.py

# 4. Run (GSM8K example)
export HF_DATASETS_CACHE=./eval_data && export HF_DATASETS_OFFLINE=1 && export HF_ALLOW_CODE_EVAL=1

# Info-Gain on LLaDA
accelerate launch --num_processes 1 dllm/pipelines/info_gain/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 --model info_gain_llada --apply_chat_template \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,variant=info_gain,use_cache=prefix,threshold=0.9,candidate_number=8,position_temperature=0.1,max_new_tokens=256,steps=256,block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

# LookUM on LLaDA (change variant only)
accelerate launch --num_processes 1 dllm/pipelines/info_gain/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 --model info_gain_llada --apply_chat_template \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,variant=lookum,use_cache=prefix,threshold=0.9,candidate_number=8,position_temperature=0.1,max_new_tokens=256,steps=256,block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

# One-click eval (all tasks: GSM8K + MATH + HumanEval + MBPP)
bash examples/info-gain/llada/eval.sh --variant info_gain --num_gpu 4
bash examples/info-gain/llada/eval.sh --variant lookum --num_gpu 4
bash examples/info-gain/dream/eval.sh --variant info_gain --num_gpu 4
bash examples/info-gain/dream/eval.sh --variant lookum --num_gpu 4
```

See [`dllm/README.md`](dllm/README.md) for full documentation.

## Table of Contents

- [Motivation](#motivation)
  - [The Problem with Greedy Sampling](#the-problem-with-greedy-sampling)
  - [Case Studies](#case-studies)
  - [Key Observations](#key-observations)
- [Info-Gain Sampler](#info-gain-sampler)
  - [Objective Function](#objective-function)
  - [Implementation](#implementation)
  - [Efficient Implementation](#efficient-implementation)
- [Installation](#installation)
- [Model Preparation](#model-preparation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
  - [Task-Specific Scripts](#task-specific-scripts-recommended)
  - [Unified Script](#unified-script-evalsh)
  - [Programmatic Usage](#programmatic-usage)
- [Reproducing Paper Experiments](#reproducing-paper-experiments)
  - [Experiment Details](#experiment-details)
  - [Experimental Setup](#experimental-setup)
  - [Main Results](#main-results)
  - [Ablation Studies](#ablation-studies)
- [License](#license)
- [Citation](#citation)

---

## Motivation

Masked Diffusion Models (MDMs) have emerged as a powerful alternative to autoregressive models (ARMs) for discrete sequence generation. By leveraging bidirectional attention, MDMs break free from strict left-to-right generation, granting unprecedented flexibility in decoding paths. This flexibility unlocks superior performance in tasks requiring bidirectional attention, such as code infilling, biological sequence design, and long-horizon planning tasks.

However, this potential remains largely untapped due to a **training-inference mismatch**. While MDMs are trained under random masking patterns, inference entails a multi-step, order-sensitive decoding process. Navigating the large space of possible decoding orders requires a sampler that carefully selects which tokens to reveal next. Consequently, generation quality is heavily dependent on the effectiveness of the sampler.

### The Problem with Greedy Sampling

Existing samplers predominantly rely on **local certainty heuristics** (such as confidence, entropy, or margin) to greedily select the next decoding target. These methods aim to minimize error accumulation by prioritizing the most certain positions. However, such samplers are often non-robust due to the **myopia of local heuristics**: they ignore the long-term impact of current decoding decisions on future uncertainty. Consequently, they frequently prioritize tokens that appear syntactically *confident* but are semantically suboptimal, leading to error propagation and compromised generation quality.

**Key Question**: Is greedy optimization sufficient to minimize cumulative uncertainty across steps?

To quantify uncertainty throughout a generation process, we introduce **Cumulative Entropy** $\tilde{H}$ over trajectory $\tau = z_T \rightarrow z_{T-1} \rightarrow \ldots \rightarrow z_0$:

$$\tilde{H}(\tau) := \sum_{t=T}^{1} C(a_t \mid z_t)$$

where $C(a_t \mid z_t) = \sum_{\ell \in A_t} H^{(\ell)}(z_t)$ represents the sum of marginal entropy for tokens selected at step $t$.

### Case Studies

**Case Study 1: One-way Multiplication**

The model is tasked with generating an equation $a \times b = c$, where $a$ and $b$ are decimal factors and $c$ is a binary product. This task is inherently one-way: computing a product from factors is straightforward, while factoring is computationally difficult.

Two decoding paths emerge:
- **(i) Product-first**: Decode the binary product $c$ first (low uncertainty: choice from $\{0,1\}$)
- **(ii) Factor-first**: Decode the decimal factors $a$ and $b$ first (high uncertainty: 10 possible values each)

A greedy sampler mistakenly favors path (i) because binary digits exhibit lower per-token uncertainty. However, by minimizing immediate uncertainty, the greedy strategy commits to a product without fixing the factors, leading to incorrect equations and high residual uncertainty. Empirically, the greedy certainty-based sampler (Entropy) prioritizes path (i) with 73.2% probability.

Conversely, an optimal strategy would resolve the higher-uncertainty factors $a$ and $b$ first; once fixed, $c$ can be determined with nearly zero uncertainty. Our Info-Gain Sampler effectively addresses this challenge by prioritizing factor decoding with 84% probability.

**Case Study 2: Binary Judgment**

The model judges the truth of an arithmetic statement using the template: `[reasoning-masks] The answer is (Yes/No): [answer-mask]`. The answer token typically exhibits lower local uncertainty (binary choice: Yes/No), whereas reasoning steps involve much higher uncertainty.

Greedy samplers tend to decode the answer token prematurely, making a commitment before the underlying reasoning is resolved. This leads to incorrect judgments and leaves high residual uncertainty in the reasoning positions. As shown in experiments, greedy samplers achieve only 67-73% accuracy with high cumulative entropy (31.68-35.60), while an autoregressive baseline achieves 90% accuracy with lower cumulative entropy (25.19).

### Key Observations

**Observation 1**: Existing greedy certainty-based samplers often fail to find near-optimal decoding paths. An optimal decoding action should be evaluated not only by its own prediction certainty but also by the *information gain* it provides for the remainder of the generation process.

**Observation 2**: MDMs' bidirectional architecture enables efficient information gain estimation in one forward pass, bypassing expensive iterative computations. Unlike ARMs, which have a next-token bottleneck, MDMs can simultaneously evaluate the impact of any decoding action on the uncertainty of the entire sequence in a single forward pass.

These observations motivate the Info-Gain Sampler, which balances immediate certainty with information gain to prioritize globally informative decisions and yield more robust decoding trajectories.

## Info-Gain Sampler

The Info-Gain Sampler leverages the bidirectional nature of MDMs to balance the immediate uncertainty cost of a decoding decision against its expected information gain over the remaining masked positions.

### Objective Function

We first define **state uncertainty** as the average marginal entropy over the masked positions in state $z_t$:

$$\mathcal{H}(z_t) = \frac{1}{|\mathcal{M}_t|} \sum_{\ell \in \mathcal{M}_t} H^{(\ell)}(z_t)$$

The state uncertainty quantifies the information remaining to be resolved by the model and can be computed efficiently via a single forward pass.

The **information gain** of action $a_t$ is defined as the reduction in state uncertainty (equivalently, the decrease in marginal entropy over the remaining masked positions) it induces:

$$\text{IG}(a_t; z_t) := \mathcal{H}(z_t) - \mathcal{H}(z_{t-1})$$

where $z_{t-1} = \text{Apply}(z_t, a_t)$ denotes the state obtained after executing action $a_t$ from state $z_t$.

The total impact of a decoding action $a_t$ is decomposed into two components:

1. **Immediate Cost**: The uncertainty of the tokens being decoded in the current step, measured by the sum of marginal entropy over the chosen positions $C(a_t \mid z_t)$.

2. **Information Gain**: The reduction in the uncertainty over the remaining mask positions, quantified by $\text{IG}(a_t; z_t)$.

To balance these two components, we define the Info-Gain Sampler objective as:

$$J_{IG}(a_t \mid z_t) = \underbrace{\text{IG}(a_t; z_t)}_{\text{Information Gain}} - \underbrace{C(a_t \mid z_t)}_{\text{Immediate Cost}}$$

### Implementation

At each decoding step, Info-Gain Sampler follows a **three-step cycle** to determine and execute the most informative action:

1. **Sampling**: We sample a candidate set $\mathcal{C} = \{a_t^{(1)}, \dots, a_t^{(N)}\}$ of diverse actions using the *Action Sampler*. This explores the combinatorially large action space by proposing multiple potential actions.

2. **Evaluation**: We compute the objective $J_{IG}(a_t \mid z_t)$ for all candidates $a_t$ in the set $\mathcal{C}$. Crucially, this evaluation is highly efficient as it requires only a single batched forward pass to estimate the future information gain for all candidates simultaneously.

3. **Transition**: The optimal action is selected as $a_t^* = \arg\max_{a \in \mathcal{C}} J_{IG}(a \mid z_t)$. We then execute this action to transition to the next state $z_{t-1}^*$, repeating the cycle until all masked positions are filled.

**Action Sampler**: We explore the large action space by generating a candidate set $\mathcal{C}$ of size $N$ through a two-stage sampling process:
- **Token Sampling**: Drawing tokens $v_\ell$ from $p_\theta$ with token temperature $\tau_{\text{token}}$
- **Position Sampling**: Selecting positions $\ell \in \mathcal{M}_t$ using a softmax over certainty scores $\phi(\ell, z_t)$ with position temperature $\tau_{\text{pos}}$

Each candidate action $a_t = \{(\ell, v_\ell)\}$ is formed by pairing these samples, providing a diverse and high-quality set for evaluation.

### Efficient Implementation

To ensure efficiency, candidate evaluations are performed in parallel within a single batched forward pass. We further optimize the sampler by:

- **Block-wise computation**: Restricting information-gain computation to the current active block $\mathcal{B}$, which enables effective KV caching
- **High-confidence bypass**: If the maximum token probability exceeds a threshold $\gamma$, the corresponding positions are directly fixed into the action set. This hybrid approach significantly reduces inference latency while preserving planning quality

Because Info-Gain effectively reduces uncertainty during decoding, the high-confidence bypass is triggered more frequently, making the mechanism exceptionally efficient.

---

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0 (with CUDA support recommended)
- CUDA-capable GPU (recommended for model inference)

### Install Dependencies

```bash
git clone git@github.com:yks23/Information-Gain-Sampler.git
cd Information-Gain-Sampler

conda create -n info-gain python=3.10
conda activate info-gain

# Install core dependencies
pip install -r requirements.txt

# Optional: For multimodal evaluation (FID, GenEval)
pip install tensorflow scipy mmdet open_clip_torch clip_benchmark pandas
```

## Model Preparation

### Supported Models

The framework supports the following models:

| Model | Type | HuggingFace Path | Local Path |
|-------|------|------------------|------------|
| **TraDo** | MDM | `Gen-Verse/TraDo-8B-Instruct`<br>`Gen-Verse/TraDo-4B-Instruct`<br>`Gen-Verse/TraDo-8B-Thinking` | `/model/trado` |
| **LLaDA** | MDM | `GSAI-ML/LLaDA-8B-Instruct` | `/model/llada` |
| **Dream** | MDM | `Dream-org/Dream-v0-Instruct-7B` | `/model/dream` |
| **SDAR** | MDM | `JetLM/SDAR-8B-Chat` | `/model/sdar` |
| **MMaDA** | MDM | `Gen-Verse/MMaDA-8B-MixCoT` (required) | `/model/mmada` |

**Detection Rules** (checked in order):
1. **TraDo**: Contains "trado" → `TraDoAdapter`
2. **Dream**: Contains "dream" → `DreamAdapter`
3. **SDAR**: Contains "sdar" → `SDARAdapter`
4. **LLaDA**: Contains "llada" → `LLaDAAdapter`
5. **MMaDA**: Contains "mmada" → `MMaDAAdapter`
6. **Mistral**: Contains "mistral" → `MistralAdapter`
7. **Qwen**: Contains "qwen" → `QwenAdapter`
8. **Default**: Unknown models → `LLaDAAdapter` (assumes MDM model)

**Usage**:

```python
from src.models import get_model_adapter

# Load from ./model/ directory (preferred - faster, no download needed)
# Models should be placed in ./model/ directory at project root
adapter = get_model_adapter("trado", device="cuda:0")  # Looks in ./model/trado/
adapter = get_model_adapter("llada", device="cuda:0")  # Looks in ./model/llada/
adapter = get_model_adapter("dream", device="cuda:0")  # Looks in ./model/dream/
adapter = get_model_adapter("sdar", device="cuda:0")  # Looks in ./model/sdar/
adapter = get_model_adapter("mmada", device="cuda:0")  # Looks in ./model/mmada/

# Load from absolute path (also supported)
adapter = get_model_adapter("/model/llada", device="cuda:0")
adapter = get_model_adapter("/model/dream", device="cuda:0")
adapter = get_model_adapter("/model/sdar", device="cuda:0")

# Load from HuggingFace Hub (auto-downloads if not cached)
adapter = get_model_adapter("Gen-Verse/TraDo-8B-Instruct", device="cuda:0")
adapter = get_model_adapter("GSAI-ML/LLaDA-8B-Instruct", device="cuda:0")
adapter = get_model_adapter("Dream-org/Dream-v0-Instruct-7B", device="cuda:0")
adapter = get_model_adapter("JetLM/SDAR-8B-Chat", device="cuda:0")
adapter = get_model_adapter("Gen-Verse/MMaDA-8B-MixCoT", device="cuda:0")
```

**Model Directory Structure**:

Models should be organized in the `./model/` directory at the project root:

```
Information-Gain-Sampler/
├── model/                    # Model directory (./model/, git-ignored)
│   ├── trado/                # TraDo model (TraDo-8B-Instruct, TraDo-4B-Instruct, etc.)
│   │   ├── config.json
│   │   ├── model-*.safetensors
│   │   └── tokenizer.json
│   ├── llada/                # LLaDA model
│   │   ├── config.json
│   │   ├── model-*.safetensors
│   │   └── tokenizer.json
│   ├── dream/                # Dream model
│   │   ├── config.json
│   │   └── ...
│   ├── sdar/                 # SDAR model
│   │   └── ...
│   ├── mmada/                # MMaDA model (for multimodal tasks)
│   │   ├── config.json
│   │   └── ...
│   └── ...                   # Other models
```

**Note**: For local models, ensure the model directory contains:
- `config.json` or `config.yaml` - Model configuration
- Model weights (`.safetensors` or `.bin` files) - Model parameters
- `tokenizer.json` and related tokenizer files - Tokenizer configuration

**Model Download Sizes** (approximate):
- TraDo-8B-Instruct: ~16GB
- LLaDA-8B-Instruct: ~16GB
- Dream-v0-Instruct-7B: ~14GB
- MMaDA-8B-MixCoT: ~16GB (required for text-to-image generation)

**Download Models from HuggingFace**:

```bash
# TraDo models
huggingface-cli download Gen-Verse/TraDo-8B-Instruct --local-dir ./model/trado

# LLaDA
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada

# Dream
huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir ./model/dream

# SDAR
huggingface-cli download JetLM/SDAR-8B-Chat --local-dir ./model/sdar

# MMaDA (for multimodal tasks - MixCoT version required)
huggingface-cli download Gen-Verse/MMaDA-8B-MixCoT --local-dir ./model/mmada
```

### Multimodal Models

For text-to-image evaluation with MMaDA, you need two models:

1. **MMaDA model** (main text-to-image model):
   - **HuggingFace**: `Gen-Verse/MMaDA-8B-MixCoT` (**required** - use MixCoT version for text-to-image generation)
   - **Direct download**:
     ```bash
     # Using huggingface-cli (recommended)
     huggingface-cli download Gen-Verse/MMaDA-8B-MixCoT --local-dir ./model/mmada
     ```
   - Local path: `./model/mmada/` (preferred) or `./mmada-mix/` (fallback)
   - Purpose: Generates images from text prompts
   - Size: ~8B parameters, ~16GB disk space
   - **Note**: MMaDA-8B-MixCoT is the required version for text-to-image generation tasks. The Base version is not suitable for this evaluation framework.

2. **VQ model** (vector quantization model - MAGVITv2):
   - **HuggingFace**: `showlab/magvitv2`
   - **Direct download**:
     ```bash
     # Using huggingface-cli (recommended)
     huggingface-cli download showlab/magvitv2 --local-dir ./model/magvitv2
     ```
   - Local path: `./model/magvitv2/` (preferred) or `./magvitv2/` (fallback)
   - Purpose: Encodes/decodes images to/from discrete tokens
   - Size: ~600M parameters, ~1.2GB disk space

**Model Loading Priority**:
1. `model/mmada/` and `model/magvitv2/` (preferred)
2. Project root: `mmada-mix/` and `magvitv2/` (fallback)
3. Config file paths (last resort)

**Setup Instructions**:
- Download models using HuggingFace `from_pretrained()` or `huggingface-cli download`
- Place models in the `./model/` directory (preferred) or project root directory
- Ensure sufficient disk space: ~20GB for both models
- **Note**: Models will be automatically downloaded on first use if using HuggingFace paths directly

See [src/benchmarks/multimodal_tasks/multimodal_eval/README.md](src/benchmarks/multimodal_tasks/multimodal_eval/README.md) for detailed setup and configuration instructions.

## Data Preparation

### Baseline Files

Baseline frequency files (`data/baseline/reference_corpus*.json`) are used for PC-Sampler heuristic. To generate them:

```bash
# Generate baseline from a reference corpus
python utils/calculate_p_baseline.py \
    --input_file /path/to/reference_corpus.jsonl \
    --output_file data/baseline/reference_corpus.json \
    --model_name GSAI-ML/LLaDA-8B-Instruct  # or your model name
```

**What the script does:**
1. Loads the reference corpus (JSONL format)
2. Tokenizes all text using the specified model's tokenizer
3. Calculates token frequency distribution across the corpus
4. Saves the baseline distribution as a JSON file with format:
   ```json
   {
     "num_token": <total_tokens>,
     "p_baseline_dict": {<token_id>: <frequency>, ...}
   }
   ```

**When to generate separate baselines:**
- Different tokenizers produce different token IDs → Generate separate baselines for Dream and LLaDA models
- Different corpus domains → Generate domain-specific baselines (e.g., code corpus for code tasks)
- Different model vocabularies → Each model family needs its own baseline

**Recommended baseline corpus**: Use a large, diverse text corpus (e.g., Wikipedia, Common Crawl subset) that matches your task domain.

### Multimodal Data

For multimodal evaluation, you need the following files:

1. **GenEval prompts**: 
   - Location: `src/benchmarks/multimodal_tasks/multimodal_eval/prompts/generation_prompts.txt`
   - Contains: Text prompts for text-to-image generation evaluation
   - Format: One prompt per line
   - Metadata: `src/benchmarks/multimodal_tasks/multimodal_eval/prompts/evaluation_metadata.jsonl` (JSONL format with prompt metadata)

2. **ImageNet reference statistics** (for FID evaluation):
   - Location: `data/VIRTUAL_imagenet512.npz`
   - Contains: Pre-computed InceptionV3 features for ImageNet 512×512 images
   - Purpose: Reference distribution for FID calculation
   - **Download**:
     ```bash
     # Download to data/ directory
     wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz \
          -O data/VIRTUAL_imagenet512.npz
     ```
   - **Direct URL**: `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz`
   - File size: ~200MB (compressed)

3. **Mask2Former model** (for GenEval object detection):
   - Download location: `models/mask2former/` (relative to project root)
   - Model file: `mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth`
   - **Download**: 
     ```bash
     # Option 1: Use mmdetection's model zoo (recommended)
     mkdir -p models/mask2former
     wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
          -O models/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
     ```
   - **Direct URL**: `https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth`
   - File size: ~200MB
   - Purpose: Object detection and segmentation for GenEval evaluation

## Quick Start

### Task-Specific Scripts (Recommended)

We provide specialized scripts for different task types and algorithms:

**Reasoning Tasks** (code, math, logic):
```bash
cd scripts
python eval_reasoning.py --task humaneval --model_name GSAI-ML/LLaDA-8B-Instruct --mode info-gain
python eval_reasoning.py --task math500 --model_name /model/llada --mode pc_sampler
```

**Creative Writing Task**:
```bash
cd scripts
python eval_writing.py --model_name GSAI-ML/LLaDA-8B-Instruct --mode info-gain --gen_length 512
```

**Info-Gain Algorithm** (all tasks):
```bash
cd scripts
python eval_info_gain.py --task humaneval --model_name GSAI-ML/LLaDA-8B-Instruct --candidate_number 8
```

**Baseline Algorithms** (original, pc_sampler, eb_sampler, etc.):
```bash
cd scripts
python eval_baselines.py --task humaneval --model_name /model/llada --mode pc_sampler
```

**Multimodal Tasks** (text-to-image):
```bash
cd scripts
python eval_multimodal.py --pipeline all  # Full pipeline
python eval_multimodal.py --pipeline generate  # Only generation
python eval_multimodal.py --pipeline geneval --image_dir ./output_geneval  # Only evaluation
```

### Unified Script (Eval.sh)

Alternatively, you can use the unified CLI-parameterised `Eval.sh` script:

```bash
cd scripts

# LLaDA on HumanEval with Info-Gain Sampler
bash Eval.sh --task humaneval --model GSAI-ML/LLaDA-8B-Instruct --mode info-gain \
    --candidate_number 8 --position_temperature 0.2

# Dream on MATH-500 with PC-Sampler
bash Eval.sh --task math500 --model /model/dream --mode pc_sampler

# Sudoku with Info-Gain
bash Eval.sh --task sudoku --model /model/llada --mode info-gain --candidate_number 8

# Countdown without few-shot
bash Eval.sh --task countdown --model /model/llada --mode info-gain --no_shot
```

Built-in task defaults (gen_length, steps, block_length, data_path) are applied automatically. Any parameter can be overridden via CLI flags. Run `bash Eval.sh --help` for full usage.

### Programmatic Usage

```python
from src.models import get_model_adapter
from src.generators.base import generate
from src.prompts.model_templates import apply_model_template
import torch

# Load model (auto-detects Dream / LLaDA / SDAR / AR)
adapter = get_model_adapter("GSAI-ML/LLaDA-8B-Instruct", device="cuda:0")
tokenizer = adapter.tokenizer
model = adapter.model

# Build prompt
query = "What is 2 + 2?"
messages = [{"role": "user", "content": query}]
prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
prompt = tokenizer(prompt_str)['input_ids']
prompt = torch.tensor(prompt).to("cuda:0").unsqueeze(0)

# Generate with Info-Gain Sampler
output = generate(
    model=model,
    prompt=prompt,
    steps=256,
    gen_length=256,
    block_length=32,
    baseline_name="../data/baseline/reference_corpus.json",
    temperature=0.0,
    candidate_number=8,          # >1 enables Info-Gain mode
    position_temperature=0.2,    # >0 enables position sampling
    heuristic='confidence',
    mask_id=adapter.mask_id,
    adapter=adapter,              # Model adapter (auto-detects model-specific behavior)
    use_kv_cache=True,           # Enable KV-cache optimization (optional)
)

# Decode result
generated_text = tokenizer.batch_decode(output[:, prompt.shape[1]:], skip_special_tokens=True)[0]
print(generated_text)
```

For a complete example, see [src/example_usage.py](src/example_usage.py).

---

## Reproducing Paper Experiments

We provide a unified script to reproduce all experiments from the paper in one command:

```bash
# Run all experiments (may take several hours/days depending on hardware)
bash scripts/reproduce_paper.sh

# Run on specific device
bash scripts/reproduce_paper.sh --device cuda:1

# Skip specific experiments (comma-separated: exp1,exp2,exp3,exp4)
bash scripts/reproduce_paper.sh --skip exp2,exp4
```

### Experiment Details

The script reproduces four main experiments from the paper:

1. **Experiment 1: Fully-Attention MDM Reasoning (Dream-7B-Instruct)**
   - Tasks: GSM8K, MATH-500, HumanEval, MBPP, Sudoku, Countdown
   - Parameters: `position_temp=0.1`, `candidate_number=8`, `K=1,2`
   - Each task runs 5 times for statistical significance
   - Results: `results/paper_reproduction/exp1_*/`

2. **Experiment 2: Semi-Autoregressive MDM Reasoning**
   - Models: SDAR-8B-Chat, TraDo-8B-Instruct
   - Tasks: GSM8K, MATH-500, HumanEval, MBPP
   - Parameters: `token_temp=0.7`, `block_length=16`, `K=1,2`
   - Each configuration runs 5 times
   - Results: `results/paper_reproduction/exp2_*/`

3. **Experiment 3: Multimodal Text-to-Image Generation (MMaDa)**
   - Evaluations: GenEval and ImageNet-512 FID
   - Parameters: `position_temp=0.4`, `candidate_number=8`, 50-step cosine scheduler
   - Results: `results/paper_reproduction/exp3_multimodal/`

4. **Experiment 4: Creative Writing (SDAR-8B-Chat)**
   - Temperatures: 0.5, 1.0, 1.5
   - Parameters: `K=1,2`, `position_temp=0.1`, `candidate_number=8`
   - Each configuration runs 5 times
   - Results: `results/paper_reproduction/exp4_*/`

### Experimental Setup

**Hyperparameters**:
- Reasoning tasks: Position temperature $\tau_{\text{pos}} = 0.1$, candidate number $N = 8$, acceleration threshold $\gamma = 0.8$
- Text-to-image: Position temperature $\tau_{\text{pos}} = 0.4$, candidate number $N = 8$, 50-step cosine scheduler
- Evaluation metrics: Pass@1 accuracy, cumulative entropy $\tilde{H}$

**Baselines**: We compare against Uniform, Entropy, Confidence, Margin, KLASS, and PC-Sampler methods.

### Main Results

#### Full-Attention MDM (Dream-7B-Instruct)

Info-Gain Sampler consistently outperforms all baselines on Dream-7B-Instruct with only a marginal increase in generation time (+24%) and GPU memory usage (+20%) via effective acceleration techniques.

- **Average accuracy improvement**: 3.6% (K=2) and 2.9% (K=1) over the best-performing baselines
- **Cumulative entropy reduction**: Only 47.8% (K=2) and 50.8% (K=1) of the best-performing baseline's cumulative entropy
- This confirms Info-Gain's ability to find more globally optimized trajectories

#### Semi-Autoregressive MDM

Results for Semi-AR models (SDAR-8B-Chat, TraDo-8B-Instruct) further validate the robustness of Info-Gain Sampler.

- **Average accuracy improvement**: Over 20.3% (SDAR-8B-Chat) and 20.8% (TraDo-8B-Instruct) at K=1 settings
- **Cumulative entropy reduction**: Significant reduction across different architectures (e.g., SDAR from 210.3 to 74.1 at K=2)
- Notably, while introducing non-zero token temperature ($\tau_{\text{token}} = 0.7$) degrades baseline performance, Info-Gain Sampler maintains a substantial lead

#### Text-to-Image Generation

In multimodal settings, Info-Gain Sampler excels in both alignment and fidelity.

- **GenEval**: Highest average score of 58.2 (vs. 56.3 for Margin baseline)
  - Significantly improves "positional" (25.0 vs. 19.0) and "attribute" (32.0 vs. 29.0) sub-scores
- **ImageNet-512**: Substantial improvements
  - FID: from 43.3 to 38.1
  - IS (Inception Score): from 53.3 to 63.0

#### Creative Writing

For creative writing, Info-Gain Sampler consistently outperforms all baselines across various token temperatures.

- **Average win-rate**: 63.1% across all settings and baselines
- **Peak performance**: At high temperature ($\tau_{\text{token}} = 1.5$), achieves 80.3% win-rate against Entropy baseline
- By prioritizing informative actions through its lookahead mechanism, Info-Gain Sampler exhibits superior robustness to temperature scaling, effectively balancing creativity and coherence

### Ablation Studies

#### Optimization of Cumulative Uncertainty

Info-Gain Sampler significantly outperforms baselines in optimizing cumulative uncertainty:

1. The Info-Gain heuristic balances immediate cost with future gains, yielding non-linear entropy growth that stabilizes earlier than the greedy Entropy baseline
2. Cumulative entropy shows a strong **negative correlation with accuracy** (Pearson's $r = -0.70$), validating it as a reliable proxy for decoding quality

#### Comparison of Info-Gain Variants

We compare Info-Gain Sampler ($B=1$), Info-Gain Beam Search ($B>1$), and Best-of-N (BoN) under a fixed computational budget:

1. **Info-Gain Sampler ($B=1$)** performs near the Pareto frontier, achieving near-optimal results while remaining highly parallelizable and avoiding complex KV-cache management
2. Both Info-Gain variants significantly outperform **BoN**, proving that global planning via information gain is superior to simply increasing independent samples
3. Increasing **Beam Size** under given expansion budget yields marginal uncertainty reduction but incurs higher memory overhead

#### Compatibility with Temperature Sampling

Info-Gain Sampler maintains stable, low trajectory uncertainty across various temperature scales without sensitive tuning. Importantly, low cumulative entropy reflects more optimized decoding rather than mode collapse, as evidenced by preserved diversity and competitive win rates in creative writing. In contrast, other baselines are highly sensitive to temperature changes, leading to decoding instability.

---

## License

MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{info-gain-sampler,
  title={Improve Sampling for Masked Diffusion Models via Information Gain},
  author={Kaisen Yang, Jayden Teoh, Kaicheng Yang, Yitong Zhang, Alex Lamb},
  year={2026},
  journal={arXiv preprint},
}
```
