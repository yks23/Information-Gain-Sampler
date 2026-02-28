# Improving Sampling for Masked Diffusion Models via Information Gain

[中文版 README](README_zh.md) | [English README](README.md)

A unified decoding framework for Masked Diffusion Models (MDMs) that combines trajectory planning with information-gain maximization. This repository provides an implementation of the **Info-Gain Sampler**, a flexible decoding strategy that balances immediate certainty with long-term information gain to achieve more robust generation quality.

**Key Features:**
- **Information-Gain Optimization**: Maximizes information gain while minimizing immediate uncertainty cost
- **Efficient Parallel Evaluation**: Batched candidate evaluation in a single forward pass
- **Multiple Cache Modes**: Supports no cache, prefix cache, and dual cache for different speed/quality trade-offs
- **Unified Interface**: Works seamlessly with LLaDA, Dream, MMaDA, and other MDM architectures
- **Task-Agnostic**: Applicable to text generation, code completion, and multimodal tasks

> **Note**: This repository is under active development. We also provide a production-ready adaptation for the [dllm](https://github.com/ZHZisZZ/dllm) framework. The `dllm/` directory is a Git submodule containing our Info-Gain sampler implementation with full dllm integration.
> 
> **Beam Search**: The beam search feature is not yet fully organized and may have incomplete implementations. We recommend using the Info-Gain sampler (single candidate or with `candidate_number > 1`) for production use.

**Initialize submodule:**

```bash
# Clone repository with submodules
git clone --recurse-submodules git@github.com:yks23/Information-Gain-Sampler.git

# Or if already cloned, initialize submodule
git submodule update --init --recursive
```

Then navigate to the `dllm/` directory and refer to [`dllm/README.md`](dllm/README.md) for using the dllm framework integration.

## Table of Contents

- [Motivation](#motivation)
- [Info-Gain Sampler](#info-gain-sampler)
- [Installation](#installation)
- [Model Preparation](#model-preparation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Project Status](#project-status)
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

Greedy samplers suffer from myopia: they tend to prioritize decoding positions with low local uncertainty, but ignore the impact of these decisions on global uncertainty. For example, when generating the equation $a \times b = c$, greedy samplers prioritize decoding the binary product $c$ (low uncertainty) rather than first resolving the higher-uncertainty factors $a$ and $b$, leading to incorrect equations. In judgment tasks, greedy samplers decode answer tokens prematurely, making commitments before reasoning is complete, achieving only 67-73% accuracy, while Info-Gain Sampler finds better decoding paths by prioritizing information gain.

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

At each decoding step, Info-Gain Sampler follows a **three-step cycle**:

1. **Sampling**: Sample a candidate set $\mathcal{C} = \{a_t^{(1)}, \dots, a_t^{(N)}\}$ of diverse actions
2. **Evaluation**: Compute the objective $J_{IG}(a_t \mid z_t)$ for all candidates (efficiently done via a single batched forward pass)
3. **Transition**: Select the optimal action $a_t^* = \arg\max_{a \in \mathcal{C}} J_{IG}(a \mid z_t)$ and execute, repeating until all masked positions are filled

### Efficient Implementation

- **Parallel Candidate Evaluation**: All candidates are evaluated simultaneously in a single batched forward pass, leveraging MDMs' bidirectional architecture
- **KV-Cache Support**: Optional prefix cache and dual cache modes for faster inference (disabled by default for multimodal tasks)
- **Dynamic Thresholding**: High-confidence bypass mechanism (threshold $\gamma$) automatically triggers when uncertainty is sufficiently reduced, significantly reducing inference latency
- **No External Dependencies**: Core Info-Gain functions are self-contained, avoiding complex dependency chains

---

## Installation

**Requirements**: Python >= 3.8, PyTorch >= 2.0.0 (with CUDA support recommended), CUDA-capable GPU

```bash
git clone --recurse-submodules git@github.com:yks23/Information-Gain-Sampler.git
cd Information-Gain-Sampler
git submodule update --init --recursive

conda create -n info-gain python=3.10
conda activate info-gain

# Install core dependencies
pip install -r requirements.txt

# Optional: dllm framework integration (see dllm/README.md)
cd dllm/ && pip install -e . && cd ..

# Optional: multimodal evaluation
pip install tensorflow scipy mmdet open_clip_torch clip_benchmark pandas
```

## Model Preparation

### Supported Models

| Model | Type | HuggingFace Path | Local Path |
|-------|------|------------------|------------|
| **TraDo** | MDM | `Gen-Verse/TraDo-8B-Instruct` | `./model/trado` |
| **LLaDA** | MDM | `GSAI-ML/LLaDA-8B-Instruct` | `./model/llada` |
| **Dream** | MDM | `Dream-org/Dream-v0-Instruct-7B` | `./model/dream` |
| **SDAR** | MDM | `JetLM/SDAR-8B-Chat` | `./model/sdar` |
| **MMaDA** | MDM | `Gen-Verse/MMaDA-8B-MixCoT` | `./model/mmada` |

**Usage**:

```python
from src.models import get_model_adapter

# Load from local directory (recommended)
adapter = get_model_adapter("llada", device="cuda:0")  # Looks in ./model/llada/

# Load from HuggingFace Hub (auto-downloads)
adapter = get_model_adapter("GSAI-ML/LLaDA-8B-Instruct", device="cuda:0")
```

**Download Models**:

```bash
# Example: Download LLaDA model
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada
```

**Multimodal Models**: Text-to-image tasks require MMaDA model (`Gen-Verse/MMaDA-8B-MixCoT`) and VQ model (`showlab/magvitv2`). See [src/benchmarks/multimodal_tasks/multimodal_eval/README.md](src/benchmarks/multimodal_tasks/multimodal_eval/README.md) for detailed instructions.

## Data Preparation

### Baseline Files

PC-Sampler requires baseline frequency files (`data/baseline/reference_corpus*.json`):

```bash
python utils/calculate_p_baseline.py \
    --input_file /path/to/reference_corpus.jsonl \
    --output_file data/baseline/reference_corpus.json \
    --model_name GSAI-ML/LLaDA-8B-Instruct
```

### Multimodal Data

Multimodal evaluation requires the following files:

- **GenEval prompts**: `src/benchmarks/multimodal_tasks/multimodal_eval/prompts/generation_prompts.txt`
- **ImageNet reference statistics** (for FID evaluation):
  ```bash
  wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz \
       -O data/VIRTUAL_imagenet512.npz
  ```
- **Mask2Former model** (for GenEval object detection):
  ```bash
  mkdir -p models/mask2former
  wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
       -O models/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
  ```

## Quick Start

### Task-Specific Scripts

```bash
cd scripts

# Reasoning tasks (math, code, etc.)
# HumanEval (code completion)
python eval_reasoning.py --task humaneval \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --threshold 0.8 \
    --gen_length 512 \
    --steps 256

# MATH-500 (mathematical reasoning)
python eval_reasoning.py --task math500 \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --threshold 0.8 \
    --gen_length 512 \
    --steps 256

# GSM8K (grade school math)
python eval_reasoning.py --task gsm8k \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --threshold 0.8

# Creative writing tasks
python eval_writing.py \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --threshold 0.8 \
    --gen_length 512 \
    --steps 512 \
    --temperature 0.7

# Multimodal tasks (text-to-image)
# Full pipeline: generation + evaluation
python eval_multimodal.py --pipeline all \
    --mmada_model_path ./model/mmada \
    --vq_model_path ./model/magvitv2

# Only generate images
python eval_multimodal.py --pipeline generate \
    --mmada_model_path ./model/mmada \
    --vq_model_path ./model/magvitv2

# Only evaluate existing images
python eval_multimodal.py --pipeline geneval --image_dir ./output_geneval
```

### Unified Script

```bash
cd scripts

# Info-Gain Sampler with cache support
bash Eval.sh \
    --task humaneval \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --use_cache prefix \
    --threshold 0.8 \
    --gen_length 512 \
    --steps 256

# With dual cache (faster but requires more memory)
bash Eval.sh \
    --task math500 \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --use_cache dual \
    --threshold 0.8

# Run bash Eval.sh --help for full usage
```

### dllm Framework Evaluation

For production use, we recommend the dllm integration:

```bash
cd dllm
accelerate launch --num_processes 4 \
    dllm/pipelines/info_gain/llada/eval.py \
    --tasks "gsm8k" \
    --model "llada" \
    --apply_chat_template \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,use_cache=prefix,threshold=0.8,candidate_number=8,position_temperature=0.2,max_new_tokens=256,steps=256,block_size=32"
```

## Project Status

### Ongoing

- Evaluation code organization and cleanup
- Protein generation quality testing
- Performance optimization for large-scale evaluation
- Beam search feature organization and refactoring

### Done

- Published arXiv paper ([arXiv:2602.18176](https://arxiv.org/abs/2602.18176))
- dllm framework integration with full cache support
- Info-Gain implementation for text and multimodal tasks
- Unified evaluation pipeline for reasoning, writing, and multimodal tasks

---

## License

MIT License.

## Citation

If you use this code in your research, please cite:

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
