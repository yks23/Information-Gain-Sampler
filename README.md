# Info-Gain Sampler for Masked Diffusion Models

[中文版 README](README_zh.md) | [English README](README.md)

A unified decoding framework for Masked Diffusion Models (MDMs) that combines trajectory planning with information-gain maximization. This repository provides an implementation of the **Info-Gain Sampler**, a flexible decoding strategy that supports multiple heuristic functions and can adapt to various generation tasks.

## Overview

The Info-Gain Sampler extends the PC-Sampler framework with information-theoretic action selection. It supports:

- **Multiple heuristic functions**: confidence, PC-value, negative entropy, margin, and uniform sampling
- **Flexible trajectory control**: position-aware weighting and stochastic position sampling
- **Unified interface**: all baseline methods (entropy, margin, confidence, etc.) are implemented as special cases of the base `generate` function
- **Multiple models**: TraDo, LLaDA, Dream, SDAR, MMaDA, and auto-regressive baselines (Mistral, Qwen)
- **KV-Cache optimization**: Automatic KV-cache support for all MDM models to accelerate generation
- **Multiple evaluation tasks**: HumanEval, MBPP, MATH-500, GSM8K, GPQA, Sudoku, Countdown, Creativity Writing
- **Multimodal evaluation**: GenEval for text-to-image generation (FID, CLIP Score, etc.)

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

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0 (with CUDA support recommended)
- CUDA-capable GPU (recommended for model inference)

### Install Dependencies

```bash
git clone <repository-url>
cd Uncode-new

# Install core dependencies
pip install -r requirements.txt

# Optional: For multimodal evaluation (FID, GenEval)
pip install tensorflow scipy mmdet open_clip_torch clip_benchmark pandas
```

### Verify Installation

```bash
python -c "import torch; import transformers; print('Installation successful!')"
```

## Data Preparation

### Text Task Datasets

All text task datasets should be placed in the `data/` directory:

```
data/
├── humaneval.jsonl          # HumanEval dataset
├── mbpp.jsonl               # MBPP dataset (or sanitized-mbpp.json)
├── math500.jsonl            # MATH-500 dataset
├── gsm8k.jsonl              # GSM8K dataset
├── gpqa.jsonl               # GPQA dataset
├── countdown.jsonl          # Countdown dataset
├── sudoku.csv               # Sudoku dataset
└── baseline/                # Baseline frequency files
    ├── reference_corpus.json
    ├── reference_corpus_dream.json
    └── reference_corpus_llada.json
```

**Dataset Sources and Download Instructions:**
- **HumanEval**: Download `HumanEval.jsonl.gz` from [OpenAI's repository](https://github.com/openai/human-eval), extract and place as `data/humaneval.jsonl`
- **MBPP**: Download from [Google's repository](https://github.com/google-research/google-research/tree/master/mbpp) or use HuggingFace Datasets: `datasets.load_dataset("mbpp")`, save as `data/mbpp.jsonl`
- **MATH-500**: Extract 500 problems from [MATH dataset](https://github.com/hendrycks/math), save as `data/math500.jsonl`
- **GSM8K**: Download from [HuggingFace Datasets](https://huggingface.co/datasets/gsm8k): `datasets.load_dataset("gsm8k", "main")`, save as `data/gsm8k.jsonl`
- **GPQA**: Download from [GPQA repository](https://github.com/idavidrein/gpqa), save as `data/gpqa.jsonl`
- **Countdown**: Included in repository at `data/countdown.jsonl` (or download from source)
- **Sudoku**: Included in repository at `data/sudoku.csv` (or download from source)
- **Creativity Writing**: Included in repository at `src/benchmarks/text_tasks/creativity_writing/data/creativity_writing.jsonl` (200 prompts)

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

### Multimodal Models

For multimodal (text-to-image) evaluation, you need to place the following models in the `model/` directory:

```
model/
├── mmada/              # MMaDA text-to-image generation model
│   ├── config.json
│   ├── model-*.safetensors
│   └── tokenizer.json
└── magvitv2/           # MAGVITv2 VQ model (for image decoding)
    ├── config.json
    ├── model-*.safetensors
    └── ...
```

**Model Loading Priority**:
1. `model/mmada/` and `model/magvitv2/` (preferred)
2. Project root: `mmada-mix/` and `magvitv2/` (fallback)
3. Config file paths (last resort)

The system will automatically detect and load models from the `model/` directory.

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
     
     # Option 2: Use download script (if available in MMaDA project)
     # cd /path/to/MMaDA/geneval/evaluation
     # bash download_models.sh ../../models/mask2former/
     ```
   - **Direct URL**: `https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth`
   - File size: ~200MB
   - Purpose: Object detection and segmentation for GenEval evaluation

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
| Mistral | AR | `mistralai/Mistral-7B-Instruct-v0.2` | `/model/mistral` |
| Qwen | AR | `Qwen/Qwen-7B-Chat` | `/model/qwen` |

### Download Models

Models can be loaded from **local paths** (preferred) or **HuggingFace Hub**. The framework automatically detects the model type and handles both cases.

**Loading Priority**:
1. **Local Path** (preferred): If the path exists as a directory, it will be used directly
2. **HuggingFace Hub**: If not a local path, it will be loaded from HuggingFace Hub

**Model Type Detection** (case-insensitive):
- **Local Path**: Detected from directory name and `config.json` if available
- **HuggingFace Hub**: Detected from model name substring

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
adapter = get_model_adapter("Dream-org/Dream-v0-Instruct-7B", device="cuda:0")
adapter = get_model_adapter("Gen-Verse/MMaDA-8B-MixCoT", device="cuda:0")
```

**Model Directory Structure**:

Models should be organized in the `./model/` directory at the project root:

```
Uncode-new/
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
- TraDo-4B-Instruct: ~8GB
- TraDo-8B-Thinking: ~16GB
- LLaDA-8B-Instruct: ~16GB
- Dream-v0-Instruct-7B: ~14GB
- MMaDA-8B-MixCoT: ~16GB (required for text-to-image generation)
- Mistral-7B-Instruct-v0.2: ~13GB
- Qwen-7B-Chat: ~14GB

**Download Models from HuggingFace**:

```bash
# TraDo models
huggingface-cli download Gen-Verse/TraDo-8B-Instruct --local-dir ./model/trado
huggingface-cli download Gen-Verse/TraDo-4B-Instruct --local-dir ./model/trado-4b
huggingface-cli download Gen-Verse/TraDo-8B-Thinking --local-dir ./model/trado-thinking

# LLaDA
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada

# Dream
huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir ./model/dream

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
     
     # Or place in model/ directory
     huggingface-cli download Gen-Verse/MMaDA-8B-MixCoT --local-dir ./model/mmada
     
     # Or using Python
     from huggingface_hub import snapshot_download
     snapshot_download(repo_id="Gen-Verse/MMaDA-8B-MixCoT", local_dir="./model/mmada")
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
     
     # Or place in model/ directory
     huggingface-cli download showlab/magvitv2 --local-dir ./model/magvitv2
     
     # Or using Python
     from huggingface_hub import snapshot_download
     snapshot_download(repo_id="showlab/magvitv2", local_dir="./model/magvitv2")
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

## Testing

Before using the repository, please run the comprehensive test suite to ensure everything works correctly. See [TEST_CHECKLIST.md](TEST_CHECKLIST.md) for detailed testing instructions and commands.

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

### Output Structure

All results are saved in `results/paper_reproduction/` with the following structure:

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

Each result file contains the evaluation metrics (accuracy, FID, IS, etc.) for that specific run. The script automatically aggregates results across multiple runs.

### Requirements

- All required models must be downloaded and placed in the `model/` directory
- All datasets must be prepared in the `data/` directory
- Sufficient GPU memory (recommended: 24GB+ for large models)
- Estimated time: 1-3 days depending on hardware and number of experiments

## Quick Start

### Task-Specific Scripts (Recommended)

We provide specialized scripts for different task types and algorithms:

**Reasoning Tasks** (code, math, logic):
```bash
cd scripts
python eval_reasoning.py --task humaneval --model_name GSAI-ML/LLaDA-8B-Instruct --mode info-gain
python eval_reasoning.py --task math500 --model_name /path/to/model --mode pc_sampler
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
python eval_baselines.py --task humaneval --model_name /path/to/model --mode pc_sampler
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
bash Eval.sh --task math500 --model /path/to/Dream-v0-Instruct-7B --mode pc_sampler

# Sudoku with Info-Gain
bash Eval.sh --task sudoku --model /path/to/model --mode info-gain --candidate_number 8

# Countdown without few-shot
bash Eval.sh --task countdown --model /path/to/model --mode info-gain --no_shot
```

Built-in task defaults (gen_length, steps, block_length, data_path) are applied automatically. Any parameter can be overridden via CLI flags. Run `bash Eval.sh --help` for full usage.

### Programmatic Usage

```python
from src.models import get_model_adapter
from src.generators import generate, generate_with_info_gain
from src.prompts import get_task_prompt
from src.prompts.model_templates import apply_model_template
import torch

# Load model (auto-detects Dream / LLaDA / SDAR / AR)
adapter = get_model_adapter("GSAI-ML/LLaDA-8B-Instruct", device="cuda:0")
tokenizer = adapter.tokenizer
model = adapter.model

# Build prompt
input_data = {"problem": "What is 2 + 2?"}
query = get_task_prompt("math500", input_data, use_shot=True)
prompt_str = apply_model_template(adapter, tokenizer, query, task="math500")
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

### Why Info-Gain is Effective

**State uncertainty** serves as an effective indicator for determining whether the current decoding state lies close to the **training data manifold**. When decoded content is logically coherent and fluently expressed, the state resides near the data manifold, resulting in a concentrated predictive probability distribution and low uncertainty. Conversely, if the model lacks sufficient training on certain text forms, decoding enters inadequately covered regions, exhibiting dispersed probability distributions and high uncertainty.

Existing greedy certainty-based samplers cannot recognize data manifold deviation signals reflected by state uncertainty. They select the most certain action at each step, but locally certain actions do not necessarily correspond to actions that keep subsequent states on the data manifold.

In contrast, the Info-Gain Sampler actively perceives and utilizes state uncertainty through its information-gain term. When a candidate action would cause the state to deviate from the data manifold, the resulting state exhibits increased uncertainty, which negatively impacts the information-gain term and prevents such actions from being prioritized. This mechanism enables the Info-Gain Sampler to preserve logical coherence and fluent expression throughout the decoding path, even under high sampling temperatures.

### Key Concepts

**Info-Gain Sampler** is a decoding strategy that selects actions (token positions to decode) by maximizing information gain. It works in two modes:

1. **Traditional mode** (`candidate_number=1`):
   - Greedy selection based on heuristic scores
   - Equivalent to traditional uncertainty-based samplers

2. **Info-Gain mode** (`candidate_number>1`):
   - Samples multiple candidate actions
   - Evaluates each candidate by computing information gain
   - Selects the action that maximizes immediate cost − information gain

### Heuristic Functions

| Heuristic | Description |
|-----------|-------------|
| `confidence` (default) | Model confidence (probability of predicted token) |
| `pc` | PC-Sampler heuristic with frequency-based calibration |
| `neg_entropy` | Negative entropy (higher entropy = lower score) |
| `margin` | Margin between top-1 and top-2 probabilities |
| `uniform` | Uniform random sampling |

### Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `candidate_number` | Number of candidate actions to evaluate | 1 | `=1`: Greedy selection (baseline mode)<br>`>1`: Info-Gain mode (evaluates multiple candidates) |
| `position_temperature` | Temperature for stochastic position sampling | 0.0 | `=0`: Deterministic selection (greedy)<br>`>0`: Stochastic sampling with Gumbel noise |
| `heuristic` | Heuristic function for action scoring | `confidence` | See heuristic functions table above |
| `beam_size` | Beam search queue size | 1 | `=1`: Single path search<br>`>1`: Multi-path beam search |
| `use_kv_cache` | Enable KV-cache optimization | False | Caches prefix states to speed up generation (30-50% faster) |
| `use_block_causal_mask` | Use block-causal attention mask | False | Enables bidirectional attention within blocks, causal across blocks |

### Baseline Methods as Special Cases

All baseline decoding methods are implemented as special cases of the base `generate` function:

- **`original`**: `candidate_number=1`, `heuristic='confidence'`
- **`pc_sampler`**: `candidate_number=1`, `heuristic='pc'`
- **`entropy`**: `candidate_number=1`, `heuristic='neg_entropy'`
- **`margin`**: `candidate_number=1`, `heuristic='margin'`

## Supported Models

| Model | Type | Adapter | KV-Cache | Notes |
|-------|------|---------|----------|-------|
| LLaDA | MDM | `LLaDAAdapter` | ✅ (truncate) | Default mask_id=126336 |
| Dream | MDM | `DreamAdapter` | ✅ (native) | Logits shifted by one position |
| SDAR | MDM | `SDARAdapter` | ✅ (truncate) | Interface ready; model implementation pending |
| Mistral | AR Baseline | `MistralAdapter` | ❌ | Standard chat template |
| Qwen | AR Baseline | `QwenAdapter` | ❌ | Standard chat template |

**Model Detection**: Model type is automatically detected by analyzing keywords in the model name or path. The detection is case-insensitive and works for both HuggingFace Hub paths and local paths. Use `get_model_adapter(model_name, device)` for automatic detection and loading.

**KV-Cache Support**: All MDM models support KV-cache optimization to speed up generation:
- **Dream models**: Use native `store_kv` parameter for efficient cache management (no truncation needed)
- **LLaDA/SDAR models**: Require cache truncation after forward passes (handled automatically)
- **Performance**: 30-50% speedup for long sequences, especially with Info-Gain sampling

## Evaluation Tasks

### Text Tasks

| Task | Dataset | Description |
|------|---------|-------------|
| `humaneval` | `humaneval.jsonl` | Python code completion |
| `mbpp` | `mbpp.jsonl` | Python code generation |
| `math500` | `math500.jsonl` | Mathematical reasoning |
| `gsm8k` | `gsm8k.jsonl` | Grade school math |
| `gpqa` | `gpqa.jsonl` | Graduate-level QA |
| `sudoku` | `sudoku.csv` | 4×4 Sudoku puzzle solving |
| `countdown` | `countdown.jsonl` | Arithmetic operations game |
| `creativity_writing` | `creativity_writing.jsonl` | Creative story writing (200 prompts) |

### Multimodal Tasks

| Task | Description | Metrics |
|------|-------------|---------|
| `geneval` | Text-to-image generation evaluation | FID, sFID, IS, Precision, Recall, CLIP Score |

See [src/benchmarks/text_tasks/creativity_writing/README.md](src/benchmarks/text_tasks/creativity_writing/README.md) for detailed creative writing evaluation instructions.
See [src/benchmarks/multimodal_tasks/multimodal_eval/README.md](src/benchmarks/multimodal_tasks/multimodal_eval/README.md) for multimodal evaluation instructions.

## KV-Cache Optimization

KV-cache optimization significantly speeds up generation by caching previously computed key-value states. All MDM models (Dream, LLaDA, SDAR) support KV-cache:

```python
output = generate(
    model=model,
    prompt=prompt,
    steps=256,
    gen_length=256,
    block_length=32,
    adapter=adapter,
    use_kv_cache=True,  # Enable KV-cache
    # ... other parameters
)
```

**How it works:**
1. **Prefill phase**: Process the prompt tokens once and cache their key-value states
2. **Block-by-block generation**: For each generation block, only process new tokens while reusing cached prefix states
3. **Cache update**: After completing each block, update the cache with the new block's key-value states
4. **Lookahead optimization**: In Info-Gain mode, candidate evaluation reuses the committed prefix cache

**Implementation details:**
- **Dream models**: Use native `store_kv=False` parameter to prevent cache growth during lookahead
- **LLaDA/SDAR models**: Automatically truncate cache back to committed length after each forward pass
- **Block completion**: Cache is updated once per block completion, not per denoising step

**Performance**: KV-cache can reduce generation time by 30-50% for long sequences, especially with Info-Gain sampling where multiple candidates are evaluated.

## Evaluation

All evaluation tasks can be run using the unified `Eval.sh` script. Each task has built-in defaults for generation parameters, which can be overridden via command-line flags.

### Text Task Evaluation

#### HumanEval (Code Completion)

```bash
cd scripts

# Info-Gain Sampler (recommended)
bash Eval.sh \
    --task humaneval \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --heuristic confidence \
    --use_kv_cache

# PC-Sampler baseline
bash Eval.sh \
    --task humaneval \
    --model /path/to/model \
    --mode pc_sampler \
    --lambd 0.25 \
    --alpha 100

# Original (confidence-based)
bash Eval.sh \
    --task humaneval \
    --model /path/to/model \
    --mode original
```

**Output**: Results saved to `results/humaneval_<mode>_<timestamp>/`

#### MBPP (Code Generation)

```bash
bash Eval.sh \
    --task mbpp \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2
```

**Output**: Results saved to `results/mbpp_<mode>_<timestamp>/`

#### MATH-500 (Mathematical Reasoning)

```bash
bash Eval.sh \
    --task math500 \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --temperature 0.7
```

**Output**: Results saved to `results/math500_<mode>_<timestamp>/`

#### GSM8K (Grade School Math)

```bash
bash Eval.sh \
    --task gsm8k \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --gen_length 512 \
    --steps 512
```

**Output**: Results saved to `results/gsm8k_<mode>_<timestamp>/`

#### GPQA (Graduate-Level QA)

```bash
bash Eval.sh \
    --task gpqa \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8
```

**Output**: Results saved to `results/gpqa_<mode>_<timestamp>/`

#### Sudoku (4×4 Puzzle Solving)

```bash
bash Eval.sh \
    --task sudoku \
    --model /path/to/model \
    --mode info-gain \
    --candidate_number 8
```

**Note**: Sudoku uses a special prompt format with embedded mask tokens. The task automatically:
- Calculates the number of empty cells (mask tokens) in the puzzle
- Sets `gen_length=0` (no additional generation needed)
- Sets `steps` equal to the number of empty cells
- Sets `block_length` equal to the number of empty cells (single block)

**Output**: Results saved to `results/sudoku_<mode>_<timestamp>/`

#### Countdown (Arithmetic Operations)

```bash
# With few-shot examples (default)
bash Eval.sh \
    --task countdown \
    --model /path/to/model \
    --mode info-gain \
    --candidate_number 8

# Without few-shot examples
bash Eval.sh \
    --task countdown \
    --model /path/to/model \
    --mode info-gain \
    --candidate_number 8 \
    --no_shot
```

**Output**: Results saved to `results/countdown_<mode>_<timestamp>/`

#### Creativity Writing

```bash
# Generate outputs
bash Eval.sh \
    --task creativity_writing \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --gen_length 512

# Output saved to src/benchmarks/text_tasks/creativity_writing/outputs/<model>_<mode>.json
```

**Evaluate with LLM-as-Judge**:

```bash
cd Creativity_writing

# Pairwise comparison (compare two models)
python judge.py \
    --model_outputs outputs/model_a.json \
    --reference_outputs outputs/model_b.json \
    --judge_model gpt-4o \
    --mode pairwise

# Single-score rating
python judge.py \
    --model_outputs outputs/model_a.json \
    --judge_model gpt-4o \
    --mode single
```

**Note**: 
- Set `OPENAI_API_KEY` environment variable before running the judge script
- For custom API endpoints, set `OPENAI_API_BASE` (e.g., `https://api.openai.com/v1`)
- The judge script supports both OpenAI-compatible APIs and custom endpoints

**Output**: 
- Generation: `src/benchmarks/text_tasks/creativity_writing/outputs/<model>_<mode>.json`
- Judge results: `src/benchmarks/text_tasks/creativity_writing/outputs/judge_<mode>_<timestamp>.json`

### Multimodal Task Evaluation

#### GenEval (Text-to-Image Generation)

**Step 1: Generate Images**

```bash
cd src/benchmarks/multimodal_tasks/multimodal_eval

# Edit scripts/run_generate.sh to set model paths, then:
bash scripts/run_generate.sh

# Or directly:
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

**Step 2: Evaluate with GenEval**

```bash
# GenEval evaluation (object detection + color classification)
bash scripts/run_eval_geneval.sh ./output_geneval

# View detailed scores
python view_scores.py results/geneval_results.jsonl
```

**Step 3: CLIP Score Evaluation**

```bash
bash scripts/run_eval_clip.sh ./output_geneval

# Output: results/clip_scores.json
```

**Step 4: FID / IS / Precision / Recall**

```bash
# FID evaluation (requires data/VIRTUAL_imagenet512.npz)
bash scripts/run_eval_fid.sh ./data/VIRTUAL_imagenet512.npz ./output_geneval

# Or directly:
python eval_fid.py \
    ./data/VIRTUAL_imagenet512.npz \
    ./output_geneval \
    --batch-size 64
```

**One-Click Pipeline** (recommended for full evaluation):

```bash
cd src/benchmarks/multimodal_tasks/multimodal_eval
bash scripts/run_all.sh
```

**Pipeline Steps**:
1. **Generation**: Generates images from GenEval prompts (saves to `output_geneval/`)
2. **GenEval**: Evaluates object detection, counting, color, and spatial relationships
3. **CLIP Score**: Calculates semantic alignment between images and prompts
4. **FID**: Computes distribution similarity metrics (requires `data/VIRTUAL_imagenet512.npz`)

**Automatic Downloads**:
- **InceptionV3 model** (for FID calculation): Automatically downloaded from OpenAI on first use
  - URL: `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb`
  - Location: `src/benchmarks/multimodal_tasks/multimodal_eval/classify_image_graph_def.pb` (auto-downloaded)
  - Size: ~100MB

**Output Locations**:
- Generated images: `src/benchmarks/multimodal_tasks/multimodal_eval/output_geneval/`
- GenEval results: `src/benchmarks/multimodal_tasks/multimodal_eval/results/geneval_results.jsonl`
- CLIP scores: `src/benchmarks/multimodal_tasks/multimodal_eval/results/clip_scores.json`
- FID metrics: Printed to console (also saved to log files)

#### ImageNet FID End-to-End

```bash
cd src/benchmarks/multimodal_tasks/multimodal_eval

# Full run (50K images)
bash scripts/run_imagenet_fid.sh

# Test run (100 images)
bash scripts/run_imagenet_fid.sh --test

# Eval-only (skip generation)
bash scripts/run_imagenet_fid.sh --eval-only
```

**Output**: 
- FID scores printed to console in real-time
- Detailed results saved to `results/imagenet_fid_<timestamp>.txt`
- Generated images saved to `output_imagenet/` (50K images for full run, 100 for test)

### Common Evaluation Options

All tasks support the following common options:

```bash
bash Eval.sh \
    --task <task_name> \
    --model <model_path> \
    --mode <generation_mode> \
    --device cuda:0 \                    # GPU device
    --temperature 0.7 \                  # Sampling temperature
    --gen_length 256 \                   # Override default gen_length
    --steps 256 \                        # Override default steps
    --block_length 32 \                  # Override default block_length
    --use_kv_cache \                     # Enable KV-cache optimization
    --result_dir results/custom_dir \    # Custom output directory
    --result_path results/custom.json   # Custom output file path
```

**Generation Modes**:
- `original`: Confidence-based greedy selection
- `pc_sampler`: PC-Sampler with frequency calibration
- `eb_sampler`: Entropy-based sampler
- `fast_dllm`: Fast dLLM with dynamic thresholding
- `entropy`: Negative entropy heuristic
- `margin`: Margin heuristic
- `info-gain`: Info-Gain Sampler (recommended)

**Info-Gain Specific Options**:
- `--candidate_number N`: Number of candidate actions to evaluate (default: 1)
  - `N=1`: Greedy selection based on heuristic scores (baseline mode)
  - `N>1`: Info-Gain mode - evaluates N candidates and selects the one with maximum information gain
- `--position_temperature T`: Temperature for stochastic position sampling (default: 0.0)
  - `T=0`: Deterministic selection (always picks top-k positions)
  - `T>0`: Stochastic sampling with Gumbel noise (adds exploration)
- `--heuristic H`: Heuristic function for scoring positions (default: `confidence`)
  - Options: `confidence`, `pc`, `neg_entropy`, `margin`, `uniform`
  - See "Heuristic Functions" section for details
- `--tokens_per_step K`: Number of tokens to decode per step (default: 1)
  - `K=1`: Standard decoding (one token per step)
  - `K>1`: K-step decoding (decodes K tokens simultaneously)

### Using eval.py Directly

For more control, use `eval.py` directly:

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

### Using lm-evaluation-harness

For integration with the lm-evaluation-harness framework:

```bash
python -m src.utils.lm_eval_adapter \
    --model llada_dist \
    --model_args model_path=GSAI-ML/LLaDA-8B-Base,mode=info-gain \
    --tasks lambada_openai \
    --batch_size 32
```

## Reproducibility

This section provides complete end-to-end examples for reproducing evaluation results.

### Example 1: HumanEval with Info-Gain Sampler

**Step 1: Environment Setup**

```bash
# Clone repository
git clone <repository-url>
cd Uncode-new

# Install dependencies
pip install -r requirements.txt
```

**Step 2: Data Preparation**

```bash
# Download HumanEval dataset
# Place it in data/humaneval.jsonl

# Generate baseline file (if needed)
python utils/calculate_p_baseline.py \
    --input_file /path/to/reference_corpus.jsonl \
    --output_file data/baseline/reference_corpus_llada.json \
    --model_name GSAI-ML/LLaDA-8B-Instruct
```

**Step 3: Run Evaluation**

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

**Step 4: View Results**

```bash
# Results are saved to results/humaneval_info-gain_<timestamp>/
# Check the result files for accuracy and pass rates
cat results/humaneval_info-gain_*/results.txt
```

### Example 2: MATH-500 with PC-Sampler

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

### Example 3: Creativity Writing with LLM-as-Judge

**Step 1: Generate Outputs**

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

**Step 2: Evaluate with Judge**

```bash
cd Creativity_writing

# Set API key
export OPENAI_API_KEY="your-api-key"

# Run judge evaluation
python judge.py \
    --model_outputs outputs/llada_8b_instruct_info-gain_confidence_K1.json \
    --judge_model gpt-4o \
    --mode single
```

### Example 4: Multimodal Evaluation (GenEval + FID)

**Step 1: Setup Multimodal Environment**

```bash
cd src/benchmarks/multimodal_tasks/multimodal_eval

# Install additional dependencies
pip install tensorflow scipy mmdet open_clip_torch clip_benchmark pandas

# Clone mmdetection (if needed)
cd ..
git clone https://github.com/open-mmlab/mmdetection.git
cd src/benchmarks/multimodal_tasks/multimodal_eval
```

**Step 2: Prepare Models**

```bash
# Ensure MMaDA and VQ models are available
# Edit scripts/run_generate.sh to set paths:
# mmada_model_path=../mmada-mix
# vq_model_path=../magvitv2
```

**Step 3: Run Full Evaluation**

```bash
bash scripts/run_all.sh
```

This will:
1. Generate images from GenEval prompts
2. Run GenEval evaluation (object detection + color)
3. Calculate CLIP Score
4. Calculate FID, IS, Precision, Recall

**Step 4: View Results**

```bash
# GenEval results
python view_scores.py results/geneval_results.jsonl

# CLIP Score
cat results/clip_scores.json

# FID results (printed to console)
```

### Reproducing Paper Results

To reproduce results from papers or previous experiments, ensure consistency in:

1. **Model versions**: Use the exact same model checkpoints (same commit hash or version tag)
   - Check model commit hashes if using HuggingFace models
   - For local models, ensure weights are identical

2. **Hyperparameters**: Match all generation parameters exactly
   - `temperature`, `steps`, `gen_length`, `block_length`
   - `candidate_number`, `position_temperature`, `heuristic`
   - `lambd`, `alpha` (for PC-Sampler)
   - `use_kv_cache` flag (affects generation speed but not results)

3. **Data splits**: Use the same dataset versions and splits
   - Ensure dataset files are identical (same number of samples, same order)
   - For tasks with train/test splits, use the same split

4. **Baseline files**: Use baseline files generated from the same reference corpus
   - Same corpus source and size
   - Same tokenizer/model used for tokenization
   - Verify baseline file contents match

5. **Random seeds**: Set random seeds if reproducibility is critical
   - Note: Current implementation does not expose seed parameter
   - For deterministic results, ensure PyTorch/CUDA random state is controlled externally

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, commit messages, and pull request process.

## License

[Add your license information here]

---

**Note**: This repository is actively maintained. For issues, questions, or contributions, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).

