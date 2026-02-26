# Info-Gain / LookUM Sampler

Resources and examples for inferencing **LLaDA** and **Dream** with lookahead-based samplers.

## Setup

```bash
# 1. Install dllm
pip install -e .

# 2. (For evaluation) Install the dllm-fork of lm-evaluation-harness
#    This fork contains custom tasks (humaneval_instruct_llada, mbpp_instruct_llada, etc.)
git clone --branch dllm https://github.com/ZHZisZZ/lm-evaluation-harness.git lm-evaluation-harness
pip install -e "lm-evaluation-harness[ifeval,math]"

# 3. (Optional) Pre-download evaluation datasets for offline use
python scripts/prepare_eval_data.py
```

> **Important**: The standard `pip install lm_eval` does NOT contain the custom tasks
> required for evaluation. You must install the dllm-fork above.

## Variants

| Variant | Objective (maximise) | `--variant` |
|---------|---------------------|-------------|
| **Info-Gain** | $J(a) = \text{IG}(a) - C(a)$ | `info_gain` (default) |
| **LookUM** | $J(a) = \text{IG}(a)$ | `lookum` |

where $\text{IG}(a) = \mathcal{H}(z_t) - \bar{H}_{\text{next}}(a)$ is the information gain and $C(a)$ is the immediate cost (entropy sum over chosen positions). LookUM drops the $C(a)$ term, selecting purely by future uncertainty reduction.

## Files

```
dllm/pipelines/info_gain
├── core.py                         # Shared: entropy, candidate generation, scoring
├── __init__.py
├── dream/
│   ├── __init__.py
│   ├── models/
│   │   ├── configuration_dream.py
│   │   └── modeling_dream.py
│   └── sampler.py                  # Dream sampler (Info-Gain & LookUM)
└── llada/
    ├── __init__.py
    ├── models/
    │   ├── configuration_llada.py
    │   └── modeling_llada.py
    └── sampler.py                  # LLaDA sampler (Info-Gain & LookUM)

examples/info-gain
├── README.md
├── dream/
│   └── sample.py
└── llada/
    └── sample.py
```

## Inference

Info-Gain (default):

```shell
python examples/info-gain/llada/sample.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --use_cache prefix --threshold 0.9 \
    --candidate_number 8 --position_temperature 0.1

python examples/info-gain/dream/sample.py \
    --model_name_or_path "Dream-org/Dream-v0-Instruct-7B" \
    --use_cache prefix --threshold 0.9 \
    --candidate_number 8 --position_temperature 0.1
```

LookUM (add `--variant lookum`):

```shell
python examples/info-gain/llada/sample.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --variant lookum --use_cache prefix --threshold 0.9

python examples/info-gain/dream/sample.py \
    --model_name_or_path "Dream-org/Dream-v0-Instruct-7B" \
    --variant lookum --use_cache prefix --threshold 0.9
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--variant` | Scoring variant: `info_gain` or `lookum` | `info_gain` |
| `--use_cache` | Cache mode: `none`, `prefix`, `dual` (dual requires `replace_position`) | `prefix` |
| `--threshold` | High-confidence bypass threshold (skip lookahead when top-1 prob ≥ threshold) | `0.9` |
| `--candidate_number` | Number of candidate unmask actions | `8` |
| `--position_temperature` | Position sampling temperature (0 = greedy) | `0.1` |
| `--steps` | Total denoising steps | `512` |
| `--block_size` | Block size for block-based generation | `32` |
| `--temperature` | Token sampling temperature | `0.0` |

## Algorithm

Since $\mathcal{H}(z_t)$ is constant w.r.t. action $a$ at each step, the objectives simplify to:

- **Info-Gain**: maximise $-C(a) - \bar{H}_{\text{next}}(a)$
- **LookUM**: maximise $-\bar{H}_{\text{next}}(a)$

At each denoising step:

1. **High-confidence bypass**: Positions with top-1 probability ≥ `threshold` are directly fixed (no lookahead).
2. **Candidate generation**: `candidate_number` diverse unmask actions via joint token–position Gumbel sampling.
3. **Lookahead**: Batch forward pass to compute $\bar{H}_{\text{next}}$ for each candidate.
4. **Selection**: Pick the candidate with the highest $J(a)$.
5. **Logits caching**: Reuse the winner's lookahead logits as next step's base logits (saves one forward).
6. **Last-step fast path**: When all remaining masks will be filled, skip lookahead and sample directly.
