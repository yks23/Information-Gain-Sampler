# Info-Gain Sampler

Resources and examples for inferencing **LLaDA** and **Dream** with **Info-Gain Sampler**.

## Files

```
# Pipeline modules for Info-Gain
dllm/pipelines/info_gain
├── __init__.py
├── dream/
│   ├── __init__.py
│   ├── models/
│   │   ├── configuration_dream.py
│   │   └── modeling_dream.py
│   └── sampler.py                  # Info-Gain Dream sampler
└── llada/
    ├── __init__.py
    ├── models/
    │   ├── configuration_llada.py
    │   └── modeling_llada.py
    └── sampler.py                  # Info-Gain LLaDA sampler

# Example entry points
examples/info-gain
├── README.md
├── dream/
│   └── sample.py
└── llada/
    └── sample.py
```

## Inference

Sampling with the Info-Gain LLaDA sampler (prefix cache + confidence threshold + Info-Gain):

```shell
python examples/info-gain/llada/sample.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --use_cache prefix \
    --threshold 0.9 \
    --candidate_number 8 \
    --position_temperature 0.1
```

Sampling with the Info-Gain Dream sampler:

```shell
python examples/info-gain/dream/sample.py \
    --model_name_or_path "Dream-org/Dream-v0-Instruct-7B" \
    --use_cache prefix \
    --threshold 0.9 \
    --candidate_number 8 \
    --position_temperature 0.1
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use_cache` | Cache mode: `none`, `prefix` | `prefix` |
| `--threshold` | High-confidence bypass threshold. Positions with top-1 probability >= threshold skip Info-Gain evaluation | `0.9` |
| `--candidate_number` | Number of candidate unmask actions for Info-Gain evaluation | `8` |
| `--position_temperature` | Temperature for Gumbel-perturbed position sampling (0 = greedy) | `0.1` |
| `--steps` | Total denoising steps | `512` |
| `--block_size` | Block size for block-based generation | `32` |
| `--temperature` | Token sampling temperature | `0.0` |

## Algorithm

At each denoising step, the Info-Gain Sampler:

1. **High-confidence bypass**: If `threshold` is set and positions have top-1 probability >= threshold, directly fix those tokens (no lookahead needed).
2. **Candidate generation**: Generate `candidate_number` diverse unmask actions via Gumbel-perturbed position sampling with `position_temperature`.
3. **Lookahead evaluation**: Batch forward pass to evaluate next-step entropy for each candidate.
4. **Selection**: Score = action_entropy + next_avg_entropy. Select the candidate with the lowest score.
