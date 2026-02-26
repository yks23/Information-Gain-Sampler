# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a Python ML research codebase — "Info-Gain Sampler for Masked Diffusion Models". It is NOT a web application. It provides a unified decoding framework for MDMs. See `README.md` for full details.

### Key caveats

- **`src/models/` is missing from the repository.** The `src.models` package (containing `get_model_adapter`, adapter classes like `LlamaAdapter`, `DreamAdapter`, etc.) is imported throughout the codebase but was never committed. Without it, scripts that load models (`scripts/eval.py`, `src/example_usage.py`) will fail with `ModuleNotFoundError`. The core generation logic in `src/generators/` works independently.
- **No GPU in Cloud VM.** Model inference requires CUDA-capable GPUs with ≥16 GB VRAM (7B–8B parameter models). The core generation functions (`src/generators/`) can be tested on CPU with mock tensors.
- **No automated tests.** The repo has zero test files. Validation is done by running evaluation scripts against benchmark datasets.
- **No linting configuration.** No `.flake8`, `pyproject.toml[tool.ruff]`, or similar config exists. Use `ruff check --select=E9,F63,F7,F82` for critical error checks. Pre-existing lint issues exist in `src/benchmarks/multimodal_tasks/multimodal_eval/mmada_utils/training/` (undefined names, syntax error) — these are auxiliary training scripts and do not affect core functionality.

### Dependencies

Managed via `requirements.txt` (pip). Install with `pip install -r requirements.txt`. No conda environment is strictly required — Python 3.8+ works (3.10 recommended by README).

### Running core logic on CPU

```python
import sys; sys.path.insert(0, '.')
from src.generators.base import generate, add_gumbel_noise, get_num_transfer_tokens
from src.generators.info_gain import compute_entropy_info_gain
```

These modules can be imported and tested without GPU or model weights.

### Running evaluations (requires GPU + models + src/models/)

See `README.md` "Quick Start" section. Key entry points:
- `scripts/eval_reasoning.py` — reasoning tasks
- `scripts/eval_writing.py` — creative writing
- `scripts/eval_multimodal.py` — text-to-image
- `scripts/Eval.sh` — unified CLI wrapper
