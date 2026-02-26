#!/usr/bin/env python3
"""
Pre-download evaluation datasets into the HuggingFace cache.

This ensures lm-evaluation-harness can load datasets offline.
Datasets are cached in the same location that `datasets.load_dataset()` uses,
so no extra configuration is needed at eval time.

Usage:
    # On a machine WITH internet:
    python scripts/prepare_eval_data.py

    # Then on the eval machine (can be offline):
    export HF_DATASETS_OFFLINE=1
    bash examples/info-gain/llada/eval.sh

    # If the cache is at a non-default location, set it consistently:
    export HF_DATASETS_CACHE=/path/to/cache
    python scripts/prepare_eval_data.py          # download
    export HF_DATASETS_OFFLINE=1                 # then eval
"""

import os

# Dataset definitions: (hub_path, subset)
# These MUST match the dataset_path / dataset_name in lm-eval task YAMLs.
DATASETS = {
    "gsm8k":        ("openai/gsm8k", "main"),
    "mbpp":         ("google-research-datasets/mbpp", "full"),
    "humaneval":    ("openai/openai_humaneval", None),
    "math_algebra":              ("EleutherAI/hendrycks_math", "algebra"),
    "math_counting_and_prob":    ("EleutherAI/hendrycks_math", "counting_and_probability"),
    "math_geometry":             ("EleutherAI/hendrycks_math", "geometry"),
    "math_intermediate_algebra": ("EleutherAI/hendrycks_math", "intermediate_algebra"),
    "math_num_theory":           ("EleutherAI/hendrycks_math", "number_theory"),
    "math_prealgebra":           ("EleutherAI/hendrycks_math", "prealgebra"),
    "math_precalc":              ("EleutherAI/hendrycks_math", "precalculus"),
}


def main():
    from datasets import config, load_dataset

    cache_dir = os.environ.get("HF_DATASETS_CACHE", config.HF_DATASETS_CACHE)
    print(f"Cache directory: {cache_dir}")
    print(f"Downloading {len(DATASETS)} dataset(s)...\n")

    # De-duplicate by (path, subset) to avoid downloading the same repo multiple times
    seen = set()
    for name, (path, subset) in sorted(DATASETS.items()):
        key = (path, subset)
        if key in seen:
            continue
        seen.add(key)

        label = f"{path}" + (f" [{subset}]" if subset else "")

        # Check cache first
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        try:
            ds = load_dataset(path, subset)
            total = sum(len(s) for s in ds.values())
            print(f"  ✓ {label:50s} {total} examples (cached)")
            os.environ.pop("HF_DATASETS_OFFLINE", None)
            continue
        except Exception:
            pass
        os.environ.pop("HF_DATASETS_OFFLINE", None)

        # Download
        print(f"  ↓ {label} ...")
        try:
            ds = load_dataset(path, subset)
            total = sum(len(s) for s in ds.values())
            splits = ", ".join(f"{k}={len(v)}" for k, v in ds.items())
            print(f"  ✓ {total} examples ({splits})")
        except Exception as e:
            print(f"  ✗ {e}")
        print()

    print("Done.")
    print()
    print("To run evaluation offline, set before eval commands:")
    print(f"  export HF_DATASETS_CACHE={cache_dir}")
    print("  export HF_DATASETS_OFFLINE=1")


if __name__ == "__main__":
    main()
