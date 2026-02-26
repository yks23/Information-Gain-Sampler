#!/usr/bin/env python3
"""
Pre-download evaluation datasets for offline use.

Usage:
    # Download to default HF cache
    python scripts/prepare_eval_data.py

    # Download to a specific local directory (recommended for portability)
    python scripts/prepare_eval_data.py --local_dir ./eval_data

    # Then run eval pointing to the same directory:
    #   export HF_DATASETS_CACHE=./eval_data
    #   export HF_DATASETS_OFFLINE=1
    #   bash examples/info-gain/llada/eval.sh
"""

import os
import sys
import argparse

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


def try_cache(path, subset):
    """Try loading from cache only (no network)."""
    from datasets import load_dataset
    old_offline = os.environ.get("HF_DATASETS_OFFLINE")
    old_hub = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return load_dataset(path, subset)
    except Exception:
        return None
    finally:
        # Restore original values
        for key, old_val in [("HF_DATASETS_OFFLINE", old_offline), ("HF_HUB_OFFLINE", old_hub)]:
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val


def main():
    parser = argparse.ArgumentParser(description="Download evaluation datasets")
    parser.add_argument("--local_dir", type=str, default=None,
                        help="Download datasets to this directory. "
                             "Use the same path as HF_DATASETS_CACHE when running eval.")
    args = parser.parse_args()

    if args.local_dir:
        abs_dir = os.path.abspath(args.local_dir)
        os.makedirs(abs_dir, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = abs_dir
        print(f"Download directory: {abs_dir}")
    else:
        from datasets import config
        abs_dir = os.environ.get("HF_DATASETS_CACHE", config.HF_DATASETS_CACHE)
        print(f"Using default HF cache: {abs_dir}")

    print()

    seen = set()
    for name, (path, subset) in sorted(DATASETS.items()):
        key = (path, subset)
        if key in seen:
            continue
        seen.add(key)
        label = f"{path}" + (f" [{subset}]" if subset else "")

        # Check cache first
        ds = try_cache(path, subset)
        if ds is not None:
            n = sum(len(s) for s in ds.values())
            print(f"  ✓ {label:55s} {n:>5d} examples (cached)")
            continue

        # Download
        print(f"  ↓ {label} ...")
        try:
            from datasets import load_dataset
            ds = load_dataset(path, subset)
            n = sum(len(s) for s in ds.values())
            print(f"  ✓ {label:55s} {n:>5d} examples (downloaded)")
        except Exception as e:
            err = str(e).split('\n')[0][:80]
            print(f"  ✗ {label}: {err}")

    print()
    print("=" * 60)
    print("To run evaluation with these datasets:")
    print()
    print(f"  export HF_DATASETS_CACHE={abs_dir}")
    print("  export HF_DATASETS_OFFLINE=1")
    print("  export HF_ALLOW_CODE_EVAL=1")
    print()
    print("  cd dllm/")
    print("  bash examples/info-gain/llada/eval.sh --variant info_gain")
    print("=" * 60)


if __name__ == "__main__":
    main()
