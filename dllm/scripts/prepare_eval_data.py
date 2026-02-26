#!/usr/bin/env python3
"""
Download evaluation datasets for offline use.

Usage:
    python scripts/prepare_eval_data.py                    # download all
    python scripts/prepare_eval_data.py --data_dir ./data  # custom dir
    python scripts/prepare_eval_data.py --tasks gsm8k mbpp # specific tasks

After downloading, set the environment variable before running eval:
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
"""

import argparse
import os

DATASETS = {
    "gsm8k": ("openai/gsm8k", "main"),
    "mbpp": ("google-research-datasets/mbpp", "full"),
    "humaneval": ("openai/openai_humaneval", None),
    "minerva_math_algebra": ("EleutherAI/hendrycks_math", "algebra"),
    "minerva_math_counting_and_prob": ("EleutherAI/hendrycks_math", "counting_and_probability"),
    "minerva_math_geometry": ("EleutherAI/hendrycks_math", "geometry"),
    "minerva_math_intermediate_algebra": ("EleutherAI/hendrycks_math", "intermediate_algebra"),
    "minerva_math_num_theory": ("EleutherAI/hendrycks_math", "number_theory"),
    "minerva_math_prealgebra": ("EleutherAI/hendrycks_math", "prealgebra"),
    "minerva_math_precalc": ("EleutherAI/hendrycks_math", "precalculus"),
}

ALL_TASKS = {
    "gsm8k": ["gsm8k"],
    "mbpp": ["mbpp"],
    "humaneval": ["humaneval"],
    "minerva_math": [
        "minerva_math_algebra",
        "minerva_math_counting_and_prob",
        "minerva_math_geometry",
        "minerva_math_intermediate_algebra",
        "minerva_math_num_theory",
        "minerva_math_prealgebra",
        "minerva_math_precalc",
    ],
    "all": list(DATASETS.keys()),
}


def download_dataset(name, path, subset, cache_dir):
    from datasets import load_dataset

    print(f"  Downloading: {path}" + (f" [{subset}]" if subset else ""))
    try:
        ds = load_dataset(path, subset, cache_dir=cache_dir)
        total = sum(len(split) for split in ds.values())
        print(f"  ✓ {name}: {total} examples ({', '.join(f'{k}={len(v)}' for k, v in ds.items())})")
    except Exception as e:
        print(f"  ✗ {name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download evaluation datasets")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Cache directory. Default: HuggingFace default cache (~/.cache/huggingface/datasets)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=list(ALL_TASKS.keys()),
        help="Tasks to download (default: all)",
    )
    args = parser.parse_args()

    # Resolve task list
    dataset_names = set()
    for task in args.tasks:
        dataset_names.update(ALL_TASKS.get(task, [task]))

    cache_dir = args.data_dir
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Also set HF cache env so datasets lib uses it
        os.environ["HF_DATASETS_CACHE"] = cache_dir

    print(f"Downloading {len(dataset_names)} dataset(s)...")
    if cache_dir:
        print(f"Cache dir: {cache_dir}")
    print()

    for name in sorted(dataset_names):
        if name not in DATASETS:
            print(f"  ? Unknown dataset: {name}")
            continue
        path, subset = DATASETS[name]
        download_dataset(name, path, subset, cache_dir)

    print()
    print("Done. To run evaluation offline, set:")
    print("  export HF_DATASETS_OFFLINE=1")
    if cache_dir:
        print(f"  export HF_DATASETS_CACHE={os.path.abspath(cache_dir)}")


if __name__ == "__main__":
    main()
