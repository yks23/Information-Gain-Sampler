#!/usr/bin/env python3
"""
Check and verify the evaluation environment.

Usage:
    python scripts/check_eval_env.py           # check everything
    python scripts/check_eval_env.py --fix     # check + auto-download missing datasets
"""

import importlib
import os
import sys
import argparse


def check_mark(ok):
    return "✓" if ok else "✗"


def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check_packages():
    section("1. Python packages")
    all_ok = True

    pkgs = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
        ("lm_eval", "lm_eval"),
        ("dllm", "dllm"),
        ("peft", "peft"),
        ("sympy", "sympy"),
        ("math_verify", "math_verify"),
    ]

    for display, module in pkgs:
        try:
            m = importlib.import_module(module)
            ver = getattr(m, "__version__", "ok")
            print(f"  {check_mark(True)} {display:20s} {ver}")
        except ImportError:
            print(f"  {check_mark(False)} {display:20s} NOT INSTALLED")
            all_ok = False

    return all_ok


def check_lm_eval_fork():
    section("2. lm-eval-harness (dllm fork)")

    try:
        from lm_eval.tasks import TaskManager
        tm = TaskManager()
        all_tasks = tm.all_tasks
    except Exception as e:
        print(f"  {check_mark(False)} Cannot load TaskManager: {e}")
        return False

    custom_tasks = [
        "humaneval_instruct_llada",
        "mbpp_instruct_llada",
        "humaneval_instruct_dream",
        "mbpp_instruct_dream",
        "gsm8k",
        "minerva_math_algebra",
    ]

    all_ok = True
    for task in custom_tasks:
        found = task in all_tasks
        print(f"  {check_mark(found)} task: {task}")
        if not found:
            all_ok = False

    if not all_ok:
        print()
        print("  FIX: Install the dllm-fork of lm-evaluation-harness:")
        print("    git clone --branch dllm https://github.com/ZHZisZZ/lm-evaluation-harness.git lm-evaluation-harness")
        print('    pip install -e "lm-evaluation-harness[ifeval,math]"')

    return all_ok


def check_info_gain_pipeline():
    section("3. Info-Gain / LookUM pipeline")

    checks = [
        ("core", "dllm.pipelines.info_gain.core"),
        ("llada sampler", "dllm.pipelines.info_gain.llada.sampler"),
        ("llada eval", "dllm.pipelines.info_gain.llada.eval"),
        ("dream sampler", "dllm.pipelines.info_gain.dream.sampler"),
        ("dream eval", "dllm.pipelines.info_gain.dream.eval"),
    ]

    all_ok = True
    for name, module in checks:
        try:
            importlib.import_module(module)
            print(f"  {check_mark(True)} {name}")
        except Exception as e:
            print(f"  {check_mark(False)} {name}: {e}")
            all_ok = False

    if not all_ok:
        print()
        print("  FIX: pip install -e .")

    return all_ok


def _try_load(path, subset):
    """Try loading dataset: cache first, then online."""
    from datasets import load_dataset

    # 1) Try cache only (no network)
    old_offline = os.environ.get("HF_DATASETS_OFFLINE")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        ds = load_dataset(path, subset)
        return ds, "cached"
    except Exception:
        pass
    finally:
        if old_offline is None:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
        else:
            os.environ["HF_DATASETS_OFFLINE"] = old_offline

    # 2) Try online
    try:
        ds = load_dataset(path, subset)
        return ds, "downloaded"
    except Exception as e:
        return None, str(e).split('\n')[0][:80]


def check_datasets(fix=False):
    section("4. Evaluation datasets")

    datasets_to_check = [
        ("openai/gsm8k", "main", "gsm8k"),
        ("google-research-datasets/mbpp", "full", "mbpp"),
        ("openai/openai_humaneval", None, "humaneval"),
        ("EleutherAI/hendrycks_math", "algebra", "minerva_math (sample)"),
    ]

    from datasets import config
    cache_dir = os.environ.get("HF_DATASETS_CACHE", config.HF_DATASETS_CACHE)
    print(f"  Cache: {cache_dir}")
    print()

    all_ok = True
    for path, subset, label in datasets_to_check:
        ds, status = _try_load(path, subset)
        if ds is not None:
            n = sum(len(s) for s in ds.values())
            print(f"  {check_mark(True)} {label:30s} {n} examples ({status})")
        else:
            print(f"  {check_mark(False)} {label:30s} not in cache")
            all_ok = False
            if fix:
                print(f"       Downloading {path} ...")
                try:
                    from datasets import load_dataset
                    ds = load_dataset(path, subset)
                    n = sum(len(s) for s in ds.values())
                    print(f"       {check_mark(True)} Downloaded: {n} examples")
                    all_ok = True  # fixed
                except Exception as e2:
                    err = str(e2).split('\n')[0][:80]
                    print(f"       {check_mark(False)} Failed: {err}")

    if not all_ok and not fix:
        print()
        print("  FIX: python scripts/prepare_eval_data.py")
        print("  Or:  python scripts/check_eval_env.py --fix")

    return all_ok


def check_env_vars():
    section("5. Environment variables")

    vars_to_check = [
        ("HF_DATASETS_CACHE", None, "dataset cache location"),
        ("HF_DATASETS_OFFLINE", None, "offline mode (set to 1 if no internet)"),
        ("HF_ALLOW_CODE_EVAL", "1", "required for HumanEval/MBPP"),
        ("HF_DATASETS_TRUST_REMOTE_CODE", "True", "for remote code datasets"),
    ]

    for var, recommended, desc in vars_to_check:
        val = os.environ.get(var)
        if val is not None:
            print(f"  {check_mark(True)} {var}={val}")
        elif recommended:
            print(f"  {check_mark(False)} {var} not set (recommended: {recommended}) — {desc}")
        else:
            print(f"  · {var} not set — {desc}")


def main():
    parser = argparse.ArgumentParser(description="Check evaluation environment")
    parser.add_argument("--fix", action="store_true", help="Auto-download missing datasets")
    args = parser.parse_args()

    print("=" * 60)
    print("  Info-Gain / LookUM Evaluation Environment Check")
    print("=" * 60)

    r1 = check_packages()
    r2 = check_info_gain_pipeline()
    r3 = check_lm_eval_fork()
    r4 = check_datasets(fix=args.fix)
    check_env_vars()

    section("Summary")
    results = [("Packages", r1), ("Pipeline", r2), ("lm-eval fork", r3), ("Datasets", r4)]
    all_ok = all(r for _, r in results)
    for name, ok in results:
        print(f"  {check_mark(ok)} {name}")

    print()
    if all_ok:
        print("  All checks passed. Ready to run evaluation.")
        print()
        print("  Quick start:")
        print("    cd dllm/")
        print("    bash examples/info-gain/llada/eval.sh --variant info_gain")
    else:
        print("  Some checks failed. See FIX instructions above.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
