#!/usr/bin/env python3
"""
Unified entry point for Info-Gain Sampler evaluation.

Quick start:
    python run.py --config configs/gsm8k_info_gain.yaml
    python run.py --config configs/humaneval_info_gain.yaml --model dream
    python run.py --config configs/math500_info_gain.yaml --max_samples 5

Any key in the YAML config can be overridden on the command line:
    python run.py --config configs/gsm8k_info_gain.yaml --model sdar --candidate_number 16

Available configs:
    configs/gsm8k_info_gain.yaml      GSM8K with Info-Gain
    configs/math500_info_gain.yaml    MATH-500 with Info-Gain
    configs/humaneval_info_gain.yaml  HumanEval with Info-Gain
    configs/mbpp_info_gain.yaml       MBPP with Info-Gain
    configs/writing_info_gain.yaml    Creative writing with Info-Gain
    configs/gsm8k_original.yaml       GSM8K baseline (greedy)
"""

import os
import sys
import argparse
import types

# Add project root and scripts/ to path
_root = os.path.dirname(os.path.abspath(__file__))
_scripts = os.path.join(_root, "scripts")
for p in (_root, _scripts):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_args(config_path: str, overrides: list[str]) -> types.SimpleNamespace:
    """
    Load a YAML config, then apply CLI overrides (--key value pairs).
    Returns a SimpleNamespace compatible with scripts/eval.py:main().
    """
    cfg = load_yaml(config_path)

    # Parse overrides: --key value or --flag (boolean)
    i = 0
    while i < len(overrides):
        tok = overrides[i]
        if tok.startswith("--"):
            key = tok[2:].replace("-", "_")
            if i + 1 < len(overrides) and not overrides[i + 1].startswith("--"):
                val_str = overrides[i + 1]
                i += 2
                # Type coercion: try int, float, bool, then str
                if val_str.lower() in ("true", "false"):
                    cfg[key] = val_str.lower() == "true"
                else:
                    try:
                        cfg[key] = int(val_str)
                    except ValueError:
                        try:
                            cfg[key] = float(val_str)
                        except ValueError:
                            cfg[key] = val_str if val_str.lower() != "null" else None
            else:
                # Boolean flag
                cfg[key] = True
                i += 1
        else:
            i += 1

    # Defaults expected by eval.py:main()
    defaults = dict(
        task=None,
        model_name=None,
        device="cuda:0",
        mode="info-gain",
        variant="info_gain",
        algorithm=None,
        gen_length=256,
        steps=256,
        block_length=32,
        temperature=0.0,
        candidate_number=8,
        position_temperature=0.2,
        threshold=0.8,
        use_cache="prefix",
        use_kv_cache=False,
        no_shot=False,
        max_samples=None,
        data_path=None,
        result_path=None,
        result_dir=None,
        heuristic="confidence",
        tokens_per_step=None,
        lambd=0.25,
        alpha=10,
        gamma=0.01,
        thread=0.9,
        baseline_name=None,
        beam_size=1,
    )
    defaults.update(cfg)

    # `model` in config → `model_name` in eval.py
    if "model" in defaults and defaults.get("model_name") is None:
        defaults["model_name"] = defaults.pop("model")
    else:
        defaults.pop("model", None)

    # `dynamic_threshold` alias
    if "threshold" in defaults and "dynamic_threshold" not in defaults:
        defaults["dynamic_threshold"] = defaults["threshold"]

    return types.SimpleNamespace(**defaults)


def main():
    parser = argparse.ArgumentParser(
        description="Info-Gain Sampler — unified evaluation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        # Allow unknown args so we can pass them through as overrides
        add_help=True,
    )
    parser.add_argument(
        "--config", required=False, default=None, metavar="YAML",
        help="Path to experiment config (e.g. configs/gsm8k_info_gain.yaml)",
    )
    # Show config list shortcut
    parser.add_argument(
        "--list_configs", action="store_true",
        help="List available configs and exit",
    )

    known, overrides = parser.parse_known_args()

    # Allow --list_configs without --config
    if known.list_configs or (len(sys.argv) == 2 and sys.argv[1] == "--list_configs"):
        cfg_dir = os.path.join(_root, "configs")
        yamls = sorted(f for f in os.listdir(cfg_dir) if f.endswith(".yaml") and f != "base.yaml")
        print("Available configs:")
        for y in yamls:
            print(f"  configs/{y}")
        return

    if known.config is None:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(known.config):
        print(f"Error: config not found: {known.config}")
        sys.exit(1)

    args = build_args(known.config, overrides)

    if args.task is None:
        print("Error: 'task' must be set in the config or via --task.")
        sys.exit(1)
    if args.model_name is None:
        print("Error: 'model' must be set in the config or via --model.")
        sys.exit(1)

    from eval import main as eval_main
    eval_main(args)


if __name__ == "__main__":
    main()
