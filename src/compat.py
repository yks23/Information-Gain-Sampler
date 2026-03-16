"""
Environment compatibility fixes.

This module applies patches needed to work around version incompatibilities
in the current environment (e.g., torchvision 0.19.0 + PyTorch 2.10.0).

Import this module early (before transformers or torchvision) to apply patches.
"""

import sys
import torch


def _patch_torchvision():
    """
    Fix torchvision 0.19.0 + PyTorch 2.10.0+ incompatibility.

    In newer PyTorch (2.10.0+), torchvision 0.19.0's _meta_registrations.py fails
    because operators like torchvision::nms aren't registered. We register stub
    implementations and then re-import torchvision cleanly.
    """
    # First try: maybe torchvision works without any patching
    try:
        import torchvision  # noqa: F401
        return  # Already works, nothing to do
    except (RuntimeError, AttributeError):
        pass

    # Clean up the partially initialized torchvision from sys.modules
    _torchvision_keys = [k for k in sys.modules if k == 'torchvision' or k.startswith('torchvision.')]
    for key in _torchvision_keys:
        del sys.modules[key]

    # Register stub operators that torchvision requires
    try:
        lib = torch.library.Library('torchvision', 'DEF')
        lib.define('nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor')
        lib.impl('nms', lambda dets, scores, iou_threshold: torch.zeros(0, dtype=torch.long), 'CPU')
        lib.impl('nms', lambda dets, scores, iou_threshold: torch.zeros(0, dtype=torch.long), 'CUDA')
        lib.impl_abstract('nms', lambda dets, scores, iou_threshold: dets.new_empty((0,), dtype=torch.long))
    except Exception:
        pass

    try:
        lib2 = torch.library.Library('torchvision', 'IMPL')
        # Also try registering roi_align if needed
    except Exception:
        pass

    # Retry torchvision import
    try:
        import torchvision  # noqa: F401
    except Exception:
        pass  # Best-effort; transformers will handle gracefully if torchvision unavailable


# Apply patch at import time
_patch_torchvision()
