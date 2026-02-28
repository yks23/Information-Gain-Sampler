"""
Generators module: core generation functions for Masked Diffusion Models.

Submodules:
    - base: Core generate() function and shared utilities
    - eb_sampler: Entropy-Based Sampler
    - fast_dllm: Fast dLLM generation with dynamic thresholding

Note: Info-Gain sampler is now directly available from dllm.pipelines.info_gain
"""

from .base import (
    generate_with_beam_search,
    add_gumbel_noise,
    get_num_transfer_tokens,
    get_transfer_index,
    apply_eos_penalty,
    load_baseline,
    pc_sampler_function,
    BeamCandidate,
    _kv_cache_forward,
    _truncate_kv_cache,
)
# Backward compatibility alias
generate = generate_with_beam_search
from .eb_sampler import generate_with_eb_sampler
from .fast_dllm import generate_with_fast_dllm, get_transfer_index_dynamic

