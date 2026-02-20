"""
Generators module: core generation functions for Masked Diffusion Models.

Submodules:
    - base: Core generate() function and shared utilities
    - info_gain: Info-Gain Sampler (瞬时熵 + 下一步平均熵)
    - eb_sampler: Entropy-Based Sampler
    - fast_dllm: Fast dLLM generation with dynamic thresholding
"""

from .base import (
    generate,
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
from .info_gain import (
    beam_search_expand_candidate,
    compute_entropy_info_gain,
    _lookahead_with_kv_cache,
    generate_with_info_gain,
)
from .eb_sampler import generate_with_eb_sampler
from .fast_dllm import generate_with_fast_dllm, get_transfer_index_dynamic

