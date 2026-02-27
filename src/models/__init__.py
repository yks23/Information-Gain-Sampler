"""
Model adapters for different Masked Diffusion Models.

Usage:
    from src.models import get_model_adapter

    # Load from local path in ./model/ directory (preferred)
    adapter = get_model_adapter("llada", device="cuda:0")      # ./model/llada/
    adapter = get_model_adapter("dream", device="cuda:0")       # ./model/dream/
    adapter = get_model_adapter("sdar", device="cuda:0")        # ./model/sdar/
    
    # Load from absolute path
    adapter = get_model_adapter("/model/llada", device="cuda:0")
    
    # Load from HuggingFace Hub
    adapter = get_model_adapter("GSAI-ML/LLaDA-8B-Instruct", device="cuda:0")
    adapter = get_model_adapter("JetLM/SDAR-8B-Chat", device="cuda:0")
    
    # adapter.model, adapter.tokenizer, adapter.mask_id are available
    # adapter.forward(x) returns logits with model-specific processing
"""

import os
from pathlib import Path
from .base import BaseModelAdapter
from .dream import DreamAdapter
from .llada import LLaDAAdapter
from .sdar import SDARAdapter

# Auto-regressive baselines
from .ar import LlamaAdapter, MistralAdapter, QwenAdapter


def _is_local_path(model_name: str) -> bool:
    """
    Check if model_name is a local file path.
    
    Returns True if:
    - It's an absolute path (starts with /)
    - It's a relative path that exists as a directory
    - It contains path separators (/, \)
    - It exists in the default model/ directory
    """
    # Absolute path
    if os.path.isabs(model_name):
        return os.path.exists(model_name) and os.path.isdir(model_name)
    
    # Relative path - check if it exists
    if os.path.exists(model_name) and os.path.isdir(model_name):
        return True
    
    # Check in default model/ directory
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "model"
    model_path = model_dir / model_name
    if model_path.exists() and model_path.is_dir():
        return True
    
    # Contains path separators (likely a path)
    if '/' in model_name or '\\' in model_name:
        # Check if parent directory exists
        path_obj = Path(model_name)
        if path_obj.parent.exists():
            return True
    
    return False


def _detect_model_type_from_path(model_path: str) -> str:
    """
    Detect model type from local path by examining directory name and contents.
    
    Returns: 'dream', 'sdar', 'llada', 'mistral', 'qwen', 'llama', or 'llada' (default)
    """
    path_lower = model_path.lower()
    path_name = os.path.basename(model_path).lower()
    
    # Check directory name
    if 'dream' in path_name or 'dream' in path_lower:
        return 'dream'
    elif 'sdar' in path_name or 'sdar' in path_lower:
        return 'sdar'
    elif 'llada' in path_name or 'llada' in path_lower:
        return 'llada'
    elif 'mistral' in path_name or 'mistral' in path_lower:
        return 'mistral'
    elif 'qwen' in path_name or 'qwen' in path_lower:
        return 'qwen'
    elif 'llama' in path_name or 'llama' in path_lower:
        return 'llama'
    
    # Check config.json if exists
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_type = config.get('model_type', '').lower()
                if 'dream' in model_type:
                    return 'dream'
                elif 'llada' in model_type or 'llama' in model_type:
                    # Check architecture to distinguish
                    arch = config.get('architectures', [])
                    if arch and 'Dream' in str(arch):
                        return 'dream'
                    elif arch and 'LLaDA' in str(arch):
                        return 'llada'
        except Exception:
            pass
    
    # Default to LLaDA for MDM models
    return 'llada'


def get_model_adapter(model_name: str, device: str = "cuda:0") -> BaseModelAdapter:
    """
    Auto-detect model type and return the appropriate adapter.

    Supports both local paths and HuggingFace Hub model names.
    Local paths are preferred and checked first.
    
    Args:
        model_name: Local path to model directory OR HuggingFace Hub model name
        device: Device to load model on (e.g., "cuda:0", "cpu")
    
    Detection logic:
        1. If model_name is a local path (exists as directory):
           - Detect model type from path name and config.json
           - Use local path directly
        2. If model_name is a HuggingFace Hub name:
           - Detect model type from name substring (case-insensitive)
        
        Detection order (by model name/path substring, case-insensitive):
        1. 'dream'   -> DreamAdapter
        2. 'sdar'    -> SDARAdapter
        3. 'llada'   -> LLaDAAdapter
        4. 'mistral' -> MistralAdapter
        5. 'qwen'    -> QwenAdapter
        6. 'llama'   -> LlamaAdapter
        7. default   -> LLaDAAdapter
    
    Examples:
        # Local path in ./model/ directory (preferred)
        adapter = get_model_adapter("llada", device="cuda:0")      # ./model/llada/
        adapter = get_model_adapter("dream", device="cuda:0")      # ./model/dream/
        adapter = get_model_adapter("sdar", device="cuda:0")       # ./model/sdar/
        
        # Absolute path
        adapter = get_model_adapter("/model/llada", device="cuda:0")
        adapter = get_model_adapter("/model/dream", device="cuda:0")
        
        # HuggingFace Hub
        adapter = get_model_adapter("GSAI-ML/LLaDA-8B-Instruct", device="cuda:0")
        adapter = get_model_adapter("Dream-org/Dream-v0-Instruct-7B", device="cuda:0")
        adapter = get_model_adapter("JetLM/SDAR-8B-Chat", device="cuda:0")
    """
    # Check if it's a local path
    is_local = _is_local_path(model_name)
    
    if is_local:
        # Check in ./model/ directory first (project root/model/)
        project_root = Path(__file__).parent.parent.parent
        model_dir = project_root / "model"
        model_path_in_model_dir = model_dir / model_name
        
        # Resolve to absolute path
        if model_path_in_model_dir.exists() and model_path_in_model_dir.is_dir():
            # Model found in ./model/ directory (preferred)
            model_path = str(model_path_in_model_dir.resolve())
            print(f"[INFO] Model found in ./model/ directory: {model_path}")
        elif os.path.isabs(model_name) and os.path.exists(model_name) and os.path.isdir(model_name):
            # Use provided absolute path directly
            model_path = os.path.abspath(model_name)
            print(f"[INFO] Using provided absolute path: {model_path}")
        elif os.path.exists(model_name) and os.path.isdir(model_name):
            # Use provided relative path directly
            model_path = os.path.abspath(model_name)
            print(f"[INFO] Using provided relative path: {model_path}")
        else:
            raise FileNotFoundError(
                f"Model path does not exist: {model_name}\n"
                f"  Checked in ./model/ directory: {model_path_in_model_dir}\n"
                f"  Checked as provided path: {os.path.abspath(model_name)}\n"
                f"  Expected location: {model_dir.resolve()}/<model_name>\n"
                f"  Available models in ./model/: {', '.join([d.name for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]) if model_dir.exists() else 'none'}"
            )
        
        if not os.path.isdir(model_path):
            raise NotADirectoryError(f"Model path is not a directory: {model_path}")
        
        # Detect model type from path
        model_type = _detect_model_type_from_path(model_path)
        print(f"[INFO] Loading model from local path: {model_path}")
        print(f"[INFO] Detected model type: {model_type}")
        
        # Use the detected path
        actual_model_name = model_path
    else:
        # HuggingFace Hub model name
        model_type = None
        name_lower = model_name.lower()
        actual_model_name = model_name

        # Detect from name
        if 'dream' in name_lower:
            model_type = 'dream'
        elif 'sdar' in name_lower:
            model_type = 'sdar'
        elif 'llada' in name_lower:
            model_type = 'llada'
        elif 'mistral' in name_lower:
            model_type = 'mistral'
        elif 'qwen' in name_lower:
            model_type = 'qwen'
        elif 'llama' in name_lower:
            model_type = 'llama'
        else:
            # Default to LLaDA for unknown MDM models
            model_type = 'llada'
        
        print(f"[INFO] Loading model from HuggingFace Hub: {actual_model_name}")
        print(f"[INFO] Detected model type: {model_type}")
    
    # Select adapter class
    if model_type == 'dream':
        adapter_cls = DreamAdapter
    elif model_type == 'sdar':
        adapter_cls = SDARAdapter
    elif model_type == 'llada':
        adapter_cls = LLaDAAdapter
    elif model_type == 'mistral':
        adapter_cls = MistralAdapter
    elif model_type == 'qwen':
        adapter_cls = QwenAdapter
    elif model_type == 'llama':
        adapter_cls = LlamaAdapter
    else:
        # Default to LLaDA for unknown MDM models
        adapter_cls = LLaDAAdapter

    # Create adapter with actual model name/path
    adapter = adapter_cls(actual_model_name, device)
    return adapter

