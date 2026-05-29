"""
Reproducibility utilities
"""
import random
import os
import warnings
import numpy as np
import torch
from typing import Optional

# Suppress known non-critical warnings
warnings.filterwarnings('ignore', message='.*cumsum_cuda_kernel does not have a deterministic.*')
warnings.filterwarnings('ignore', message='.*Setting `pad_token_id` to `eos_token_id`.*')


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set CUBLAS workspace config for CUDA >= 10.2 determinism
        if torch.cuda.is_available():
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # For PyTorch 1.8+
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                # Use warn_only=True to allow non-deterministic ops with warning
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older PyTorch without warn_only parameter
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass

    os.environ['PYTHONHASHSEED'] = str(seed)


def get_reproducibility_info() -> dict:
    """Get information about current reproducibility state"""
    info = {
        'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'not set'),
        'numpy_seed': 'set',
        'torch_seed': 'set',
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cudnn_deterministic'] = torch.backends.cudnn.deterministic
        info['cudnn_benchmark'] = torch.backends.cudnn.benchmark

    return info


class SeedContext:
    """Context manager for temporary seed setting"""

    def __init__(self, seed: int):
        self.seed = seed
        self.saved_state = None

    def __enter__(self):
        self.saved_state = {
            'random': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            self.saved_state['cuda'] = torch.cuda.get_rng_state_all()

        set_seed(self.seed, deterministic=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.saved_state['random'])
        np.random.set_state(self.saved_state['numpy'])
        torch.set_rng_state(self.saved_state['torch'])
        if torch.cuda.is_available() and 'cuda' in self.saved_state:
            torch.cuda.set_rng_state_all(self.saved_state['cuda'])
        return False
