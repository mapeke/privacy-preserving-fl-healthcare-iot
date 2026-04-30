"""Reproducibility helpers.

Sets seeds for Python's ``random``, NumPy, and PyTorch (CPU + CUDA), and optionally enables
deterministic CuDNN. Call :func:`set_global_seed` once at the start of every entry point.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python ``random``, NumPy, and PyTorch.

    Args:
        seed: Non-negative integer seed.
        deterministic_torch: If True, force CuDNN into deterministic mode. Slightly slower but
            removes a major source of non-reproducibility.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
