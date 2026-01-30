'''
Utility functions for setting random seeds across libraries. USED FOR REPRODUCIBILITY.
'''
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def set_seed(seed: int) -> None:
    # Set random seed for reproducibility across random, numpy, and torch (All the same seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
