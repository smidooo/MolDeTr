"""Deterministic-run helpers.

``set_seed`` seeds Python, NumPy, and PyTorch (CPU + CUDA) so any run touching the stochastic
augmentation pipeline (``moldetr/dataloader/data_augmentation.py``, ``shimming.py``) is
reproducible. Matches the seed (42) used for the article's evaluation.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs used by the pipeline for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: give each worker a deterministic, distinct seed."""
    worker_seed = (np.random.get_state()[1][0] + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(int(worker_seed))
