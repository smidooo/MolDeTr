"""Deterministic-run helpers.

``set_seed`` seeds Python, NumPy, and PyTorch (CPU + CUDA) so any run touching the stochastic
augmentation pipeline (``moldetr/dataloader/data_augmentation.py``) is reproducible. Matches the
seed (42) used for the article's evaluation.

The shim branch of that pipeline does not run in this distribution, but the raise is not the
reason: ``augment_distortions`` pins ``toss_coin`` to ``0.99``, so neither the shim branch nor the
line-broadening branch is reached. ``add_shim_distortions`` raising (its GPL-derived simulator was
removed -- see ``THIRD_PARTY.md``) is what a *direct* caller gets instead of an ``AttributeError``.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs used by the pipeline for reproducible runs.

    Requires the ``model`` extra: torch is imported first, on purpose. Seeding Python and NumPy
    before reaching for it left a torch-free install with a half-reseeded process *and* an
    exception, which the caller has no way to unpick.
    """
    import torch  # noqa: PLC0415  (deliberately first: fail before mutating any global RNG)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: give each worker a deterministic, distinct seed."""
    worker_seed = (np.random.get_state()[1][0] + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(int(worker_seed))
