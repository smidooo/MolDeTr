"""Helper functions for loss computation (Hungarian-match permutation indexing)."""

"""Helper functions for loss computation (Hungarian-match permutation indexing)."""

import torch

from typing import Callable

Get_Best_Class_Parameters_Partial = Callable[[torch.Tensor], torch.Tensor]


def get_src_permutation_idx(
    indices: list[tuple[torch.tensor, torch.tensor]]
) -> tuple[torch.tensor, torch.tensor]:
    """
    get indices of inputs to be permuted according to Hungarian algorithm.
    Parameters
    ----------
    indices: list of tuples containing the indices of the inputs and targets to be permuted

    Returns: batch index and source index
    -------

    """
    # permute predictions following indices
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
