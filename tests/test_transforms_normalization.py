"""Source-level round-trip + invariance tests for the dataloader transforms/normalization (weight-free).

The `[sum, min, max, std]` coupling embedding must be order-invariant (it's how a variable-length,
order-free coupling set becomes a fixed vector), and normalize<->untransform must round-trip so the
predicted parameters decode back to physical units without drift.
"""

import torch

from moldetr.dataloader.normalization import permutation_invariant_coupling_constant_embedding
from moldetr.dataloader.transforms import Normalize


def test_coupling_embedding_is_permutation_invariant():
    a = permutation_invariant_coupling_constant_embedding([7.0, 8.0, 2.0])
    b = permutation_invariant_coupling_constant_embedding([2.0, 8.0, 7.0])
    assert a == b  # sum, min, max, std are all order-independent


def test_coupling_embedding_values():
    sum_, min_, max_, std_ = permutation_invariant_coupling_constant_embedding([3.0, 9.0])
    assert (sum_, min_, max_) == (12.0, 3.0, 9.0)
    assert abs(std_ - 3.0) < 1e-9  # std of [3, 9] is 3


def test_normalize_untransform_round_trips_float(extrema):
    norm = Normalize(extrema)
    for index in range(min(4, len(extrema))):
        v = 42.0
        assert abs(norm.untransform(norm.transform(v, index), index) - v) < 1e-6


def test_normalize_untransform_round_trips_tensor(extrema):
    norm = Normalize(extrema)
    n = len(extrema)
    raw = torch.arange(1, n + 1, dtype=torch.float32)
    normed = torch.tensor([norm.transform(float(raw[i]), i) for i in range(n)])
    for i in range(n):
        assert torch.allclose(norm.untransform(normed, i), raw[i], atol=1e-5)
