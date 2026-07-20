"""Tests for the training metrics (moldetr.metrics.multiplet_metrics) — previously zero public coverage.

The class-accuracy metrics take a pre-applied matcher. On a perfect match (each query predicts its target's
class + parameters) the without-empty accuracy is 1.0; both metrics always return a finite scalar in [0, 1].
"""

from functools import partial

import torch

from moldetr.config import CostWeighting
from moldetr.matcher.matcher import matching
from moldetr.metrics.multiplet_metrics import (
    accuracy_with_empty_object,
    accuracy_without_empty_object,
)

N_CLASSES, N_PARAMS = 5, 7


def _zero_giou(a, b):
    return torch.zeros(a.shape[0], a.shape[1])


def _matcher_partial():
    return partial(
        matching,
        calculate_giou=_zero_giou,
        n_classes=N_CLASSES,
        cost_weighting=CostWeighting(1.0, 1.0, None),
        parameter_cost_weights=None,
    )


def _perfect_match_batch(q=5, n_t=3):
    """(outputs, targets) where query i confidently predicts target i's class + exact parameters."""
    tgt = torch.zeros(n_t, 1 + N_PARAMS)
    classes = [0, 1, 2][:n_t]
    tgt[:, 0] = torch.tensor([float(c) for c in classes])
    for i in range(n_t):
        tgt[i, 1:] = float(i + 1)
    outputs = torch.zeros(
        1, 1, q, N_CLASSES + N_PARAMS
    )  # (bs, n_groups=1, queries, n_classes+params)
    for i in range(n_t):
        outputs[0, 0, i, classes[i]] = 10.0  # confident correct class
        outputs[0, 0, i, N_CLASSES:] = tgt[i, 1:]  # exact params -> matched by the L1 cost
    outputs[0, 0, 3, N_CLASSES:] = 99.0  # decoys
    outputs[0, 0, 4, N_CLASSES:] = -99.0
    targets = {"targets": [tgt], "num_targets": [n_t]}
    return outputs, targets


def test_without_empty_accuracy_is_one_on_a_perfect_match():
    outputs, targets = _perfect_match_batch()
    acc = accuracy_without_empty_object(outputs, targets, _matcher_partial(), N_CLASSES, n_groups=1)
    assert torch.isclose(acc, torch.tensor(1.0), atol=1e-6)


def test_accuracy_metrics_return_a_finite_fraction():
    outputs, targets = _perfect_match_batch()
    for metric in (accuracy_with_empty_object, accuracy_without_empty_object):
        acc = metric(outputs, targets, _matcher_partial(), N_CLASSES, n_groups=1)
        assert torch.isfinite(acc)
        assert 0.0 <= float(acc) <= 1.0
