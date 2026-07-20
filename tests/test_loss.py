"""Tests for the loss cluster (moldetr.loss.individual_losses) — previously zero coverage.

Covers the primitives (weighted L1, sigmoid focal, 1-D inter/union) and one matcher->loss integration:
the parameter loss is zero on a perfect match. All weight-free / CPU.
"""

import torch

from moldetr.config import CostWeighting
from moldetr.loss.individual_losses import (
    calculate_inter_union,
    parameter_loss,
    sigmoid_focal_loss,
    weighted_l1_loss,
)
from moldetr.matcher.matcher import matching

N_CLASSES = 5
N_PARAMS = 7


def test_weighted_l1_loss_zero_on_exact_match_with_finite_grad():
    x = torch.randn(4, N_PARAMS, requires_grad=True)
    assert torch.allclose(
        weighted_l1_loss(x, x.detach().clone(), torch.ones(N_PARAMS)), torch.zeros(4, N_PARAMS)
    )
    weighted_l1_loss(x, torch.zeros(4, N_PARAMS), torch.ones(N_PARAMS)).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_weighted_l1_loss_scales_with_weight():
    out, tgt = torch.zeros(3), torch.ones(3)
    assert torch.allclose(weighted_l1_loss(out, tgt, torch.full((3,), 2.0)), torch.full((3,), 2.0))


def test_sigmoid_focal_loss_rewards_confident_correct_over_confident_wrong():
    targets = torch.zeros(1, 1, N_CLASSES)
    targets[0, 0, 2] = 1.0  # true class is index 2
    correct = torch.full((1, 1, N_CLASSES), -10.0)
    correct[0, 0, 2] = 10.0  # confident + right
    wrong = torch.full((1, 1, N_CLASSES), -10.0)
    wrong[0, 0, 0] = 10.0  # confident + wrong
    loss_correct = sigmoid_focal_loss(correct, targets)
    loss_wrong = sigmoid_focal_loss(wrong, targets)
    assert torch.isfinite(loss_correct) and loss_correct < loss_wrong


def test_inter_union_identical_lines_have_iou_one():
    lines = torch.tensor([[2.0], [5.0]])  # (2, N): row0 = left edge, row1 = right edge
    inter, union, _ = calculate_inter_union(lines, lines)
    assert torch.allclose(inter, union)  # intersection == union -> IoU 1


def test_inter_union_disjoint_lines_have_zero_intersection():
    out = torch.tensor([[0.0], [1.0]])
    tgt = torch.tensor([[2.0], [3.0]])
    inter, union, _ = calculate_inter_union(out, tgt)
    assert torch.allclose(inter, torch.zeros_like(inter))
    assert torch.all(union > 0)


def test_parameter_loss_is_zero_on_a_perfect_match():
    """Integration: the matcher pairs each target with its exact-param query -> L1 parameter loss is 0."""
    q, n_t = 5, 3
    tgt = torch.zeros(n_t, 1 + N_PARAMS)
    tgt[:, 0] = torch.tensor([1.0, 2.0, 3.0])
    for i in range(n_t):
        tgt[i, 1:] = float(i + 1)
    outputs = torch.zeros(1, q, N_CLASSES + N_PARAMS)
    for i in range(n_t):
        outputs[0, i, N_CLASSES:] = tgt[i, 1:]  # query i params == target i params
    outputs[0, 3, N_CLASSES:] = 99.0
    outputs[0, 4, N_CLASSES:] = -99.0
    targets = {"targets": [tgt], "num_targets": [n_t]}
    indices = matching(
        outputs,
        targets,
        lambda a, b: torch.zeros(a.shape[0], a.shape[1]),
        n_classes=N_CLASSES,
        cost_weighting=CostWeighting(1.0, 1.0, None),
        parameter_cost_weights=None,
    )
    loss = parameter_loss(
        outputs,
        targets,
        n_classes=N_CLASSES,
        parameter_weighting=torch.ones(N_PARAMS),
        indices=indices,
        reduction="sum",
    )
    assert abs(float(loss)) < 1e-5
