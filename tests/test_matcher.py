"""Tests for the Hungarian matcher (moldetr.matcher.matcher) — previously zero coverage.

The matcher assigns model queries to targets via scipy's linear_sum_assignment. These lock:
- it returns a valid partial permutation (each query/target used at most once),
- it recovers the correct assignment when a query's parameters exactly match a target (an MFT),
- it no longer writes a stray cost.txt into the working directory (regression guard for that fix).
"""

import os

import torch

from moldetr.config import CostWeighting
from moldetr.matcher.matcher import matching

N_CLASSES = 5
N_PARAMS = 7  # PARAM_NAMES: center, line_width, bbox_range, cc1..cc4


def _zero_giou(out_param_matrix, tgt_param_matrix):
    """Trivial GIoU stub -> a constant cost that doesn't bias the assignment (isolates class + L1)."""
    return torch.zeros(out_param_matrix.shape[0], out_param_matrix.shape[1])


def _match(outputs, tgt):
    targets = {"targets": [tgt], "num_targets": [tgt.shape[0]]}
    return matching(
        outputs,
        targets,
        _zero_giou,
        n_classes=N_CLASSES,
        cost_weighting=CostWeighting(1.0, 1.0, None),
        parameter_cost_weights=None,
    )


def test_matching_returns_a_valid_partial_permutation():
    torch.manual_seed(0)
    q, n_t = 6, 3
    outputs = torch.randn(1, q, N_CLASSES + N_PARAMS)
    tgt = torch.randn(n_t, 1 + N_PARAMS)
    tgt[:, 0] = torch.tensor([1.0, 2.0, 3.0])  # class labels
    ((row, col),) = _match(outputs, tgt)
    assert len(row) == len(col) == min(q, n_t)
    assert len(set(row.tolist())) == len(row)  # no query used twice
    assert len(set(col.tolist())) == len(col)  # no target used twice
    assert set(col.tolist()) == set(range(n_t))  # every target matched
    assert all(0 <= r < q for r in row.tolist())


def test_matching_recovers_identity_when_params_align():
    """MFT: with uniform class logits, the L1 parameter cost matches each target to its exact-param query."""
    q, n_t = 5, 3
    tgt = torch.zeros(n_t, 1 + N_PARAMS)
    tgt[:, 0] = torch.tensor([1.0, 2.0, 3.0])
    for i in range(n_t):
        tgt[i, 1:] = float(i + 1)  # distinct parameters per target
    outputs = torch.zeros(1, q, N_CLASSES + N_PARAMS)  # class logits all 0 -> no class bias
    for i in range(n_t):
        outputs[0, i, N_CLASSES:] = tgt[i, 1:]  # query i's params == target i's params (L1 cost 0)
    outputs[0, 3, N_CLASSES:] = 99.0  # decoy queries, far from every target
    outputs[0, 4, N_CLASSES:] = -99.0
    ((row, col),) = _match(outputs, tgt)
    mapping = dict(zip(col.tolist(), row.tolist()))  # target -> query
    for i in range(n_t):
        assert mapping[i] == i


def test_matching_does_not_write_cost_txt(tmp_path, monkeypatch):
    """Regression guard: matching() must not litter the CWD with a debug cost.txt."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(1)
    outputs = torch.randn(1, 4, N_CLASSES + N_PARAMS)
    tgt = torch.randn(2, 1 + N_PARAMS)
    tgt[:, 0] = torch.tensor([1.0, 2.0])
    _match(outputs, tgt)
    assert not (tmp_path / "cost.txt").exists()
    assert not os.path.exists("cost.txt")
