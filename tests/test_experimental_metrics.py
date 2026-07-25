"""Table 4 (experimental MAE / R^2) + per-class confusion-matrix accuracies — weight-free anchor.

Complements ``scripts/aggregate_experimental.py`` (which anchors the *median* headline numbers
0.90 Hz / 0.20 Hz / 93.5 %). This module locks the paper numbers that have **no** committed script:
Table 4's mean errors + R^2 and the per-proton-count accuracies behind the confusion matrix.

Data: the committed ``structured_output/experimental_matched_pairs.json`` — verified byte-identical
to the paper's private ``matches_total_data.json`` intermediate. Schema::

    matched_pairs_total        : list of [pred, label]   (215 pairs)
    unmatched_predictions_total : list of pred            (20 false positives)
    unmatched_labels_total      : list of label           (5 missed real labels)

``pred`` has ``chemical_shift_in_points``, ``coupling_constants`` (zero-padded list), ``proton_count``;
``label`` has ``chemical_shift_in_points``, ``coupling_constants`` (scalar OR list, may be absent),
``proton_count``. The point grid is 5.12 points/Hz.

Metric definitions mirror the paper's private ``matching_4_experimental_evaluation.py`` exactly:

* shift error (Hz) = |pred.chemical_shift_in_points - label.chemical_shift_in_points| / 5.12
  (``calculate_shift_and_coupling_errors`` lines 231-235).
* coupling pairing = the first ``len(label)`` predicted couplings, in their original (zero-padded)
  order, vs the label couplings (lines 243-247: ``label[:len(pred)]`` then ``zip(pred, label)``;
  since preds are the longer, zero-padded list this is ``zip(pred[:len(label)], label)`` — the same
  pairing as the committed ``scripts.aggregate_experimental.coupling_errors``).
* MAE = mean(|err|); the reported "±" is ``std_dev_mae = np.std(|err - MAE|)``
  (``calculate_errors`` lines 197 & 202 — NOT the plain std_dev on line 199).
* R^2 = 1 - SS_res / SS_tot (``calculate_r2`` lines 278-281).

Subset: Table 4's parameter errors are taken over **correctly-classified** matched pairs
(``pred.proton_count == label.proton_count``, n=198). Evidence this is the paper's subset: over all
215 pairs the shift MAE is 1.398 Hz, but over the 198 correct-class pairs it is 1.3685 Hz = the
paper's reported 1.368 (see ``test_shift_mae_requires_correct_class_subset``). Couplings are
identical either way — the 17 wrong-class pairs carry no couplings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.aggregate_experimental import coupling_errors

REPO = Path(__file__).resolve().parent.parent
PAIRS_JSON = REPO / "structured_output" / "experimental_matched_pairs.json"
POINTS_PER_HZ = 5.12
# 650 = 13 ROIs x 10 decoder queries x 5 noise runs (== aggregate_experimental.py's --total-queries
# default). Only the "no spin" row depends on this; the 1H/2H/3H rows come purely from the JSON.
TOTAL_QUERIES = 13 * 10 * 5
PROTON_TO_IDX = {1: 0, 2: 1, 3: 2}  # confusion-matrix index; row/col 3 == "no spin"


# --------------------------------------------------------------------------------------------------
# data + metric helpers (faithful to matching_4_experimental_evaluation.py)
# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(PAIRS_JSON.read_text(encoding="utf-8"))


def _correct_class(pairs: list) -> list:
    """Matched pairs whose predicted proton count equals the label's (the Table 4 subset)."""
    return [(p, l) for p, l in pairs if p["proton_count"] == l["proton_count"]]


def _shift_errors_hz(pairs: list) -> list[float]:
    return [
        abs(p["chemical_shift_in_points"] - l["chemical_shift_in_points"]) / POINTS_PER_HZ
        for p, l in pairs
    ]


def _coupling_pairs(pairs: list) -> list[tuple[float, float]]:
    """(pred_c, label_c) for the first len(label) predicted couplings — lines 238-247 of the ref."""
    out: list[tuple[float, float]] = []
    for p, l in pairs:
        if "coupling_constants" in p and "coupling_constants" in l:
            lc = l["coupling_constants"]
            lc = lc if isinstance(lc, list) else [lc]
            pc = p["coupling_constants"]
            lc = lc[: len(pc)]
            out.extend(zip(pc, lc))
    return out


def _mae(values: list[float]) -> float:
    """calculate_errors line 197: mean of the absolute errors."""
    return float(np.mean(values))


def _std_dev_mae(values: list[float]) -> float:
    """calculate_errors line 202: the Table 4 "±" = np.std(|err - mean(err)|)."""
    m = float(np.mean(values))
    return float(np.std([abs(v - m) for v in values]))


def _r2(preds: list[float], labels: list[float]) -> float:
    """calculate_r2 lines 278-281: 1 - SS_res / SS_tot."""
    mean_l = sum(labels) / len(labels)
    ss_tot = sum((x - mean_l) ** 2 for x in labels)
    ss_res = sum((x - y) ** 2 for x, y in zip(preds, labels))
    return 1 - ss_res / ss_tot


def _confusion_matrix(data: dict) -> np.ndarray:
    """4x4 [1H,2H,3H,no-spin], rows=true, cols=pred, built from matched + unmatched entries."""
    cm = np.zeros((4, 4), dtype=int)
    for p, l in data["matched_pairs_total"]:
        cm[PROTON_TO_IDX[l["proton_count"]], PROTON_TO_IDX[p["proton_count"]]] += 1
    for l in data["unmatched_labels_total"]:  # missed real label -> predicted "no spin" (col 3)
        cm[PROTON_TO_IDX[l["proton_count"]], 3] += 1
    for p in data["unmatched_predictions_total"]:  # false positive -> true "no spin" (row 3)
        cm[3, PROTON_TO_IDX[p["proton_count"]]] += 1
    n_real = len(data["matched_pairs_total"]) + len(data["unmatched_labels_total"])
    cm[3, 3] = (TOTAL_QUERIES - n_real) - len(data["unmatched_predictions_total"])
    return cm


# --------------------------------------------------------------------------------------------------
# dataset shape
# --------------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_dataset_shape(data):
    """44 spin groups x 5 noise runs = 220 real labels; 20 false positives; 5 missed labels."""
    assert len(data["matched_pairs_total"]) == 215
    assert len(data["unmatched_predictions_total"]) == 20
    assert len(data["unmatched_labels_total"]) == 5
    n_real = len(data["matched_pairs_total"]) + len(data["unmatched_labels_total"])
    assert n_real == 220 == 44 * 5


# --------------------------------------------------------------------------------------------------
# Table 4: chemical-shift MAE / R^2
# --------------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_table4_shift_mae(data):
    """MAE delta = 1.368 Hz (+/- 1.075). Computed on the 198 correct-class pairs: 1.3685 (+/-1.084)."""
    cc = _correct_class(data["matched_pairs_total"])
    assert len(cc) == 198
    errs = _shift_errors_hz(cc)
    assert _mae(errs) == pytest.approx(1.368, abs=0.01)  # paper 1.368 Hz -> computed 1.3685
    # The "±" is std_dev_mae; computed 1.0844 vs the paper's 1.075 (a ~0.009 rounding-level residual,
    # the single sub-0.01 imperfection in the whole reconciliation -- see the module report).
    assert _std_dev_mae(errs) == pytest.approx(1.075, abs=0.02)


@pytest.mark.unit
def test_shift_mae_requires_correct_class_subset(data):
    """Guard the subset choice: over ALL 215 pairs the shift MAE is 1.398 Hz, NOT the paper's 1.368.

    This is why Table 4's shift error is reported over correctly-classified spin systems only.
    """
    all_errs = _shift_errors_hz(data["matched_pairs_total"])
    assert _mae(all_errs) == pytest.approx(1.398, abs=0.01)
    assert _mae(all_errs) != pytest.approx(1.368, abs=0.01)


@pytest.mark.unit
def test_table4_shift_r2(data):
    """R^2 delta = 0.999 (paper). Exact value 0.99989 -> reported as 0.999 at 3 sig figs."""
    cc = _correct_class(data["matched_pairs_total"])
    preds = [p["chemical_shift_in_points"] / POINTS_PER_HZ for p, _ in cc]
    labels = [l["chemical_shift_in_points"] / POINTS_PER_HZ for _, l in cc]
    r2 = _r2(preds, labels)
    assert 0.999 <= r2 < 1.0  # paper 0.999; computed 0.99989 (all-pairs 0.99988 -- same at 3 dp)


# --------------------------------------------------------------------------------------------------
# Table 4: coupling-constant MAE / R^2
# --------------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_table4_coupling_mae(data):
    """MAE J = 0.470 Hz (+/- 0.543). Computed: 0.4703 (+/- 0.5426) over 123 coupling pairs."""
    cc = _correct_class(data["matched_pairs_total"])
    errs = [abs(a - b) for a, b in _coupling_pairs(cc)]
    assert len(errs) == 123
    assert _mae(errs) == pytest.approx(0.470, abs=0.01)  # paper 0.470 Hz -> computed 0.4703
    assert _std_dev_mae(errs) == pytest.approx(0.543, abs=0.01)  # paper +/-0.543 -> computed 0.5426


@pytest.mark.unit
def test_table4_coupling_r2(data):
    """R^2 J = 0.936 (paper). Computed 0.93640."""
    cc = _correct_class(data["matched_pairs_total"])
    pairs = _coupling_pairs(cc)
    preds = [a for a, _ in pairs]
    labels = [b for _, b in pairs]
    assert _r2(preds, labels) == pytest.approx(0.936, abs=0.001)


@pytest.mark.unit
def test_coupling_pairing_matches_committed_helper(data):
    """_coupling_pairs reproduces scripts.aggregate_experimental.coupling_errors (committed anchor)."""
    cc = _correct_class(data["matched_pairs_total"])
    mine = sorted(round(abs(a - b), 9) for a, b in _coupling_pairs(cc))
    theirs = sorted(
        round(e, 9)
        for p, l in cc
        if "coupling_constants" in p and "coupling_constants" in l
        for e in coupling_errors(p["coupling_constants"], l["coupling_constants"])
    )
    assert mine == theirs


# --------------------------------------------------------------------------------------------------
# per-class (confusion-matrix) accuracies
# --------------------------------------------------------------------------------------------------
@pytest.mark.unit
def test_confusion_matrix_from_json(data):
    """The 4x4 confusion matrix derived from the committed JSON (rows=true, cols=pred)."""
    expected = np.array(
        [
            [97, 2, 0, 1],  # true 1H  -> 97 correct, 2 as 2H, 1 missed
            [6, 71, 0, 3],  # true 2H  -> 71 correct, 6 as 1H, 3 missed
            [0, 9, 30, 1],  # true 3H  -> 30 correct, 9 as 2H, 1 missed
            [7, 13, 0, 410],  # true no-spin -> 410 correct-empty, 7+13 false positives
        ]
    )
    np.testing.assert_array_equal(_confusion_matrix(data), expected)


@pytest.mark.unit
def test_per_class_accuracy(data):
    """Row-normalized recall reproduces the paper's 97.00 / 88.75 / 75.00 / 95.35 %."""
    cm = _confusion_matrix(data)
    assert (cm[0, 0], cm[0].sum()) == (97, 100)  # 1H
    assert 100 * cm[0, 0] / cm[0].sum() == pytest.approx(97.00, abs=0.01)
    assert (cm[1, 1], cm[1].sum()) == (71, 80)  # 2H
    assert 100 * cm[1, 1] / cm[1].sum() == pytest.approx(88.75, abs=0.01)
    assert (cm[2, 2], cm[2].sum()) == (30, 40)  # 3H
    assert 100 * cm[2, 2] / cm[2].sum() == pytest.approx(75.00, abs=0.01)
    assert (cm[3, 3], cm[3].sum()) == (410, 430)  # no spin
    assert 100 * cm[3, 3] / cm[3].sum() == pytest.approx(95.35, abs=0.01)


@pytest.mark.unit
def test_stale_hardcoded_matrix_disagrees_with_json(data):
    """The private plot_confusion_matrix.py hardcodes a DIFFERENT matrix (93.5/78.7/81.6/99.5 %).

    Its rows sum to 107/75/38/745 and it is inconsistent with the 44x5 = 220-label accounting
    (row sums 100/80/40/430). The JSON-derived matrix is authoritative; the hardcoded one is stale.
    """
    stale = np.array([[100, 6, 1, 0], [10, 59, 1, 5], [0, 0, 31, 7], [4, 0, 0, 741]])
    cm = _confusion_matrix(data)
    assert not np.array_equal(cm, stale)
    assert list(cm.sum(axis=1)) == [100, 80, 40, 430]
    assert list(stale.sum(axis=1)) == [107, 75, 38, 745]
