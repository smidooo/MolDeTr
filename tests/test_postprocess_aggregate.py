"""Unit tests for the decode + aggregate helpers."""

import numpy as np

from moldetr.postprocess import decode_predictions, _untransform, PROTON_COUNTS
from scripts.aggregate_experimental import coupling_errors


EXTREMA = {
    "center_position_in_points": [0, 6143],
    "line_width_in_points": [1.5, 11.3],
    "bounding_box_range_in_points": [0.0, 408.8],
    "coupling_constant_1_in_points": [0.0, 294.3],
    "coupling_constant_2_in_points": [0.0, 102.4],
    "coupling_constant_3_in_points": [0.0, 102.4],
    "coupling_constant_4_in_points": [0.0, 51.1],
}


def test_untransform_endpoints():
    normed = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    phys = _untransform(normed, EXTREMA)
    assert phys[0] == 6143  # center at max
    assert phys[1] == 1.5  # line width at min


def test_coupling_errors_pairs_top_k():
    # label has one true coupling; prediction padded with zeros
    assert (
        coupling_errors([7.25, 0.0, 0.0], [7.13]) == [round(abs(7.25 - 7.13), 10)]
        or abs(coupling_errors([7.25, 0.0, 0.0], [7.13])[0] - 0.12) < 1e-9
    )


def test_decode_merges_duplicate_queries():
    # two near-identical high-confidence queries at the same shift -> one detection
    n_classes, n_params = 5, 7
    q = np.zeros(n_classes + n_params)
    q[2] = 5.0  # strong class-2 logit -> proton_count 3
    q[n_classes] = 0.5  # center at mid-range
    output = np.stack([q, q.copy(), q.copy()])
    preds = decode_predictions(output, EXTREMA, points_per_hz=5.12, threshold=0.3)
    assert len(preds) == 1
    assert preds[0]["proton_count"] == PROTON_COUNTS[2] == 3


def test_decode_drops_low_confidence():
    q = np.full(12, -10.0)  # all logits very negative -> no object
    preds = decode_predictions(q[None, :], EXTREMA, points_per_hz=5.12, threshold=0.3)
    assert preds == []


def test_decode_emits_single_max_coupling():
    """The 4 regression "coupling" slots are a permutation-invariant embedding
    ``[sum, min, max, std]`` (see moldetr.dataloader.normalization), NOT four couplings.
    The live decode must report only the physical max(J) = the 'max' slot (phys[5])."""
    n_classes = 5
    q = np.zeros(n_classes + 7)
    q[2] = 5.0  # strong class-2 logit -> kept, proton_count 3
    q[n_classes + 0] = 0.5  # center (mid-range)
    q[n_classes + 3] = 0.5  # sum slot  -> ~28.7 Hz, must NOT appear as a coupling
    q[n_classes + 4] = 0.1  # min slot
    q[n_classes + 5] = 0.4  # max slot  -> the one physical coupling
    q[n_classes + 6] = 0.2  # std slot
    preds = decode_predictions(q[None, :], EXTREMA, points_per_hz=5.12, threshold=0.3)
    assert len(preds) == 1
    couplings = preds[0]["coupling_constants_hz"]
    assert len(couplings) == 1  # exactly one coupling, not four embedding stats
    expected_max_j = 0.4 * 102.4 / 5.12  # max slot (extrema [0,102.4]) -> Hz
    assert abs(couplings[0] - expected_max_j) < 1e-9


def test_decode_drops_max_coupling_below_min():
    """If the 'max' slot is below ``min_coupling_hz`` the multiplet has no reportable
    coupling — even though the larger sum/min/std slots are non-zero (they are not couplings)."""
    n_classes = 5
    q = np.zeros(n_classes + 7)
    q[2] = 5.0
    q[n_classes + 0] = 0.5  # center
    q[n_classes + 3] = 0.5  # sum slot large (~28.7 Hz) -> must be ignored
    q[n_classes + 5] = 0.02  # max slot -> ~0.4 Hz, below min_coupling_hz (0.5)
    preds = decode_predictions(q[None, :], EXTREMA, points_per_hz=5.12, threshold=0.3)
    assert len(preds) == 1
    assert preds[0]["coupling_constants_hz"] == []


def test_per_class_accuracy_counts_unmatched_labels_as_miss():
    """An unmatched label is a miss for its class (model predicted 'no spin' there) — this is what
    makes the per-class denominator match the confusion matrix / the paper's per-class numbers."""
    from scripts.aggregate_experimental import per_class_accuracy

    matched = [
        ({"proton_count": 1}, {"proton_count": 1}),  # 1H correct
        ({"proton_count": 2}, {"proton_count": 1}),  # 1H label, predicted 2H -> miss
        ({"proton_count": 2}, {"proton_count": 2}),  # 2H correct
    ]
    unmatched_labels = [{"proton_count": 1}]  # a 1H label the model missed entirely
    acc = per_class_accuracy(matched, unmatched_labels)
    assert acc[1] == (1, 3)  # 1 correct of 3 total 1H (2 matched + 1 unmatched)
    assert acc[2] == (1, 1)
    # Without the unmatched labels the 1H denominator drops to the 2 matched.
    assert per_class_accuracy(matched)[1] == (1, 2)


def test_regression_stats_mae_and_r2():
    """MAE and R² (Table 4) on a hand-checkable case: exact shifts, a known coupling error."""
    from scripts.aggregate_experimental import regression_stats

    pairs = [
        (
            {"chemical_shift_in_points": 512.0, "coupling_constants": [7.0]},
            {"chemical_shift_in_points": 512.0, "coupling_constants": [7.0]},
        ),
        (
            {"chemical_shift_in_points": 1024.0, "coupling_constants": [8.0]},
            {"chemical_shift_in_points": 1024.0, "coupling_constants": [10.0]},
        ),
    ]
    r = regression_stats(pairs)
    assert r["mae_dshift_hz"] == 0.0 and r["r2_dshift"] == 1.0  # shifts exact
    assert r["mae_dJ_hz"] == 1.0  # coupling errors |0|, |2| -> mean 1.0
    assert abs(r["r2_dJ"] - (1 - 4 / 4.5)) < 1e-9  # ss_res=4, ss_tot=4.5
