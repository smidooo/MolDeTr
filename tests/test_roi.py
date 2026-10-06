"""Weight-free tests for ``moldetr.roi``: the benchmark-ROI loader and the paper's matcher.

Both exist because of a measured failure, not a hypothetical one (CI, 2026-10-06, real checkpoint,
13 Zenodo ROIs x 5 noise seeds, paper matcher): a downloader who builds the model input from
``spectrum_raw`` instead of ``spectrum_padded`` gets 110/220 labels matched with 116 false positives,
against 204/220 and 17 for the frame the labels live in. Seven ROIs start ``spectrum_raw`` at a
non-zero ``metadata.padding_before``, so the wrong array shifts every peak 195-390 Hz off its label.

The fixtures below rebuild that layout in miniature: a short raw ROI embedded at an offset inside a
6144-point padded spectrum, labels indexed in the padded frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moldetr.labels import Multiplet
from moldetr.roi import load_roi, match_hungarian

N = 6144
PAD_BEFORE = 1500
RAW_LEN = 2000
PEAK_IN_RAW = 700  # the one peak, in raw-frame points
PEAK_PADDED = PAD_BEFORE + PEAK_IN_RAW  # where the label lives


def _write_roi(path: Path, *, with_padded: bool = True, padded_len: int = N) -> Path:
    x = np.arange(RAW_LEN)
    # Complex, as in the deposit: the real part is the absorption line, the imaginary part dispersive.
    raw = np.exp(-0.5 * ((x - PEAK_IN_RAW) / 3.0) ** 2) * (1 + 0.5j) * 1e9
    padded = np.zeros(padded_len, dtype=complex)
    padded[PAD_BEFORE : PAD_BEFORE + RAW_LEN] = raw
    arrays = {
        "spectrum_raw": raw,
        "ppm_axis_padded": np.linspace(10.0, -5.0, padded_len),
        "metadata": np.array(
            {
                "points_per_hz": 5.12,
                "padding_before": PAD_BEFORE,
                "padding_after": padded_len - PAD_BEFORE - RAW_LEN,
                "compound": "Test compound",
                "base_frequency_mhz": 80.0,
            },
            dtype=object,
        ),
        "ground_truth": np.array(
            [
                {
                    "proton_count": 2,
                    "chemical_shift_in_points": float(PEAK_PADDED),
                    "coupling_constants": [7.5],
                    "chemical_shift_ppm": 4.0,
                }
            ],
            dtype=object,
        ),
    }
    if with_padded:
        arrays["spectrum_padded"] = padded
    np.savez(path, **arrays)
    return path


def test_load_roi_feeds_the_padded_frame_real_part(tmp_path: Path) -> None:
    roi = load_roi(_write_roi(tmp_path / "roi_SX.npz"))
    assert roi.spectrum.shape == (N,)
    assert roi.spectrum.dtype == np.float64
    assert not np.iscomplexobj(roi.spectrum)
    # The peak must sit where the LABEL says, i.e. in the padded frame, not at the raw index.
    assert int(np.argmax(roi.spectrum)) == PEAK_PADDED
    assert roi.labels[0].center_in_points == PEAK_PADDED
    # Real part, not magnitude: magnitude would be |1+0.5j| = 1.118 x larger at the peak.
    assert roi.spectrum.max() == pytest.approx(1e9, rel=1e-6)


def test_load_roi_surfaces_the_frame_metadata(tmp_path: Path) -> None:
    roi = load_roi(_write_roi(tmp_path / "roi_SX.npz"))
    assert roi.name == "roi_SX"
    assert roi.padding_before == PAD_BEFORE
    assert roi.padding_after == N - PAD_BEFORE - RAW_LEN
    assert roi.points_per_hz == pytest.approx(5.12)
    assert roi.compound == "Test compound"
    assert roi.ppm_axis is not None and roi.ppm_axis.shape == (N,)
    assert isinstance(roi.labels[0], Multiplet)
    assert roi.labels[0].proton_count == 2
    assert roi.labels[0].coupling_constants_hz == [7.5]


def test_load_roi_refuses_a_file_without_the_padded_frame(tmp_path: Path) -> None:
    # Falling back to spectrum_raw is exactly the trap; refusing is the only safe behaviour.
    with pytest.raises(ValueError, match="spectrum_padded"):
        load_roi(_write_roi(tmp_path / "roi_SX.npz", with_padded=False))


def test_load_roi_refuses_a_wrong_length_padded_spectrum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="6144"):
        load_roi(_write_roi(tmp_path / "roi_SX.npz", padded_len=N - 1))


def _pred(points: float, protons: int, j: float = 7.0) -> dict:
    return {
        "chemical_shift_in_points": float(points),
        "proton_count": protons,
        "coupling_constants_hz": [j],
    }


def _label(points: float, protons: int, j: list[float] | None = None) -> Multiplet:
    return Multiplet(proton_count=protons, center_in_points=float(points), coupling_constants_hz=j or [])


def test_hungarian_assigns_co_located_labels_by_proton_count() -> None:
    """The S2/S7 shape: three labels at one identical point (H = 2, 2, 1).

    Shift alone cannot tell them apart, so the paper's cost lets proton agreement decide. A shift-only
    matcher would pair them in list order and score this case 1/3 (labels 2,2,1 vs preds 1,2,2).
    """
    labels = [_label(1086, 2), _label(1086, 2), _label(1086, 1)]
    preds = [_pred(1080, 1), _pred(1090, 2), _pred(1100, 2)]
    res = match_hungarian(preds, labels)
    assert res.n_matched == 3
    assert res.n_correct == 3
    assert (res.false_positives, res.false_negatives) == (0, 0)


def test_hungarian_prefers_proton_agreement_over_a_closer_shift_inside_the_gate() -> None:
    # One proton mismatch costs as much as an 80 Hz shift error, so inside the 20 Hz gate a 9.8 Hz
    # pair with the right proton count beats a 0 Hz pair with the wrong one.
    labels = [_label(1000, 1)]
    preds = [_pred(1000, 2), _pred(1050, 1)]
    res = match_hungarian(preds, labels)
    assert res.n_matched == 1
    assert res.pairs[0][0]["chemical_shift_in_points"] == 1050
    assert res.n_correct == 1
    assert res.false_positives == 1


def test_hungarian_excludes_pairs_beyond_the_gate() -> None:
    # 25 Hz apart: not a match at all, so one false positive AND one false negative.
    labels = [_label(1000, 2)]
    preds = [_pred(1000 + 25 * 5.12, 2)]
    res = match_hungarian(preds, labels)
    assert res.n_matched == 0
    assert (res.false_positives, res.false_negatives) == (1, 1)
    assert res.dshift_hz == []


def test_hungarian_reports_shift_in_hz_and_largest_coupling_error() -> None:
    labels = [_label(2000, 3, j=[7.5])]
    preds = [_pred(2000 + 5.12, 3, j=7.0)]
    res = match_hungarian(preds, labels)
    assert res.dshift_hz == [pytest.approx(1.0)]
    assert res.dj_hz == [pytest.approx(0.5)]


def test_summarize_pools_counts_and_uses_both_accuracy_denominators() -> None:
    """Paper Table 1(d) quotes proton accuracy over matched pairs (92.1 %); per label (198/220) is
    the other honest denominator. Both must come out, from the same pooled counts."""
    from moldetr.roi import summarize

    a = match_hungarian([_pred(1000, 2), _pred(3000, 1)], [_label(1000, 2), _label(2000, 1)])
    b = match_hungarian([_pred(1000 + 5.12, 1)], [_label(1000, 2)])
    s = summarize([a, b])
    assert (s.n_labels, s.n_matched, s.false_positives, s.false_negatives) == (3, 2, 1, 1)
    assert s.n_correct == 1
    assert s.proton_acc_per_label == pytest.approx(1 / 3)
    assert s.proton_acc_per_match == pytest.approx(1 / 2)
    assert s.median_dshift_hz == pytest.approx(0.5)  # median of [0.0, 1.0]


def test_summarize_of_nothing_matched_reports_nan_not_a_crash() -> None:
    from moldetr.roi import summarize

    s = summarize([match_hungarian([], [_label(1000, 1)])])
    assert (s.n_labels, s.n_matched, s.false_negatives) == (1, 0, 1)
    assert np.isnan(s.median_dshift_hz) and np.isnan(s.proton_acc_per_match)
    assert s.proton_acc_per_label == 0.0


def test_hungarian_with_no_predictions_counts_every_label_missed() -> None:
    res = match_hungarian([], [_label(1000, 1), _label(2000, 2)])
    assert (res.n_matched, res.false_positives, res.false_negatives) == (0, 0, 2)
