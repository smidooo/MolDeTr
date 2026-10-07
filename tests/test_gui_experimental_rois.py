"""The GUI's Detect decode, on all 13 Zenodo experimental ROIs, against the article's numbers.

Every ROI goes in the way an upload does (``app._load`` with ``trusted=False``, so no pickle),
through ``validate_spectrum`` -> ``run`` -> ``decode_predictions`` with the file's ppm bounds,
under noise seeds 0-4 (the article pooled ~5 noisy repeats: 215 matched pairs over 220 labels).
Scored with the article's own matcher, ``moldetr.roi.match_hungarian``.

Measured 2026-10-06 on CI (run 37518511396, real checkpoint, this exact path). The bounds are
asserted on the article-faithful matcher; the lenient one (keeps valid pairs in spectra the
article's code zeroes) gave 204 matched, 17 FP, 86.8 % of labels:

=====================  ========  ======================  ====================
quantity               measured  bound                   article (Table 1(d))
=====================  ========  ======================  ====================
labels                 220       == 220                  220
matched (<= 20 Hz)     200       >= 190                  215
false positives        21        <= 32                   20
median |dd| (Hz)       0.917     0.60 - 1.20             0.885
median |dJ| (Hz)       0.174     <= 0.30                 0.199
H acc, matched pairs   93.5 %    >= 88 %                 92.1 %
H acc, all labels      85.0 %    >= 78 %                 90.0 % (198/220)
=====================  ========  ======================  ====================

|dJ| is like-for-like: every label carries at most one J, and the article scored it against the
largest predicted J, the one J this decode reports. The labels gap (200 vs 215) is the decode
merging closely overlapped multiplets (S2, S7), see scripts/evaluate_experimental.py.

What the bounds exclude, measured the same day on the same 220 labels (lenient matcher): building
the input from ``spectrum_raw`` padded at the end (110 matched, 116 false positives, 49.5 %) or
from the magnitude (105 false positives, 68.6 %). Seed 3 alone gives a 1.35 Hz median, which is
why one seed is not enough and why the band is on the 5-seed pool.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.model

REPO = Path(__file__).resolve().parent.parent
ROI_DIR = Path(os.environ.get("MOLDETR_ROI_NPZ_DIR", str(REPO / "structured_output")))
ROI_FILES = sorted(ROI_DIR.glob("roi_S*.npz"))
SEEDS = range(5)
THRESHOLD = 0.3
N_SPIN_SYSTEMS = 44  # in the 13 ROIs (structured_output/README.md)


def _require_rois() -> None:
    if len(ROI_FILES) < 13:
        pytest.skip(f"expected 13 Zenodo roi_S*.npz in {ROI_DIR}, found {len(ROI_FILES)}")


@pytest.fixture(scope="module")
def gui_runs():
    """(roi, seed) -> (predictions, labels) for every ROI and noise seed, via the GUI's path."""
    ckpt = os.environ.get("MOLDETR_CHECKPOINT")
    if not ckpt or not Path(ckpt).exists():
        pytest.skip("real checkpoint absent (set MOLDETR_CHECKPOINT)")
    _require_rois()
    import app
    from moldetr.inference import build_model, load_checkpoint, run
    from moldetr.postprocess import decode_predictions, load_extrema
    from moldetr.roi import load_roi

    model = load_checkpoint(build_model(), ckpt)
    extrema = load_extrema(app.EXTREMA)
    runs = {}
    for path in ROI_FILES:
        raw, cal = app._load(str(path), trusted=False)  # exactly how an upload is read
        amps = app.validate_spectrum(raw, points_per_hz=5.12)
        labels = load_roi(path).labels
        for seed in SEEDS:
            preds = decode_predictions(
                run(model, amps, noise_seed=seed),
                extrema,
                5.12,
                ppm_left=cal.get("ppm_left"),
                ppm_right=cal.get("ppm_right"),
                threshold=THRESHOLD,
            )
            runs[(path.stem, seed)] = (preds, labels)
    return runs


def test_the_gui_reads_an_upload_in_the_frame_the_labels_use():
    """``app._load`` must return ``spectrum_padded`` (the labels' frame), not ``spectrum_raw``."""
    _require_rois()
    import app
    from moldetr.roi import load_roi

    for path in ROI_FILES:
        raw, _cal = app._load(str(path), trusted=False)
        assert np.array_equal(np.real(raw), load_roi(path).spectrum), path.name


def test_gui_detect_reproduces_the_article_on_all_rois(gui_runs):
    from moldetr.roi import match_hungarian, summarize

    def scored(faithful: bool):
        runs = gui_runs.values()
        return summarize([match_hungarian(p, lab, article_faithful=faithful) for p, lab in runs])

    def describe(tag: str, x) -> str:
        return (
            f"{tag}: labels={x.n_labels} matched={x.n_matched} FP={x.false_positives} "
            f"FN={x.false_negatives} median|dd|={x.median_dshift_hz:.3f} Hz "
            f"median|dJ|={x.median_dj_hz:.3f} Hz H/match={x.proton_acc_per_match:.3f} "
            f"H/label={x.proton_acc_per_label:.3f}"
        )

    s, faithful = scored(False), scored(True)
    report = describe("lenient", s) + " | " + describe("faithful", faithful)
    # Shown in the nightly log on every run, pass or fail: the measured numbers, not a verdict.
    warnings.warn(report, stacklevel=1)
    assert faithful.n_labels == N_SPIN_SYSTEMS * len(SEEDS), report
    assert faithful.n_matched >= 190, report
    assert faithful.false_positives <= 32, report
    assert 0.60 <= faithful.median_dshift_hz <= 1.20, report
    assert faithful.median_dj_hz <= 0.30, report
    assert faithful.proton_acc_per_match >= 0.88, report
    assert faithful.proton_acc_per_label >= 0.78, report
    # The lenient matcher only ever keeps pairs the faithful one drops, never fewer.
    assert s.n_matched >= faithful.n_matched, report


def test_app_predict_table_is_the_decode_this_test_scores(gui_runs):
    """Ties the replicated path to ``app.predict`` itself: seed 0 is what the Detect button runs."""
    import app

    n_rows = 0
    for path in ROI_FILES:
        table, _fig, msg = app.predict(str(path), THRESHOLD, app.AUTO, None, None, 5.12)
        preds, _labels = gui_runs[(path.stem, 0)]
        assert table is not None, f"{path.name}: {msg}"
        shown = list(zip(table["PROTONS"], table["δ (PPM)"]))
        scored = [(f"{p['proton_count']} H", f"{p['chemical_shift_ppm']:.3f}") for p in preds]
        assert shown == scored, f"{path.name}: the table is not the decode this file scores"
        n_rows += len(shown)
    assert n_rows >= 40, "every ROI came back empty; the comparison above proved nothing"
