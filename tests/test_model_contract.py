"""Local-only contract: the REAL checkpoint's output matches the stub's assumptions.

Guards the whole weight-free CI strategy — if the real model's output shape ever diverges from the
`(1, 80, 12)` block the ``fake_model`` + ``decode_predictions`` assume, these fail. Run locally with:

    MOLDETR_CHECKPOINT=/path/to/model_spin_system_ABCDEFG_exp2.pth pytest -m model
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def real_model():
    ckpt = os.environ.get("MOLDETR_CHECKPOINT")
    if not ckpt or not Path(ckpt).exists():
        pytest.skip("real checkpoint absent (set MOLDETR_CHECKPOINT) — local-only contract test")
    from moldetr.inference import build_model, load_checkpoint

    return load_checkpoint(build_model(), ckpt)


def test_real_output_shape_matches_stub_contract(real_model, valid_spectrum):
    from moldetr.inference import run

    out = run(real_model, valid_spectrum)
    assert tuple(out.shape) == (80, 12)  # exactly what conftest.fake_model returns


# The vanillin oracle. CAPTURED from a live decode on 2026-07-25 against checkpoint md5
# faf842d1a1d8beae67e0544e28f226b5 — not transcribed from the docs. It happens to agree with the
# same values read off the GUI (δ 7.419/7.388/6.961; J 8.7/2.0/8.2) to every published digit, which
# is worth stating: the from-weights decode and the committed-JSON aggregate path are not required
# to agree (docs/SCOPE.md:88 keeps them separate). Here they do.
VANILLIN_SHIFTS_PPM = (7.4186, 7.3879, 6.9611)
VANILLIN_MAX_J_HZ = (8.738, 1.966, 8.247)

# δ: 0.005 ppm is 1.5 Hz at 300 MHz — far below any real decode regression, far above float noise.
# J: the three couplings are ~2 and ~8 Hz, so 0.2 Hz pins them without being brittle.
TOL_PPM = 0.005
TOL_J_HZ = 0.2


def _vanillin_predictions(model, example_paths, extrema):
    """The live decode on roi_S8, sorted high→low ppm (the order the GUI table shows)."""
    from moldetr.inference import run
    from moldetr.postprocess import decode_predictions
    from moldetr.reproducibility import set_seed
    from moldetr.validation import validate_spectrum

    # NB: this does NOT control the injected noise -- normalize_spectrum builds its own
    # RandomState(noise_seed=0), which global seeding cannot reach. Kept for the torch/numpy
    # global state the model path touches; the noise realisation is already deterministic.
    set_seed(42)
    data = np.load(example_paths["roi_S8"], allow_pickle=True)
    axis = np.asarray(data["ppm_axis_padded"], dtype=float)
    amp = validate_spectrum(data["spectrum_padded"], points_per_hz=5.12)
    preds = decode_predictions(
        run(model, amp),
        extrema,
        5.12,
        ppm_left=float(axis[0]),
        ppm_right=float(axis[-1]),
        threshold=0.3,
    )
    return sorted(preds, key=lambda p: -p["chemical_shift_ppm"])


def test_vanillin_oracle_reproduces_the_published_numbers(real_model, example_paths, extrema):
    """The end-to-end regression oracle: real weights → real decode → the documented ABX pattern.

    Everything else in the suite either runs the stub or checks structure. This is the only test
    that would notice the model silently producing *different physics* — a checkpoint swapped for a
    differently-trained one, a decode change that shifts every δ, a coupling slot re-indexed.
    """
    preds = _vanillin_predictions(real_model, example_paths, extrema)

    assert len(preds) == 3, f"vanillin is an ABX system — expected 3 multiplets, got {len(preds)}"
    assert [p["proton_count"] for p in preds] == [1, 1, 1]

    for p, expected in zip(preds, VANILLIN_SHIFTS_PPM):
        assert p["chemical_shift_ppm"] == pytest.approx(expected, abs=TOL_PPM)
    for p, expected in zip(preds, VANILLIN_MAX_J_HZ):
        assert p["coupling_constants_hz"][0] == pytest.approx(expected, abs=TOL_J_HZ)

    assert all(p["confidence"] > 0.9 for p in preds), (
        f"confidences dropped: {[round(p['confidence'], 3) for p in preds]}"
    )


def test_vanillin_max_j_deviates_from_ground_truth_as_documented(
    real_model, example_paths, extrema
):
    """Pins the *known* limitation rather than papering over it.

    Ground truth is J = 8.0 / 2.0 / 8.0; the live decode gives 8.74 / 1.97 / 8.25. That ~0.7 Hz
    gap on the two large couplings is the documented decode-path split — `max J` from weights only
    approximates, while the committed `structured_output` path is what reproduces the paper's
    0.20 Hz median. Asserting a *bound* on the deviation means the split staying this size is
    verified, and a decode change that widened it would fail here instead of quietly degrading.
    """
    preds = _vanillin_predictions(real_model, example_paths, extrema)
    data = np.load(example_paths["roi_S8"], allow_pickle=True)
    gt = sorted(data["ground_truth"], key=lambda g: -g["chemical_shift_ppm"])

    deviations = [
        abs(p["coupling_constants_hz"][0] - max(g["coupling_constants"])) for p, g in zip(preds, gt)
    ]
    assert max(deviations) < 1.0, f"max-J deviation from GT grew beyond 1 Hz: {deviations}"
    # And the shifts, unlike the couplings, really are tight against ground truth.
    for p, g in zip(preds, gt):
        assert p["chemical_shift_ppm"] == pytest.approx(g["chemical_shift_ppm"], abs=0.01)


def test_real_model_decodes_example_to_physical_predictions(real_model, example_paths, extrema):
    from moldetr.inference import run
    from moldetr.postprocess import PROTON_COUNTS, decode_predictions

    data = np.load(example_paths["roi_S8"], allow_pickle=True)  # vanillin ABX, ppm-calibrated
    axis = np.asarray(data["ppm_axis_padded"], dtype=float)
    preds = decode_predictions(
        run(real_model, np.asarray(data["spectrum_padded"])),
        extrema,
        5.12,
        ppm_left=float(axis[0]),
        ppm_right=float(axis[-1]),
        threshold=0.3,
    )
    assert preds, "the real model should detect at least one multiplet on the vanillin example"
    for p in preds:
        assert p["proton_count"] in PROTON_COUNTS
        assert 0.0 <= p["confidence"] <= 1.0
        assert 0 <= p["chemical_shift_in_points"] <= 6143
