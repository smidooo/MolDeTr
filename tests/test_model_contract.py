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
