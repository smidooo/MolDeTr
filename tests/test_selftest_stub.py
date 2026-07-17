"""Self-test of the conftest stub machinery — the linchpin for every Detect/Simulate test.

If these fail, the whole weight-free CI strategy is invalid, so they assert real decoded
detections (never merely "not an error"). Mirrors the decode contract in moldetr.inference.run +
moldetr.postprocess.decode_predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from moldetr.inference import run
from moldetr.postprocess import decode_predictions


@pytest.mark.unit
def test_fake_model_output_shape(fake_model, valid_spectrum):
    out = run(fake_model, valid_spectrum)
    assert tuple(out.shape) == (80, 12)  # (n_groups*num_queries, num_classes+num_params)


@pytest.mark.unit
def test_fake_model_decodes_to_three_sorted_detections(fake_model, extrema, valid_spectrum):
    preds = decode_predictions(run(fake_model, valid_spectrum), extrema, 5.12, threshold=0.3)
    assert [p["proton_count"] for p in preds] == [1, 2, 3]  # decode sorts by ascending center
    centers = [p["chemical_shift_in_points"] for p in preds]
    assert centers == sorted(centers)
    assert all(c2 - c1 > 20 for c1, c2 in zip(centers, centers[1:]))  # un-merged by _merge NMS
    assert all(abs(p["coupling_constants_hz"][0] - 8.0) < 0.1 for p in preds)  # 0.4*102.38/5.12


@pytest.mark.unit
def test_make_fake_model_controls_count_and_coupling(make_fake_model, extrema, valid_spectrum):
    model = make_fake_model([{"proton": 6, "center_frac": 0.5, "coupling_frac": 0.0}])
    preds = decode_predictions(run(model, valid_spectrum), extrema, 5.12, threshold=0.3)
    assert len(preds) == 1 and preds[0]["proton_count"] == 6
    assert preds[0]["coupling_constants_hz"] == []  # coupling_frac 0 → below min_coupling_hz


@pytest.mark.unit
def test_threshold_one_drops_all(fake_model, extrema, valid_spectrum):
    assert decode_predictions(run(fake_model, valid_spectrum), extrema, 5.12, threshold=1.0) == []


@pytest.mark.unit
def test_patch_model_enables_predict(patch_model, valid_spectrum, tmp_npz):
    app = patch_model
    path = tmp_npz(spectrum_padded=valid_spectrum, ppm_axis_padded=np.linspace(10, 0, 6144))
    table, fig, msg = app.predict(path, 0.3, app.AUTO, None, None, 5.12)
    assert "Detected **3** multiplet(s)" in msg
    assert not table.empty and len(table) == 3
    assert fig is not None


@pytest.mark.unit
def test_patch_model_checkpoint_gate_passes(patch_model, valid_spectrum, tmp_npz):
    # A real file traverses the checkpoint-exists branch (a None file would short-circuit at app.py:130
    # BEFORE the gate at :132). The dummy checkpoint exists → gate passes → the stubbed model runs.
    app = patch_model
    path = tmp_npz(spectrum_padded=valid_spectrum, ppm_axis_padded=np.linspace(10, 0, 6144))
    _t, _f, msg = app.predict(path, 0.3, app.AUTO, None, None, 5.12)
    assert "Checkpoint not found" not in msg
    assert "Detected" in msg
