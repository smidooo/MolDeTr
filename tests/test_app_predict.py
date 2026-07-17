"""Detect tab — `app.predict` / `app.predict_ui` scenario matrix (stubbed model, weight-free)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _valid_npz_with_ppm(tmp_npz, spec, left=10.0, right=0.0):
    return tmp_npz(spectrum_padded=spec, ppm_axis_padded=np.linspace(left, right, 6144))


def _download_path(btn) -> str | None:
    """Normalize a gr.DownloadButton value to a filesystem path (gradio may wrap it)."""
    v = btn.value
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("path")
    return getattr(v, "path", None)


# --- checkpoint gate + no-file (checkpoint-independent) -------------------------------------------


@pytest.mark.unit
def test_no_file_message(app_module):
    _t, _f, msg = app_module.predict(None, 0.3, app_module.AUTO, None, None, 5.12)
    assert msg == "Load a `.npz`/`.npy` spectrum, or pick an example below."


@pytest.mark.unit
def test_checkpoint_absent_message(app_module, tmp_npz, valid_spectrum, monkeypatch):
    monkeypatch.setattr(app_module, "CHECKPOINT", str(Path("does") / "not" / "exist.pth"))
    path = _valid_npz_with_ppm(tmp_npz, valid_spectrum)
    _t, _f, msg = app_module.predict(path, 0.3, app_module.AUTO, None, None, 5.12)
    assert "Checkpoint not found" in msg and "10.5281/zenodo.21217102" in msg


# --- validation errors (stub patched, but rejection happens before the model) --------------------


@pytest.mark.unit
def test_wrong_length_rejected(patch_model, tmp_npz):
    app = patch_model
    path = tmp_npz(spec=np.abs(np.random.RandomState(2).rand(5000)))
    table, fig, msg = app.predict(path, 0.3, app.AUTO, None, None, 5.12)
    assert table is None and fig is None
    assert msg.startswith("Invalid spectrum:") and "exactly 6144" in msg


@pytest.mark.unit
def test_nan_rejected(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    bad = valid_spectrum.copy()
    bad[5] = np.inf
    _t, _f, msg = app.predict(tmp_npz(spec=bad), 0.3, app.AUTO, None, None, 5.12)
    assert msg.startswith("Invalid spectrum:") and "NaN or Inf" in msg


@pytest.mark.unit
def test_complex_warns_but_detects(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    path = _valid_npz_with_ppm(tmp_npz, valid_spectrum.astype(np.complex64))
    table, _f, msg = app.predict(path, 0.3, app.AUTO, None, None, 5.12)
    assert "Detected **3** multiplet(s)" in msg
    assert "using its real part" in msg
    assert len(table) == 3


# --- ppm mode × bounds → shift-column header -----------------------------------------------------


@pytest.mark.unit
def test_auto_with_calibration_is_ppm(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    table, _f, _m = app.predict(
        _valid_npz_with_ppm(tmp_npz, valid_spectrum), 0.3, app.AUTO, None, None, 5.12
    )
    assert "δ (PPM)" in table.columns


@pytest.mark.unit
def test_auto_without_calibration_is_hz(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    table, _f, _m = app.predict(tmp_npz(spec=valid_spectrum), 0.3, app.AUTO, None, None, 5.12)
    assert "δ (HZ)" in table.columns


@pytest.mark.unit
def test_manual_both_bounds_is_ppm(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    table, _f, _m = app.predict(tmp_npz(spec=valid_spectrum), 0.3, app.MANUAL, 8.0, 2.0, 5.12)
    assert "δ (PPM)" in table.columns


@pytest.mark.unit
def test_manual_single_bound_falls_back_to_hz(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    table, _f, _m = app.predict(tmp_npz(spec=valid_spectrum), 0.3, app.MANUAL, 8.0, None, 5.12)
    assert "δ (HZ)" in table.columns  # MANUAL without both bounds → Hz


@pytest.mark.unit
def test_none_mode_is_hz(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    table, _f, _m = app.predict(
        _valid_npz_with_ppm(tmp_npz, valid_spectrum), 0.3, app.NONE, None, None, 5.12
    )
    assert "δ (HZ)" in table.columns  # NONE overrides the file calibration


# --- threshold -----------------------------------------------------------------------------------


@pytest.mark.unit
def test_threshold_one_detects_nothing(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    table, _f, msg = app.predict(
        _valid_npz_with_ppm(tmp_npz, valid_spectrum), 1.0, app.AUTO, None, None, 5.12
    )
    assert table.empty
    assert msg == "No multiplets passed the detection threshold — try lowering it."


# --- predict_ui downloads ------------------------------------------------------------------------


@pytest.mark.unit
def test_downloads_enabled_and_parse(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    path = _valid_npz_with_ppm(tmp_npz, valid_spectrum)
    table, _f, _m, csv_btn, json_btn = app.predict_ui(path, 0.3, app.AUTO, None, None, 5.12)
    assert bool(csv_btn.interactive) and bool(json_btn.interactive)
    csv_path = _download_path(csv_btn)
    assert csv_path and csv_path.endswith(".csv") and Path(csv_path).exists()
    back = pd.read_csv(csv_path)
    assert list(back.columns) == list(table.columns) and len(back) == len(table)


@pytest.mark.unit
def test_downloads_disabled_when_empty(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    path = _valid_npz_with_ppm(tmp_npz, valid_spectrum)
    table, _f, _m, csv_btn, json_btn = app.predict_ui(path, 1.0, app.AUTO, None, None, 5.12)
    assert table.empty
    assert not csv_btn.interactive and not json_btn.interactive
    assert _download_path(csv_btn) is None and _download_path(json_btn) is None
