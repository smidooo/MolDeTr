"""`app._load` — file parsing + ppm calibration resolution (checkpoint-independent)."""

from __future__ import annotations

import numpy as np
import pytest


def _obj0d(value):
    """A 0-d object array holding one Python value (how the export stores `metadata`)."""
    a = np.empty((), dtype=object)
    a[()] = value
    return a


@pytest.mark.unit
def test_npz_prefers_ppm_axis_padded(app_module, tmp_npz, valid_spectrum):
    axis = np.linspace(9.0, 1.0, 6144)
    path = tmp_npz(spectrum_padded=valid_spectrum, ppm_axis_padded=axis)
    arr, cal = app_module._load(path)
    assert arr.shape == (6144,)
    assert cal["ppm_left"] == pytest.approx(9.0)
    assert cal["ppm_right"] == pytest.approx(1.0)


@pytest.mark.unit
def test_npz_falls_back_to_metadata_ppm(app_module, tmp_npz, valid_spectrum):
    path = tmp_npz(spec=valid_spectrum, metadata=_obj0d({"left_ppm": 8.0, "right_ppm": 2.0}))
    arr, cal = app_module._load(path)
    assert arr.shape == (6144,)
    assert (cal["ppm_left"], cal["ppm_right"]) == (8.0, 2.0)


@pytest.mark.unit
def test_npz_spec_only_has_no_calibration(app_module, tmp_npz, valid_spectrum):
    _arr, cal = app_module._load(tmp_npz(spec=valid_spectrum))
    assert cal == {}


@pytest.mark.unit
def test_npz_first_key_fallback(app_module, tmp_npz, valid_spectrum):
    # No spectrum_padded / spec key → the first array in the archive is taken as the spectrum.
    arr, cal = app_module._load(tmp_npz(some_weird_key=valid_spectrum))
    assert arr.shape == (6144,) and cal == {}


@pytest.mark.unit
def test_npy_load_no_calibration(app_module, tmp_npy, valid_spectrum):
    arr, cal = app_module._load(tmp_npy(valid_spectrum))
    assert arr.shape == (6144,) and cal == {}


@pytest.mark.unit
def test_real_example_roi_has_ppm_axis(app_module, example_paths):
    arr, cal = app_module._load(example_paths["roi_S10"])
    assert arr.shape[0] == 6144
    assert cal.get("ppm_left") is not None and cal.get("ppm_right") is not None


@pytest.mark.unit
def test_real_synthetic_example_is_complex_without_ppm(app_module, example_paths):
    arr, cal = app_module._load(example_paths["synthetic"])
    assert np.iscomplexobj(arr)  # synthetic_example.npz ships complex64
    assert cal == {}  # spec-only → no calibration → Auto falls back to Hz
