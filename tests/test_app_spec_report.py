"""`app._spec_report` — the post-upload input-check panel (checkpoint-independent, never rejects)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_none_file_returns_empty(app_module):
    assert app_module._spec_report(None, 5.12) == ""


@pytest.mark.unit
def test_valid_spectrum_all_checks_pass(app_module, tmp_npz, valid_spectrum):
    axis = np.linspace(10.0, 0.0, 6144)
    path = tmp_npz(spectrum_padded=valid_spectrum, ppm_axis_padded=axis)
    r = app_module._spec_report(path, 5.12)
    assert "**Input check**" in r
    assert "Length: **6144** points ✓" in r
    assert "**5.12** points/Hz → **1200 Hz** window ✓" in r
    assert "Data type: real ✓" in r
    assert "Finite values: ✓" in r
    assert "ppm axis in file: yes ✓ (Auto works)" in r


@pytest.mark.unit
def test_wrong_length_flagged(app_module, tmp_npz):
    path = tmp_npz(spec=np.abs(np.random.RandomState(1).rand(5000)))
    r = app_module._spec_report(path, 5.12)
    assert "Length: **5000** points ✗ needs exactly 6144" in r


@pytest.mark.unit
def test_nan_flagged(app_module, tmp_npz, valid_spectrum):
    bad = valid_spectrum.copy()
    bad[10] = np.nan
    r = app_module._spec_report(tmp_npz(spec=bad), 5.12)
    assert "Finite values: ✗ contains NaN/Inf" in r


@pytest.mark.unit
def test_complex_flagged_as_recoverable(app_module, tmp_npz, valid_spectrum):
    r = app_module._spec_report(tmp_npz(spec=valid_spectrum.astype(np.complex64)), 5.12)
    assert "Data type: complex — the real (absorption) part is used" in r


@pytest.mark.unit
def test_no_ppm_axis_reported(app_module, tmp_npz, valid_spectrum):
    r = app_module._spec_report(tmp_npz(spec=valid_spectrum), 5.12)
    assert "ppm axis in file: no — use Manual or None" in r


@pytest.mark.unit
def test_wrong_resolution_warns(app_module, tmp_npz, valid_spectrum):
    r = app_module._spec_report(tmp_npz(spec=valid_spectrum), 10.0)
    assert "**614 Hz** window ⚠ not 1200 Hz — predictions may be unreliable" in r


@pytest.mark.unit
def test_blank_resolution_falls_back_to_default(app_module, tmp_npz, valid_spectrum):
    """An empty `gr.Number` arrives as ``None`` and genuinely means "unset" → use the default."""
    r = app_module._spec_report(tmp_npz(spec=valid_spectrum), None)
    assert "**5.12** points/Hz → **1200 Hz** window ✓" in r


@pytest.mark.unit
@pytest.mark.parametrize("pph", [0, -5.12])
def test_non_positive_resolution_is_reported_not_replaced(app_module, tmp_npz, valid_spectrum, pph):
    """Changed deliberately: this used to assert 0 fell back to 5.12 alongside ``None``.

    Lumping them together was the bug. ``None`` is "the user told us nothing"; ``0`` is "the user
    told us zero", and answering that with a ✓ input check computed from 5.12 reports a resolution
    nobody entered. Negatives never even hit the old guard — they are truthy — so the panel showed
    a negative Hz window as though it were fine.
    """
    r = app_module._spec_report(tmp_npz(spec=valid_spectrum), pph)
    assert r.startswith("⚠ Invalid input:") and "positive" in r


@pytest.mark.unit
def test_unreadable_file(app_module, tmp_path):
    bad = tmp_path / "corrupt.npz"
    bad.write_bytes(b"this is not a valid npz archive")
    r = app_module._spec_report(str(bad), 5.12)
    assert r.startswith("⚠ Could not read the file:")
