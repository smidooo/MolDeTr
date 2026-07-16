"""`plotting.assignment_rows` (δ≠Δ table) + `plotting.spectrum_figure` (axis-branch selection)."""

from __future__ import annotations

import numpy as np
import pytest

from plotting import assignment_rows, spectrum_figure


def _pred(**over):
    p = {
        "proton_count": 2,
        "chemical_shift_in_points": 3000.0,
        "chemical_shift_ppm": 7.5,
        "chemical_shift_hz": 384.0,
        "coupling_constants_hz": [8.2],
        "linewidth_hz": 1.25,
    }
    p.update(over)
    return p


@pytest.mark.unit
def test_assignment_rows_ppm_formatting():
    (row,) = assignment_rows([_pred()], ppm=True)
    assert row == {
        "#": 1,
        "PROTONS": "2 H",
        "δ (PPM)": "7.500",
        "MAX J (HZ)": "8.2",
        "LINE WIDTH (HZ)": "1.25",
    }


@pytest.mark.unit
def test_assignment_rows_hz_header_and_formatting():
    (row,) = assignment_rows([_pred()], ppm=False)
    assert "δ (HZ)" in row and row["δ (HZ)"] == "384.0"


@pytest.mark.unit
def test_delta_header_never_becomes_capital_delta():
    # "δ".upper() == "Δ" (reads as "difference" in NMR) — headers must keep the lowercase literal.
    for ppm in (True, False):
        header = "".join(assignment_rows([_pred()], ppm=ppm)[0].keys())
        assert "δ" in header
        assert "Δ" not in header


@pytest.mark.unit
def test_assignment_rows_missing_values_dashed():
    (row,) = assignment_rows(
        [_pred(chemical_shift_ppm=None, coupling_constants_hz=[], linewidth_hz=None)], ppm=True
    )
    assert row["δ (PPM)"] == "–"
    assert row["MAX J (HZ)"] == "–"
    assert row["LINE WIDTH (HZ)"] == "–"


@pytest.mark.unit
def test_assignment_rows_empty():
    assert assignment_rows([], ppm=True) == []


@pytest.fixture
def amp():
    return np.abs(np.random.RandomState(0).rand(6144))


@pytest.mark.unit
def test_spectrum_figure_ppm_axis(amp):
    fig = spectrum_figure(amp, [_pred()], ppm_left=10.0, ppm_right=0.0, points_per_hz=5.12)
    assert fig.layout.xaxis.title.text == "Chemical shift δ (ppm)"
    assert fig.layout.xaxis.autorange == "reversed"  # NMR convention
    assert len(fig.data) == 2  # spectrum trace + marker trace


@pytest.mark.unit
def test_spectrum_figure_hz_axis(amp):
    fig = spectrum_figure(amp, [_pred()], ppm_left=None, ppm_right=None, points_per_hz=5.12)
    assert fig.layout.xaxis.title.text == "ν (Hz, window-relative)"
    assert fig.layout.xaxis.autorange is True


@pytest.mark.unit
def test_spectrum_figure_point_index_axis(amp):
    fig = spectrum_figure(amp, [], ppm_left=None, ppm_right=None, points_per_hz=None)
    assert fig.layout.xaxis.title.text == "Point index"
    assert len(fig.data) == 1  # no predictions → no marker trace
