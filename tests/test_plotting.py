"""`plotting.assignment_rows` (δ≠Δ table) + `plotting.spectrum_figure` (axis-branch selection).

Also covers the three pure coordinate helpers (`_shift_to_x`, `_apex`, `_ppm_to_index`). They are
private, but every marker the GUI draws is positioned by them, so an off-by-one or a missing clamp
is a visibly wrong figure with no other test standing between it and the user.
"""

from __future__ import annotations

import numpy as np
import pytest

from app_ui.plotting import (
    MARKER_COLORS,
    _apex,
    _ppm_to_index,
    _shift_to_x,
    assignment_rows,
    spectrum_figure,
)


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


@pytest.mark.unit
def test_spectrum_figure_marker_colours_cycle_past_the_three_brand_colours(amp):
    """A 4th detection reuses colour 1 — MARKER_COLORS is a cycle, not a limit of three.

    Every other figure test uses ≤ 3 predictions, so `(i - 1) % len(MARKER_COLORS)` has never been
    exercised past its first pass. An `IndexError` or a dropped 4th marker would live exactly here.
    """
    preds = [_pred(chemical_shift_in_points=float(800 * i)) for i in range(1, 6)]
    fig = spectrum_figure(amp, preds, ppm_left=10.0, ppm_right=0.0, points_per_hz=5.12)

    (markers,) = [t for t in fig.data if t.mode == "markers+text"]
    assert list(markers.text) == ["1", "2", "3", "4", "5"]  # numbering never restarts
    assert list(markers.marker.color) == [
        MARKER_COLORS[0],
        MARKER_COLORS[1],
        MARKER_COLORS[2],
        MARKER_COLORS[0],  # wraps here
        MARKER_COLORS[1],
    ]
    assert len(fig.layout.shapes) == 5  # one stem per detection, none silently dropped


# --- pure coordinate helpers ----------------------------------------------------------------------


@pytest.mark.unit
def test_shift_to_x_maps_endpoints_and_midpoint_onto_the_ppm_axis():
    """Point 0 → ppm_left, point n-1 → ppm_right, midpoint → halfway (axis descends, NMR order)."""
    n = 6144
    assert _shift_to_x(0, n, 10.0, 0.0) == pytest.approx(10.0)
    assert _shift_to_x(n - 1, n, 10.0, 0.0) == pytest.approx(0.0)
    assert _shift_to_x((n - 1) / 2, n, 10.0, 0.0) == pytest.approx(5.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "left,right,n",
    [(None, 0.0, 6144), (10.0, None, 6144), (None, None, 6144), (10.0, 0.0, 1)],
)
def test_shift_to_x_falls_through_to_point_index(left, right, n):
    """Half a calibration is not a calibration: pts passes through unscaled, as does a n=1 axis."""
    assert _shift_to_x(1234.0, n, left, right) == 1234.0


@pytest.mark.unit
def test_apex_takes_the_local_max_inside_the_half_window():
    amp = np.zeros(200)
    amp[100], amp[118] = 1.0, 5.0  # 118 == xi + half, the last included index
    assert _apex(amp, 100) == 5.0


@pytest.mark.unit
def test_apex_ignores_a_taller_peak_one_point_outside_the_window():
    """The ±18-point window is what puts a marker on *its own* peak, not the spectrum's tallest.

    Discriminating case: widening the window by a single point flips this to 9.0.
    """
    amp = np.zeros(200)
    amp[100], amp[119] = 1.0, 9.0  # 119 == xi + half + 1, the first excluded index
    assert _apex(amp, 100) == 1.0


@pytest.mark.unit
def test_apex_clamps_at_the_left_edge():
    """`lo` clamps to 0; a negative slice start would silently wrap to the end of the array."""
    amp = np.zeros(50)
    amp[0] = 3.0
    assert _apex(amp, 2) == 3.0


@pytest.mark.unit
def test_apex_falls_back_to_the_global_max_when_the_window_misses_the_array():
    """A centre past the end yields an empty slice; the fallback keeps a bad centre from crashing."""
    amp = np.zeros(50)
    amp[10] = 7.0
    assert _apex(amp, 500) == 7.0


@pytest.mark.unit
def test_ppm_to_index_maps_the_axis_endpoints():
    n = 6144
    assert _ppm_to_index(10.0, n, 10.0, 0.0) == 0
    assert _ppm_to_index(0.0, n, 10.0, 0.0) == n - 1
    assert _ppm_to_index(5.0, n, 10.0, 0.0) == pytest.approx((n - 1) / 2, abs=1)


@pytest.mark.unit
@pytest.mark.parametrize("ppm_val,expected", [(99.0, 0), (-99.0, 6143)])
def test_ppm_to_index_clamps_out_of_range_values(ppm_val, expected):
    """Out-of-window ppm must clamp into [0, n-1] — unclamped, this indexes outside the spectrum."""
    assert _ppm_to_index(ppm_val, 6144, 10.0, 0.0) == expected


@pytest.mark.unit
@pytest.mark.parametrize("n,left,right", [(6144, 5.0, 5.0), (1, 10.0, 0.0)])
def test_ppm_to_index_degenerate_axis_returns_zero(n, left, right):
    """A zero-width axis or a single-point spectrum would divide by zero without the guard."""
    assert _ppm_to_index(3.0, n, left, right) == 0
