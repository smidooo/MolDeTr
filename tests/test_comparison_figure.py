"""Tests for :func:`plotting.comparison_figure` — the GT-vs-prediction overlay.

Pins the intuitive encoding: matched-in-tolerance vs matched-off connectors (green/amber),
missed ground truth (red-outlined triangle), spurious prediction (dashed-ring circle), and
prediction-marker opacity scaling with confidence.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from app_ui.plotting import MATCH_OFF, MATCH_OK, MISS_INK, comparison_figure

N = 6144


def _amp() -> np.ndarray:
    a = np.zeros(N)
    a[3000] = 1.0
    return a


def _connector_colors(fig: go.Figure) -> list:
    return [
        t.line.color
        for t in fig.data
        if t.mode == "lines" and t.x is not None and len(t.x) == 2 and t.x[0] is not None
    ]


def test_matched_missed_spurious_encoding() -> None:
    matched = [
        # tiny Δδ (0.08 Hz) -> within tol -> green connector
        (
            {"shift_ppm": 5.0, "proton_count": 2},
            {"chemical_shift_ppm": 5.001, "proton_count": 2, "confidence": 0.9},
        ),
        # Δδ = 0.10 ppm * 80 MHz = 8 Hz > tol -> amber connector
        (
            {"shift_ppm": 3.0, "proton_count": 1},
            {"chemical_shift_ppm": 3.10, "proton_count": 1, "confidence": 0.5},
        ),
        # no prediction -> missed GT
        ({"shift_ppm": 1.0, "proton_count": 3}, None),
    ]
    spurious = [{"chemical_shift_ppm": 7.0, "proton_count": 1, "confidence": 0.6}]
    fig = comparison_figure(
        _amp(), matched, spurious, ppm_left=8.0, ppm_right=0.0, base_freq_mhz=80.0, tol_hz=2.0
    )
    assert isinstance(fig, go.Figure)

    colors = _connector_colors(fig)
    assert MATCH_OK in colors, "in-tolerance match must draw a green connector"
    assert MATCH_OFF in colors, "off match must draw an amber connector"

    missed = [
        t
        for t in fig.data
        if t.marker
        and t.marker.symbol == "triangle-down"
        and t.marker.line
        and t.marker.line.color == MISS_INK
    ]
    assert len(missed) == 1, "missed GT must be one red-outlined triangle"

    spur = [t for t in fig.data if t.marker and t.marker.symbol == "circle-open-dot"]
    assert len(spur) == 1, "spurious prediction must be one dashed-ring circle"


def test_prediction_opacity_scales_with_confidence() -> None:
    def _pred_opacity(fig: go.Figure) -> list:
        return [
            t.marker.opacity
            for t in fig.data
            if t.marker and t.marker.symbol == "circle" and t.marker.opacity is not None
        ]

    hi = [
        (
            {"shift_ppm": 5.0, "proton_count": 1},
            {"chemical_shift_ppm": 5.0, "proton_count": 1, "confidence": 0.95},
        )
    ]
    lo = [
        (
            {"shift_ppm": 5.0, "proton_count": 1},
            {"chemical_shift_ppm": 5.0, "proton_count": 1, "confidence": 0.40},
        )
    ]
    fh = comparison_figure(_amp(), hi, ppm_left=8.0, ppm_right=0.0, base_freq_mhz=80.0)
    fl = comparison_figure(_amp(), lo, ppm_left=8.0, ppm_right=0.0, base_freq_mhz=80.0)
    assert max(_pred_opacity(fh)) > max(_pred_opacity(fl))


def test_connector_amber_on_proton_count_mismatch() -> None:
    """A spot-on δ but wrong proton count is an ``off`` match (amber), never green — the connector
    colour must agree with the table's ``✓ match`` (which requires both δ and H)."""
    matched = [
        (
            {"shift_ppm": 5.0, "proton_count": 2},
            {"chemical_shift_ppm": 5.0, "proton_count": 3, "confidence": 0.9},
        )
    ]
    fig = comparison_figure(
        _amp(), matched, [], ppm_left=8.0, ppm_right=0.0, base_freq_mhz=80.0, tol_hz=2.0
    )
    colors = _connector_colors(fig)
    assert MATCH_OFF in colors and MATCH_OK not in colors
