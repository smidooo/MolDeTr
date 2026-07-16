"""Smoke test for the annotated-spectrum plot (headless Agg backend)."""

import numpy as np

from moldetr.visualization import plot_spectrum


def _preds():
    return [
        {
            "proton_count": 2,
            "chemical_shift_in_points": 2000.0,
            "chemical_shift_ppm": 7.31,
            "coupling_constants_hz": [7.2],
            "linewidth_hz": 1.4,
        },
        {
            "proton_count": 1,
            "chemical_shift_in_points": 4200.0,
            "chemical_shift_ppm": 3.05,
            "coupling_constants_hz": [],
            "linewidth_hz": 1.1,
        },
    ]


def test_plot_spectrum_writes_png_and_returns_rows(tmp_path):
    amp = np.abs(np.random.RandomState(0).rand(6144))
    out = tmp_path / "pred.png"
    fig, rows = plot_spectrum(amp, _preds(), ppm_left=10.0, ppm_right=0.0, save_path=str(out))
    assert fig is not None
    assert out.exists() and out.stat().st_size > 0
    # one assignment row per prediction, numbered 1..N, with the expected columns
    assert len(rows) == 2
    assert [r["#"] for r in rows] == [1, 2]
    assert rows[0]["protons"] == "2 H"
    assert "δ (ppm)" in rows[0] and rows[0]["δ (ppm)"] == "7.310"
    assert rows[0]["max J (Hz)"] == "7.2" and rows[1]["max J (Hz)"] == "–"


def test_plot_spectrum_no_table_still_returns_rows(tmp_path):
    amp = np.abs(np.random.RandomState(1).rand(6144))
    fig, rows = plot_spectrum(amp, _preds(), show_table=False)
    assert fig is not None and len(rows) == 2
