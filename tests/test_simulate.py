"""Validate ``moldetr.simulate`` against closed-form NMR line-shape theory.

The quantum-mechanical simulator is checked against *analytical* ground truth, never against
itself:

* (a) an isolated spin  -> one Lorentzian at the right ppm, FWHM equal to the requested width;
* (b) a first-order AX pair (Δδ >> J) -> two doublets, each split by exactly ``J`` Hz, 1:1 lines;
* (c) a strongly-coupled AB pair (Δν ≈ 2J) -> the four AB lines at their exact positions with the
  classic "roofing" intensity ratio ``(1 + sin2θ) / (1 − sin2θ)``, ``sin2θ = J / √(Δ² + J²)``.

Test (c) is the one that fails if the simulator silently used the first-order (weak-coupling)
approximation instead of full Hamiltonian diagonalisation.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths

from moldetr.simulate import simulate

# Canonical MolDeTr grid: 400 MHz, a 3 ppm window = 1200 Hz over 6144 points -> 5.12 points/Hz.
BASE_FREQ = 400.0
LEFT_PPM = 8.0
RIGHT_PPM = 5.0
N_POINTS = 6144
PTS_PER_HZ = 5.12
WINDOW_HZ = (LEFT_PPM - RIGHT_PPM) * BASE_FREQ  # 1200.0
HZ_PER_POINT = WINDOW_HZ / (N_POINTS - 1)  # ppm-axis sample spacing, in Hz
PPM_PER_POINT = (LEFT_PPM - RIGHT_PPM) / (N_POINTS - 1)


def _peaks(spec: np.ndarray, height_frac: float = 0.05, distance: int = 15) -> np.ndarray:
    """Indices of local maxima taller than ``height_frac`` of the global max."""
    idx, _ = find_peaks(spec, height=height_frac * float(spec.max()), distance=distance)
    return idx


def _hz(idx: np.ndarray, ppm_axis: np.ndarray) -> np.ndarray:
    """Convert sample indices to their transition frequency in Hz via the returned ppm axis."""
    return ppm_axis[idx] * BASE_FREQ


def test_single_uncoupled_spin_position_and_width() -> None:
    """(a) One spin -> a single Lorentzian at the correct ppm with FWHM = requested width."""
    shift, fwhm = 6.5, 2.0
    spec, ppm = simulate([shift], [[0.0]], [fwhm], BASE_FREQ, LEFT_PPM, RIGHT_PPM, N_POINTS)

    assert spec.shape == (N_POINTS,)
    assert ppm.shape == (N_POINTS,)
    # ppm axis runs high -> low (NMR convention), index 0 = left_ppm.
    assert ppm[0] == LEFT_PPM and ppm[-1] == RIGHT_PPM

    idx = _peaks(spec)
    assert idx.size == 1, f"expected 1 peak, found {idx.size}"

    peak_ppm = float(ppm[idx[0]])
    assert abs(peak_ppm - shift) < 2 * PPM_PER_POINT, f"peak at {peak_ppm:.4f} ppm, want {shift}"

    width_pts = float(peak_widths(spec, idx, rel_height=0.5)[0][0])
    width_hz = width_pts * HZ_PER_POINT
    assert abs(width_hz - fwhm) < 1.0 * HZ_PER_POINT, f"FWHM {width_hz:.3f} Hz, want {fwhm}"


def test_first_order_ax_two_doublets() -> None:
    """(b) AX (Δδ >> J) -> two doublets, each split by J Hz (= J*5.12 points), 1:1 intensities."""
    shifts, j_hz = [7.0, 6.0], 7.0
    couplings = [[0.0, j_hz], [j_hz, 0.0]]
    spec, ppm = simulate(shifts, couplings, [1.0, 1.0], BASE_FREQ, LEFT_PPM, RIGHT_PPM, N_POINTS)

    idx = _peaks(spec, height_frac=0.3, distance=20)
    assert idx.size == 4, f"expected 4 peaks, found {idx.size}"

    order = np.argsort(_hz(idx, ppm))
    idx_sorted = idx[order]
    hz = _hz(idx_sorted, ppm)  # ascending Hz
    low, high = hz[:2], hz[2:]

    # Two doublets centred on the two shifts.
    assert abs(float(np.mean(low)) / BASE_FREQ - 6.0) < 0.02
    assert abs(float(np.mean(high)) / BASE_FREQ - 7.0) < 0.02

    # Splitting inside each doublet equals J (in Hz and, as asked, in points via 5.12 pts/Hz).
    assert abs((low[1] - low[0]) - j_hz) < 0.3
    assert abs((high[1] - high[0]) - j_hz) < 0.3
    split_pts = abs(int(idx_sorted[1]) - int(idx_sorted[0]))
    assert abs(split_pts - j_hz * PTS_PER_HZ) < 2, f"{split_pts} pts, want {j_hz * PTS_PER_HZ:.2f}"

    # Intensities ~1:1 within each doublet (first order -> negligible roofing).
    h = spec[idx_sorted]
    assert abs(h[0] - h[1]) / max(h[0], h[1]) < 0.05
    assert abs(h[2] - h[3]) / max(h[2], h[3]) < 0.05


def test_ab_roofing_positions_and_intensity_ratio() -> None:
    """(c) AB (Δν ≈ 2J) -> exact 4-line positions and analytical inner/outer roofing ratio."""
    j_hz, delta_hz, center = 12.0, 24.0, 6.5  # Δν = 2J
    half_ppm = (delta_hz / BASE_FREQ) / 2.0
    shifts = [center + half_ppm, center - half_ppm]
    couplings = [[0.0, j_hz], [j_hz, 0.0]]
    spec, ppm = simulate(shifts, couplings, [2.0, 2.0], BASE_FREQ, LEFT_PPM, RIGHT_PPM, N_POINTS)

    idx = _peaks(spec, height_frac=0.05, distance=15)
    assert idx.size == 4, f"expected 4 AB lines, found {idx.size}"

    order = np.argsort(_hz(idx, ppm))
    idx_sorted = idx[order]
    hz = _hz(idx_sorted, ppm)  # ascending: [outer-, inner-, inner+, outer+]

    # Exact AB line positions: nu0 ± (C ∓ J/2), C = ½√(Δ² + J²).
    nu0 = center * BASE_FREQ
    c_ab = 0.5 * np.sqrt(delta_hz**2 + j_hz**2)
    expected = np.sort(
        [
            nu0 - (c_ab + j_hz / 2),
            nu0 - (c_ab - j_hz / 2),
            nu0 + (c_ab - j_hz / 2),
            nu0 + (c_ab + j_hz / 2),
        ]
    )
    assert np.allclose(hz, expected, atol=0.35), f"lines {hz}, want {expected}"

    # Each outer/inner pair on one side is split by J.
    assert abs((hz[1] - hz[0]) - j_hz) < 0.35
    assert abs((hz[3] - hz[2]) - j_hz) < 0.35

    # Roofing: inner lines taller than outer, ratio = (1+sin2θ)/(1-sin2θ).
    h = spec[idx_sorted]
    inner = 0.5 * (h[1] + h[2])
    outer = 0.5 * (h[0] + h[3])
    sin2theta = j_hz / np.sqrt(j_hz**2 + delta_hz**2)
    expected_ratio = (1.0 + sin2theta) / (1.0 - sin2theta)
    ratio = inner / outer
    assert inner > outer, "inner AB lines must roof above the outer lines"
    assert abs(ratio - expected_ratio) / expected_ratio < 0.05, (
        f"roofing ratio {ratio:.3f}, analytical {expected_ratio:.3f}"
    )
