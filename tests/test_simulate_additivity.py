"""Integral scaling: one proton must contribute the same area in every simulated spectrum.

Without this, spectra cannot be added. `simulate()` peak-normalises (`spectrum /= spectrum.max()`),
so a 1H singlet and a 3H methyl come out the same height and summing two spin systems is
meaningless. Area, not height, tracks proton count, and the two diverge as soon as line widths do.

There is a second, less obvious trap. Simulating two uncoupled systems *jointly* does not give the
sum of simulating them separately: in the joint 2^(nA+nB) Hilbert space every A transition is
repeated once per B spin state, so A's lines carry a 2^nB degeneracy factor (and vice versa). The
naive superposition identity therefore fails on raw intensities. Normalising each system's **total
transition intensity to its proton count** cancels exactly that factor, which is what makes blocks
additive and is the property these tests pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from moldetr.simulate import simulate

BASE_FREQ_MHZ = 80.0
LEFT_PPM, RIGHT_PPM = 15.0, 0.0
N_POINTS = 6144


def _area(spectrum: np.ndarray, ppm_axis: np.ndarray, lo_ppm: float, hi_ppm: float) -> float:
    """Numerically integrate a ppm window.

    The ppm axis descends (NMR convention), so ``trapezoid`` returns a negative value; take the
    magnitude rather than reordering, which would only obscure the axis direction.
    """
    mask = (ppm_axis >= lo_ppm) & (ppm_axis <= hi_ppm)
    return abs(float(np.trapezoid(spectrum[mask], ppm_axis[mask])))


def _sim(shifts, couplings, widths, **kw):
    return simulate(
        shifts, couplings, widths, BASE_FREQ_MHZ, LEFT_PPM, RIGHT_PPM, N_POINTS, **kw
    )


@pytest.mark.unit
def test_peak_scaling_is_still_the_default() -> None:
    """Existing callers must be untouched: the default keeps max-normalising to 1.0."""
    spectrum, _ = _sim([7.5, 6.9], np.array([[0.0, 8.0], [8.0, 0.0]]), [1.0, 1.0])
    assert spectrum.max() == pytest.approx(1.0)


@pytest.mark.unit
def test_one_proton_has_the_same_area_in_every_spectrum() -> None:
    """A 1H singlet and a 3H singlet in one window must integrate 1 : 3.

    This is the user-facing invariant. It is checked *within* one spectrum, so it cannot be
    satisfied trivially by a global rescale.
    """
    shifts = [7.5] + [1.2] * 3  # one aromatic H, one methyl
    couplings = np.zeros((4, 4))
    spectrum, ppm = _sim(shifts, couplings, [1.0] * 4, scale="protons")

    one_h = _area(spectrum, ppm, 7.0, 8.0)
    three_h = _area(spectrum, ppm, 0.7, 1.7)
    assert one_h > 0.0
    assert three_h / one_h == pytest.approx(3.0, rel=0.02)


@pytest.mark.unit
def test_uncoupled_systems_superpose_exactly() -> None:
    """simulate(A ∪ B) == simulate(A) + simulate(B) when A and B share no coupling.

    This is what lets the app decompose the coupling matrix into blocks, simulate each, and sum.
    It fails on raw intensities because of the 2^n degeneracy factor described in the module
    docstring, so it is a genuine check on the scaling convention rather than a tautology.
    """
    a_shifts, a_j = [3.5, 3.5], np.zeros((2, 2))
    b_shifts, b_j = [1.2, 1.2, 1.2], np.zeros((3, 3))
    a_j[0, 1] = a_j[1, 0] = 12.0  # keep A genuinely coupled

    joint_shifts = a_shifts + b_shifts
    joint_j = np.zeros((5, 5))
    joint_j[:2, :2] = a_j  # no cross terms: two independent blocks

    joint, _ = _sim(joint_shifts, joint_j, [1.0] * 5, scale="protons")
    a_only, _ = _sim(a_shifts, a_j, [1.0] * 2, scale="protons")
    b_only, _ = _sim(b_shifts, b_j, [1.0] * 3, scale="protons")

    assert np.allclose(joint, a_only + b_only, atol=1e-9)


@pytest.mark.unit
def test_total_area_tracks_proton_count() -> None:
    """Two systems with different spin counts scale their whole-spectrum area accordingly."""
    two, ppm = _sim([5.0, 5.0], np.zeros((2, 2)), [1.0] * 2, scale="protons")
    six, _ = _sim([5.0] * 6, np.zeros((6, 6)), [1.0] * 6, scale="protons")

    assert _area(six, ppm, 4.0, 6.0) / _area(two, ppm, 4.0, 6.0) == pytest.approx(3.0, rel=0.02)


@pytest.mark.unit
def test_unknown_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="scale"):
        _sim([5.0], np.zeros((1, 1)), [1.0], scale="bogus")
