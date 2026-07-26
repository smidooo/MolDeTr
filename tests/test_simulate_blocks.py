"""Block decomposition: independent spin systems are disconnected components of the J matrix.

Two reasons this exists, and both matter:

* **Correctness.** Users want to define several spin systems and see them summed. Leaving the cross
  couplings at zero already says "independent", so the blocks fall out of the single matrix with no
  extra UI concept.
* **Cost.** The Hamiltonian is ``2**n``. A 3+2+2 layout is ``2**7 = 128`` states as one system but
  ``8 + 4 + 4 = 16`` as three blocks, which is the difference between usable and not on a free
  Colab CPU.

Splitting is only legitimate because per-proton scaling makes uncoupled blocks superpose exactly
(see ``test_simulate_additivity``).
"""

from __future__ import annotations

import numpy as np
import pytest

from moldetr.simulate import coupling_blocks, simulate_systems

BASE_FREQ_MHZ = 80.0
LEFT_PPM, RIGHT_PPM = 15.0, 0.0
N_POINTS = 6144


@pytest.mark.unit
def test_blocks_split_on_zero_cross_coupling() -> None:
    j = np.zeros((5, 5))
    j[0, 1] = j[1, 0] = 7.0  # {0,1}
    j[2, 3] = j[3, 4] = 5.0  # {2,3,4} via a chain
    j[3, 2] = j[4, 3] = 5.0
    assert coupling_blocks(j) == [[0, 1], [2, 3, 4]]


@pytest.mark.unit
def test_every_spin_appears_exactly_once() -> None:
    """Uncoupled spins are their own singleton blocks; nothing is dropped or duplicated."""
    j = np.zeros((4, 4))
    j[1, 2] = j[2, 1] = 3.0
    blocks = coupling_blocks(j)
    assert sorted(i for b in blocks for i in b) == [0, 1, 2, 3]
    assert [0] in blocks and [3] in blocks


@pytest.mark.unit
def test_coupling_below_tolerance_does_not_join_a_block() -> None:
    """A numerically tiny J is not a real coupling; it must not fuse two blocks into a 2**n cost."""
    j = np.zeros((2, 2))
    j[0, 1] = j[1, 0] = 1e-12
    assert coupling_blocks(j) == [[0], [1]]


@pytest.mark.unit
def test_block_split_matches_simulating_the_whole_system() -> None:
    """The decomposed result is identical to simulating everything at once, per proton.

    This is the test that licenses the optimisation: if it ever fails, the speed-up is changing the
    physics.
    """
    shifts = [3.5, 3.5, 1.2, 1.2, 1.2]
    j = np.zeros((5, 5))
    j[0, 1] = j[1, 0] = 12.0
    j[2, 3] = j[3, 2] = 6.0

    blocked, ppm = simulate_systems(
        shifts, j, [1.0] * 5, BASE_FREQ_MHZ, LEFT_PPM, RIGHT_PPM, N_POINTS, scale="protons"
    )
    from moldetr.simulate import simulate

    whole, _ = simulate(
        shifts, j, [1.0] * 5, BASE_FREQ_MHZ, LEFT_PPM, RIGHT_PPM, N_POINTS, scale="protons"
    )
    assert np.allclose(blocked, whole, atol=1e-9)
    assert ppm.shape == (N_POINTS,)


@pytest.mark.unit
def test_per_block_line_widths_are_honoured() -> None:
    """Widths are per spin; two blocks with different widths must not average into each other.

    `simulate()` collapses widths to one mean, so a single call cannot give two groups different
    line shapes. Decomposing is what makes per-group width real, which is why this is asserted here
    rather than assumed.
    """
    shifts = [7.5, 1.2]
    j = np.zeros((2, 2))  # independent
    narrow, _ = simulate_systems(
        shifts, j, [0.5, 3.0], BASE_FREQ_MHZ, LEFT_PPM, RIGHT_PPM, N_POINTS
    )
    uniform, _ = simulate_systems(
        shifts, j, [1.75, 1.75], BASE_FREQ_MHZ, LEFT_PPM, RIGHT_PPM, N_POINTS
    )
    # Same mean width, different distribution: a mean-collapsing implementation returns the same
    # array for both, so this inequality is exactly the regression guard.
    assert not np.allclose(narrow, uniform, atol=1e-6)


@pytest.mark.unit
def test_peak_scaling_preserves_relative_integrals() -> None:
    """The default peak rescale must not disturb the 1:3 ratio it is applied on top of.

    Distortion magnitudes are calibrated against a peak of 1 (training renormalises to max(Re)=1
    before phase/noise/baseline), so the summed spectrum is rescaled before distorting. That is only
    safe if it is a single global divisor.
    """
    shifts = [7.5] + [1.2] * 3
    j = np.zeros((4, 4))
    peaked, ppm = simulate_systems(
        shifts, j, [1.0] * 4, BASE_FREQ_MHZ, LEFT_PPM, RIGHT_PPM, N_POINTS
    )
    protons, _ = simulate_systems(
        shifts, j, [1.0] * 4, BASE_FREQ_MHZ, LEFT_PPM, RIGHT_PPM, N_POINTS, scale="protons"
    )

    assert peaked.max() == pytest.approx(1.0)

    def ratio(spec):
        mask_1h = (ppm >= 7.0) & (ppm <= 8.0)
        mask_3h = (ppm >= 0.7) & (ppm <= 1.7)
        a1 = abs(float(np.trapezoid(spec[mask_1h], ppm[mask_1h])))
        a3 = abs(float(np.trapezoid(spec[mask_3h], ppm[mask_3h])))
        return a3 / a1

    assert ratio(peaked) == pytest.approx(ratio(protons), rel=1e-9)
    assert ratio(peaked) == pytest.approx(3.0, rel=0.02)


@pytest.mark.unit
def test_spin_count_guard_rejects_an_unsimulatable_block() -> None:
    """One oversized block must fail fast rather than allocate a 2**n Hilbert space."""
    n = 15
    j = np.zeros((n, n))
    for i in range(n - 1):  # one fully connected chain -> a single 2**15 block
        j[i, i + 1] = j[i + 1, i] = 5.0
    with pytest.raises(ValueError, match="spins"):
        simulate_systems([1.0] * n, j, [1.0] * n, BASE_FREQ_MHZ, LEFT_PPM, RIGHT_PPM, N_POINTS)
