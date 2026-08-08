"""Block decomposition: independent spin systems are disconnected components of the J matrix.

Two reasons this exists, and both matter:

* **Correctness.** Users want to define several spin systems and see them summed. Leaving the cross
  couplings at zero already says "independent", so the blocks fall out of the matrix itself. The
  Simulate tab now *also* offers a second matrix editor, but that is presentation only: it lays the
  two systems out block-diagonally and calls the same ``simulate_systems`` once, which is why these
  tests still cover it without knowing it exists.
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


@pytest.mark.unit
def test_an_upper_triangle_matrix_blocks_the_same_as_a_symmetric_one() -> None:
    """`simulate` documents that only the upper triangle is read, so blocking must agree.

    `coupling_blocks` walks the *full* row, so a matrix filled only above the diagonal used to split
    differently from the symmetric matrix meaning the same thing. Two spins both coupled to a third
    came out as `[[0, 2], [1]]` instead of `[[0, 1, 2]]`: the depth-first walk reaches spin 2 from
    spin 0, then finds nothing in row 2 because `j[2, 1]` is the empty half, and by the time it
    starts from spin 1 the shared partner is already claimed.

    Nothing hit it before because the only producer, `sp.build_coupling_matrix`, fills both halves.
    A matrix editor whose contract is "fill the upper triangle" produces exactly this input, and the
    result is a silently wrong spectrum — a coupled three-spin system simulated as a pair plus a
    singlet.
    """
    symmetric = np.zeros((3, 3))
    symmetric[0, 2] = symmetric[2, 0] = 7.0
    symmetric[1, 2] = symmetric[2, 1] = 5.0
    upper = np.triu(symmetric, 1)

    assert coupling_blocks(upper) == coupling_blocks(symmetric) == [[0, 1, 2]]

    shifts, widths = [7.5, 6.9, 3.5], [1.0] * 3
    from_upper, _ = simulate_systems(
        shifts, upper, widths, BASE_FREQ_MHZ, LEFT_PPM, RIGHT_PPM, N_POINTS
    )
    from_symmetric, _ = simulate_systems(
        shifts, symmetric, widths, BASE_FREQ_MHZ, LEFT_PPM, RIGHT_PPM, N_POINTS
    )
    assert np.allclose(from_upper, from_symmetric, atol=1e-12)


@pytest.mark.unit
def test_the_lower_triangle_is_ignored_by_blocking_and_by_simulation_alike() -> None:
    """ "Only the upper triangle is read" has to mean the same thing to both, or they disagree.

    `simulate` builds its Hamiltonian from `couplings[i, j]` with `i < j`, so a coupling written
    *below* the diagonal is simply not there. `coupling_blocks` must reach the same verdict — if it
    symmetrised the whole matrix instead of the upper half, it would fuse spins into one block that
    `simulate` then simulates uncoupled, and the two halves of the pipeline would describe different
    spin systems.

    So this is not a claim that the lower triangle *should* be read. It pins that both sides ignore
    it, which is what makes the upper triangle a usable contract for a matrix editor.
    """
    below_only = np.zeros((2, 2))
    below_only[1, 0] = 12.0
    empty = np.zeros((2, 2))

    assert coupling_blocks(below_only) == coupling_blocks(empty) == [[0], [1]]

    shifts, widths = [3.50, 3.60], [1.0, 1.0]
    from_below, _ = simulate_systems(
        shifts, below_only, widths, BASE_FREQ_MHZ, LEFT_PPM, RIGHT_PPM, N_POINTS
    )
    from_empty, _ = simulate_systems(
        shifts, empty, widths, BASE_FREQ_MHZ, LEFT_PPM, RIGHT_PPM, N_POINTS
    )
    assert np.allclose(from_below, from_empty, atol=1e-12)


@pytest.mark.unit
def test_a_j_matrix_too_small_for_the_shifts_is_rejected() -> None:
    """An undersized J matrix must not silently define how many spins exist.

    `coupling_blocks` only reads `j.shape[0]`, so a 2x2 matrix against three shifts used to yield
    two blocks and drop the third proton — area 2.0 where the caller asked for 3, with no error.
    The per-block slice hid it from `simulate`'s own shape check, which only ever saw 2 of each.
    """
    with pytest.raises(ValueError, match="couplings_hz"):
        simulate_systems(
            [1.2, 3.5, 7.5],
            np.zeros((2, 2)),
            [1.0] * 3,
            BASE_FREQ_MHZ,
            LEFT_PPM,
            RIGHT_PPM,
            N_POINTS,
        )


@pytest.mark.unit
def test_a_surplus_line_width_is_rejected() -> None:
    """One width per spin, checked against the caller's shifts rather than a block's slice."""
    with pytest.raises(ValueError, match="widths_hz"):
        simulate_systems(
            [1.2, 3.5],
            np.zeros((2, 2)),
            [1.0] * 3,
            BASE_FREQ_MHZ,
            LEFT_PPM,
            RIGHT_PPM,
            N_POINTS,
        )


@pytest.mark.unit
def test_unknown_scale_is_rejected_by_the_block_path_too() -> None:
    """A misspelled scale must raise, not silently return a per-proton spectrum.

    The inner `simulate` call hardcodes `scale="protons"`, so it never sees the caller's value, and
    `if scale == "peak"` simply falls through. The result peaks near 0.37 instead of 1.0 and feeds
    `distort()`, whose magnitudes are calibrated against a peak of 1 — every distortion would land
    roughly 2.7x too strong.
    """
    with pytest.raises(ValueError, match="scale"):
        simulate_systems(
            [5.0],
            np.zeros((1, 1)),
            [1.0],
            BASE_FREQ_MHZ,
            LEFT_PPM,
            RIGHT_PPM,
            N_POINTS,
            scale="Peak",
        )


@pytest.mark.unit
def test_a_spin_with_no_observable_transition_is_rejected_under_proton_scaling() -> None:
    """`scale="protons"` promises area == proton count, so a silent zero must not be allowed.

    `_transitions` keeps only `delta_e > 0.0`, so a lone spin at exactly 0.0 ppm produces no
    transitions at all: its block contributes nothing and the total area is short by one proton with
    no warning. 0.0 ppm is the right-hand edge of the app's own 15 -> 0 window, so a user can type
    it. Peak scaling is left alone — it makes no area promise, and that behaviour predates this PR.
    """
    with pytest.raises(ValueError, match="no observable transition"):
        simulate_systems(
            [0.0, 5.0],
            np.zeros((2, 2)),
            [1.0, 1.0],
            BASE_FREQ_MHZ,
            LEFT_PPM,
            RIGHT_PPM,
            N_POINTS,
            scale="protons",
        )
