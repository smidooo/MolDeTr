"""The spin matrix is the Simulate tab's single source of truth for what is being simulated.

One grid carries both halves of a spin-system definition: the **diagonal** holds each spin's
chemical shift in ppm, the **upper triangle** holds the pairwise couplings in Hz. That is not a UI
affectation — `simulate` reads only `couplings[i, j]` for `i < j`, so the upper triangle is already
the contract, and several independent spin systems fall out of one matrix by leaving the cross terms
at zero rather than needing a separate "system" concept in the interface.

These tests cover the translation layer between the editor grid and the simulator, which is where a
misread row or a swapped index turns into a silently wrong spectrum rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_the_diagonal_is_read_as_shifts_and_the_triangle_as_couplings(app_module) -> None:
    """The grid's first column is a static label, so every value is offset by one."""
    rows = [
        ["A", 7.5, 8.0],
        ["B", 0.0, 6.9],
    ]

    shifts, couplings = app_module._matrix_to_system(rows)

    assert shifts == [7.5, 6.9]
    assert couplings.shape == (2, 2)
    assert couplings[0, 1] == pytest.approx(8.0)


@pytest.mark.unit
def test_a_blank_cell_reads_as_zero(app_module) -> None:
    """Gradio hands back empty strings and None for cleared cells; neither may reach the simulator.

    `float("")` raises and `float(None)` raises, so without explicit coercion a user clearing one
    coupling would get a traceback rather than an uncoupled spin.
    """
    rows = [
        ["A", 7.5, "", None],
        ["B", 0.0, 6.9, "  "],
        ["C", 0.0, 0.0, 1.2],
    ]

    shifts, couplings = app_module._matrix_to_system(rows)

    assert shifts == [7.5, 6.9, 1.2]
    assert np.count_nonzero(couplings) == 0


@pytest.mark.unit
def test_the_lower_triangle_is_ignored_so_the_grid_matches_the_simulator(app_module) -> None:
    """Whatever the lower half shows, only the upper half defines the system.

    `simulate` and `coupling_blocks` both read the upper triangle, so the parser must too. If it
    mirrored the lower half back instead, a stale value left below the diagonal would couple two
    spins that the plotted spectrum treats as independent.
    """
    rows = [
        ["A", 7.5, 8.0],
        ["B", 99.0, 6.9],  # 99 sits below the diagonal and must not be read
    ]

    _shifts, couplings = app_module._matrix_to_system(rows)

    assert couplings[0, 1] == pytest.approx(8.0)
    assert float(np.max(np.abs(couplings))) == pytest.approx(8.0)


@pytest.mark.unit
def test_a_non_numeric_cell_is_reported_rather_than_swallowed(app_module) -> None:
    """A typo must name the cell, since a silently dropped coupling looks like a valid spectrum."""
    rows = [
        ["A", 7.5, "eight"],
        ["B", 0.0, 6.9],
    ]

    with pytest.raises(ValueError, match="row 1"):
        app_module._matrix_to_system(rows)


@pytest.mark.unit
def test_growing_the_matrix_keeps_what_was_already_typed(app_module) -> None:
    """Changing the spin count must not discard the user's work.

    Rebuilding the grid from scratch on every slider step is the easy implementation and the one
    that loses a half-entered system on a mis-click.
    """
    rows = [
        ["A", 7.5, 8.0],
        ["B", 0.0, 6.9],
    ]

    grown = app_module._resize_matrix(rows, 3)

    assert len(grown) == 3
    assert grown[0][1] == pytest.approx(7.5)
    assert grown[0][2] == pytest.approx(8.0)
    assert grown[1][2] == pytest.approx(6.9)
    assert grown[2][3] == pytest.approx(0.0)  # the new spin starts at 0 ppm, uncoupled


@pytest.mark.unit
def test_shrinking_the_matrix_drops_only_the_removed_spins(app_module) -> None:
    rows = [
        ["A", 7.5, 8.0, 1.0],
        ["B", 0.0, 6.9, 2.0],
        ["C", 0.0, 0.0, 1.2],
    ]

    shrunk = app_module._resize_matrix(rows, 2)

    assert len(shrunk) == 2
    assert [len(r) for r in shrunk] == [3, 3]
    assert shrunk[0][1] == pytest.approx(7.5)
    assert shrunk[0][2] == pytest.approx(8.0)


@pytest.mark.unit
def test_row_labels_are_pople_letters(app_module) -> None:
    """A, B, C ... name the spins the way the spin-system literature does, and the paper's table."""
    grid = app_module._resize_matrix([], 4)

    assert [row[0] for row in grid] == ["A", "B", "C", "D"]


@pytest.mark.unit
def test_simulating_from_the_grid_matches_the_phenotype_it_was_filled_from(patch_model) -> None:
    """Driving the tab from the matrix must not change the spectrum a preset used to produce.

    The grid replaces a shift textbox plus one global J, so this is the check that the replacement
    describes the same spin system rather than merely a plausible one. Compared against
    `simulate_systems` called with the phenotype's own arrays, so the reference does not come from
    the code under test.
    """
    app = patch_model
    rows, width_rows = app._phenotype_grid("ethyl")

    cache = app._simulate_stage(rows, width_rows)
    assert not isinstance(cache, str), cache

    pheno = app.sp.PHENOTYPES["ethyl"]
    expected, _ppm = app.simulate_systems(
        pheno["shifts_ppm"],
        app.sp.build_coupling_matrix(len(pheno["shifts_ppm"]), pheno["couplings"]),
        pheno["widths_hz"],
        app.sp.BASE_FREQ_MHZ,
        app.sp.LEFT_PPM,
        app.sp.RIGHT_PPM,
        app.sp.N_POINTS,
    )
    assert np.allclose(cache["spectrum"], expected, atol=1e-12)
    assert [(g["shift_ppm"], g["proton_count"]) for g in cache["gt_groups"]] == [(3.5, 2), (1.2, 3)]


@pytest.mark.unit
def test_the_width_table_has_one_row_per_spin_system_not_per_peak(app_module) -> None:
    """The table's grain is the coupling block, because that is what the simulator can honour.

    `simulate` collapses widths to a single mean *within* a block: for ethyl, per-spin widths of
    (1, 1, 1, 3, 3) give a spectrum bit-identical to a uniform 1.8. A row per ground-truth group
    would therefore put two controls on screen that silently average into one. Ethyl's five spins
    are a single coupled system, so it gets one row — even though the spectrum shows two multiplets.
    """
    _rows, widths = app_module._phenotype_grid("ethyl")
    assert len(widths) == 1
    assert widths[0][1] == 5  # all five protons, one system

    # Two genuinely independent systems do get a row each.
    shifts = [7.5, 1.2]
    couplings = np.zeros((2, 2))
    assert len(app_module._width_rows(shifts, couplings)) == 2


@pytest.mark.unit
def test_independent_systems_can_carry_different_line_widths(patch_model) -> None:
    """The payoff of simulating blocks apart: two systems, two line shapes, in one window.

    A mean-collapsing implementation returns the same array whichever way the widths are split, so
    the inequality below is exactly the regression guard.
    """
    app = patch_model
    grid = app._resize_matrix([], 2)
    grid[0][1], grid[1][2] = 7.5, 1.2  # two uncoupled spins

    uneven = app._simulate_stage(grid, [["δ 7.5", 1, 0.5], ["δ 1.2", 1, 3.0]])
    uniform = app._simulate_stage(grid, [["δ 7.5", 1, 1.75], ["δ 1.2", 1, 1.75]])

    assert not isinstance(uneven, str) and not isinstance(uniform, str)
    assert not np.allclose(uneven["spectrum"], uniform["spectrum"], atol=1e-6)


@pytest.mark.unit
def test_a_bad_cell_becomes_a_message_not_a_traceback(patch_model) -> None:
    """The stage's error channel has to cover the grid too, or a typo takes the tab down."""
    app = patch_model
    rows, width_rows = app._phenotype_grid("ethyl")
    rows[0][1] = "seven point five"

    out = app._simulate_stage(rows, width_rows)

    assert isinstance(out, str)
    assert "row 1" in out


@pytest.mark.unit
def test_every_phenotype_round_trips_through_the_grid(app_module) -> None:
    """The dropdown pre-fills the matrix, so each preset must survive the trip unchanged.

    This is what makes the matrix the single source of truth rather than a second one: the phenotype
    stops being an input to simulation and becomes a way to populate the grid.
    """
    for name in app_module.PHENOTYPE_CHOICES:
        pheno = app_module.sp.PHENOTYPES[name]
        rows, _widths = app_module._phenotype_grid(name)
        shifts, couplings = app_module._matrix_to_system(rows)

        assert shifts == pytest.approx(pheno["shifts_ppm"]), name
        expected = app_module.sp.build_coupling_matrix(len(pheno["shifts_ppm"]), pheno["couplings"])
        assert np.allclose(np.triu(couplings, 1), np.triu(expected, 1)), name
