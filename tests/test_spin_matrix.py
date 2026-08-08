"""The spin matrix is the Simulate tab's single source of truth for what is being simulated.

One grid carries both halves of a spin-system definition: the **diagonal** holds each spin's
chemical shift in ppm, the **upper triangle** holds the pairwise couplings in Hz. That is not a UI
affectation — `simulate` reads only `couplings[i, j]` for `i < j`, so the upper triangle is already
the contract, and several independent spin systems fall out of one matrix by leaving the cross terms
at zero.

A second matrix editor now offers the same thing as an explicit choice, for discoverability; it
concatenates block-diagonally into exactly the single matrix these tests describe, so nothing here
changes. `test_two_spin_systems` covers that composition.

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
    assert widths[0][2] == 5  # all five protons, one system

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

    uneven = app._simulate_stage(grid, [["A", "7.5", 1, 0.5], ["B", "1.2", 1, 3.0]])
    uniform = app._simulate_stage(grid, [["A", "7.5", 1, 1.75], ["B", "1.2", 1, 1.75]])

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


@pytest.mark.unit
def test_editing_the_matrix_clears_the_cache_and_re_derives_the_widths(app_module) -> None:
    """Drive the handler the way the graph does, and check both of its outputs.

    An earlier version of this test asserted `invalidate_cache() is None`, which cannot fail while
    the function exists: deleting the `.change` wiring entirely, and mis-wiring its output to the
    status box instead of the cache, both left it green. Asserting on the returned cache *and* the
    rebuilt table is what makes it discriminating; `test_the_matrix_edit_handler_is_wired_to_the_cache`
    covers the wiring itself.
    """
    grid = app_module._resize_matrix([], 3)
    grid[0][1], grid[1][2], grid[2][3] = 7.5, 3.5, 1.2  # three uncoupled spins
    table = app_module._width_rows(*app_module._matrix_to_system(grid))
    table[0][3], table[1][3], table[2][3] = 0.5, 1.5, 3.0

    cache, rebuilt = app_module.matrix_edited(grid, table)

    assert cache is None  # the cached spectrum is dropped
    assert [row[3] for row in rebuilt] == [0.5, 1.5, 3.0]  # untouched systems keep their widths

    grid[0][2] = 8.0  # couple A-B: three systems become two
    _cache, merged = app_module.matrix_edited(grid, table)

    assert [row[0] for row in merged] == ["A, B", "C"]
    assert merged[1][3] == pytest.approx(3.0)  # δ 1.2 keeps its own width, not a neighbour's


@pytest.mark.unit
def test_retyping_a_shift_keeps_the_line_width_the_user_set(app_module) -> None:
    """Editing δ must not silently reset that system's line width.

    Rows are keyed on the spins a system contains ("A", "A, B"), not on its shifts, precisely so
    this holds. Keying on the shift list — the obvious first choice, since that is what the row
    displays — made every δ edit rewrite the label, so the width fell back to the 1.0 default with
    nothing on screen to say it had.
    """
    grid = app_module._resize_matrix([], 2)
    grid[0][1], grid[1][2] = 7.5, 1.2
    table = [["A", "7.5", 1, 2.5], ["B", "1.2", 1, 0.4]]

    grid[0][1] = 7.0  # the user retypes the first shift
    cache, kept = app_module.matrix_edited(grid, table)

    assert cache is None
    assert [row[3] for row in kept] == [2.5, 0.4]  # widths survive
    assert kept[0][1] == "7"  # the δ column is derived, so it follows the edit


@pytest.mark.unit
def test_a_broken_cell_leaves_the_width_table_alone(app_module) -> None:
    """A grid that cannot be parsed mid-edit must not destroy what the user typed.

    Returning an empty table here — the first implementation — pushed `[]` into the component and
    silently reverted every width to 1.0 on the next simulate.
    """
    grid = app_module._resize_matrix([], 2)
    grid[0][1] = "seven point five"
    table = [["A", "7.5", 1, 2.5], ["B", "1.2", 1, 0.4]]

    cache, kept = app_module.matrix_edited(grid, table)

    assert cache is None
    assert kept == table


@pytest.mark.unit
def test_a_cleared_cache_prompts_instead_of_rendering(app_module) -> None:
    """The prompt is what the user sees after an edit, so it has to name the next action."""
    table, fig, msg = app_module.redistort(None, False, 3.0, 6.0, 0.0, 0.0, 0.3, False, 130.0)

    assert (table, fig) == (None, None)
    assert "Simulate & Predict" in msg


@pytest.mark.unit
def test_a_stale_width_table_never_lands_on_the_wrong_spins(app_module) -> None:
    """Width rows are matched to spin systems by label, not by position in the table.

    Typing a coupling into the matrix merges two blocks into one, so the table's rows no longer
    line up with the systems. Matched positionally, row 2's width lands on block 2 — which is now a
    *different* set of spins — and the spin at δ 1.2 is simulated with the width shown beside the
    label "δ 3.5". The screen and the spectrum disagree, with no error.
    """
    shifts = [7.5, 3.5, 1.2]
    uncoupled = np.zeros((3, 3))
    table = app_module._width_rows(shifts, uncoupled)
    table[0][3], table[1][3], table[2][3] = 0.5, 1.5, 3.0
    assert app_module._widths_per_spin(shifts, uncoupled, table) == [0.5, 1.5, 3.0]

    coupled = np.zeros((3, 3))
    coupled[0, 1] = 8.0  # A and B are now one system; the table still shows three rows

    applied = app_module._widths_per_spin(shifts, coupled, table)

    # δ 1.2 keeps its own row's 3.0 rather than inheriting the row labelled "δ 3.5".
    assert applied[2] == pytest.approx(3.0)


@pytest.mark.unit
def test_a_ragged_grid_is_reported_not_raised(app_module) -> None:
    """Both grids allow adding rows, so a row can be shorter than the matrix is wide.

    `_matrix_to_system` indexes `row[k + 1]` unguarded, so a short row raised `IndexError` — and
    `_simulate_stage` catches only `ValueError`, so it escaped the error-string channel entirely and
    took the tab down with a Gradio toast instead of naming the problem.
    """
    ragged = [
        ["A", 7.5, 8.0, 1.0],
        ["B", 0.0, 6.9],  # one cell short
        ["C", 0.0, 0.0, 1.2],
    ]

    with pytest.raises(ValueError, match="row 2"):
        app_module._matrix_to_system(ragged)


@pytest.mark.unit
def test_a_short_width_row_is_reported_not_raised(app_module) -> None:
    """Same hole on the width table, reachable the same way."""
    shifts = [7.5, 1.2]
    couplings = np.zeros((2, 2))

    with pytest.raises(ValueError, match="width"):
        app_module._widths_per_spin(shifts, couplings, [["A", "7.5", 1], ["B", "1.2", 1, 1.0]])


@pytest.mark.unit
def test_a_non_finite_cell_names_the_cell(app_module) -> None:
    """NaN and inf pass `float()` and then fail much later, describing the wrong cause.

    A NaN shift produces a spin system with no observable transition, so the user is told the
    system is degenerate rather than that they typed something unusable into a specific cell.
    """
    for bad in ("nan", "inf", "-inf", float("nan")):
        with pytest.raises(ValueError, match="row 1"):
            app_module._matrix_to_system([["A", bad, 7.0], ["B", 2.0, 0.0]])


@pytest.mark.unit
def test_extra_columns_are_reported_rather_than_dropped(app_module) -> None:
    """`n` comes from the row count, so surplus columns used to vanish without comment."""
    with pytest.raises(ValueError, match="row 1"):
        app_module._matrix_to_system([["A", 1.0, 7.0, 5.0], ["B", 2.0, 0.0, 5.0]])


@pytest.mark.unit
def test_every_phenotype_ground_truth_matches_the_apps_own_derivation(app_module) -> None:
    """The hand-written GT in `PHENOTYPES` must agree with what the app derives from the matrix.

    These are the only two places ground truth exists, and they are reached by different paths: the
    CLI returns `pheno["gt_groups"]` verbatim, while the Simulate tab throws it away and re-derives
    from the edited grid via `_build_gt_groups`. So a wrong hand-written entry is invisible in the
    GUI and wrong in the CLI, with nothing to reconcile them -- which is exactly the gap this test
    closes, now that there are 25 Table S2 presets rather than 3 hand-checked ones.

    Comparison is order-insensitive on groups: `_build_gt_groups` orders by first appearance and the
    literals are written top-down, but that is incidental, not contract.
    """
    import simulate_and_predict as sp

    def key(groups):
        return sorted((round(g["shift_ppm"], 6), g["proton_count"], g["max_j_hz"]) for g in groups)

    for name, pheno in sp.PHENOTYPES.items():
        matrix = sp.build_coupling_matrix(len(pheno["shifts_ppm"]), pheno["couplings"])
        derived = app_module._build_gt_groups(pheno["shifts_ppm"], matrix)
        assert key(pheno["gt_groups"]) == key(derived), (
            f"{name}: declared ground truth disagrees with the app's derivation\n"
            f"  declared: {key(pheno['gt_groups'])}\n"
            f"  derived : {key(derived)}"
        )


@pytest.mark.unit
def test_every_phenotype_fits_the_editor_and_the_simulator(app_module) -> None:
    """No preset may exceed the grid cap or the exact-diagonalisation block cap.

    `MAX_MATRIX_SPINS` bounds the editor; `simulate_systems` raises above `MAX_BLOCK_SPINS` because
    the Hamiltonian is 2**n. A preset that violates either is unreachable in the GUI or raises on
    use, and with 25 of them nobody is going to click through each one.
    """
    from moldetr.simulate import MAX_BLOCK_SPINS, coupling_blocks

    import simulate_and_predict as sp

    for name, pheno in sp.PHENOTYPES.items():
        n = len(pheno["shifts_ppm"])
        assert n <= app_module.MAX_MATRIX_SPINS, f"{name}: {n} spins exceeds the editor cap"
        matrix = sp.build_coupling_matrix(n, pheno["couplings"])
        for block in coupling_blocks(matrix):
            assert len(block) <= MAX_BLOCK_SPINS, f"{name}: block of {len(block)} spins"
