"""Two independent spin systems, composed from two editors instead of one block-diagonal grid.

The physics for this already existed and is pinned elsewhere: `simulate_systems` splits the coupling
matrix into blocks, simulates each on a per-proton scale and sums them, so a block-diagonal matrix is
*exactly* two systems simulated apart and added (`test_simulate_blocks`, `test_simulate_additivity`).
What was missing was the interface — the preset dropdown replaces the whole grid, so a second system
had to be hand-assembled cell by cell, and users had to learn the zero-cross-coupling convention to
know that was even possible.

These tests therefore assert a **composition** property, not new physics: driving the tab from two
panels must produce the same spectrum as the single combined matrix, and the reference is built from
the phenotypes' own arrays so it does not come from the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

#: Two presets that are already shipped, chosen because they are the maintainer's own example and
#: because 2 + 4 spins stays inside a single editor's cap — so this pair also works as the reference
#: for the combined-matrix comparison.
FIRST = "AB"
SECOND = "AA'BB'"

#: add_noise, snr, phase0, broaden, baseline, threshold, satellites, satellite J — the eight
#: distortion arguments, shared by both panels because they describe the spectrometer, not a system.
DISTORTION = (False, 3.0, 0.0, 0.0, 0.0, 0.3, False, 130.0)


def _combined_reference(app, first: str, second: str):
    """Simulate `first` + `second` as one block-diagonal system, from the phenotype arrays.

    Deliberately *not* built from the grid helpers: this is the independent reference the two-panel
    path is measured against, so it must not share an implementation with it.
    """
    a, b = app.sp.PHENOTYPES[first], app.sp.PHENOTYPES[second]
    na, nb = len(a["shifts_ppm"]), len(b["shifts_ppm"])

    couplings = np.zeros((na + nb, na + nb))
    couplings[:na, :na] = app.sp.build_coupling_matrix(na, a["couplings"])
    couplings[na:, na:] = app.sp.build_coupling_matrix(nb, b["couplings"])

    return app.simulate_systems(
        list(a["shifts_ppm"]) + list(b["shifts_ppm"]),
        couplings,
        list(a["widths_hz"]) + list(b["widths_hz"]),
        app.sp.BASE_FREQ_MHZ,
        app.sp.LEFT_PPM,
        app.sp.RIGHT_PPM,
        app.sp.N_POINTS,
    )


@pytest.mark.unit
def test_two_panels_reproduce_the_single_block_diagonal_matrix(patch_model) -> None:
    """The composition property: panel 1 + panel 2 == the combined matrix, pointwise.

    This is the check that the second panel is wired into the *same* computation rather than a
    parallel one. The tolerance is 1e-12 because both sides call `simulate_systems` on identical
    inputs — anything larger would be hiding a real difference in how the blocks were assembled.
    """
    app = patch_model
    grid_a, widths_a = app._phenotype_grid(FIRST)
    grid_b, widths_b = app._phenotype_grid(SECOND)

    cache = app._simulate_stage(grid_a, widths_a, True, grid_b, widths_b)
    assert not isinstance(cache, str), cache

    expected, _ppm = _combined_reference(app, FIRST, SECOND)
    assert np.allclose(cache["spectrum"], expected, atol=1e-12)


@pytest.mark.unit
def test_the_second_system_contributes_its_own_ground_truth_groups(patch_model) -> None:
    """Both systems must reach the comparison, not just the spectrum.

    A implementation that summed the spectra but left `gt_groups` derived from panel 1 alone would
    pass the spectrum check above and still report every peak of the second system as spurious.
    """
    app = patch_model
    grid_a, widths_a = app._phenotype_grid(FIRST)
    grid_b, widths_b = app._phenotype_grid(SECOND)

    one = app._simulate_stage(grid_a, widths_a)
    both = app._simulate_stage(grid_a, widths_a, True, grid_b, widths_b)
    assert not isinstance(one, str) and not isinstance(both, str)

    assert len(both["gt_groups"]) > len(one["gt_groups"])
    shifts_alone = {g["shift_ppm"] for g in one["gt_groups"]}
    assert shifts_alone < {g["shift_ppm"] for g in both["gt_groups"]}


@pytest.mark.unit
def test_the_label_counts_both_systems(patch_model) -> None:
    """The status line already reports `N spin(s) in M system(s)`; M must now be able to exceed 1."""
    app = patch_model
    grid_a, widths_a = app._phenotype_grid(FIRST)
    grid_b, widths_b = app._phenotype_grid(SECOND)

    cache = app._simulate_stage(grid_a, widths_a, True, grid_b, widths_b)

    assert not isinstance(cache, str), cache
    assert cache["label"] == "6 spin(s) in 2 system(s)"


@pytest.mark.unit
def test_a_disabled_second_panel_changes_nothing(patch_model) -> None:
    """The default-off guard, and the reason every pre-existing golden stays valid.

    `test_simulate_two_stage`'s byte-exact CSV and `test_e2e_client`'s 8-positional-arg call both
    rest on this: the second panel's controls exist and hold real values, and must still be
    completely inert until it is switched on. Asserting it here turns that from a claim into a
    checked property.
    """
    app = patch_model
    grid_a, widths_a = app._phenotype_grid("ethyl")
    grid_b, widths_b = app._phenotype_grid(SECOND)

    before = app._simulate_stage(grid_a, widths_a)
    after = app._simulate_stage(grid_a, widths_a, False, grid_b, widths_b)
    assert not isinstance(before, str) and not isinstance(after, str)

    assert after["label"] == before["label"]
    assert np.array_equal(after["spectrum"], before["spectrum"])
    assert after["gt_groups"] == before["gt_groups"]


@pytest.mark.unit
def test_the_combined_system_may_exceed_one_editors_spin_cap(patch_model) -> None:
    """`MAX_MATRIX_SPINS` bounds *an editor*, not the spectrum — which is the point of two panels.

    ethyl (5) + AA'BB'C (5) is 10 spins, above the 8-spin grid cap, and legal: the real limit is
    `MAX_BLOCK_SPINS` = 10 **per coupled block**, and neither block here comes close. A merged-grid
    design would have rejected this pair for no physical reason.
    """
    app = patch_model
    grid_a, widths_a = app._phenotype_grid("ethyl")
    grid_b, widths_b = app._phenotype_grid("AA'BB'C")
    assert len(grid_a) + len(grid_b) > app.MAX_MATRIX_SPINS

    cache = app._simulate_stage(grid_a, widths_a, True, grid_b, widths_b)

    assert not isinstance(cache, str), cache
    assert cache["label"] == "10 spin(s) in 2 system(s)"


@pytest.mark.unit
def test_the_wired_callback_carries_the_second_panel_through(patch_model) -> None:
    """`simulate_to_state` is the function Gradio actually calls, so the panel must reach *it*.

    `_simulate_stage` accepting the arguments proves nothing about the button: the tab is wired to
    `simulate_to_state`, and a second panel that stops at the helper would be inert in the browser
    while every other unit test here stayed green.

    Scope, stated because it is easy to over-read: this calls the callback **directly**, so it pins
    that the parameters are threaded through to the simulation and reported — not that the button
    hands them over in the right order. Gradio binds positionally, and a permutation of the wired
    inputs keeps this green. `test_ui_graph.py::test_the_simulate_click_wires_its_inputs_in_the_
    callbacks_parameter_order` is what covers that half.
    """
    app = patch_model
    grid_a, widths_a = app._phenotype_grid(FIRST)
    grid_b, widths_b = app._phenotype_grid(SECOND)

    cache, _table, _fig, msg = app.simulate_to_state(
        grid_a, widths_a, *DISTORTION, True, grid_b, widths_b
    )

    assert not isinstance(cache, str), cache
    assert cache["label"] == "6 spin(s) in 2 system(s)"
    assert "6 spin(s) in 2 system(s)" in msg


@pytest.mark.unit
def test_the_second_panel_has_its_own_controls_in_the_graph(app_module) -> None:
    """Distinct `elem_id`s, because six browser tests locate `#sim-matrix` in Playwright strict mode.

    Reusing panel 1's ids would be valid Python, invalid HTML, and would turn every one of those
    locators into a strict-mode violation rather than a clear failure here.
    """
    demo = app_module.build_ui()
    present = {b.elem_id for b in demo.blocks.values() if getattr(b, "elem_id", None)}

    assert {
        "sim-second-enabled",
        "sim-preset-2",
        "sim-nspins-2",
        "sim-matrix-2",
        "sim-widths-2",
    } <= present


@pytest.mark.unit
def test_the_second_system_is_switched_off_in_the_shipped_graph(app_module) -> None:
    """The default-off property, asserted where it actually has to hold: the built UI.

    `test_a_disabled_second_panel_changes_nothing` proves the Python path is inert when the flag is
    False. This is the other half — that the flag *ships* False, which is what makes the
    8-positional-arg `gradio_client` call and every byte-exact golden still describe the app.
    """
    demo = app_module.build_ui()
    boxes = [b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "sim-second-enabled"]

    assert len(boxes) == 1
    assert boxes[0].value is False


@pytest.mark.unit
@pytest.mark.parametrize("empty", [[], None], ids=["cleared-grid", "no-grid"])
def test_switching_the_second_system_on_with_nothing_in_it_is_harmless(patch_model, empty) -> None:
    """An enabled but empty second panel must degrade to one system, not to an error.

    Both spellings reach this: a user clearing every row leaves `[]`, and a `gradio_client` caller
    that omits the tail leaves `None`. Neither is a mistake worth a message — there is simply no
    second system — so the result has to equal the single-system spectrum rather than raise, return
    the "add at least one spin" prompt, or build a degenerate zero-width block.
    """
    app = patch_model
    grid_a, widths_a = app._phenotype_grid(FIRST)

    alone = app._simulate_stage(grid_a, widths_a)
    with_empty = app._simulate_stage(grid_a, widths_a, True, empty, empty)
    assert not isinstance(alone, str) and not isinstance(with_empty, str), with_empty

    assert with_empty["label"] == alone["label"]
    assert np.array_equal(with_empty["spectrum"], alone["spectrum"])


@pytest.mark.unit
def test_a_bad_cell_in_the_second_panel_becomes_a_message_not_a_traceback(patch_model) -> None:
    """The error channel has to cover panel 2, or a typo there takes the tab down.

    Panel 1 already has this guard; the failure mode it prevents is identical and the second grid
    is no less editable.
    """
    app = patch_model
    grid_a, widths_a = app._phenotype_grid(FIRST)
    grid_b, widths_b = app._phenotype_grid(SECOND)
    grid_b[0][1] = "seven point five"

    out = app._simulate_stage(grid_a, widths_a, True, grid_b, widths_b)

    assert isinstance(out, str)
    assert "row 1" in out
