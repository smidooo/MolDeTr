"""The Blocks graph as a contract: component count, event wiring, arity, elem_ids, no orphans.

Gradio resolves event wiring at *call* time. A `.click()` whose input list drifts out of step with
its callback signature builds fine, renders fine, and raises only when a user presses the button —
which in a frozen paper companion means the first person to try the demo finds it, not CI.

These tests read the built graph directly (`demo.blocks`, `demo.fns`) so they cost one `build_ui()`
and no browser. The counts are deliberate change detectors: this repo is a frozen release, so a
component appearing or vanishing should require someone to update a number on purpose.
"""

from __future__ import annotations

import inspect

import pytest

#: 56 -> 60 when the 13C-satellite controls landed: a checkbox, a J slider, their Row, and the
#: Row the noise checkbox moved into. Was unchanged across the matrix rebuild by coincidence,
#: not by stasis: the shift textbox, the two
#: number boxes and their row went away, and a spin-count slider, two dataframes and a hint line
#: arrived. `gr.State` holds the simulation cache but is not itself a Block.
#: 60 -> 67 when the optional second spin system landed. Six were written by hand — the Accordion,
#: its enable checkbox, a preset Dropdown, a spin-count Slider and two Dataframes — and the seventh
#: is a `Form` **Gradio generated**, because it wraps runs of consecutive form-like components. That
#: is why this number is read off the failure rather than derived: hand-counting the source misses it.
N_COMPONENTS = 67
#: load_example · _spec_report ×2 · predict_ui · simulate_to_state · redistort ×8 (the two
#: 13C-satellite controls joined the live triggers) · and, **per spin-system panel**, preset_grid ·
#: resize_spin_matrix · matrix_edited · invalidate_cache — four each. 17 -> 21 for the second
#: panel's four, then **-> 22** for `invalidate_on_second_toggle`: the enable checkbox changes what
#: is simulated, so it clears the cache like the grids do. It was the one such control wired to
#: nothing, and only `test_every_control_that_defines_the_spectrum_clears_the_cache` could see that
#: — this count was already satisfied at 21. Gradio derives an endpoint id for every wiring, so the
#: re-distort handlers are named one per control: `api_name=False` becomes the literal "false",
#: "false_1", ..., which is the auto-derived surface these tests exist to prevent.
N_EVENTS = 22

#: Label of the Simulate tab's line-broadening slider, addressed by the OOD-copy guard below.
BROADEN_LABEL = "Broadening FWHM (Hz; 0 = off)"

# elem_ids the CSS and the browser tests address by name. `md-check`/`md-plot` carry no CSS rule of
# their own (see the L7 exclusion list) but are still selector anchors for the e2e suite.
EXPECTED_ELEM_IDS = {
    "md-check",
    "md-examples",
    "md-file",
    "md-plot",
    "md-ppm",
    "md-table",
    "sim-matrix",
    "sim-matrix-2",
    "sim-nspins",
    "sim-nspins-2",
    "sim-preset",
    "sim-preset-2",
    "sim-second-enabled",
    "sim-widths",
    "sim-widths-2",
}


@pytest.fixture(scope="module")
def demo():
    """One built Blocks graph for the whole module — `build_ui()` is not cheap.

    Imports `app` directly rather than taking the function-scoped `app_module` fixture: these tests
    only read the graph, so there is nothing to isolate per test, and the module scope keeps six
    `build_ui()` calls down to one.
    """
    import app

    return app.build_ui()


def _required_params(fn) -> list[str]:
    """Positional parameters with no default — what Gradio must supply from the input list."""
    params = inspect.signature(fn).parameters.values()
    return [
        p.name
        for p in params
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]


@pytest.mark.unit
def test_component_and_event_counts_are_the_frozen_surface(demo):
    """Change detector. If this fails because you added UI on purpose, update the constant *and*
    the browser-tier expectations that walk the same surface.
    """
    assert len(demo.blocks) == N_COMPONENTS
    assert len(demo.fns) == N_EVENTS


@pytest.mark.unit
def test_every_event_arity_matches_its_callback_signature(demo):
    """The failure this catches is invisible until the button is pressed: Gradio passes the wired
    inputs positionally, so one extra `gr.Number` in an input list is a `TypeError` at click time.
    """
    mismatches = [
        f"{d.fn.__name__}: {len(d.inputs)} wired inputs vs {len(_required_params(d.fn))} "
        f"required params {_required_params(d.fn)}"
        for d in demo.fns.values()
        if len(d.inputs) != len(_required_params(d.fn))
    ]
    assert not mismatches, "event/callback arity drift:\n  " + "\n  ".join(mismatches)


@pytest.mark.unit
def test_the_simulate_click_wires_its_inputs_in_the_callbacks_parameter_order(demo):
    """Arity is not order. Gradio binds the input list positionally, so three controls of the right
    count in the wrong order type-check, build, render, and hand `simulate_to_state` a grid where it
    expects a bool.

    `test_every_event_arity_matches_its_callback_signature` compares counts only — 13 wired vs 13
    required — and stays green through any permutation. The failure would surface as
    "Invalid spin matrix" on a matrix the user can plainly see is valid, or worse, as a plausible
    spectrum built from the wrong panel.

    Only the spin-system controls carry `elem_id`s, so this pins the head and the tail — the two
    regions where a parameter was actually appended — rather than the anonymous distortion sliders
    in between.
    """
    import app as app_module

    click = next(d for d in demo.fns.values() if d.api_name == "simulate_and_detect")
    wired = [getattr(c, "elem_id", None) for c in click.inputs]
    params = _required_params(app_module.simulate_to_state)

    assert len(wired) == len(params), f"{len(wired)} wired vs {len(params)} params"
    assert wired[:2] == ["sim-matrix", "sim-widths"]
    assert params[:2] == ["matrix_rows", "width_rows"]
    # The tail is what this diff appended, and appending is only safe if it stays the tail.
    assert wired[-3:] == ["sim-second-enabled", "sim-matrix-2", "sim-widths-2"]
    assert params[-3:] == ["second_enabled", "matrix_rows2", "width_rows2"]


@pytest.mark.unit
def test_no_event_wires_a_component_from_outside_this_graph(demo):
    """Orphan check: a component built outside the `with gr.Blocks()` context (or left over from a
    previous `build_ui()`) wires up silently and then never updates in the browser.
    """
    owned = set(demo.blocks)
    stray = {
        (d.fn.__name__, role, c._id)
        for d in demo.fns.values()
        for role, comps in (("input", d.inputs), ("output", d.outputs))
        for c in comps
        if c._id not in owned
    }
    assert not stray, f"components wired but not owned by this Blocks graph: {sorted(stray)}"


@pytest.mark.unit
def test_expected_elem_ids_are_all_present(demo):
    """`elem_id`s are the contract between the Python graph, `CUSTOM_CSS`, and the browser tests —
    the one name that must agree across all three.
    """
    present = {b.elem_id for b in demo.blocks.values() if getattr(b, "elem_id", None)}
    assert EXPECTED_ELEM_IDS <= present, f"missing elem_ids: {sorted(EXPECTED_ELEM_IDS - present)}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "elem_id",
    ["sim-matrix", "sim-widths", "sim-matrix-2", "sim-widths-2", "sim-second-enabled"],
)
def test_every_control_that_defines_the_spectrum_clears_the_cache(demo, elem_id):
    """A control that changes *what is simulated* must invalidate the cached spectrum.

    The distortion sliders deliberately do not: they re-distort the cache, which is the whole point
    of keeping it. But anything describing the spin system itself has to clear it, or the next slider
    drag re-renders a spectrum the controls no longer describe — plot, table and spurious count all
    labelled with the old system.

    Written because `sim-second-enabled` was the one such control wired to nothing. It reached the
    button's input list, so a fresh press was always right, and every other test passed: simulate
    with the box off, tick it, nudge a slider, and the tab still showed one system. Counting inputs
    cannot catch that — only asking which controls *trigger* an invalidation can.
    """
    import gradio as gr

    control = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == elem_id)
    state_ids = {b._id for b in demo.blocks.values() if isinstance(b, gr.State)}

    clearing = [
        d.api_name
        for d in demo.fns.values()
        if any(t[0] == control._id for t in getattr(d, "targets", []))
        and {c._id for c in d.outputs} & state_ids
    ]

    assert clearing, f"{elem_id} changes the spectrum but triggers no cache invalidation"


@pytest.mark.unit
@pytest.mark.parametrize("elem_id", ["sim-matrix", "sim-matrix-2"])
def test_the_spin_grid_labels_every_column_it_can_ever_grow_to(demo, elem_id):
    """`headers` and `datatype` are frozen at build time; the grid is not.

    No handler returns `gr.update`, so a matrix sized from its seed preset keeps those headers
    forever. Resize past the seed and the surplus columns fall back to positional indices: the
    second panel seeds from a 2-spin preset, so choosing `AA'BB'` rendered `spin | A | B | 4 | 5`
    instead of `A B C D` — visible in the shipped screenshot before this was fixed.

    Sizing both to `MAX_MATRIX_SPINS` costs nothing, because Gradio renders as many columns as the
    *data* has and only reads `headers` for their labels — a 2-spin grid still shows two columns.
    """
    import app as app_module

    grid = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == elem_id)
    expected = app_module.MAX_MATRIX_SPINS + 1  # the static label column, then one per spin

    assert len(grid.headers) == expected, f"{elem_id} cannot label a full-width grid"
    assert len(grid.datatype) == expected, f"{elem_id} would type a full-width grid as text"


@pytest.mark.unit
def test_elem_ids_are_unique(demo):
    """Two components sharing an id is valid Python and invalid HTML: the CSS rule then styles
    whichever one the DOM happens to hand over first, and the browser tests silently assert on it.
    """
    ids = [b.elem_id for b in demo.blocks.values() if getattr(b, "elem_id", None)]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate elem_ids: {sorted(duplicates)}"


@pytest.mark.unit
def test_every_event_is_addressable_over_the_api(demo):
    """`gradio_client` reaches an event only through its `api_name`; an unnamed event is invisible
    to the in-process e2e tier no matter how central it is to the UI.
    """
    unnamed = [d.fn.__name__ for d in demo.fns.values() if not d.api_name]
    assert not unnamed, f"events with no api_name (unreachable from gradio_client): {unnamed}"

    names = [str(d.api_name) for d in demo.fns.values()]
    assert len(set(names)) == len(names), f"duplicate api_names: {names}"

    # Left implicit, Gradio derives the endpoint id from the callback name — which published
    # `/_spec_report` and `/_spec_report_1`, i.e. a public API named after private helpers, with
    # the `_1` suffix decided by registration order.
    derived = [n for n in names if n.startswith("_")]
    assert not derived, f"api_names derived from private callbacks: {derived}"


@pytest.mark.unit
def test_editable_dataframes_hand_back_plain_lists(demo):
    """The Simulate grids must be `type="array"`, not Gradio's default `type="pandas"`.

    The handlers index rows positionally (`row[i + 1]`, since column 0 is the spin label). A pandas
    component satisfies that shape in every direct-call test — those pass a list of lists in — and
    then delivers a DataFrame over the wire, where iterating yields *column names*: the first row
    becomes the string "spin" and the parser reports a character as a bad cell. Only the
    `gradio_client` tier can see it, so this pins the setting where it is cheap to check.
    """
    import gradio as gr

    editable = [
        b
        for b in demo.blocks.values()
        if isinstance(b, gr.Dataframe) and getattr(b, "interactive", None) is not False
    ]
    assert editable, "expected the Simulate matrix and width tables"
    for block in editable:
        assert block.type == "array", f"{block.elem_id or block} is {block.type!r}"


@pytest.mark.unit
@pytest.mark.parametrize("suffix", ["", "_2"])
def test_the_matrix_edit_handler_clears_the_cache_and_rebuilds_the_widths(demo, suffix):
    """Assert the wiring, not just that a function exists.

    The first version of this guard called `invalidate_cache()` and checked it returned `None`,
    which cannot fail while the function is defined: deleting the `.change` wiring outright, and
    mis-wiring its output to the status box so the cache was never cleared, both left it green. What
    actually has to hold is that editing the matrix is wired to the state block *and* to the width
    table, so this reads the built graph.

    Parametrised over both spin-system panels. Resolving the handler by `api_name` means an
    unparametrised version keeps testing panel 1 forever: the second panel could be wired to nothing
    and this would stay green, which is the failure mode the whole test was written against.
    """
    import gradio as gr

    ids = f"sim-widths{suffix.replace('_', '-')}", f"sim-matrix{suffix.replace('_', '-')}"
    edit = next(d for d in demo.fns.values() if d.api_name == f"matrix_edited{suffix}")
    state_ids = {b._id for b in demo.blocks.values() if isinstance(b, gr.State)}
    widths = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == ids[0])
    matrix = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == ids[1])

    output_ids = {c._id for c in edit.outputs}
    assert output_ids & state_ids, "a matrix edit must clear the cached spectrum"
    assert widths._id in output_ids, "a matrix edit must re-derive the line-width table"
    assert {matrix._id, widths._id} <= {c._id for c in edit.inputs}

    # The width table clears the cache too, and must not be wired to rewrite itself.
    width_edit = next(
        d for d in demo.fns.values() if d.api_name == f"invalidate_on_width_edit{suffix}"
    )
    assert {c._id for c in width_edit.outputs} <= state_ids


def test_broadening_slider_does_not_claim_it_is_out_of_distribution(demo):
    """The broadening slider must not tell users the model never saw line broadening.

    It did, until 2026-07-27. That claim came from reading `augment_distortions` in its current
    state, where `toss_coin = 0.99` is hardcoded and the shim/broadening branches are dead — but
    that literal postdates the shipped weights. The checkpoint was last written 2024-10-14; the pin
    landed 2024-12-01, and at the weights' date the line still read `np.random.uniform(0, 1)`, so
    training applied broadening on roughly a third of its spectra.

    Guarding the copy rather than the wording: any future text that reasserts "trained without line
    broadening" is wrong for the shipped checkpoint, and nothing else in the suite would catch it.
    """
    sliders = [b for b in demo.blocks.values() if getattr(b, "label", None) == BROADEN_LABEL]
    assert len(sliders) == 1, f"expected exactly one {BROADEN_LABEL!r} slider, got {len(sliders)}"
    info = (getattr(sliders[0], "info", None) or "").lower()
    assert info, "the broadening slider should explain its relation to the training distribution"
    for phrase in ("without line broadening", "never saw", "outside the model's training"):
        assert phrase not in info, f"broadening slider still claims OOD via {phrase!r}: {info}"
