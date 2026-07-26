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

#: Unchanged across the matrix rebuild by coincidence, not by stasis: the shift textbox, the two
#: number boxes and their row went away, and a spin-count slider, two dataframes and a hint line
#: arrived. `gr.State` holds the simulation cache but is not itself a Block.
N_COMPONENTS = 56
#: load_example · _spec_report ×2 · predict_ui · preset_grid · resize_spin_matrix ·
#: simulate_to_state · redistort ×6 · invalidate_cache ×2. Gradio derives an endpoint id for every
#: re-distort handlers are named one per control — `api_name=False` becomes the literal "false",
#: "false_1", ..., which is the auto-derived surface these tests exist to prevent.
N_EVENTS = 15

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
    "sim-nspins",
    "sim-preset",
    "sim-widths",
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
