"""The Simulate tab in a real browser: the matrix drives it, and sliders do not re-simulate.

Two claims this PR makes that no other tier can check.

**The matrix is the input.** Everything below the surface is exercised by direct calls and by
`gradio_client`, but only a browser shows whether the grid a user actually edits reaches the
handler — Gradio's Dataframe has its own client-side state, and a cell can look edited without the
value ever leaving the front end.

**A distortion slider re-distorts rather than re-simulating.** That is the whole point of splitting
the stages, and it is a claim about *what did not happen*. Counting `simulate_systems` calls on the
server while driving the real UI is the only way to see it end to end: the in-process tests count
calls without a browser, and the browser tests before this one could see the result change without
knowing what it cost.
"""

from __future__ import annotations

import time

import pytest

# Guard before the import, matching the other browser modules: pytest *collects* every file before
# deselecting by marker, so a bare `from playwright...` breaks collection — and therefore every
# lane — on the CI legs that do not install it.
pytest.importorskip("playwright")
from playwright.sync_api import Page, expect  # noqa: E402

pytestmark = pytest.mark.browser


@pytest.fixture
def simulate_page(page: Page, served_app_url: str) -> Page:
    """The Simulate tab, after one completed simulation, so the cache in `gr.State` is populated."""
    page.goto(served_app_url)
    page.get_by_role("tab", name="Simulate").click()
    page.locator("#sim-matrix").wait_for()
    page.get_by_role("button", name="Simulate & Predict").click()
    expect(page.get_by_text("ground-truth multiplet(s)", exact=False)).to_be_visible(timeout=30_000)
    return page


def test_the_matrix_and_width_tables_are_on_screen(simulate_page: Page) -> None:
    """The grid replaced a textbox and two number boxes; all four anchors must exist to be styled."""
    for elem_id in ("#sim-preset", "#sim-nspins", "#sim-matrix", "#sim-widths"):
        expect(simulate_page.locator(elem_id)).to_be_visible()


def test_moving_a_distortion_slider_does_not_re_simulate(
    simulate_page: Page, patch_model, monkeypatch
) -> None:
    """The claim, measured on the server while the real UI drives it.

    Counting rather than timing: a timing assertion would pass on a fast machine even if the
    eigendecomposition ran again, and flake on a loaded CI runner even when it did not.
    """
    app = patch_model
    counts = {"simulate": 0, "redistort": 0}
    real_simulate, real_detect = app.simulate_systems, app._detect_stage

    def counting_simulate(*a, **kw):
        counts["simulate"] += 1
        return real_simulate(*a, **kw)

    def counting_detect(*a, **kw):
        counts["redistort"] += 1
        return real_detect(*a, **kw)

    monkeypatch.setattr(app, "simulate_systems", counting_simulate)
    monkeypatch.setattr(app, "_detect_stage", counting_detect)

    phase = simulate_page.get_by_role("slider", name="Zeroth-order phase (deg; 0 = off)")
    expect(phase).to_be_visible()
    # A real pointer drag, not `fill()`. Gradio's `.release` listens for the pointer coming back up
    # on the range input, so setting `value` programmatically changes the number on screen and
    # triggers nothing — which is how the first version of this test passed while asserting that a
    # re-distort had *not* re-simulated, when in fact no re-distort had happened at all.
    box = phase.bounding_box()
    assert box is not None, "the phase slider has no layout box to drag"
    simulate_page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] / 2)
    simulate_page.mouse.down()
    simulate_page.mouse.move(box["x"] + box["width"] * 0.8, box["y"] + box["height"] / 2, steps=8)
    simulate_page.mouse.up()
    # Poll the server-side counter rather than a fixed sleep or a DOM guess: the assertion below is
    # only meaningful once the re-distort actually ran, and a fixed wait either flakes on a loaded
    # runner or wastes seconds on a fast one.
    deadline = time.monotonic() + 30.0
    while counts["redistort"] == 0 and time.monotonic() < deadline:
        simulate_page.wait_for_timeout(250)

    assert counts["redistort"] >= 1, (
        "moving the phase slider triggered no re-distort at all, so the zero simulation count "
        "below would prove nothing — check the `.release` wiring"
    )
    assert counts["simulate"] == 0, (
        "moving a distortion slider re-ran the spin dynamics; the cached spectrum in gr.State "
        "should have been re-distorted instead"
    )
