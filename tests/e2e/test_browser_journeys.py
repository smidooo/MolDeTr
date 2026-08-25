"""The user journeys, driven through the real DOM.

`test_browser_detect.py` proves the happy path renders. These cover the branches a user actually
takes and the direct-call tests cannot reach: every bundled example (including the one with no ppm
axis, which must fall back to a Hz axis), each ppm mode as selected by *clicking a radio* rather
than passing a string, the empty-result state, the download buttons' real file contents, the Plotly
zoom/reset interaction, and the full Simulate round trip.

Weight-free — the in-process app runs the stubbed 3-detection model via `patch_model`.
"""

from __future__ import annotations

import csv
import json

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import Page, expect  # noqa: E402

from tests.plot_wait import sliced_wait  # noqa: E402

pytestmark = pytest.mark.browser

DETECT = "Detect multiplets"


def _detect_with_example(page: Page, url: str, example: str) -> None:
    page.goto(url)
    page.locator("#md-file").wait_for()
    page.locator("#md-examples").get_by_role("button", name=example).click()
    expect(page.locator("#md-check")).to_contain_text("Input check")
    page.get_by_role("button", name=DETECT).click()


@pytest.mark.parametrize(
    "example,shift_header",
    [
        ("guajazulene", "δ (PPM)"),  # ppm_axis_padded present → ppm axis
        ("vanillin", "δ (PPM)"),
        ("synthetic", "δ (HZ)"),  # spec-only, no calibration → Hz fallback
    ],
)
def test_each_example_detects_with_the_right_shift_unit(
    page: Page, served_app_url: str, example: str, shift_header: str
) -> None:
    """The AUTO→Hz fallback is a real branch: `synthetic_example.npz` ships no ppm axis, so the
    table must switch units rather than silently label Hz values as ppm.
    """
    _detect_with_example(page, served_app_url, example)
    expect(page.get_by_text("Detected", exact=False)).to_be_visible(timeout=30_000)
    expect(page.locator("#md-table")).to_contain_text(shift_header)


def test_none_mode_overrides_a_file_calibration(page: Page, served_app_url: str) -> None:
    """Clicking "None (report in Hz)" must beat the ppm axis the file carries — asserted through
    the radio itself, since the direct-call test passes the mode as a string and never proves the
    control is wired to it.
    """
    page.goto(served_app_url)
    page.locator("#md-file").wait_for()
    page.locator("#md-examples").get_by_role("button", name="guajazulene").click()
    page.locator("#md-ppm").get_by_text("None (report in Hz)").click()
    page.get_by_role("button", name=DETECT).click()

    expect(page.get_by_text("Detected", exact=False)).to_be_visible(timeout=30_000)
    expect(page.locator("#md-table")).to_contain_text("δ (HZ)")


def test_manual_mode_reveals_its_bounds_and_reports_ppm(page: Page, served_app_url: str) -> None:
    page.goto(served_app_url)
    page.locator("#md-file").wait_for()
    page.locator("#md-examples").get_by_role("button", name="synthetic").click()
    page.locator("#md-ppm").get_by_text("Manual (window ppm)").click()

    left = page.get_by_label("Window left ppm (Manual only)")
    right = page.get_by_label("Window right ppm (Manual only)")
    expect(left).to_be_visible()
    expect(right).to_be_visible()
    left.fill("8")
    right.fill("2")
    page.get_by_role("button", name=DETECT).click()

    expect(page.get_by_text("Detected", exact=False)).to_be_visible(timeout=30_000)
    expect(page.locator("#md-table")).to_contain_text("δ (PPM)")  # manual bounds → ppm units


def test_threshold_at_maximum_reports_no_multiplets(page: Page, served_app_url: str) -> None:
    """The empty state is the one a confused user hits first; it must say what to do next."""
    page.goto(served_app_url)
    page.locator("#md-file").wait_for()
    page.locator("#md-examples").get_by_role("button", name="guajazulene").click()
    # Settle before touching the threshold, exactly as `_detect_with_example` above does. Selecting
    # an example populates the Detect tab's inputs asynchronously; without this wait the `fill`
    # below races that population, and when the example lands second it restores the threshold to
    # its default. Detection then finds multiplets, the empty state never renders, and the
    # assertion burns its full 30 s. This was the only test in the file that skipped the settle,
    # and it is why `browser e2e (firefox)` went red on an otherwise docs-only commit.
    expect(page.locator("#md-check")).to_contain_text("Input check")
    # "Detection threshold" labels BOTH a number box and a range slider (Gradio renders a paired
    # control), and the same label appears again on the Simulate tab — so scope to the Detect tab
    # and pick the number input explicitly rather than relying on a unique label.
    page.get_by_role("tabpanel").first.get_by_test_id("number-input").last.fill("1")
    page.get_by_role("button", name=DETECT).click()

    expect(page.get_by_text("No multiplets passed the detection threshold")).to_be_visible(
        timeout=30_000
    )


def test_download_buttons_deliver_the_detected_rows(page: Page, served_app_url: str) -> None:
    """Clicking the buttons and reading the bytes.

    The unit tests check that `predict_ui` wrote files; only this proves the browser can actually
    fetch them and that what arrives is the table the user is looking at.
    """
    _detect_with_example(page, served_app_url, "guajazulene")
    expect(page.get_by_text("Detected", exact=False)).to_be_visible(timeout=30_000)

    with page.expect_download() as csv_info:
        page.get_by_role("button", name="Download CSV").click()
    rows = list(csv.DictReader(csv_info.value.path().read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 3, f"expected the 3 stubbed detections, got {len(rows)}"
    assert "PROTONS" in rows[0] and any("δ" in k for k in rows[0])

    with page.expect_download() as json_info:
        page.get_by_role("button", name="Download JSON").click()
    payload = json.loads(json_info.value.path().read_text(encoding="utf-8"))
    assert [r["PROTONS"] for r in payload] == [r["PROTONS"] for r in rows], (
        "the CSV and JSON exports disagree about what was detected"
    )


def _expect_plotly_or_report(page: Page):
    """Wait for the Plotly canvas, sampling the DOM while it waits, and report honestly on failure.

    `browser e2e (webkit)` has failed around this wait four times, in two distinct classes:

    - **Plotly never initialised** — the #41 merge run (31247518046) and the #49 merge run
      (31266452657): the page lives out the full 30 s, `window.Plotly` never appears, and a
      `ResizeObserver loop completed with undelivered notifications` pageError fires ~1.9 s in.
      The diagnostic answers this truthfully after a timeout.
    - **The browser target dies** — the #78 merge run (31483547839) and the #92 merge run
      (31585679098): the webkit process dies seconds into the journey. Against a dead target the
      post-hoc diagnostic could only ever raise `Target crashed` itself, so this wrapper reported
      "the Plotly canvas never appeared within 30 s" about a page that had stopped existing at
      ~1.9 s.

    The mechanics live in `tests.plot_wait.sliced_wait` (unit-pinned there): short slices with a
    probe between them, target-death classified against five driver phrasings, everything else
    propagating unchanged, and the last playwright call log chained onto the timeout report.
    """
    plot = page.locator("#md-plot .js-plotly-plot")
    sliced_wait(
        lambda: plot_diagnostic_state(page),
        lambda slice_s: expect(plot).to_be_visible(timeout=slice_s),
    )
    return plot


_PLOT_DIAGNOSTIC_JS = """() => {
    const entries = performance.getEntriesByType('resource')
        .filter(e => /PlotlyPlot-[^/]*\\.js/.test(e.name));
    const first = entries[0];
    const chunk = {
        requested: entries.length,
        bytes: first ? first.encodedBodySize : null,
        ms: first ? Math.round(first.duration) : null,
        geoAssets: typeof window.PlotlyGeoAssets !== 'undefined',
    };
    chunk.status = entries.length === 0
        ? 'never-requested-or-in-flight'
        : (first.encodedBodySize > 0 ? 'loaded' : 'requested-but-empty');
    const c = document.querySelector('#md-plot');
    if (!c) return {container: 'ABSENT', chunk};
    const r = c.getBoundingClientRect();
    const cs = getComputedStyle(c);
    return {
        box: `${Math.round(r.width)}x${Math.round(r.height)}`,
        display: cs.display,
        visibility: cs.visibility,
        children: c.childElementCount,
        hasPlotlyDiv: !!c.querySelector('.js-plotly-plot'),
        plotlyLoaded: typeof window.Plotly !== 'undefined',
        chunk,
        html: c.innerHTML.slice(0, 300),
    };
}"""


def plot_diagnostic_state(page: Page) -> dict:
    """What `#md-plot` and the lazily-imported Plotly chunk actually are, right now.

    Shared by the failure path above and by the non-vacuity test below, so the thing asserted to
    discriminate is the same code that runs when a lane goes red.

    `plotlyLoaded` is kept exactly as #53 shipped it. It was checked rather than assumed on
    2026-08-10: on a healthy webkit run it reports `true`, so it does discriminate and issue #51's
    close condition that reads `plotlyLoaded: false` as a script/bundle problem is sound.

    `chunk` is the addition. gradio's Plot component lazy-imports `PlotlyPlot-*.js` with no
    `.catch()` and no retry, memoising only on success, and its failure branch renders the `Empty`
    placeholder -- which has a non-zero box and a non-zero child count. So `plotlyLoaded: false`
    says the library is absent but not *why*, and container geometry cannot separate *never
    mounted* from *mounted and empty*. Resource timing can: no entry means the chunk was never
    fetched, an entry with zero bytes means it was fetched and rejected, and bytes plus a duration
    means it arrived and the failure is downstream of the network. Measured on a healthy run:
    `{'requested': 1, 'bytes': 1275128, 'ms': 285, 'geoAssets': True, 'status': 'loaded'}`.
    """
    return page.evaluate(_PLOT_DIAGNOSTIC_JS)


def _assert_diagnostic_is_not_vacuous(page: Page) -> None:
    """The failure probe must be able to say "yes", or its "no" means nothing.

    Asserted inside the zoom test rather than as a test of its own, deliberately: it needs a drawn
    canvas, the zoom test has already produced one, and a separate test would add a third full app
    load to a tier that has now twice shown load sensitivity (#51 on webkit, and
    `test_container_max_width_comes_from_custom_css` on firefox). Same guarantee, no extra load.

    Why it exists at all: a diagnostic is only trustworthy once it has been *seen* reporting the
    healthy case. Reading a bundle and concluding a field cannot work is not the same as measuring
    it -- that exact reasoning was tried on 2026-08-10 and was **wrong**. A grep showed the gradio
    chunk assigning `window.PlotlyGeoAssets` and `window.PlotlyLocales` but never `window.Plotly`,
    which looked like proof that #53's `plotlyLoaded` was vacuous. Running it says otherwise:
    `window.Plotly` is defined once the chunk evaluates, so the field was right all along.

    Same non-vacuity discipline as `test_browser_a11y.py::test_button_name_settle_is_not_vacuous`,
    applied to the failure report instead of to a settle.
    """
    state = plot_diagnostic_state(page)
    assert state["hasPlotlyDiv"] is True, f"canvas visible but probe disagrees: {state}"
    assert state["plotlyLoaded"] is True, (
        f"the canvas drew, so `plotlyLoaded` must be true or the field is vacuous: {state}"
    )
    assert state["chunk"]["status"] == "loaded", (
        f"the canvas drew, so the chunk must read as loaded; probe said {state['chunk']}"
    )
    assert state["chunk"]["bytes"] > 0, f"a drawn canvas implies non-zero chunk bytes: {state}"


def test_spectrum_plot_zooms_and_resets(page: Page, served_app_url: str) -> None:
    """Drag-to-zoom is advertised in the caption, so it is part of the contract.

    Zoom state lives in Plotly's x-axis range; a double-click resets it to autorange. Reading the
    range rather than screenshotting keeps this robust across engines.
    """
    _detect_with_example(page, served_app_url, "guajazulene")
    expect(page.get_by_text("Detected", exact=False)).to_be_visible(timeout=30_000)
    plot = _expect_plotly_or_report(page)
    _assert_diagnostic_is_not_vacuous(page)

    box = plot.bounding_box()
    assert box, "the Plotly canvas has no layout box"
    y = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + box["width"] * 0.35, y)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.65, y, steps=12)
    page.mouse.up()

    zoomed = plot.evaluate("el => el.layout.xaxis.autorange")
    assert zoomed is False, "box-zoom did not pin the x-axis range (autorange still on)"

    plot.dblclick(position={"x": box["width"] / 2, "y": box["height"] / 2})
    page.wait_for_function(
        "() => document.querySelector('#md-plot .js-plotly-plot').layout.xaxis.autorange !== false",
        timeout=15_000,
    )


def test_simulate_tab_round_trip(page: Page, served_app_url: str) -> None:
    """The whole second tab, end to end: the matrix as loaded, simulate, detect, read the table.

    The Simulate handler carries two tabular payloads and is the widest surface in the app; this is
    the only test that drives it through real controls rather than a function call or an API
    payload.
    """
    page.goto(served_app_url)
    page.get_by_role("tab", name="Simulate").click()
    expect(page.locator("#sim-matrix")).to_be_visible()

    page.get_by_role("button", name="Simulate & Predict").click()

    # `.first`: the status renders the label in markdown bold, so "Simulated" lives in a <strong>
    # nested inside the paragraph and matches both elements.
    expect(page.get_by_text("Simulated", exact=False).first).to_be_visible(timeout=60_000)
    expect(page.get_by_text("ground-truth multiplet(s)", exact=False)).to_be_visible()
    # The label now reports the system the matrix describes rather than a phenotype name.
    expect(page.get_by_text("spin(s) in", exact=False).first).to_be_visible()
