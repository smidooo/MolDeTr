"""Selector liveness: every rule in ``CUSTOM_CSS`` must still match something in the served DOM.

The per-rule tests in ``test_browser_branding.py`` prove the stylesheet is *attached*. This proves
it still *applies*: Gradio renames internal classes and restructures wrappers between minor
versions, so a rule can go on being served while quietly matching nothing. Styling then degrades
one element at a time, with every existing test green.

Parsing ``CUSTOM_CSS`` rather than listing selectors by hand is the point — a rule added to the
stylesheet is covered the moment it is written, with no second place to remember to update.
"""

from __future__ import annotations


import pytest

pytest.importorskip("playwright")
from playwright.sync_api import Page, expect  # noqa: E402

from app_ui.theme import CUSTOM_CSS  # noqa: E402

from tests.css_at_rules import (  # noqa: E402
    parse_selectors as _parse_selectors,
    strip_pseudo as _strip_pseudo,
)

pytestmark = pytest.mark.browser

# Selectors that legitimately match nothing in the page state below, with the reason each is
# unreachable. Anything not listed here MUST match at least one element.
KNOWN_UNMATCHABLE = {
    # Pure interaction states: no element carries them at rest, and hovering one element would
    # not make the *rule* verifiable for the others.
    "#md-examples button:hover",
}


def _match_counts(page: Page, selectors: list[str]) -> dict[str, int]:
    """How many elements each selector matches, evaluated in one round trip."""
    return page.evaluate(
        """(sels) => Object.fromEntries(
            sels.map(s => {
                try { return [s, document.querySelectorAll(s).length]; }
                catch (e) { return [s, -1]; }   // -1 = the browser rejected the selector
            })
        )""",
        selectors,
    )


@pytest.fixture
def populated_page(page: Page, served_app_url: str) -> Page:
    """The app after a detection, so table rules have something to match.

    `#md-table table` and `#md-table thead th` only exist once the Dataframe has rows, so a
    liveness check on the landing page would have to exclude them — and would then never notice
    if they died. Running one detection first is what makes them checkable.
    """
    page.goto(served_app_url)
    page.locator("#md-file").wait_for()
    page.locator("#md-examples").get_by_role("button", name="guajazulene").click()
    # Wait for the file to actually land before pressing Detect. `load_example` sets the File
    # component and `spectrum.change` then repopulates the input-check panel; clicking Detect
    # before that propagates makes `predict` receive None and answer "Load a .npz/.npy spectrum",
    # so the wait below times out. Chromium usually wins that race and Firefox/WebKit do not —
    # which is how it stayed invisible until the cross-browser matrix.
    expect(page.locator("#md-check")).to_contain_text("Input check")
    page.get_by_role("button", name="Detect multiplets").click()
    # Wait on the status text, not on a row: Gradio's Dataframe keeps hidden <tr> elements in the
    # DOM before any data arrives, so a visibility wait on `tbody tr` times out against rows that
    # exist but are not shown. The status line is the unambiguous "results are in" signal.
    expect(page.get_by_text("Detected", exact=False)).to_be_visible(timeout=30_000)
    page.wait_for_selector("#md-table table", state="attached", timeout=30_000)
    return page


def test_every_custom_css_selector_still_matches_something(populated_page: Page) -> None:
    """The regression this exists for: a Gradio upgrade renames a wrapper, one rule stops
    applying, and nothing else in the suite notices because the stylesheet is still attached.
    """
    selectors = _parse_selectors(CUSTOM_CSS)
    assert selectors, "parsed no selectors out of CUSTOM_CSS — the parser is broken, not the CSS"

    to_check = [s for s in selectors if s not in KNOWN_UNMATCHABLE]
    counts = _match_counts(populated_page, [_strip_pseudo(s) for s in to_check])

    dead = [orig for orig in to_check if counts.get(_strip_pseudo(orig), 0) < 1]
    assert not dead, (
        "CSS rules that no longer match anything in the served DOM:\n  "
        + "\n  ".join(dead)
        + "\n(If a rule is intentionally state-only, add it to KNOWN_UNMATCHABLE with a reason.)"
    )


def test_the_liveness_check_would_notice_a_dead_selector(populated_page: Page) -> None:
    """Proves the check above has teeth.

    A meta-test that only ever runs against a healthy page cannot distinguish "every selector
    matches" from "my matching logic returns a truthy value for everything". Feeding it a selector
    that is valid CSS but present in no MolDeTr build must come back as dead.
    """
    counts = _match_counts(populated_page, ["#md-file", "#md-this-element-does-not-exist"])

    assert counts["#md-file"] >= 1, "the control selector should match — the page state is wrong"
    assert counts["#md-this-element-does-not-exist"] == 0


def test_known_unmatchable_entries_are_still_unmatchable(populated_page: Page) -> None:
    """Keeps the exclusion list honest.

    An exclusion that has quietly become matchable is a rule losing its only coverage. If this
    fails, delete the entry from KNOWN_UNMATCHABLE — the selector can be checked for real now.
    """
    still_excluded = _match_counts(populated_page, [_strip_pseudo(s) for s in KNOWN_UNMATCHABLE])
    # The *stripped* form is expected to match (that is why stripping is worth doing); what must
    # stay unmatchable is the full selector including its pseudo-class.
    full = _match_counts(populated_page, sorted(KNOWN_UNMATCHABLE))
    assert all(c >= 1 for c in still_excluded.values()), (
        f"the non-pseudo part of an excluded rule went dead: {still_excluded}"
    )
    assert all(c == 0 for c in full.values()), f"an exclusion is now matchable, drop it: {full}"


def test_no_selector_is_syntactically_invalid(populated_page: Page) -> None:
    """`querySelectorAll` throws on a malformed selector, which the browser otherwise swallows
    when parsing the stylesheet — the rule is simply dropped and every visual test stays green.
    """
    selectors = [_strip_pseudo(s) for s in _parse_selectors(CUSTOM_CSS)]
    counts = _match_counts(populated_page, selectors)
    invalid = [s for s, c in counts.items() if c == -1]
    assert not invalid, f"malformed selectors (browser rejected them): {invalid}"
