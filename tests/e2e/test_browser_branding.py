"""Branding reaches the served DOM — the layer that catches a silently-unstyled app.

Gradio 6 moved ``theme=``/``css=`` off ``Blocks(...)`` onto ``.launch()``. Nothing raises if they
are omitted: the app serves, every callback works, every other browser test passes — and the users
get default Gradio styling. Line coverage cannot see it either (``app_ui/theme.py`` reports 100 %
because its constants are all module-level assignments, whether or not anything applies them).

So these assertions target computed styles that **only** ``CUSTOM_CSS`` can produce — never inline
styles from ``HEADER_HTML``, and never a Gradio default.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import Page, expect  # noqa: E402

pytestmark = pytest.mark.browser


UNSTYLED = "The app is being served WITHOUT its stylesheet."

# Why these use `expect(...).to_have_css` instead of reading the computed style once:
# `wait_for_selector` resolves when the ELEMENT exists, which is not when the STYLESHEET applies.
# WebKit needs ~300 ms more (measured: `.gradio-container` max-width is `none` at t+0.0 s and
# `1320px` at t+0.3 s); Chromium and Firefox happened to be ready on the first read, so a one-shot
# assertion passed there by luck. `expect` polls until the timeout, which is what makes this a
# branding check rather than a race.


def test_dropzone_border_colour_comes_from_custom_css(page: Page, served_app_url: str) -> None:
    """`#md-file` gets border-color #c6d2e1 from CUSTOM_CSS.

    Note: ``border-style: dashed`` is NOT a valid discriminator — Gradio's default dropzone is
    already dashed, so that assertion passes even on a completely unstyled app. The *colour* is
    ours alone.
    """
    page.goto(served_app_url)
    expect(page.locator("#md-file"), UNSTYLED).to_have_css("border-top-color", "rgb(198, 210, 225)")


def test_footer_is_hidden_by_custom_css(page: Page, served_app_url: str) -> None:
    """CUSTOM_CSS hides Gradio's footer; unstyled, it renders and is visible."""
    page.goto(served_app_url)
    page.wait_for_selector("#md-header")
    # Every footer must be hidden, not merely the first, so poll the whole set.
    page.wait_for_function(
        """() => {
            const f = [...document.querySelectorAll('footer')];
            return f.length > 0 && f.every(e => getComputedStyle(e).display === 'none');
        }""",
        timeout=15_000,
    )


def test_container_max_width_comes_from_custom_css(page: Page, served_app_url: str) -> None:
    """CUSTOM_CSS pins the container to 1320px; the Gradio default is not this value."""
    page.goto(served_app_url)
    expect(page.locator(".gradio-container"), UNSTYLED).to_have_css("max-width", "1320px")


# ---- the font must not be able to take the stylesheet down with it ----------
#
# CI run 31390397645 failed the assertion directly above with `max-width: none`, on a branch whose
# diff could not touch styling. Its Playwright trace shows why: `CUSTOM_CSS` opened with
# `@import url('https://fonts.googleapis.com/...')`, that request was issued at t=32556.8 ms and
# was still unresolved 4.8 s later (status -1), and `.gradio-container` stayed `none` throughout.
#
# A three-condition probe in Firefox separated the mechanism from the correlation:
#
#     control (no routing) -> 1320px at t=2.30 s
#     abort   (fails fast) -> 1320px at t=2.44 s
#     hang    (pending)    -> 'none' for the full 6 s window
#
# So a *failed* font fetch is harmless and only a *pending* one withholds the sheet — which is
# exactly the case a black-holing proxy produces for a real user, not just for CI.

FONT_CDN = "https://fonts.googleapis.com/**"
# The face the embedded woff2 provides. `600` is the weight CUSTOM_CSS asks for most.
EMBEDDED_FACE = "600 12px 'Space Grotesk'"


def test_serving_the_app_makes_no_third_party_requests(page: Page, served_app_url: str) -> None:
    """Every request the app makes must be same-origin.

    Deliberately an allowlist on origin, not a denylist of font hosts. A denylist naming
    `fonts.googleapis.com` would stay green for `fonts.bunny.net`, `use.typekit.net` or
    `cdn.jsdelivr.net/npm/@fontsource/…`, each of which reproduces the identical bug — the guard
    would go blind exactly when the defect changed hosts.
    """
    origin = re.match(r"^https?://[^/]+", served_app_url).group(0)
    foreign: list[str] = []

    def _record(request) -> None:  # noqa: ANN001 - playwright Request, imported lazily above
        if request.url.startswith(("http://", "https://")) and not request.url.startswith(origin):
            foreign.append(request.url)

    page.on("request", _record)

    page.goto(served_app_url)
    expect(page.locator(".gradio-container"), UNSTYLED).to_have_css("max-width", "1320px")

    assert foreign == [], f"app made third-party requests (expected only {origin}): {foreign}"


def test_the_embedded_webfont_actually_loads(page: Page, served_app_url: str) -> None:
    """The positive half: the two guards around this one are both absences.

    Neither "no third-party request" nor "layout survives a hang" can distinguish a working
    embedded font from a missing one — a truncated blob or a bad encode would satisfy both while
    the brand face silently fell back to `sans-serif`. The `@import` had the same blind spot; this
    change moves the font onto a new delivery path, so the positive check comes with it.
    """
    page.goto(served_app_url)
    expect(page.locator(".gradio-container"), UNSTYLED).to_have_css("max-width", "1320px")

    loaded = page.evaluate(
        "face => document.fonts.load(face).then(faces => faces.length)", EMBEDDED_FACE
    )
    assert loaded > 0, f"no @font-face matched {EMBEDDED_FACE} — the embedded woff2 did not load"
    assert page.evaluate("face => document.fonts.check(face)", EMBEDDED_FACE), (
        f"{EMBEDDED_FACE} is not usable after loading — the embedded payload is likely corrupt"
    )


def test_custom_css_applies_even_when_the_font_cdn_hangs(page: Page, served_app_url: str) -> None:
    """The layout CSS must survive a font request that never resolves.

    This is the regression guard for run 31390397645. Turns red if the webfont is ever reintroduced
    somewhere that blocks `CUSTOM_CSS` — an `@import` at the top of the sheet, or a bare
    `<link rel=stylesheet>` in `<head>` (itself render-blocking, so it would be a worse fix, not a
    better one). `route` with no `fulfill`/`abort` leaves the request pending, which the probe above
    showed is the only condition that reproduces the failure.

    `wait_until="commit"` is load-bearing. A pending subresource also holds back the `load` event, so
    the default `goto` would spend 30 s timing out on navigation and report that instead — a real
    failure, but not this one. Committing early keeps the assertion pointed at the stylesheet, which
    is what CI actually saw: there, `goto` returned at t=704 ms and the font request only began at
    t=32556.8 ms, well after the page was up.

    The explicit 30 s budget goes with it. Committing early means these seconds must cover server
    render, Gradio's JS boot *and* stylesheet application, where every other assertion in this
    module gets its 5 s after `load`. On the default budget a slow WebKit leg would go red with
    `UNSTYLED` — the precise false signature this change exists to remove.
    """
    page.route(FONT_CDN, lambda route: None)

    page.goto(served_app_url, wait_until="commit")
    expect(page.locator(".gradio-container"), UNSTYLED).to_have_css(
        "max-width", "1320px", timeout=30_000
    )
