"""Branding reaches the served DOM — the layer that catches a silently-unstyled app.

Gradio 6 moved ``theme=``/``css=`` off ``Blocks(...)`` onto ``.launch()``. Nothing raises if they
are omitted: the app serves, every callback works, every other browser test passes — and the users
get default Gradio styling. Line coverage cannot see it either (``app_ui/theme.py`` reports 100 %
because its constants are all module-level assignments, whether or not anything applies them).

So these assertions target computed styles that **only** ``CUSTOM_CSS`` can produce — never inline
styles from ``HEADER_HTML``, and never a Gradio default.
"""

from __future__ import annotations

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
