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
from playwright.sync_api import Page  # noqa: E402

pytestmark = pytest.mark.browser


def test_dropzone_border_colour_comes_from_custom_css(page: Page, served_app_url: str) -> None:
    """`#md-file` gets border-color #c6d2e1 from CUSTOM_CSS:119.

    Note: ``border-style: dashed`` is NOT a valid discriminator — Gradio's default dropzone is
    already dashed, so that assertion passes even on a completely unstyled app. The *colour* is
    ours alone.
    """
    page.goto(served_app_url)
    page.wait_for_selector("#md-file")
    colour = page.eval_on_selector("#md-file", "el => getComputedStyle(el).borderTopColor")
    assert colour == "rgb(198, 210, 225)", (
        f"#md-file border-color is {colour!r}, expected 'rgb(198, 210, 225)' (#c6d2e1) from "
        "CUSTOM_CSS. The app is being served WITHOUT its stylesheet."
    )


def test_footer_is_hidden_by_custom_css(page: Page, served_app_url: str) -> None:
    """CUSTOM_CSS:102 hides Gradio's footer; unstyled, it renders and is visible."""
    page.goto(served_app_url)
    page.wait_for_selector("#md-header")
    display = page.eval_on_selector_all(
        "footer", "els => els.map(e => getComputedStyle(e).display)"
    )
    assert display and all(d == "none" for d in display), (
        f"footer display values are {display!r}, expected every one to be 'none' from CUSTOM_CSS. "
        "The app is being served WITHOUT its stylesheet."
    )


def test_container_max_width_comes_from_custom_css(page: Page, served_app_url: str) -> None:
    """CUSTOM_CSS:101 pins the container to 1320px; the Gradio default is not this value."""
    page.goto(served_app_url)
    page.wait_for_selector(".gradio-container")
    max_width = page.eval_on_selector(".gradio-container", "el => getComputedStyle(el).maxWidth")
    assert max_width == "1320px", (
        f".gradio-container max-width is {max_width!r}, expected '1320px' from CUSTOM_CSS. "
        "The app is being served WITHOUT its stylesheet."
    )
