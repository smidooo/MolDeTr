"""Playwright browser e2e — the literal frontend user journey against the real DOM.

Opt-in (``-m browser``; needs ``pytest-playwright`` + ``playwright install chromium``). Weight-free:
the in-process app runs the stubbed model, so Detect returns real detections without the checkpoint.
Inherits the ``patch_model`` / data fixtures from ``tests/conftest.py``.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import Page, expect  # noqa: E402

pytestmark = pytest.mark.browser


def _wait_reachable(url: str, tries: int = 80) -> None:
    import httpx

    for _ in range(tries):
        try:
            httpx.get(url, timeout=1.0)
            return
        except Exception:
            time.sleep(0.25)
    raise AssertionError(f"gradio server never became reachable at {url}")


@pytest.fixture
def served_app_url(patch_model):
    """Launch the stubbed app in-process; yield its URL; close after."""
    app = patch_model
    demo = app.build_ui()
    _f, url, _s = demo.launch(
        prevent_thread_lock=True, server_name="127.0.0.1", show_error=True, quiet=True
    )
    try:
        _wait_reachable(url)
        yield url
    finally:
        demo.close()


def test_header_and_tab_strip_render(page: Page, served_app_url):
    page.goto(served_app_url)
    expect(page.locator("#md-header")).to_be_visible()
    expect(page.get_by_role("tab", name="Detect")).to_be_visible()
    expect(page.get_by_role("tab", name="Simulate")).to_be_visible()


def test_example_to_detection_journey(page: Page, served_app_url):
    page.goto(served_app_url)
    page.locator("#md-file").wait_for()
    # pick the guajazulene example → the input-check panel repopulates
    page.locator("#md-examples").get_by_role("button", name="guajazulene").click()
    expect(page.locator("#md-check")).to_contain_text("Input check")
    # run detection (stubbed model → 3 multiplets) and read the status + table
    page.get_by_role("button", name="Detect multiplets").click()
    expect(page.get_by_text("Detected", exact=False)).to_be_visible(timeout=30_000)
    expect(page.locator("#md-table")).to_be_visible()


def test_tab_switch_to_simulate(page: Page, served_app_url):
    page.goto(served_app_url)
    page.get_by_role("tab", name="Simulate").click()
    # the Simulate phenotype control is present once the tab is active
    expect(page.get_by_label("Phenotype")).to_be_visible()
