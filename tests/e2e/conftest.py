"""Shared fixtures for the Playwright (``-m browser``) tier, plus a Windows event-loop repair.

``nbmake``'s pytest plugin calls ``asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())``
from ``pytest_addoption`` (nbmake/pytest_plugin.py), so it runs on *every* pytest invocation on
Windows — not just notebook runs. Selector loops cannot spawn subprocesses, so Playwright's driver
dies with a bare ``NotImplementedError`` during fixture setup and all browser tests error out.

Restoring the Proactor policy for this directory fixes the browser tier locally. Linux CI is
unaffected either way (its selector loops support subprocesses), so this is guarded to win32.
"""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

if sys.platform == "win32":  # pragma: no cover - platform-specific
    _proactor = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
    if _proactor is not None and not isinstance(asyncio.get_event_loop_policy(), _proactor):
        asyncio.set_event_loop_policy(_proactor())


def wait_reachable(url: str, tries: int = 80) -> None:
    """Block until the in-process gradio server answers, or fail loudly."""
    import httpx

    for _ in range(tries):
        try:
            httpx.get(url, timeout=1.0)
            return
        except Exception:  # noqa: BLE001 - server not up yet
            time.sleep(0.25)
    raise AssertionError(f"gradio server never became reachable at {url}")


@pytest.fixture
def served_app_url(patch_model):
    """Serve the stubbed app exactly as production does; yield its URL; close after.

    Routes through ``app.launch_app`` so the theme and CSS that ship to users are the ones this
    tier asserts against. Launching bare would silently test an unstyled app — see
    ``test_browser_branding.py``.
    """
    app = patch_model
    demo, (_f, url, _s) = app.launch_app(
        prevent_thread_lock=True, server_name="127.0.0.1", show_error=True, quiet=True
    )
    try:
        wait_reachable(url)
        yield url
    finally:
        demo.close()
