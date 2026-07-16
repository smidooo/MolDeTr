"""In-process end-to-end via ``gradio_client``: launch ``build_ui()`` and drive it over HTTP.

Exercises the real Gradio request/response plumbing (serialization, file upload, event dispatch)
that the direct-function tests bypass — still weight-free (the stubbed model) and browser-free.
The handlers carry no ``api_name``, so the Detect endpoint is found by its input count (6).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("gradio_client")
from gradio_client import Client, handle_file  # noqa: E402


def _wait_reachable(url: str, tries: int = 80) -> None:
    """Poll until the launched server accepts a connection (wait-for-condition, not a fixed sleep)."""
    import httpx

    for _ in range(tries):
        try:
            httpx.get(url, timeout=1.0)
            return
        except Exception:
            time.sleep(0.25)
    raise AssertionError(f"gradio server never became reachable at {url}")


def _detect_fn_index(client: Client) -> int:
    """The Detect handler (`predict_ui`) is the dependency with exactly 6 inputs."""
    for i, dep in enumerate(client.config["dependencies"]):
        if len(dep.get("inputs", [])) == 6:
            return i
    raise AssertionError("could not locate the 6-input Detect dependency")


@pytest.fixture
def live_app(patch_model):
    """Launch the patched app in-process; yield (app, url); close after."""
    app = patch_model
    demo = app.build_ui()
    _fastapi, url, _share = demo.launch(
        prevent_thread_lock=True, server_name="127.0.0.1", show_error=True, quiet=True
    )
    try:
        _wait_reachable(url)
        yield app, url
    finally:
        demo.close()


@pytest.mark.e2e
def test_app_serves_and_exposes_api(live_app):
    _app, url = live_app
    client = Client(url, verbose=False)
    assert client.config["dependencies"]  # endpoints discovered → the app is served


@pytest.mark.e2e
def test_detect_over_gradio_client(live_app, tmp_npz, valid_spectrum):
    app, url = live_app
    npz = tmp_npz(spectrum_padded=valid_spectrum, ppm_axis_padded=np.linspace(10.0, 0.0, 6144))
    client = Client(url, verbose=False)
    out = client.predict(
        handle_file(npz), 0.3, app.AUTO, None, None, 5.12, fn_index=_detect_fn_index(client)
    )
    # predict_ui returns (table, plot, status, csv_btn, json_btn)
    status = out[2]
    assert "Detected" in status and "multiplet(s)" in status
