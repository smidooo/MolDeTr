"""In-process end-to-end via ``gradio_client``: launch ``build_ui()`` and drive it over HTTP.

Exercises the real Gradio request/response plumbing (serialization, file upload, event dispatch)
that the direct-function tests bypass — still weight-free (the stubbed model) and browser-free.

Endpoints are addressed by ``api_name``. The previous version located Detect as "the dependency
with exactly 6 inputs", which would have silently retargeted the moment any other handler grew to
six inputs — and pointed at whichever one Gradio happened to register first.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("gradio_client")
from gradio_client import Client, handle_file  # noqa: E402

# The public endpoint names wired in `build_ui()`. Named here so a rename breaks the test that
# documents the API surface, rather than only the test that happens to call it.
DETECT = "/detect"
SIMULATE = "/simulate_and_detect"
PHENOTYPE_DEFAULTS = "/phenotype_defaults"
CHECK_ON_UPLOAD = "/check_input_on_upload"
CHECK_ON_RESOLUTION = "/check_input_on_resolution_change"


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


@pytest.fixture
def live_app(patch_model):
    """Launch the patched app in-process; yield (app, url); close after."""
    app = patch_model
    demo, (_fastapi, url, _share) = app.launch_app(
        prevent_thread_lock=True, server_name="127.0.0.1", show_error=True, quiet=True
    )
    try:
        _wait_reachable(url)
        yield app, url
    finally:
        demo.close()


@pytest.mark.e2e
def test_app_serves_and_exposes_every_named_endpoint(live_app):
    """All five handlers must be reachable by name — an unnamed one is invisible to any API client."""
    _app, url = live_app
    client = Client(url, verbose=False)

    named = {d.get("api_name") for d in client.config["dependencies"]}
    expected = {
        n.lstrip("/")
        for n in (DETECT, SIMULATE, PHENOTYPE_DEFAULTS, CHECK_ON_UPLOAD, CHECK_ON_RESOLUTION)
    }
    assert expected <= named, f"missing endpoints: {sorted(expected - named)}"


@pytest.mark.e2e
def test_detect_over_gradio_client(live_app, tmp_npz, valid_spectrum):
    app, url = live_app
    npz = tmp_npz(spectrum_padded=valid_spectrum, ppm_axis_padded=np.linspace(10.0, 0.0, 6144))
    client = Client(url, verbose=False)

    out = client.predict(handle_file(npz), 0.3, app.AUTO, None, None, 5.12, api_name=DETECT)

    # predict_ui returns (table, plot, status, csv_btn, json_btn)
    status = out[2]
    assert "Detected" in status and "multiplet(s)" in status


@pytest.mark.e2e
def test_simulate_round_trip_over_gradio_client(live_app):
    """The Simulate handler now carries two tabular payloads, which serialisation can mangle.

    Direct-call tests hand the grid over as a plain list of lists; over the wire it becomes a
    dataframe payload and comes back as one. That round trip is exactly where a column offset — the
    label column that shifts every value by one — would first show up, and it is invisible to every
    other tier.
    """
    _app, url = live_app
    client = Client(url, verbose=False)
    grid, widths = _app._phenotype_grid("ethyl")
    # Over the wire a dataframe is a `{headers, data}` payload, not a bare list of lists. This is
    # also what catches a component left on Gradio's default `type="pandas"`: the direct-call tests
    # hand the handler a list of lists either way, but a pandas component delivers a DataFrame here,
    # and iterating one yields *column names* — the first "row" becomes the string "spin".
    matrix_payload = {"headers": ["spin", *(r[0] for r in grid)], "data": grid}
    width_payload = {"headers": ["spin system", "n H", "FWHM (Hz)"], "data": widths}

    out = client.predict(
        matrix_payload, width_payload, False, 3.0, 0.0, 0.0, 0.0, 0.3, api_name=SIMULATE
    )

    status = out[-1]
    assert "2 ground-truth multiplet(s)" in status and "detected" in status


@pytest.mark.e2e
def test_preset_fills_the_matrix_over_the_wire(live_app):
    """The preset dropdown returns a grid, a width table and a spin count — not form fields."""
    _app, url = live_app
    client = Client(url, verbose=False)

    grid, widths, n_spins = client.predict("aromatic_ax", api_name=PHENOTYPE_DEFAULTS)

    rows = grid["data"] if isinstance(grid, dict) else grid
    width_rows = widths["data"] if isinstance(widths, dict) else widths
    assert n_spins == 2
    assert [row[1] for row in rows] == [7.5, 0.0]  # δ on the diagonal, column offset by the label
    assert rows[0][2] == 8.0  # the AX coupling, above the diagonal
    assert len(width_rows) == 1  # one coupled system, so one line-width row


@pytest.mark.e2e
def test_input_check_reports_over_the_wire(live_app, tmp_npz, valid_spectrum):
    """The upload `.change` handler — the panel a user reads before pressing Detect."""
    _app, url = live_app
    npz = tmp_npz(spectrum_padded=valid_spectrum, ppm_axis_padded=np.linspace(10.0, 0.0, 6144))
    client = Client(url, verbose=False)

    report = client.predict(handle_file(npz), 5.12, api_name=CHECK_ON_UPLOAD)

    assert "**Input check**" in report
    assert "Length: **6144** points ✓" in report
    assert "ppm axis in file: yes ✓" in report
