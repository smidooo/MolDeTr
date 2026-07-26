"""Detect tab — `app.predict` / `app.predict_ui` scenario matrix (stubbed model, weight-free)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _valid_npz_with_ppm(tmp_npz, spec, left=10.0, right=0.0):
    return tmp_npz(spectrum_padded=spec, ppm_axis_padded=np.linspace(left, right, 6144))


def _download_path(btn) -> str | None:
    """Normalize a gr.DownloadButton value to a filesystem path (gradio may wrap it)."""
    v = btn.value
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("path")
    return getattr(v, "path", None)


# --- checkpoint gate + no-file (checkpoint-independent) -------------------------------------------


@pytest.mark.unit
def test_no_file_message(app_module):
    _t, _f, msg = app_module.predict(None, 0.3, app_module.AUTO, None, None, 5.12)
    assert msg == "Load a `.npz`/`.npy` spectrum, or pick an example below."


@pytest.mark.unit
def test_checkpoint_absent_message(app_module, tmp_npz, valid_spectrum, monkeypatch):
    monkeypatch.setattr(app_module, "CHECKPOINT", str(Path("does") / "not" / "exist.pth"))
    path = _valid_npz_with_ppm(tmp_npz, valid_spectrum)
    _t, _f, msg = app_module.predict(path, 0.3, app_module.AUTO, None, None, 5.12)
    assert "Checkpoint not found" in msg and "10.5281/zenodo.21217102" in msg


# --- validation errors (stub patched, but rejection happens before the model) --------------------


@pytest.mark.unit
def test_wrong_length_rejected(patch_model, tmp_npz):
    app = patch_model
    path = tmp_npz(spec=np.abs(np.random.RandomState(2).rand(5000)))
    table, fig, msg = app.predict(path, 0.3, app.AUTO, None, None, 5.12)
    assert table is None and fig is None
    assert msg.startswith("Invalid spectrum:") and "exactly 6144" in msg


@pytest.mark.unit
def test_nan_rejected(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    bad = valid_spectrum.copy()
    bad[5] = np.inf
    _t, _f, msg = app.predict(tmp_npz(spec=bad), 0.3, app.AUTO, None, None, 5.12)
    assert msg.startswith("Invalid spectrum:") and "NaN or Inf" in msg


@pytest.mark.unit
def test_complex_warns_but_detects(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    path = _valid_npz_with_ppm(tmp_npz, valid_spectrum.astype(np.complex64))
    table, _f, msg = app.predict(path, 0.3, app.AUTO, None, None, 5.12)
    assert "Detected **3** multiplet(s)" in msg
    assert "using its real part" in msg
    assert len(table) == 3


# --- ppm mode × bounds → shift-column header -----------------------------------------------------


@pytest.mark.unit
def test_auto_with_calibration_is_ppm(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    table, _f, _m = app.predict(
        _valid_npz_with_ppm(tmp_npz, valid_spectrum), 0.3, app.AUTO, None, None, 5.12
    )
    assert "δ (PPM)" in table.columns


@pytest.mark.unit
def test_auto_without_calibration_is_hz(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    table, _f, _m = app.predict(tmp_npz(spec=valid_spectrum), 0.3, app.AUTO, None, None, 5.12)
    assert "δ (HZ)" in table.columns


@pytest.mark.unit
def test_manual_both_bounds_is_ppm(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    table, _f, _m = app.predict(tmp_npz(spec=valid_spectrum), 0.3, app.MANUAL, 8.0, 2.0, 5.12)
    assert "δ (PPM)" in table.columns


@pytest.mark.unit
def test_manual_single_bound_falls_back_to_hz(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    table, _f, _m = app.predict(tmp_npz(spec=valid_spectrum), 0.3, app.MANUAL, 8.0, None, 5.12)
    assert "δ (HZ)" in table.columns  # MANUAL without both bounds → Hz


@pytest.mark.unit
def test_none_mode_is_hz(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    table, _f, _m = app.predict(
        _valid_npz_with_ppm(tmp_npz, valid_spectrum), 0.3, app.NONE, None, None, 5.12
    )
    assert "δ (HZ)" in table.columns  # NONE overrides the file calibration


# --- threshold -----------------------------------------------------------------------------------


@pytest.mark.unit
def test_threshold_one_detects_nothing(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    table, _f, msg = app.predict(
        _valid_npz_with_ppm(tmp_npz, valid_spectrum), 1.0, app.AUTO, None, None, 5.12
    )
    assert table.empty
    assert msg == "No multiplets passed the detection threshold — try lowering it."


# --- unreadable input: `predict` must degrade to a message, never a traceback --------------------
#
# `_spec_report` already wraps `_load` in try/except (app.py:106) and shows "⚠ Could not read the
# file". `predict` calls the same `_load` unguarded, so the identical bad file produces a rendered
# Python traceback in the GUI. These pin the parity.


@pytest.mark.unit
def test_corrupt_npz_returns_a_message_not_a_traceback(patch_model, tmp_path):
    app = patch_model
    bad = tmp_path / "corrupt.npz"
    bad.write_bytes(b"PK\x03\x04 truncated, not a real archive")

    table, fig, msg = app.predict(str(bad), 0.3, app.AUTO, None, None, 5.12)

    assert table is None and fig is None
    assert msg.startswith("⚠ Could not read the file:")


@pytest.mark.unit
def test_unsupported_extension_returns_a_message(patch_model, tmp_path):
    """`gr.File(file_types=...)` filters the browser picker, not the callback — the API path and a
    drag-drop that slips through both arrive here as an arbitrary file.
    """
    txt = tmp_path / "spectrum.txt"
    txt.write_text("1.0 2.0 3.0", encoding="utf-8")

    app = patch_model
    _t, _f, msg = app.predict(str(txt), 0.3, app.AUTO, None, None, 5.12)

    assert msg.startswith("⚠ Could not read the file:")


@pytest.mark.unit
def test_uploaded_npz_may_not_execute_a_pickled_payload(patch_model, tmp_path, valid_spectrum):
    """`np.load(..., allow_pickle=True)` on a *user-supplied* file is arbitrary code execution:
    unpickling runs `__reduce__` from the archive.

    The `metadata` branch that needs pickle is only reached when `ppm_axis_padded` is absent, and
    no bundled example takes that path — so refusing pickle for uploads costs nothing today and
    closes the hole. Trusted `examples/` files keep it (see the companion test below).
    """
    hostile = tmp_path / "uploaded.npz"
    np.savez(
        hostile,
        spectrum_padded=valid_spectrum,
        metadata=np.array({"left_ppm": 10.0, "right_ppm": 0.0}, dtype=object),
    )

    _t, _f, msg = patch_model.predict(str(hostile), 0.3, patch_model.AUTO, None, None, 5.12)

    assert msg.startswith("⚠ Could not read the file:")
    assert "pickle" in msg.lower() or "object array" in msg.lower()


@pytest.mark.unit
def test_bundled_examples_still_load(patch_model, example_paths):
    """The other half of the gate: shipping files stay readable, including the object-array keys."""
    for path in example_paths.values():
        _t, _f, msg = patch_model.predict(path, 0.3, patch_model.AUTO, None, None, 5.12)
        assert not msg.startswith("⚠ Could not read the file:"), f"{path} stopped loading: {msg}"


# --- points_per_hz: a non-positive resolution must be refused, not silently replaced --------------


@pytest.mark.unit
def test_zero_points_per_hz_is_refused_not_silently_defaulted(patch_model, tmp_npz, valid_spectrum):
    """`float(pph) if pph else POINTS_PER_HZ` treats 0 as "unset" and quietly substitutes 5.12.

    A user who clears the field and gets confident-looking results has been told nothing was wrong.
    """
    path = _valid_npz_with_ppm(tmp_npz, valid_spectrum)

    table, fig, msg = patch_model.predict(path, 0.3, patch_model.AUTO, None, None, 0)

    assert table is None and fig is None
    assert "points/Hz" in msg and "positive" in msg


@pytest.mark.unit
def test_negative_points_per_hz_is_refused(patch_model, tmp_npz, valid_spectrum):
    """A negative resolution is truthy, so it sails past the guard and yields a negative Hz window
    and a mirrored axis — wrong numbers rather than an error.
    """
    path = _valid_npz_with_ppm(tmp_npz, valid_spectrum)

    table, fig, msg = patch_model.predict(path, 0.3, patch_model.AUTO, None, None, -5.12)

    assert table is None and fig is None
    assert "points/Hz" in msg and "positive" in msg


# --- predict_ui downloads ------------------------------------------------------------------------


@pytest.mark.unit
def test_downloads_enabled_and_parse(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    path = _valid_npz_with_ppm(tmp_npz, valid_spectrum)
    table, _f, _m, csv_btn, json_btn = app.predict_ui(path, 0.3, app.AUTO, None, None, 5.12)
    assert bool(csv_btn.interactive) and bool(json_btn.interactive)
    csv_path = _download_path(csv_btn)
    assert csv_path and csv_path.endswith(".csv") and Path(csv_path).exists()
    back = pd.read_csv(csv_path)
    assert list(back.columns) == list(table.columns) and len(back) == len(table)


@pytest.mark.unit
def test_repeated_detections_reuse_one_export_directory(patch_model, tmp_npz, valid_spectrum):
    """`mkdtemp` *per click* leaked a directory on every detection — unbounded on a process that
    stays up for weeks, and invisible locally where you click twice and quit.
    """
    app = patch_model
    path = _valid_npz_with_ppm(tmp_npz, valid_spectrum)

    for _ in range(3):
        app.predict_ui(path, 0.3, app.AUTO, None, None, 5.12)

    assert app._export_dir() == app._export_dir()  # one dir, reused


@pytest.mark.unit
def test_download_links_are_content_addressed_so_the_shared_dir_is_safe(
    patch_model, make_fake_model, tmp_npz, valid_spectrum, monkeypatch
):
    """Pins the assumption that makes one reused export path safe.

    Reusing a fixed filename *looks* like a data-mixing bug — a later detection overwrites the
    file an earlier user is about to download. It is not, because ``gr.DownloadButton`` copies
    the file into Gradio's **content-addressed** cache when the event returns: the link a user
    holds is a hash of the bytes they were shown.

    That is an assumption about Gradio, not about this code, so it deserves a test rather than a
    comment. If a future Gradio served the source path directly, the shared directory would
    become a real bug — and this test is what would notice.
    """
    app = patch_model
    path = _valid_npz_with_ppm(tmp_npz, valid_spectrum)

    _t, _f, _m, csv_a, _j = app.predict_ui(path, 0.3, app.AUTO, None, None, 5.12)
    served_a = _download_path(csv_a)
    rows_a = Path(served_a).read_text(encoding="utf-8")

    # A second detection with *different* results overwrites the shared source file.
    monkeypatch.setattr(app, "_MODEL", make_fake_model([{"proton": 2, "center_frac": 0.3}]))
    _t, _f, _m, csv_b, _j = app.predict_ui(path, 0.3, app.AUTO, None, None, 5.12)
    served_b = _download_path(csv_b)

    assert served_a != served_b, "distinct results were served from the same path"
    assert Path(served_a).exists(), "the earlier download link stopped resolving"
    assert Path(served_a).read_text(encoding="utf-8") == rows_a, (
        "the earlier link now serves the later detection's numbers — the shared export directory "
        "is no longer safe and _export_dir() must go back to a unique path per detection"
    )


@pytest.mark.unit
def test_downloads_disabled_when_empty(patch_model, tmp_npz, valid_spectrum):
    app = patch_model
    path = _valid_npz_with_ppm(tmp_npz, valid_spectrum)
    table, _f, _m, csv_btn, json_btn = app.predict_ui(path, 1.0, app.AUTO, None, None, 5.12)
    assert table.empty
    assert not csv_btn.interactive and not json_btn.interactive
    assert _download_path(csv_btn) is None and _download_path(json_btn) is None
