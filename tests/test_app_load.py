"""`app._load` — file parsing + ppm calibration resolution (checkpoint-independent)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _obj0d(value):
    """A 0-d object array holding one Python value (how the export stores `metadata`)."""
    a = np.empty((), dtype=object)
    a[()] = value
    return a


@pytest.mark.unit
def test_npz_prefers_ppm_axis_padded(app_module, tmp_npz, valid_spectrum):
    axis = np.linspace(9.0, 1.0, 6144)
    path = tmp_npz(spectrum_padded=valid_spectrum, ppm_axis_padded=axis)
    arr, cal = app_module._load(path)
    assert arr.shape == (6144,)
    assert cal["ppm_left"] == pytest.approx(9.0)
    assert cal["ppm_right"] == pytest.approx(1.0)


@pytest.mark.unit
def test_npz_falls_back_to_metadata_ppm(app_module, tmp_npz, valid_spectrum):
    """`metadata` is a pickled object array, so this branch is now reachable only for a trusted
    file. The fallback itself is unchanged — `trusted=True` is what the `examples/` path supplies.
    """
    path = tmp_npz(spec=valid_spectrum, metadata=_obj0d({"left_ppm": 8.0, "right_ppm": 2.0}))
    arr, cal = app_module._load(path, trusted=True)
    assert arr.shape == (6144,)
    assert (cal["ppm_left"], cal["ppm_right"]) == (8.0, 2.0)


@pytest.mark.unit
def test_untrusted_npz_refuses_to_unpickle_metadata(app_module, tmp_npz, valid_spectrum):
    """The trust boundary, stated as a test: byte-identical file, opposite outcome.

    Unpickling runs `__reduce__` from the archive, so provenance — not content — decides. An
    upload takes the default (`trusted=False`) and is refused before any object array is touched.
    """
    path = tmp_npz(spec=valid_spectrum, metadata=_obj0d({"left_ppm": 8.0, "right_ppm": 2.0}))
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        app_module._load(path)


@pytest.mark.unit
def test_only_paths_inside_examples_are_trusted(app_module, tmp_path, example_paths):
    """`_is_bundled_example` is the whole gate, so pin both of its answers."""
    assert app_module._is_bundled_example(Path(example_paths["roi_S8"]))
    assert not app_module._is_bundled_example(tmp_path / "uploaded.npz")
    # A path that merely *mentions* examples/ elsewhere must not qualify.
    assert not app_module._is_bundled_example(tmp_path / "examples" / "spoof.npz")


@pytest.mark.unit
def test_trust_gate_does_not_depend_on_the_working_directory(app_module, monkeypatch, tmp_path):
    """`build_ui()` wires examples as the RELATIVE path "examples/roi_S8_example.npz", so
    resolving against the process CWD makes the gate depend on where the app was launched from.

    Launching as `python /abs/path/to/MolDeTr/app.py` from elsewhere silently marks every bundled
    example untrusted. That is invisible today only because no bundled example needs pickle — which
    is exactly why it would surface as a mystery the day one does.
    """
    relative = Path("examples/roi_S8_example.npz")
    monkeypatch.chdir(tmp_path)
    assert app_module._is_bundled_example(relative), (
        "a bundled example stopped being trusted purely because the CWD changed"
    )


@pytest.mark.unit
def test_a_relative_path_outside_examples_is_still_untrusted(app_module, monkeypatch, tmp_path):
    """Resolving relative paths against the repo root must not widen the gate."""
    monkeypatch.chdir(tmp_path)
    assert not app_module._is_bundled_example(Path("uploaded.npz"))
    assert not app_module._is_bundled_example(Path("../examples/roi_S8_example.npz"))


@pytest.mark.unit
def test_npz_spec_only_has_no_calibration(app_module, tmp_npz, valid_spectrum):
    _arr, cal = app_module._load(tmp_npz(spec=valid_spectrum))
    assert cal == {}


@pytest.mark.unit
def test_npz_first_key_fallback(app_module, tmp_npz, valid_spectrum):
    # No spectrum_padded / spec key → the first array in the archive is taken as the spectrum.
    arr, cal = app_module._load(tmp_npz(some_weird_key=valid_spectrum))
    assert arr.shape == (6144,) and cal == {}


@pytest.mark.unit
def test_npy_load_no_calibration(app_module, tmp_npy, valid_spectrum):
    arr, cal = app_module._load(tmp_npy(valid_spectrum))
    assert arr.shape == (6144,) and cal == {}


@pytest.mark.unit
def test_real_example_roi_has_ppm_axis(app_module, example_paths):
    arr, cal = app_module._load(example_paths["roi_S10"])
    assert arr.shape[0] == 6144
    assert cal.get("ppm_left") is not None and cal.get("ppm_right") is not None


@pytest.mark.unit
def test_real_synthetic_example_is_complex_without_ppm(app_module, example_paths):
    arr, cal = app_module._load(example_paths["synthetic"])
    assert np.iscomplexobj(arr)  # synthetic_example.npz ships complex64
    assert cal == {}  # spec-only → no calibration → Auto falls back to Hz
