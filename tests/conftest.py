"""Shared fixtures for the MolDeTr app / e2e / scripts test suite.

The 974 MB checkpoint is never available in CI, so the Detect / Simulate paths are exercised with a
deterministic ``fake_model`` patched into ``app._MODEL`` (plus ``app.CHECKPOINT`` pointed at an
existing dummy file so the checkpoint gate passes). The fake returns the exact block the real model
would: a ``(1, n_groups*num_queries, num_classes+num_params) = (1, 80, 12)`` tensor. See the decode
contract in ``moldetr.inference.run`` + ``moldetr.postprocess.decode_predictions``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
EXTREMA_PATH = REPO / "moldetr" / "assets" / "extrema.txt"

# Ensure the repo root is importable so `import app` and `import scripts.*` resolve when pytest is
# invoked from the repo root (matches how CI runs; scripts/ is not a package).
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# --- the deterministic fake model -----------------------------------------------------------------

N_QUERIES = 80  # n_groups (8) * num_queries (10)  — matches build_model() defaults
VEC = 12  # num_classes (5) + num_params (7)
_PROTON_TO_CLASS = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4}  # inverse of postprocess.PROTON_COUNTS
_ACTIVE_LOGIT = 6.0  # sigmoid(6) ≈ 0.9975 → passes any test threshold < ~0.99
_INACTIVE_LOGIT = -8.0  # sigmoid(-8) ≈ 3e-4 → dropped by any test threshold > ~0


def make_output(detections: list[dict]) -> torch.Tensor:
    """Build a ``(1, 80, 12)`` model-output tensor decoding to the given detections.

    Each detection dict: ``{"proton": 1|2|3|4|6, "center_frac": 0..1, "coupling_frac": 0..1,
    "lw_frac": 0..1}``. ``center_frac`` maps to ``center_position_in_points`` via the extrema
    (``[0, 6143]``); keep centers ≥ ~0.004 apart so ``_merge`` (20-pt NMS) does not collapse them.
    Queries beyond ``len(detections)`` get all-low logits and are dropped by ``decode_predictions``.
    """
    out = np.zeros((N_QUERIES, VEC), dtype=np.float32)
    out[:, :5] = _INACTIVE_LOGIT
    for i, d in enumerate(detections):
        out[i, :5] = _INACTIVE_LOGIT
        out[i, _PROTON_TO_CLASS[d["proton"]]] = _ACTIVE_LOGIT
        params = out[i, 5:]  # 7 normalized regression params (postprocess.PARAM_NAMES order)
        params[0] = d.get("center_frac", 0.5)  # center_position_in_points
        params[1] = d.get("lw_frac", 0.5)  # line_width_in_points
        params[5] = d.get("coupling_frac", 0.4)  # coupling_constant_3 == the max(J) slot
    return torch.from_numpy(out[None, :, :])


class FakeModel:
    """Deterministic stand-in for the DETR model: ignores the input, returns a fixed ``(1,80,12)``."""

    def __init__(self, output: torch.Tensor) -> None:
        self._out = output

    def __call__(self, _x: torch.Tensor) -> torch.Tensor:
        return self._out

    def eval(self) -> "FakeModel":
        return self


# Protons 1/2/3 at well-separated centers, each with max J ≈ 8 Hz (0.4 * 102.38 / 5.12).
DEFAULT_DETECTIONS: list[dict] = [
    {"proton": 1, "center_frac": 0.2, "coupling_frac": 0.4},
    {"proton": 2, "center_frac": 0.5, "coupling_frac": 0.4},
    {"proton": 3, "center_frac": 0.8, "coupling_frac": 0.4},
]


@pytest.fixture
def make_fake_model():
    """Factory: build a :class:`FakeModel` from a custom detection list."""
    return lambda detections: FakeModel(make_output(detections))


@pytest.fixture
def fake_model() -> FakeModel:
    """A 3-detection fake model (protons 1/2/3 at distinct centers, max J ≈ 8 Hz)."""
    return FakeModel(make_output(DEFAULT_DETECTIONS))


# --- app patching ---------------------------------------------------------------------------------


@pytest.fixture
def app_module():
    """Import the Gradio app module (requires the ``.[app]`` extra: gradio + plotly)."""
    import app

    return app


@pytest.fixture
def patch_model(app_module, fake_model, tmp_path, monkeypatch):
    """Make ``predict`` / ``simulate_and_detect`` run weight-free: existing CHECKPOINT + fake _MODEL.

    Returns the patched ``app`` module. Use ``app.set_fake(model)`` in-test to swap detections.
    """
    ckpt = tmp_path / "fake_checkpoint.pth"
    ckpt.write_bytes(b"not-a-real-checkpoint")  # only needs to exist so the gate passes
    monkeypatch.setattr(app_module, "CHECKPOINT", str(ckpt))
    monkeypatch.setattr(app_module, "_MODEL", fake_model)
    return app_module


# --- data fixtures --------------------------------------------------------------------------------


@pytest.fixture
def valid_spectrum() -> np.ndarray:
    """A finite, real, 6144-point spectrum (abs of seeded noise) — passes ``validate_spectrum``."""
    rng = np.random.RandomState(0)
    return np.abs(rng.rand(6144)).astype(np.float64)


@pytest.fixture
def ppm_axis() -> np.ndarray:
    """A plausible descending ppm axis over the 6144-point grid (10 → 0 ppm)."""
    return np.linspace(10.0, 0.0, 6144)


@pytest.fixture
def tmp_npz(tmp_path):
    """Factory: write named arrays into a ``.npz`` and return its path string."""

    def _write(name: str = "spec.npz", **arrays) -> str:
        p = tmp_path / name
        np.savez(p, **arrays)
        return str(p)

    return _write


@pytest.fixture
def tmp_npy(tmp_path):
    """Factory: write one array to a ``.npy`` and return its path string."""

    def _write(array, name: str = "spec.npy") -> str:
        p = tmp_path / name
        np.save(p, array)
        return str(p)

    return _write


@pytest.fixture
def example_paths() -> dict[str, str]:
    """Paths to the three committed example spectra (CI-available)."""
    return {
        "roi_S10": str(EXAMPLES / "roi_S10_example.npz"),  # guajazulene 500 MHz (ppm_axis_padded)
        "roi_S8": str(EXAMPLES / "roi_S8_example.npz"),  # vanillin 300 MHz (ppm_axis_padded)
        "synthetic": str(EXAMPLES / "synthetic_example.npz"),  # complex64, spec-only (no ppm)
    }


@pytest.fixture
def extrema() -> dict:
    """The committed normalization extrema used by ``decode_predictions``."""
    from moldetr.postprocess import load_extrema

    return load_extrema(str(EXTREMA_PATH))
