"""Reproducibility: the noise-fixed live pipeline reproduces the paper's predictions on the bundled examples.

The seeded noise injection (moldetr.inference.normalize_spectrum) makes inference deterministic AND brings
the clean, FFT-resampled input in-distribution so the model reads it correctly. Two checkpoint-free tests
guard the determinism; the full example-vs-ground-truth test skips when the checkpoint is absent (CI has no
weights) and runs locally when MOLDETR_CHECKPOINT points at the .pth.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from moldetr.inference import normalize_spectrum

ROOT = Path(__file__).resolve().parent.parent
CKPT = os.environ.get(
    "MOLDETR_CHECKPOINT", str(ROOT / "moldetr" / "model" / "model_spin_system_ABCDEFG_exp2.pth")
)


def test_normalize_spectrum_is_deterministic():
    """Seeded noise -> identical output for the same input; a different seed changes it."""
    a = np.abs(np.random.RandomState(0).rand(6144)) * 1e6
    assert np.array_equal(
        normalize_spectrum(a, noise_seed=0).numpy(), normalize_spectrum(a, noise_seed=0).numpy()
    )
    assert not np.array_equal(
        normalize_spectrum(a, noise_seed=0).numpy(), normalize_spectrum(a, noise_seed=1).numpy()
    )


def test_noise_injection_present_by_default():
    """The default path injects noise (in-distribution); noise_frac=0 disables it."""
    a = np.linspace(0.0, 1.0, 6144).astype(np.float64)
    assert not np.array_equal(
        normalize_spectrum(a, noise_frac=0.0).numpy(),
        normalize_spectrum(a, noise_frac=0.005).numpy(),
    )


@pytest.mark.skipif(not Path(CKPT).exists(), reason="checkpoint absent (download from Zenodo)")
@pytest.mark.parametrize(
    "example,protons",
    [
        ("roi_S8_example.npz", [1, 1, 1]),  # vanillin ABX (300 MHz)
        ("roi_S10_example.npz", [1, 1, 1]),  # guajazulene (500 MHz)
    ],
)
def test_example_reproduces_ground_truth(example, protons):
    """With the noise fix, the live decode on a bundled example matches its ground truth."""
    from moldetr.inference import build_model, load_checkpoint, run
    from moldetr.postprocess import decode_predictions, load_extrema
    from moldetr.reproducibility import set_seed
    from moldetr.validation import validate_spectrum

    set_seed(42)
    d = np.load(ROOT / "examples" / example, allow_pickle=True)
    amp = validate_spectrum(d["spectrum_padded"], points_per_hz=5.12)
    axis = np.asarray(d["ppm_axis_padded"], dtype=float)
    ex = load_extrema(str(ROOT / "moldetr" / "assets" / "extrema.txt"))
    model = load_checkpoint(build_model(), CKPT)
    preds = sorted(
        decode_predictions(
            run(model, amp), ex, 5.12, ppm_left=float(axis[0]), ppm_right=float(axis[-1])
        ),
        key=lambda p: p["chemical_shift_ppm"],
    )
    gt = sorted(d["ground_truth"], key=lambda g: g["chemical_shift_ppm"])
    assert len(preds) == len(gt)
    assert [p["proton_count"] for p in preds] == protons
    for p, g in zip(preds, gt):
        assert abs(p["chemical_shift_ppm"] - g["chemical_shift_ppm"]) < 0.1
        if g["coupling_constants"]:
            assert abs(p["coupling_constants_hz"][0] - max(g["coupling_constants"])) < 1.5


def test_set_seed_makes_numpy_torch_and_stdlib_deterministic():
    """set_seed(n) fixes numpy, torch, and the stdlib RNG so two runs draw identically; a new seed diverges."""
    import random

    import torch

    from moldetr.reproducibility import set_seed

    def _draw():
        return (np.random.rand(3).tolist(), torch.randn(3).tolist(), random.random())

    set_seed(123)
    first = _draw()
    set_seed(123)
    assert _draw() == first
    set_seed(456)
    assert _draw() != first
