"""Tests for :mod:`moldetr.distort` -- the deterministic per-effect distortion wrapper.

Covers, per the wrapper's contract: each effect changes the spectrum when enabled alone;
determinism (same params + seed -> identical output); the all-off identity; range validation;
and that the noise knob realises roughly the requested SNR.
"""

from __future__ import annotations

import numpy as np
import pytest

from moldetr.distort import distort

N = 6144


def _spectrum() -> np.ndarray:
    """A complex spectrum with two positive Gaussian peaks (satellites need ``> 0`` samples)."""
    idx = np.arange(N)
    spec = np.zeros(N, dtype=np.complex128)
    for center, amp, width in [(2000, 1.0, 8.0), (4000, 0.6, 6.0)]:
        spec += amp * np.exp(-((idx - center) ** 2) / (2 * width**2))
    return spec


def _ppm() -> np.ndarray:
    """A monotonic high->low ppm axis, as in a real 1H spectrum."""
    return np.linspace(8.0, 0.0, N)


# Each effect enabled alone, with an in-range value that must perturb the spectrum.
_EFFECTS = [
    ("noise_snr_log10", 3.0),
    ("phase0_deg", 5.0),
    ("phase1", 1.0),
    ("baseline", True),
    ("baseline", 0.05),
    ("sat_j_hz", 130.0),
    ("sat_intensity", 0.012),
    ("broaden_hz", 2.0),
]


@pytest.mark.parametrize("param,value", _EFFECTS)
def test_each_effect_changes_spectrum(param: str, value: object) -> None:
    """Enabling any single effect must change the spectrum versus the undistorted input."""
    spec, ppm = _spectrum(), _ppm()
    out = distort(spec, ppm, **{param: value}, seed=0)
    assert out.shape == spec.shape
    assert not np.allclose(out, spec)


@pytest.mark.parametrize("param,value", _EFFECTS)
def test_input_not_mutated(param: str, value: object) -> None:
    """``distort`` copies its input; the caller's array is never mutated in place."""
    spec, ppm = _spectrum(), _ppm()
    before = spec.copy()
    distort(spec, ppm, **{param: value}, seed=0)
    assert np.array_equal(spec, before)


def test_all_none_is_identity() -> None:
    """With every effect off the output equals the input."""
    spec, ppm = _spectrum(), _ppm()
    assert np.allclose(distort(spec, ppm), spec)
    # baseline is also off for None/False specifically.
    assert np.allclose(distort(spec, ppm, baseline=None), spec)
    assert np.allclose(distort(spec, ppm, baseline=False), spec)


def test_determinism_same_params_and_seed() -> None:
    """Same inputs + seed -> byte-identical output, including the noise realisation."""
    spec, ppm = _spectrum(), _ppm()
    kwargs = dict(
        noise_snr_log10=3.0,
        phase0_deg=4.0,
        phase1=0.5,
        baseline=0.03,
        sat_j_hz=140.0,
        sat_intensity=0.01,
        broaden_hz=1.5,
        seed=7,
    )
    a = distort(spec, ppm, **kwargs)
    b = distort(spec, ppm, **kwargs)
    assert np.array_equal(a, b)


def test_seed_changes_noise_realisation() -> None:
    """A different seed changes the (noise-containing) output."""
    spec, ppm = _spectrum(), _ppm()
    a = distort(spec, ppm, noise_snr_log10=3.0, seed=0)
    b = distort(spec, ppm, noise_snr_log10=3.0, seed=1)
    assert not np.array_equal(a, b)


def test_global_rng_state_restored() -> None:
    """``distort`` seeds the global RNG internally but restores the caller's stream on return."""
    spec, ppm = _spectrum(), _ppm()
    np.random.seed(123)
    expected = np.random.rand(5)
    np.random.seed(123)
    distort(spec, ppm, noise_snr_log10=3.0, sat_j_hz=130.0, seed=999)
    assert np.array_equal(np.random.rand(5), expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise_snr_log10": 1.9},
        {"noise_snr_log10": 5.1},
        {"phase0_deg": 8.1},
        {"phase0_deg": -8.1},
        {"sat_j_hz": 39.0},
        {"sat_j_hz": 221.0},
        {"sat_intensity": 0.004},
        {"sat_intensity": 0.016},
        {"broaden_hz": -0.1},
        {"broaden_hz": 3.1},
    ],
)
def test_out_of_range_raises(kwargs: dict[str, float]) -> None:
    """Out-of-range parameters raise ``ValueError`` before any work is done."""
    spec, ppm = _spectrum(), _ppm()
    with pytest.raises(ValueError):
        distort(spec, ppm, **kwargs)


def test_noise_realises_requested_snr() -> None:
    """Noise std matches max(Re) / (2 * 10**log10) to within a loose tolerance."""
    spec, ppm = _spectrum(), _ppm()
    log10 = 3.0
    out = distort(spec, ppm, noise_snr_log10=log10, seed=0)
    noise = np.real(out) - np.real(spec)
    expected_std = float(np.max(np.real(spec))) / (2.0 * 10.0**log10)
    assert np.std(noise) == pytest.approx(expected_std, rel=0.2)


def test_noise_std_scales_inversely_with_snr() -> None:
    """Lower SNR -> proportionally more noise (~100x std swing across two decades)."""
    spec, ppm = _spectrum(), _ppm()
    std_low = np.std(np.real(distort(spec, ppm, noise_snr_log10=2.0, seed=0)) - np.real(spec))
    std_high = np.std(np.real(distort(spec, ppm, noise_snr_log10=4.0, seed=0)) - np.real(spec))
    assert std_low > std_high
    assert 50.0 < std_low / std_high < 200.0
