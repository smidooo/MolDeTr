"""Property-based tests (Hypothesis) — assert invariants that must hold for *all* inputs, not examples.

Hypothesis auto-hunts edge cases (empty, huge, subnormal, boundary) and shrinks any failure to a minimal
reproducer. These cover the input-validation contract (fuzzing) and the pure numeric conversions.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from moldetr.dataloader.dataloader import _split_lengths
from moldetr.validation import INPUT_LENGTH, validate_spectrum
from moldetr.visualization import _ppm_to_hz_factor

_FINITE32 = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=32
)


# ---------------------------------------------------------------- validate_spectrum robustness (fuzz)


@settings(max_examples=60)
@given(arrays(np.float32, INPUT_LENGTH, elements=_FINITE32))
def test_validate_accepts_any_finite_full_length_spectrum(a):
    """Any finite (6144,) array is accepted and returned as a finite real float64 array — never raises."""
    out = validate_spectrum(a, warn=False)
    assert out.shape == (INPUT_LENGTH,)
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))


@given(st.integers(min_value=1, max_value=20000).filter(lambda n: n != INPUT_LENGTH))
def test_validate_rejects_any_wrong_length(n):
    """A wrong length is a hard ValueError, never silent garbage — for every length but 6144."""
    with pytest.raises(ValueError):
        validate_spectrum(np.zeros(n, np.float32), warn=False)


@given(st.integers(min_value=0, max_value=INPUT_LENGTH - 1))
def test_validate_rejects_a_nan_anywhere(idx):
    """A single NaN at any position is rejected (no silent propagation into the model)."""
    a = np.zeros(INPUT_LENGTH, np.float32)
    a[idx] = np.nan
    with pytest.raises(ValueError):
        validate_spectrum(a, warn=False)


def test_validate_complex_input_warns_and_returns_real_part():
    a = (np.ones(INPUT_LENGTH) + 2j * np.ones(INPUT_LENGTH)).astype(np.complex64)
    with pytest.warns(UserWarning):
        out = validate_spectrum(a)  # warn=True by default
    assert out.shape == (INPUT_LENGTH,)
    assert np.allclose(out, 1.0)  # the real (absorption) part


# ---------------------------------------------------------------- pure numeric invariants


@given(st.integers(min_value=3, max_value=10_000_000))
def test_split_lengths_always_partition_n(n):
    """train/val/test lengths sum to exactly n and are non-negative — torch.random_split needs this."""
    tr, va, te = _split_lengths(n)
    assert tr + va + te == n
    assert min(tr, va, te) >= 0
    assert va <= n and te <= n


@given(
    st.floats(min_value=0.05, max_value=15.0, allow_nan=False, allow_infinity=False),  # ppm span
    st.floats(min_value=-2.0, max_value=12.0, allow_nan=False, allow_infinity=False),  # right edge
)
def test_ppm_to_hz_factor_is_positive_and_the_spectrometer_frequency(span, right):
    """For a fixed 1200 Hz window, Hz-per-ppm = 1200/|span| > 0 (the spectrometer MHz)."""
    left = right + span  # left > right by construction
    f = _ppm_to_hz_factor(left, right)
    assert f > 0
    assert abs(f - 1200.0 / span) < 1e-6 * max(1.0, f)


@given(st.floats(min_value=0.05, max_value=15.0, allow_nan=False, allow_infinity=False))
def test_ppm_to_hz_factor_scales_inversely_with_window_width(span):
    """A window twice as wide (in ppm) halves the Hz-per-ppm factor (lower field)."""
    wide = _ppm_to_hz_factor(5.0 + span, 5.0)
    wider = _ppm_to_hz_factor(5.0 + 2 * span, 5.0)
    assert abs(wider - wide / 2) < 1e-6 * max(1.0, wide)
