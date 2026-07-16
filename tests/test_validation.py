"""Tests for the input-contract validator (moldetr/validation.py)."""
import numpy as np
import pytest

from moldetr.validation import INPUT_LENGTH, validate_spectrum


def test_accepts_valid_spectrum():
    a = np.random.RandomState(0).rand(INPUT_LENGTH)
    out = validate_spectrum(a)
    assert out.shape == (INPUT_LENGTH,)
    assert out.dtype == np.float64


def test_rejects_wrong_length():
    with pytest.raises(ValueError, match="6144"):
        validate_spectrum(np.zeros(5000))


def test_rejects_nonfinite():
    a = np.ones(INPUT_LENGTH)
    a[0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_spectrum(a)


def test_complex_reduced_to_real_with_warning():
    a = np.ones(INPUT_LENGTH, dtype=complex) * (1 + 2j)
    with pytest.warns(UserWarning, match="complex"):
        out = validate_spectrum(a)
    assert not np.iscomplexobj(out)


def test_wrong_resolution_warns():
    with pytest.warns(UserWarning, match="resolution"):
        validate_spectrum(np.ones(INPUT_LENGTH), points_per_hz=10.0)


def test_ravels_2d_input():
    out = validate_spectrum(np.ones((1, INPUT_LENGTH)))
    assert out.shape == (INPUT_LENGTH,)
