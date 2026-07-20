"""Robustness / edge-case hardening for the input contract (`validate_spectrum`).

Complements the Hypothesis fuzz in test_properties.py with named, documented boundary cases — the input
*types* and *shapes* a real user throws at it (int arrays, Python lists, multi-dim, empty, negative lobes).
The point: MolDeTr fails loudly and predictably on bad input rather than returning silent garbage.
"""

import numpy as np
import pytest

from moldetr.validation import INPUT_LENGTH, validate_spectrum


def test_accepts_integer_dtype():
    """An int spectrum is upcast to float, not rejected."""
    out = validate_spectrum(np.ones(INPUT_LENGTH, dtype=np.int64), warn=False)
    assert out.dtype == np.float64 and out.shape == (INPUT_LENGTH,)


def test_accepts_python_list():
    """A plain list of the right length is accepted (np.asarray-ed)."""
    out = validate_spectrum([0.0] * INPUT_LENGTH, warn=False)
    assert out.shape == (INPUT_LENGTH,)


def test_accepts_all_zeros_and_negative_lobes():
    """All-zeros and negative intensities (real phased NMR has them) are valid, finite input."""
    assert validate_spectrum(np.zeros(INPUT_LENGTH), warn=False).shape == (INPUT_LENGTH,)
    neg = -np.abs(np.random.RandomState(0).rand(INPUT_LENGTH))
    assert np.all(validate_spectrum(neg, warn=False) <= 0)


def test_multidim_input_ravels_when_total_matches():
    """A (1, 1, 6144) or (2, 3072) array ravels to the 6144 contract; wrong total is rejected."""
    assert validate_spectrum(np.zeros((1, 1, INPUT_LENGTH)), warn=False).shape == (INPUT_LENGTH,)
    assert validate_spectrum(np.zeros((2, INPUT_LENGTH // 2)), warn=False).shape == (INPUT_LENGTH,)
    with pytest.raises(ValueError):
        validate_spectrum(np.zeros((2, INPUT_LENGTH)), warn=False)  # ravels to 12288 != 6144


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        validate_spectrum(np.zeros(0), warn=False)


def test_wrong_resolution_warns_but_returns():
    """A known-but-wrong points/Hz is a recoverable warning (not a hard failure)."""
    with pytest.warns(UserWarning):
        out = validate_spectrum(np.zeros(INPUT_LENGTH), points_per_hz=10.0)  # != 5.12
    assert out.shape == (INPUT_LENGTH,)
