"""Check a user-supplied spectrum against MolDeTr's input contract.

MolDeTr was trained on a fixed input format (see ``docs/INPUT_FORMAT.md``):

* exactly ``INPUT_LENGTH`` = 6144 points,
* sampled at ``POINTS_PER_HZ`` = 5.12 points/Hz, i.e. a 1200 Hz window,
* real-valued and finite.

Only the *global* scale is removed — the model min-max normalises each spectrum, so overall
receiver gain does not matter and no integral/proton-count reference is needed. Relative
intensities, SNR (trained on 10²–10⁵), and line shape *do* matter. A wrong length fails deep
inside the backbone with an opaque error, so :func:`validate_spectrum` checks it up front and
explains how to fix it.
"""

from __future__ import annotations

import warnings

import numpy as np

INPUT_LENGTH = 6144
POINTS_PER_HZ = 5.12
WINDOW_HZ = INPUT_LENGTH / POINTS_PER_HZ  # 1200.0 Hz


def validate_spectrum(amplitudes, points_per_hz: float | None = None, warn: bool = True):
    """Return the spectrum as a real 1-D float array, or raise if it breaks the contract.

    Raises ``ValueError`` for the hard requirements (length, finiteness). When ``warn`` is set,
    emits a :class:`UserWarning` for recoverable issues (complex input, wrong digital resolution)
    instead of failing. ``points_per_hz`` is the caller's known sampling density, if any.
    """
    a = np.asarray(amplitudes)
    if a.ndim != 1:
        a = a.ravel()

    if np.iscomplexobj(a):
        if warn:
            warnings.warn(
                "Spectrum is complex; using its real part. Pass the real (absorption) spectrum.",
                stacklevel=2,
            )
        a = np.real(a)
    a = a.astype(np.float64, copy=False)

    if a.shape[0] != INPUT_LENGTH:
        raise ValueError(
            f"Spectrum has {a.shape[0]} points, but MolDeTr needs exactly {INPUT_LENGTH}. "
            f"Resample your region to {POINTS_PER_HZ} points/Hz over a {WINDOW_HZ:.0f} Hz window "
            f"and zero-pad or crop to {INPUT_LENGTH} points (see docs/INPUT_FORMAT.md)."
        )
    if not np.all(np.isfinite(a)):
        raise ValueError("Spectrum contains NaN or Inf; clean or interpolate those points first.")

    if points_per_hz is not None and abs(points_per_hz - POINTS_PER_HZ) > 0.01:
        got_window = INPUT_LENGTH / points_per_hz
        if warn:
            warnings.warn(
                f"Digital resolution is {points_per_hz:.3f} points/Hz "
                f"(a {got_window:.0f} Hz window), not {POINTS_PER_HZ} points/Hz (1200 Hz). "
                "Predictions may be unreliable — resample to 5.12 points/Hz.",
                stacklevel=2,
            )
    return a
