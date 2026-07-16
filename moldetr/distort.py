"""Deterministic, per-effect wrapper around MolDeTr's training-time spectral augmentations.

This module re-exposes the paper's *own* augmentation math
(:mod:`moldetr.dataloader.data_augmentation`) as a clean, deterministic, per-effect API.
Each effect is applied only when its parameter is supplied, and always through the underlying
function's deterministic ("custom values") code path, so that identical ``spectrum`` + ``seed``
reproduce byte-identical output.

Distortion math is **not** reimplemented here -- ``distort`` simply calls the five underlying
``add_*`` functions with the right arguments to pin down every otherwise-random draw.

Licensing boundary
------------------
MolDeTr is Apache-2.0. The shim / field-inhomogeneity distortion
(:class:`moldetr.dataloader.shimming.ShimSim`, wrapped by
``data_augmentation.add_shim_distortions``) is GPL-derived (adapted from SHIMpanzee) and is
**deliberately not exposed here**: this module wraps only the five Apache-licensed ``add_*``
effects and never touches the shim path.

Note: ``data_augmentation`` imports ``ShimSim`` lazily -- inside ``add_shim_distortions`` itself,
the only function that uses it -- so importing ``data_augmentation`` (and therefore ``moldetr.distort``)
no longer transitively loads the GPL ``moldetr.dataloader.shimming`` module. ``distort`` neither imports
``shimming``/``ShimSim`` nor calls ``add_shim_distortions``; the GPL code is reached only if a caller
explicitly invokes the shim path.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert  # type: ignore[import-untyped]  # scipy ships no type stubs

from moldetr.dataloader.data_augmentation import (
    add_13C_satellites_with_variability,
    add_baseline_distortion,
    add_line_broadening,
    add_noise,
    add_phase_distortion,
)

__all__ = ["distort", "POINTS_PER_HZ"]

# Resampling density of the released model (points per Hz); shared by 13C-satellite spacing
# and the Hz->sigma line-broadening conversion.
POINTS_PER_HZ: float = 5.12

# FWHM = 2*sqrt(2*ln2) * sigma for a Gaussian line.
_FWHM_PER_SIGMA: float = 2.0 * np.sqrt(2.0 * np.log(2.0))

# Trained parameter ranges (mirrors data_augmentation defaults / the paper).
_SNR_LOG10_RANGE = (2.0, 5.0)
_PHASE0_ABS_MAX = 8.0
_SAT_J_RANGE = (40.0, 220.0)
_SAT_INTENSITY_RANGE = (0.005, 0.015)
_BROADEN_HZ_RANGE = (0.0, 3.0)

# Fallback (midpoint) satellite parameters when only one of the (J, intensity) pair is given.
_SAT_J_DEFAULT = 130.0
_SAT_INTENSITY_DEFAULT = 0.01
# Default baseline tilt magnitude used when ``baseline=True``.
_BASELINE_DEFAULT = 0.05


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    """Raise ``ValueError`` if ``value`` is outside the trained ``[lo, hi]`` range."""
    if not lo <= value <= hi:
        raise ValueError(f"{name}={value!r} is outside the trained range [{lo}, {hi}]")


def _validate(
    noise_snr_log10: float | None,
    phase0_deg: float | None,
    sat_j_hz: float | None,
    sat_intensity: float | None,
    broaden_hz: float | None,
) -> None:
    """Validate every supplied (non-``None``) parameter against its trained range."""
    if noise_snr_log10 is not None:
        _check_range("noise_snr_log10", noise_snr_log10, *_SNR_LOG10_RANGE)
    if phase0_deg is not None:
        _check_range("phase0_deg", phase0_deg, -_PHASE0_ABS_MAX, _PHASE0_ABS_MAX)
    if sat_j_hz is not None:
        _check_range("sat_j_hz", sat_j_hz, *_SAT_J_RANGE)
    if sat_intensity is not None:
        _check_range("sat_intensity", sat_intensity, *_SAT_INTENSITY_RANGE)
    if broaden_hz is not None:
        _check_range("broaden_hz", broaden_hz, *_BROADEN_HZ_RANGE)


def _apply_satellites(
    out: np.ndarray, sat_j_hz: float | None, sat_intensity: float | None
) -> np.ndarray:
    """Add deterministic 13C satellites by collapsing each random range to a single value.

    ``add_13C_satellites_with_variability`` has no ``use_custom_values`` path -- it always draws
    ``np.random.uniform(min, max)``. Passing ``min == max`` makes each draw return that exact
    value (``np.random.uniform(x, x) == x``), so J and intensity are pinned deterministically.
    """
    j = _SAT_J_DEFAULT if sat_j_hz is None else sat_j_hz
    intensity = _SAT_INTENSITY_DEFAULT if sat_intensity is None else sat_intensity
    result = add_13C_satellites_with_variability(
        out,
        j_coupling_min=j,
        j_coupling_max=j,
        satellite_intensity_min=intensity,
        satellite_intensity_max=intensity,
        points_per_Hz=POINTS_PER_HZ,
    )
    return np.asarray(result, dtype=np.complex128)


def _apply_phase(
    out: np.ndarray, ppm: np.ndarray, phase0_deg: float | None, phase1: float | None
) -> np.ndarray:
    """Apply a deterministic phase distortion, defaulting the unset order to 0.0.

    The custom path only triggers when BOTH ``phase_0_custom`` and ``phase_1_custom`` are not
    ``None``; otherwise the function silently falls back to random phases. So the missing order
    is filled with 0.0 (an identity for that term).
    """
    # A simulated spectrum is purely real (absorption only); multiplying a real signal by exp(iθ)
    # and keeping the real part collapses to a flat cos(θ) scaling with no dispersive line-shape
    # change. Forming the analytic signal (absorption + i·dispersion, via the Hilbert transform)
    # first makes exp(iθ) mix absorption and dispersion the way a real phasing error does. For an
    # already-analytic input the real part is unchanged, so this is a no-op there.
    analytic = hilbert(np.real(out))
    result = add_phase_distortion(
        analytic,
        ppm,
        float(ppm[0]),
        float(ppm[-1]),
        phase_0_custom=0.0 if phase0_deg is None else phase0_deg,
        phase_1_custom=0.0 if phase1 is None else phase1,
        use_custom_values=True,
    )
    return np.asarray(result, dtype=np.complex128)


def _apply_baseline(out: np.ndarray, ppm: np.ndarray, baseline: bool | float) -> np.ndarray:
    """Apply a deterministic linear baseline tilt (``+mag`` at one end, ``-mag`` at the other)."""
    magnitude = _BASELINE_DEFAULT if baseline is True else float(baseline)
    result = add_baseline_distortion(
        out,
        ppm,
        float(ppm[0]),
        float(ppm[-1]),
        sino=1.0,  # unused on the custom path
        custom_base_left=magnitude,
        custom_base_right=-magnitude,
        use_custom_values=True,
    )
    return np.asarray(result, dtype=np.complex128)


def distort(
    spectrum: np.ndarray,
    ppm_axis: np.ndarray,
    *,
    noise_snr_log10: float | None = None,
    phase0_deg: float | None = None,
    phase1: float | None = None,
    baseline: bool | float | None = None,
    sat_j_hz: float | None = None,
    sat_intensity: float | None = None,
    broaden_hz: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Apply the paper's training-time distortions deterministically, one effect at a time.

    Every effect is off unless its parameter is not ``None`` (baseline: not ``None``/``False``).
    Effects are applied in the paper's order: 13C satellites -> line broadening -> phase ->
    noise -> baseline. The returned array is kept **complex** (the caller applies ``np.real``).

    The global NumPy RNG is seeded with ``seed`` for the run (``add_noise`` always draws its
    noise realisation from ``np.random``) and its prior state is restored on return, so the
    caller's RNG stream is left untouched.

    Parameters
    ----------
    spectrum:
        Complex simulated spectrum. Copied; never mutated in place.
    ppm_axis:
        ppm value per point; its endpoints supply ``ppm_right``/``ppm_left`` for the phase and
        baseline effects.
    noise_snr_log10:
        log10 SNR exponent, 2.0-5.0 (SNR 1e2-1e5). Realised noise std ~= max(Re) / (2 * 10**x).
    phase0_deg:
        Zeroth-order phase in degrees, |.| <= 8.
    phase1:
        First-order (frequency-dependent) phase coefficient.
    baseline:
        ``None``/``False`` = off; ``True`` = default tilt; a float = tilt magnitude.
    sat_j_hz:
        13C-satellite J coupling in Hz, 40-220 (enables the satellite effect).
    sat_intensity:
        13C-satellite relative intensity, 0.005-0.015 (also enables the satellite effect).
    broaden_hz:
        Gaussian line-broadening FWHM in Hz, 0-3 (converted to a sigma in points via
        ``POINTS_PER_HZ``).
    seed:
        Seed for the noise realisation.

    Returns
    -------
    np.ndarray
        The distorted complex spectrum.
    """
    _validate(noise_snr_log10, phase0_deg, sat_j_hz, sat_intensity, broaden_hz)

    out = np.array(spectrum, dtype=np.complex128, copy=True)
    ppm = np.asarray(ppm_axis, dtype=np.float64)

    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        if sat_j_hz is not None or sat_intensity is not None:
            out = _apply_satellites(out, sat_j_hz, sat_intensity)
        if broaden_hz is not None:
            sigma_points = broaden_hz * POINTS_PER_HZ / _FWHM_PER_SIGMA
            out = add_line_broadening(out, custom_sigma=sigma_points, use_custom_values=True)
        if phase0_deg is not None or phase1 is not None:
            out = _apply_phase(out, ppm, phase0_deg, phase1)
        if noise_snr_log10 is not None:
            out, _ = add_noise(out, custom_SNR=10.0**noise_snr_log10, use_custom_values=True)
        if baseline is not None and baseline is not False:
            out = _apply_baseline(out, ppm, baseline)
    finally:
        np.random.set_state(rng_state)

    return np.asarray(out, dtype=np.complex128)
