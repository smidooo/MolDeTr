"""Pure-NumPy quantum-mechanical simulator for ¹H NMR spin systems.

The spin Hamiltonian for a set of coupled spin-½ nuclei is built in frequency units (Hz)::

    H = Σ_i ν_i Iz_i  +  Σ_{i<j} J_ij (Ix_i Ix_j + Iy_i Iy_j + Iz_i Iz_j)

with ``ν_i = δ_i · base_freq`` the Larmor offset of spin *i* (chemical shift ``δ_i`` in ppm) and
``J_ij`` the scalar coupling in Hz. ``H`` is diagonalised exactly with :func:`numpy.linalg.eigh`;
the single-quantum transitions are the positive eigenvalue differences ``E_j − E_i`` and their
intensities are ``|⟨j|F_x|i⟩|²`` with ``F_x = Σ_i Ix_i``. Because the *full* bilinear coupling
term is retained (not the weak-coupling / first-order approximation), the result is exact for
strongly-coupled systems — AB, ABX, ... — for up to ~8 spins (a ``2**8 = 256``-dimensional
Hilbert space still diagonalises instantly).

Each stick is broadened by a Lorentzian absorption line and the sum is evaluated **analytically
on the requested ppm grid** — there is no dense-simulate-then-downsample step, so the digital
resolution of the output is exactly what the caller asks for.

The method is exact spin-Hamiltonian diagonalisation (as opposed to a first-order / weak-coupling
approximation), with three deliberate design choices:

1. All inputs are explicit function arguments — nothing is read from module state.
2. The line width is a real per-call argument (``widths_hz``, a **FWHM per spin**). Because a
   transition mixes several spins, a single per-spectrum half-width is used:
   ``gamma = mean(widths_hz) / 2`` (HWHM from the mean FWHM). When every spin shares one width this
   is exact; otherwise it is the simplest faithful choice.
3. The Lorentzian sum is evaluated directly at ``n_points`` over the requested window (the model's
   5.12 points/Hz grid) — no dense-simulate-then-downsample step.

Only NumPy is required.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

# Single spin-½ operators (ħ = 1) in the |α⟩, |β⟩ (m = +½, −½) basis.
_IZ: NDArray[np.complex128] = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)
_IP: NDArray[np.complex128] = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # raising  I⁺
_IM: NDArray[np.complex128] = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)  # lowering I⁻
_IX: NDArray[np.complex128] = 0.5 * (_IP + _IM)
_IY: NDArray[np.complex128] = -0.5j * (_IP - _IM)
_I2: NDArray[np.complex128] = np.eye(2, dtype=complex)


def _embed(op: NDArray[np.complex128], pos: int, n_spins: int) -> NDArray[np.complex128]:
    """Place single-spin operator ``op`` on spin ``pos`` of an ``n_spins`` system (Kronecker)."""
    mats = [_I2] * n_spins
    mats[pos] = op
    out: NDArray[np.complex128] = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m).astype(np.complex128, copy=False)
    return out


def _build_hamiltonian(
    shifts_hz: NDArray[np.float64], couplings_hz: NDArray[np.float64], n_spins: int
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return ``(H, F_x)`` for ``H = Σ ν_i Iz_i + Σ_{i<j} J_ij (I_i · I_j)`` (both in Hz)."""
    dim = 2**n_spins
    iz = [_embed(_IZ, i, n_spins) for i in range(n_spins)]
    ix = [_embed(_IX, i, n_spins) for i in range(n_spins)]
    iy = [_embed(_IY, i, n_spins) for i in range(n_spins)]

    hamiltonian = np.zeros((dim, dim), dtype=complex)
    for i in range(n_spins):
        hamiltonian += shifts_hz[i] * iz[i]
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            j_ij = float(couplings_hz[i, j])
            if j_ij != 0.0:
                hamiltonian += j_ij * (ix[i] @ ix[j] + iy[i] @ iy[j] + iz[i] @ iz[j])

    fx = np.sum(ix, axis=0)
    return hamiltonian, fx


def _transitions(
    hamiltonian: NDArray[np.complex128],
    fx: NDArray[np.complex128],
    min_intensity: float = 1e-9,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Single-quantum transition frequencies (Hz, > 0) and intensities ``|⟨j|F_x|i⟩|²``."""
    energies, vecs = np.linalg.eigh(hamiltonian)
    # F_x in the eigenbasis: fx_eig[j, i] = ⟨j|F_x|i⟩.
    fx_eig = vecs.conj().T @ fx @ vecs
    intensity = np.abs(fx_eig) ** 2

    freqs: list[float] = []
    amps: list[float] = []
    dim = energies.shape[0]
    for i in range(dim):
        for j in range(dim):
            delta_e = float(energies[j] - energies[i])
            if delta_e > 0.0 and intensity[j, i] > min_intensity:
                freqs.append(delta_e)
                amps.append(float(intensity[j, i]))
    return np.asarray(freqs, dtype=float), np.asarray(amps, dtype=float)


def _lorentzian_sum(
    hz_axis: NDArray[np.float64],
    freqs: NDArray[np.float64],
    amps: NDArray[np.float64],
    gamma: float,
) -> NDArray[np.float64]:
    """Sum of Lorentzian absorption lines ``a · γ / (π ((x − f)² + γ²))`` on ``hz_axis``."""
    spectrum = np.zeros_like(hz_axis, dtype=float)
    for freq, amp in zip(freqs, amps):
        spectrum += amp * gamma / (np.pi * ((hz_axis - freq) ** 2 + gamma**2))
    return spectrum


def simulate(
    shifts_ppm: Sequence[float],
    couplings_hz: Sequence[Sequence[float]] | NDArray[np.float64],
    widths_hz: Sequence[float],
    base_freq_mhz: float,
    left_ppm: float,
    right_ppm: float,
    n_points: int = 6144,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a ¹H NMR spectrum by exact spin-Hamiltonian diagonalisation.

    Parameters
    ----------
    shifts_ppm:
        Chemical shift of each spin in ppm — **one entry per spin**; chemically equivalent protons
        are listed once each (e.g. a CH₃ appears three times at the same shift with zero mutual
        coupling), so ``len(shifts_ppm)`` is the number of coupled spin-½ nuclei.
    couplings_hz:
        Symmetric ``N × N`` matrix of scalar couplings in Hz (``N = len(shifts_ppm)``). Only the
        upper triangle (``i < j``) is read; the diagonal is ignored.
    widths_hz:
        Per-spin line width as **FWHM in Hz**. A single Lorentzian half-width is used for the whole
        spectrum, ``gamma = mean(widths_hz) / 2`` (HWHM); exact when all widths are equal.
    base_freq_mhz:
        Spectrometer ¹H frequency in MHz (Hz offset of a spin is ``shift_ppm · base_freq_mhz``).
    left_ppm, right_ppm:
        ppm values at the left and right edges of the output. By NMR convention the left edge is the
        higher (down-field) ppm, so ``left_ppm > right_ppm``; the returned ``ppm_axis`` runs from
        ``left_ppm`` (index 0) to ``right_ppm`` (index ``n_points − 1``). To match the MolDeTr model
        grid, choose the edges so the window is 1200 Hz (``(left_ppm − right_ppm) · base_freq_mhz``)
        and ``n_points = 6144`` (⇒ 5.12 points/Hz).
    n_points:
        Number of samples in the output spectrum (default 6144, the model's input length).

    Returns
    -------
    ``(spectrum_real, ppm_axis)``:
        ``spectrum_real`` is the real (absorption) spectrum, max-normalised to a peak of 1.0
        (0.0 if there are no transitions). ``ppm_axis`` is the matching ppm grid. The absolute
        intensity scale is not meaningful — downstream MolDeTr min-max normalises its input — but
        relative line intensities are physically correct.

    Raises
    ------
    ValueError:
        If the shapes of ``couplings_hz`` / ``widths_hz`` do not match ``len(shifts_ppm)``, if
        ``n_points < 1``, or if the mean line width is not positive.
    """
    shifts = np.asarray(shifts_ppm, dtype=float).ravel()
    n_spins = int(shifts.shape[0])
    if n_spins == 0:
        raise ValueError("shifts_ppm must contain at least one spin.")

    couplings = np.asarray(couplings_hz, dtype=float)
    if couplings.shape != (n_spins, n_spins):
        raise ValueError(
            f"couplings_hz must be {n_spins}x{n_spins} for {n_spins} spins, got {couplings.shape}."
        )

    widths = np.asarray(widths_hz, dtype=float).ravel()
    if widths.shape[0] != n_spins:
        raise ValueError(
            f"widths_hz must have one entry per spin ({n_spins}), got {widths.shape[0]}."
        )
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}.")

    gamma = float(np.mean(widths)) / 2.0  # HWHM from the mean per-spin FWHM
    if not gamma > 0.0:
        raise ValueError(f"mean line width must be positive, got FWHM mean {2 * gamma}.")

    shifts_hz = shifts * float(base_freq_mhz)
    hamiltonian, fx = _build_hamiltonian(shifts_hz, couplings, n_spins)
    freqs, amps = _transitions(hamiltonian, fx)

    ppm_axis = np.linspace(float(left_ppm), float(right_ppm), n_points)
    hz_axis = ppm_axis * float(base_freq_mhz)
    spectrum = _lorentzian_sum(hz_axis, freqs, amps, gamma)

    peak = float(spectrum.max()) if spectrum.size else 0.0
    if peak > 0.0:
        spectrum = spectrum / peak
    return spectrum, ppm_axis
