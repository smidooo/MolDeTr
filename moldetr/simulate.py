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
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Scale = Literal["peak", "protons"]

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


def _validated_couplings(
    couplings_hz: NDArray[np.float64] | Sequence[Sequence[float]], n_spins: int
) -> NDArray[np.float64]:
    """Check a coupling matrix against the convention this module reads: the upper triangle.

    ``_build_hamiltonian`` and :func:`coupling_blocks` both read only ``i < j``. A matrix whose
    lower triangle carries couplings that are *not* mirrored above the diagonal is therefore read
    as decoupled -- silently, and with a physically plausible spectrum as the result, which is the
    worst way for this to fail. Symmetric and upper-triangular matrices are both valid conventions
    and both stay accepted; anything else is rejected here rather than quietly ignored.

    **Deliberately not applied to** :func:`coupling_blocks`, :func:`simulate` or
    :func:`simulate_systems`. ``tests/test_simulate_blocks.py`` pins that blocking and simulation
    ignore the lower triangle *identically*, which is what makes the upper triangle a usable
    contract for a matrix editor; raising there would break that pairing and change behaviour on
    the path that produced the paper's numbers. This guard covers only the public spin-physics API
    added in v1.1.0, which has no such history and whose docstring already promised symmetry.
    """
    couplings = np.asarray(couplings_hz, dtype=float)
    if couplings.shape != (n_spins, n_spins):
        raise ValueError(
            f"couplings_hz must be ({n_spins}, {n_spins}) to match {n_spins} shifts, "
            f"got {couplings.shape}"
        )
    if np.any(np.tril(couplings, -1) != 0.0) and not np.allclose(couplings, couplings.T):
        raise ValueError(
            "couplings_hz has entries in its lower triangle that are not mirrored above the "
            "diagonal. Only the upper triangle is read, so those couplings would be silently "
            "ignored and the system would come out decoupled. Pass a symmetric matrix, or fill "
            "the upper triangle only."
        )
    return couplings


def _validated_spin_count(n_spins: int) -> int:
    """Bound a spin count to what this module can actually build.

    ``_embed`` allocates ``3n`` matrices of ``4**n`` complex128, so an unchecked count fails as an
    out-of-memory error rather than a diagnosable one. ``MAX_BLOCK_SPINS`` is the ceiling this
    module already states and :func:`simulate_systems` already enforces; the public entry points
    share it rather than each inventing their own.
    """
    if n_spins < 1:
        raise ValueError(f"a spin system needs at least one spin, got {n_spins}.")
    if n_spins > MAX_BLOCK_SPINS:
        raise ValueError(
            f"n_spins={n_spins} exceeds MAX_BLOCK_SPINS={MAX_BLOCK_SPINS}; the product space "
            f"would be {2**n_spins} states."
        )
    return n_spins


# --- public spin-physics API ----------------------------------------------------------------------
#
# The three functions below expose the primitives above under names that carry a compatibility
# promise. They exist because downstream packages were reaching for `_IM`, `_build_hamiltonian` and
# `_embed` directly -- private names that a rename here would break silently at a distance.
#
# The private originals are deliberately kept: they are what the shipped model was built on, and
# callers migrate on their own schedule. These wrappers add no behaviour, only a stable surface.


def build_hamiltonian(
    shifts_hz: NDArray[np.float64], couplings_hz: NDArray[np.float64]
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return ``(H, F_x)`` in Hz for a coupled spin system.

    ``H = Σ ν_i Iz_i + Σ_{i<j} J_ij (I_i · I_j)``. The spin count is read off ``shifts_hz`` rather
    than passed separately, since a mismatch between the two is a bug with no useful meaning.

    Args:
        shifts_hz: chemical shifts, one per spin, in Hz.
        couplings_hz: ``(n, n)`` scalar-coupling matrix in Hz. Only the upper triangle is read, so
            pass it symmetric or upper-triangular; a lower triangle that is not mirrored above the
            diagonal raises rather than being ignored. The diagonal is ignored either way.
    """
    shifts = np.asarray(shifts_hz, dtype=float)
    n_spins = _validated_spin_count(int(shifts.size))
    couplings = _validated_couplings(couplings_hz, n_spins)
    return _build_hamiltonian(shifts, couplings, n_spins)


def lowering_operators(n_spins: int) -> list[NDArray[np.complex128]]:
    """Return ``I⁻`` for each spin, embedded in the full ``2**n_spins`` product space.

    Useful for resolving a transition's intensity *per spin* -- something
    :func:`transitions` cannot give, because it sums over spins before returning.

    Spin 0 is the most-significant Kronecker factor, and the single-spin basis is
    ``(|alpha>, |beta>) = (m = +1/2, m = -1/2)``. A caller building its own operators to combine
    with these needs both conventions, or it gets a silently permuted result.
    """
    return [_embed(_IM, i, n_spins) for i in range(_validated_spin_count(n_spins))]


def transitions(
    hamiltonian: NDArray[np.complex128],
    fx: NDArray[np.complex128],
    min_intensity: float = 1e-9,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Single-quantum transition frequencies (Hz, > 0) and intensities ``|⟨j|F_x|i⟩|²``.

    Intensities are **already summed over spins**; use :func:`lowering_operators` if the per-spin
    contributions are needed.

    ``min_intensity`` is an absolute threshold on ``|<j|F_x|i>|**2``, not a fraction of the
    strongest line.
    """
    hamiltonian = np.asarray(hamiltonian)
    fx = np.asarray(fx)
    if fx.shape != hamiltonian.shape:
        raise ValueError(
            f"fx must have the same shape as hamiltonian {hamiltonian.shape}, got {fx.shape}. "
            "Frequencies come from the Hamiltonian alone, so a mismatched F_x returns correct "
            "line positions with wrong intensities -- the most deceptive way for this to fail."
        )
    return _transitions(hamiltonian, fx, min_intensity)


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


#: Largest block the exact diagonalisation will attempt. The Hamiltonian is ``2**n`` and
#: ``_transitions`` scans it pairwise, so 10 spins (1024 states, ~1M pairs) is already slow on the
#: free CPU runtimes this app targets. Raising this trades responsiveness for reach.
MAX_BLOCK_SPINS = 10

#: Couplings smaller than this are treated as absent when grouping spins into blocks. Without a
#: tolerance a stray 1e-15 from a matrix edit would fuse two blocks and cost 2**(n+m) states.
COUPLING_EPS_HZ = 1e-9


def _validated_system(
    shifts_ppm: Sequence[float],
    couplings_hz: Sequence[Sequence[float]] | NDArray[np.float64],
    widths_hz: Sequence[float],
    n_points: int,
    scale: Scale,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Coerce the spin-system arguments to arrays and check they describe the same system.

    Shared by :func:`simulate` and :func:`simulate_systems` so both validate the *caller's*
    arguments. ``simulate_systems`` slices per block before delegating, so without this the inner
    call would only ever see a self-consistent projection and an undersized ``couplings_hz`` would
    silently decide how many spins exist.
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
    if scale not in ("peak", "protons"):
        raise ValueError(f"scale must be 'peak' or 'protons', got {scale!r}.")
    return shifts, couplings, widths


def coupling_blocks(
    couplings_hz: Sequence[Sequence[float]] | NDArray[np.float64],
) -> list[list[int]]:
    """Group spin indices into independent spin systems.

    Two spins share a block when a non-zero coupling connects them, directly or through a chain.
    Uncoupled spins become singleton blocks. Blocks are returned in ascending order of their lowest
    index, and each block's members are sorted, so the result is deterministic.

    Only the upper triangle is read, matching :func:`simulate`, and the matrix is rebuilt
    symmetrically from it. The walk below scans whole rows, so without that step a matrix filled
    only above the diagonal — the natural output of an editor whose contract is "fill the upper
    triangle" — would split into different blocks than the symmetric matrix meaning the same thing.
    Two spins coupled to a third came out as ``[[0, 2], [1]]`` rather than ``[[0, 1, 2]]``.
    """
    given = np.asarray(couplings_hz, dtype=float)
    upper = np.triu(given, 1)
    j = upper + upper.T
    n = int(j.shape[0])
    seen: set[int] = set()
    blocks: list[list[int]] = []
    for start in range(n):
        if start in seen:
            continue
        stack, block = [start], []
        seen.add(start)
        while stack:  # depth-first walk of the coupling graph
            spin = stack.pop()
            block.append(spin)
            for other in range(n):
                if other not in seen and abs(j[spin, other]) > COUPLING_EPS_HZ:
                    seen.add(other)
                    stack.append(other)
        blocks.append(sorted(block))
    return blocks


def simulate_systems(
    shifts_ppm: Sequence[float],
    couplings_hz: Sequence[Sequence[float]] | NDArray[np.float64],
    widths_hz: Sequence[float],
    base_freq_mhz: float,
    left_ppm: float,
    right_ppm: float,
    n_points: int = 6144,
    scale: Scale = "peak",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate every independent spin system in one matrix and sum them.

    Blocks are always summed in per-proton space, so relative integrals across systems are correct
    by construction. ``scale`` only decides the final global factor:

    ``"peak"`` (default) rescales the sum to a maximum of 1.0. Relative areas are untouched — it is
    one global divisor — but the absolute scale is restored, which matters because the distortion
    parameters are calibrated against a peak of 1 (training renormalises to ``max(Re) = 1`` before
    phase, noise and baseline for the same reason). Use this when the spectrum is about to be
    distorted.

    ``"protons"`` leaves the per-proton scale in place, so total area equals the proton count. Every
    spin must have an observable transition for that to hold; see :func:`simulate`.

    Equivalent to one ``simulate(..., scale="protons")`` call over the whole matrix, but only pays
    ``2**n`` per *block* rather than for the whole system, and lets each block carry its own line
    width (``simulate`` collapses widths to a single mean, so per-group widths are only honest once
    the groups are simulated apart).

    Summing is valid because ``scale="protons"`` makes uncoupled systems superpose exactly; see the
    ``scale`` documentation on :func:`simulate`.

    Raises
    ------
    ValueError:
        If the argument shapes disagree or ``scale`` is unknown (checked here against the caller's
        arguments, before any block slicing hides the mismatch); if any single block exceeds
        :data:`MAX_BLOCK_SPINS`, since splitting cannot help — the block is genuinely one coupled
        system — so this fails fast instead of hanging; or if any block has no observable
        transition, because blocks are always summed in per-proton space.
    """
    shifts, couplings, widths = _validated_system(
        shifts_ppm, couplings_hz, widths_hz, n_points, scale
    )

    blocks = coupling_blocks(couplings)
    for block in blocks:
        if len(block) > MAX_BLOCK_SPINS:
            raise ValueError(
                f"one coupled block has {len(block)} spins, above the limit of "
                f"{MAX_BLOCK_SPINS} ({2 ** len(block)} states). Split it by removing a coupling."
            )

    total: NDArray[np.float64] | None = None
    ppm_axis: NDArray[np.float64] | None = None
    for block in blocks:
        idx = np.array(block, dtype=int)
        spectrum, ppm_axis = simulate(
            shifts[idx],
            couplings[np.ix_(idx, idx)],
            widths[idx],
            base_freq_mhz,
            left_ppm,
            right_ppm,
            n_points,
            scale="protons",
        )
        total = spectrum if total is None else total + spectrum

    if total is None or ppm_axis is None:  # no spins at all
        raise ValueError("shifts_ppm must contain at least one spin.")

    if scale == "peak":
        peak = float(total.max()) if total.size else 0.0
        if peak > 0.0:
            total = total / peak
    return total, ppm_axis


def simulate(
    shifts_ppm: Sequence[float],
    couplings_hz: Sequence[Sequence[float]] | NDArray[np.float64],
    widths_hz: Sequence[float],
    base_freq_mhz: float,
    left_ppm: float,
    right_ppm: float,
    n_points: int = 6144,
    scale: Scale = "peak",
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
    scale:
        How the intensity axis is fixed.

        ``"peak"`` (default) max-normalises the spectrum to 1.0. Convenient for a single system and
        what every existing caller expects, but **spectra scaled this way must never be added**: a
        1H singlet and a 3H methyl both come out at height 1.

        ``"protons"`` scales the transition intensities so their sum equals the number of spins, so
        the integrated area of one proton is the same in every spectrum. This is the mode to use when
        summing several spin systems into one window. It also makes independent systems superpose
        exactly: simulating two uncoupled blocks jointly repeats each block's lines once per spin
        state of the other (a ``2**n`` degeneracy factor), and normalising per proton cancels it, so
        ``simulate(A ∪ B) == simulate(A) + simulate(B)``. The area promise requires at least one
        observable transition; a system that has none raises rather than return a silent zero.

    Returns
    -------
    ``(spectrum_real, ppm_axis)``:
        ``spectrum_real`` is the real (absorption) spectrum; ``ppm_axis`` is the matching ppm grid.
        Relative line intensities are physically correct in both scaling modes.

    Raises
    ------
    ValueError:
        If the shapes of ``couplings_hz`` / ``widths_hz`` do not match ``len(shifts_ppm)``, if
        ``n_points < 1``, if ``scale`` is unknown, if the mean line width is not positive, or if
        ``scale="protons"`` is asked of a system with no observable transition.
    """
    shifts, couplings, widths = _validated_system(
        shifts_ppm, couplings_hz, widths_hz, n_points, scale
    )
    n_spins = int(shifts.shape[0])

    gamma = float(np.mean(widths)) / 2.0  # HWHM from the mean per-spin FWHM
    if not gamma > 0.0:
        raise ValueError(f"mean line width must be positive, got FWHM mean {2 * gamma}.")

    shifts_hz = shifts * float(base_freq_mhz)
    hamiltonian, fx = _build_hamiltonian(shifts_hz, couplings, n_spins)
    freqs, amps = _transitions(hamiltonian, fx)

    if scale == "protons":
        # Each Lorentzian integrates to its amplitude, so pinning the amplitude sum pins the total
        # area analytically — no dependence on the grid, the line width, or the peak height.
        total = float(amps.sum())
        if not total > 0.0:
            raise ValueError(
                f"scale='protons' promises an integrated area of {n_spins}, but this spin system "
                "has no observable transition: every single-quantum frequency is 0 Hz, which the "
                "positive-frequency filter drops. A spin at exactly 0 ppm is the usual cause. Move "
                "it off the axis origin, or use scale='peak', which makes no area promise."
            )
        amps = amps * (n_spins / total)

    ppm_axis = np.linspace(float(left_ppm), float(right_ppm), n_points)
    hz_axis = ppm_axis * float(base_freq_mhz)
    spectrum = _lorentzian_sum(hz_axis, freqs, amps, gamma)

    if scale == "peak":
        peak = float(spectrum.max()) if spectrum.size else 0.0
        if peak > 0.0:
            spectrum = spectrum / peak
    return spectrum, ppm_axis
