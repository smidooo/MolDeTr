"""Simulate -> (optionally distort) -> predict round-trip for MolDeTr.

This ties three verified building blocks into one closed loop against the released checkpoint:

1. :func:`moldetr.simulate.simulate` -- exact spin-Hamiltonian ¹H NMR simulation of a named
   *phenotype* (a small, hand-specified spin system with a known ground-truth grouping);
2. :func:`moldetr.distort.distort` -- optional, deterministic, per-effect training-time distortions
   (noise, phase, baseline, ¹³C satellites, line broadening), each bounded to its trained range;
3. the checkpoint-only predict recipe (:func:`moldetr.inference.run` +
   :func:`moldetr.postprocess.decode_predictions`) -- exactly as in ``scripts/predict.py``.

Everything is simulated on the model's own grid: ``base_freq_mhz=80``, ``left_ppm=15``,
``right_ppm=0``, ``n_points=6144`` -> a 1200 Hz window at 5.12 points/Hz. At 80 MHz the whole
0-15 ppm ¹H range fits inside that window, and J splittings (in Hz) are field-independent, so the
recovered couplings are directly comparable to the ground truth regardless of field.

Checkpoint resolution (matches ``app.py``): the ``MOLDETR_CHECKPOINT`` environment variable if set,
otherwise ``moldetr/model/model_spin_system_ABCDEFG_exp2.pth``. Weights are on Zenodo
(DOI 10.5281/zenodo.21217102).

Examples
--------
    python scripts/simulate_and_predict.py --phenotype ethyl
    python scripts/simulate_and_predict.py --phenotype aromatic_ax --snr 3.0 --phase0 5
    python scripts/simulate_and_predict.py --phenotype methoxy_singlet --plot methoxy.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from moldetr.distort import distort
from moldetr.inference import build_model, load_checkpoint, run
from moldetr.postprocess import decode_predictions, load_extrema
from moldetr.simulate import simulate
from moldetr.visualization import plot_spectrum

ROOT = Path(__file__).resolve().parent.parent

# The released model's grid: 80 MHz, 15 -> 0 ppm = 1200 Hz over 6144 points -> 5.12 points/Hz.
BASE_FREQ_MHZ = 80.0
LEFT_PPM = 15.0
RIGHT_PPM = 0.0
N_POINTS = 6144
POINTS_PER_HZ = 5.12
THRESHOLD = 0.3  # decode_predictions detection threshold (predict.py default)

DEFAULT_CHECKPOINT = os.environ.get(
    "MOLDETR_CHECKPOINT",
    str(ROOT / "moldetr" / "model" / "model_spin_system_ABCDEFG_exp2.pth"),
)
EXTREMA_PATH = str(ROOT / "moldetr" / "assets" / "extrema.txt")


class GTGroup(TypedDict):
    """A ground-truth multiplet group: chemical shift (ppm), proton count, and its largest J (Hz)."""

    shift_ppm: float
    proton_count: int
    max_j_hz: float | None  # None for a singlet (no resolvable coupling)


class Phenotype(TypedDict):
    """A named spin system: per-spin shifts/widths, pairwise couplings, and the GT grouping."""

    description: str
    shifts_ppm: list[float]  # one entry per spin (equivalent protons repeated)
    couplings: list[tuple[int, int, float]]  # (spin_i, spin_j, J in Hz)
    widths_hz: list[float]  # per-spin FWHM in Hz
    gt_groups: list[GTGroup]


# --- Phenotype ground truth --------------------------------------------------

PHENOTYPES: dict[str, Phenotype] = {
    "ethyl": {
        "description": "ethyl CH3-CH2 (a 3H triplet + a 2H quartet, J = 7 Hz)",
        "shifts_ppm": [1.2, 1.2, 1.2, 3.5, 3.5],
        # 7 Hz between each CH3 spin {0,1,2} and each CH2 spin {3,4}; CH3-CH3 and CH2-CH2 = 0.
        "couplings": [(i, j, 7.0) for i in (0, 1, 2) for j in (3, 4)],
        "widths_hz": [1.0, 1.0, 1.0, 1.0, 1.0],
        "gt_groups": [
            {"shift_ppm": 1.2, "proton_count": 3, "max_j_hz": 7.0},
            {"shift_ppm": 3.5, "proton_count": 2, "max_j_hz": 7.0},
        ],
    },
    "aromatic_ax": {
        "description": "an aromatic AX pair (two 1H doublets, J = 8 Hz)",
        "shifts_ppm": [7.5, 6.9],
        "couplings": [(0, 1, 8.0)],
        "widths_hz": [1.0, 1.0],
        "gt_groups": [
            {"shift_ppm": 7.5, "proton_count": 1, "max_j_hz": 8.0},
            {"shift_ppm": 6.9, "proton_count": 1, "max_j_hz": 8.0},
        ],
    },
    "methoxy_singlet": {
        "description": "a methoxy OCH3 singlet (3H, no coupling)",
        "shifts_ppm": [3.8, 3.8, 3.8],
        "couplings": [],
        "widths_hz": [1.0, 1.0, 1.0],
        "gt_groups": [
            {"shift_ppm": 3.8, "proton_count": 3, "max_j_hz": None},
        ],
    },
}


# --- Simulation + prediction -------------------------------------------------


def build_coupling_matrix(n_spins: int, pairs: list[tuple[int, int, float]]) -> NDArray[np.float64]:
    """Build a symmetric ``n_spins x n_spins`` Hz coupling matrix from ``(i, j, J)`` pairs.

    ``simulate`` reads only the upper triangle, but the matrix is filled symmetrically for clarity.
    """
    matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    for i, j, j_hz in pairs:
        matrix[i, j] = j_hz
        matrix[j, i] = j_hz
    return matrix


def _ppm_to_points(shift_ppm: float) -> float:
    """Point index of a ppm value on the simulation grid (index 0 = ``LEFT_PPM``)."""
    return (shift_ppm - LEFT_PPM) / (RIGHT_PPM - LEFT_PPM) * (N_POINTS - 1)


def simulate_phenotype(
    name: str, distort_kwargs: dict[str, Any] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[GTGroup]]:
    """Simulate a named phenotype (optionally distorted); return ``(amplitudes, ppm_axis, gt)``.

    ``amplitudes`` is the real (absorption) spectrum ready for the model; ``distort`` is applied
    only when ``distort_kwargs`` is non-empty, and the result is real-ified with ``np.real``.
    """
    if name not in PHENOTYPES:
        raise KeyError(f"unknown phenotype {name!r}; choose from {sorted(PHENOTYPES)}")
    pheno = PHENOTYPES[name]
    couplings = build_coupling_matrix(len(pheno["shifts_ppm"]), pheno["couplings"])
    spectrum, ppm_axis = simulate(
        pheno["shifts_ppm"],
        couplings,
        pheno["widths_hz"],
        BASE_FREQ_MHZ,
        LEFT_PPM,
        RIGHT_PPM,
        N_POINTS,
    )
    if distort_kwargs:
        spectrum = np.real(distort(spectrum, ppm_axis, **distort_kwargs))
    amplitudes = np.real(np.asarray(spectrum, dtype=np.float64))
    return amplitudes, ppm_axis, pheno["gt_groups"]


_MODEL_CACHE: dict[str, Any] = {}


def _get_model(checkpoint: str) -> Any:
    """Load (and cache) the model for ``checkpoint`` so repeated round-trips do not reload 973 MB."""
    if checkpoint not in _MODEL_CACHE:
        _MODEL_CACHE[checkpoint] = load_checkpoint(build_model(), checkpoint)
    return _MODEL_CACHE[checkpoint]


def predict(
    amplitudes: NDArray[np.float64], checkpoint: str, threshold: float = THRESHOLD
) -> list[dict[str, Any]]:
    """Run the checkpoint-only predict recipe on a real spectrum (identical to ``predict.py``)."""
    output = run(_get_model(checkpoint), amplitudes)
    preds: list[dict[str, Any]] = decode_predictions(
        output,
        load_extrema(EXTREMA_PATH),
        POINTS_PER_HZ,
        ppm_left=LEFT_PPM,
        ppm_right=RIGHT_PPM,
        threshold=threshold,
    )
    return preds


def round_trip(
    name: str, checkpoint: str, *, distort_kwargs: dict[str, Any] | None = None
) -> tuple[list[GTGroup], list[dict[str, Any]]]:
    """Simulate -> optional distort -> ``np.real`` -> predict; return ``(gt_groups, predictions)``."""
    amplitudes, _ppm_axis, gt_groups = simulate_phenotype(name, distort_kwargs)
    return gt_groups, predict(amplitudes, checkpoint)


# --- Ground-truth <-> prediction matching + reporting ------------------------


def match_to_gt(
    gt_groups: list[GTGroup], predictions: list[dict[str, Any]]
) -> list[tuple[GTGroup, dict[str, Any] | None]]:
    """Greedily pair each GT group with its nearest-δ prediction (each prediction used once)."""
    remaining = list(predictions)
    matched: list[tuple[GTGroup, dict[str, Any] | None]] = []
    for gt in gt_groups:
        if not remaining:
            matched.append((gt, None))
            continue
        target = gt["shift_ppm"]
        best = min(remaining, key=lambda p: abs(float(p["chemical_shift_ppm"]) - target))
        remaining.remove(best)
        matched.append((gt, best))
    return matched


def _comparison_row(gt: GTGroup, pred: dict[str, Any] | None) -> list[str]:
    """One formatted GT-vs-recovered table row (as strings)."""
    gt_j = "-" if gt["max_j_hz"] is None else f"{gt['max_j_hz']:.1f}"
    gt_cols = [f"{gt['shift_ppm']:.2f}", f"{gt['proton_count']}", gt_j]
    if pred is None:
        return [*gt_cols, "-", "-", "-", "-", "(no match)", "-", "-"]
    p_shift = float(pred["chemical_shift_ppm"])
    js = pred["coupling_constants_hz"]
    p_j = f"{float(js[0]):.1f}" if js else "-"
    d_shift = f"{abs(p_shift - gt['shift_ppm']):.3f}"
    d_h = f"{int(pred['proton_count']) - gt['proton_count']:+d}"
    d_j = "-" if (gt["max_j_hz"] is None or not js) else f"{abs(float(js[0]) - gt['max_j_hz']):.1f}"
    return [
        *gt_cols,
        f"{p_shift:.3f}",
        f"{int(pred['proton_count'])}",
        p_j,
        f"{float(pred['confidence']):.2f}",
        d_shift,
        d_h,
        d_j,
    ]


# ASCII headers (console-safe on Windows cp1252; the matplotlib plot keeps proper delta glyphs).
_HEADERS = [
    "GT ppm",
    "GT H",
    "GT J",
    "pred ppm",
    "pred H",
    "pred J",
    "conf",
    "|d ppm|",
    "dH",
    "|dJ|",
]


def print_comparison(
    name: str,
    matched: list[tuple[GTGroup, dict[str, Any] | None]],
    distort_kwargs: dict[str, Any] | None,
    n_preds: int,
) -> None:
    """Print the GT-vs-predicted table for one round-trip (all quantities in ppm / H / Hz)."""
    pheno = PHENOTYPES[name]
    print(f"\nPhenotype: {name} -- {pheno['description']}")
    print(f"Distortion: {distort_kwargs or 'none (clean simulation)'}")
    print(f"Detected {n_preds} multiplet(s); {len(matched)} GT group(s).\n")
    rows = [_comparison_row(gt, pred) for gt, pred in matched]
    widths = [max(len(_HEADERS[c]), *(len(r[c]) for r in rows)) for c in range(len(_HEADERS))]
    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    print(fmt.format(*_HEADERS))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))
    print("\n(ppm = chemical shift, H = proton count, J = max coupling in Hz; d = recovered - GT.)")


def _save_plot(
    amplitudes: NDArray[np.float64],
    predictions: list[dict[str, Any]],
    gt_groups: list[GTGroup],
    path: str,
    name: str,
) -> None:
    """Save a GT-overlaid annotated prediction plot via the shared MolDeTr plot style."""
    gt_overlay: list[dict[str, Any]] = [
        {"chemical_shift_in_points": _ppm_to_points(g["shift_ppm"])} for g in gt_groups
    ]
    plot_spectrum(
        amplitudes,
        predictions,
        ppm_left=LEFT_PPM,
        ppm_right=RIGHT_PPM,
        ground_truth=gt_overlay,
        title=f"MolDeTr simulate->predict: {name}",
        save_path=path,
    )
    print(f"\nSaved annotated plot to {path}")


# --- CLI ---------------------------------------------------------------------


def _distort_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    """Collect the supplied distortion knobs into ``distort`` kwargs (``None`` if none supplied)."""
    effects: dict[str, Any] = {
        "noise_snr_log10": args.snr,
        "phase0_deg": args.phase0,
        "phase1": args.phase1,
        "baseline": args.baseline,
        "sat_j_hz": args.sat_j,
        "sat_intensity": args.sat_intensity,
        "broaden_hz": args.broaden,
    }
    effects = {k: v for k, v in effects.items() if v is not None}
    if not effects:
        return None
    effects["seed"] = args.seed
    return effects


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Simulate a phenotype, predict it, compare to GT.")
    ap.add_argument("--phenotype", required=True, choices=sorted(PHENOTYPES))
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--snr", type=float, default=None, help="noise_snr_log10 (2.0-5.0)")
    ap.add_argument("--phase0", type=float, default=None, help="phase0_deg (|.| <= 8)")
    ap.add_argument("--phase1", type=float, default=None, help="first-order phase coefficient")
    ap.add_argument("--baseline", type=float, default=None, help="baseline tilt magnitude")
    ap.add_argument("--sat-j", type=float, default=None, help="13C-satellite J in Hz (40-220)")
    ap.add_argument(
        "--sat-intensity", type=float, default=None, help="13C-satellite intensity (0.005-0.015)"
    )
    ap.add_argument("--broaden", type=float, default=None, help="Gaussian broadening FWHM Hz (0-3)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="detection threshold")
    ap.add_argument("--seed", type=int, default=0, help="distortion RNG seed")
    ap.add_argument(
        "--plot", nargs="?", const="simulate_prediction.png", default=None, help="save a plot"
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Set MOLDETR_CHECKPOINT or place the weights in moldetr/model/ "
            "(Zenodo DOI 10.5281/zenodo.21217102)."
        )
    distort_kwargs = _distort_kwargs_from_args(args)
    amplitudes, _ppm_axis, gt_groups = simulate_phenotype(args.phenotype, distort_kwargs)
    preds = predict(amplitudes, args.checkpoint, threshold=args.threshold)
    matched = match_to_gt(gt_groups, preds)
    print_comparison(args.phenotype, matched, distort_kwargs, len(preds))
    if args.plot:
        _save_plot(amplitudes, preds, gt_groups, args.plot, args.phenotype)


if __name__ == "__main__":
    main()
