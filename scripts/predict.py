"""Run MolDeTr on a single 1D 1H NMR spectrum (checkpoint-only; no vendor NMR reader needed).

Examples
--------
    python scripts/predict.py --demo
    python scripts/predict.py --input path/to/roi.npz --base-freq-mhz 80.15

Weights (``model_spin_system_ABCDEFG_exp2.pth``) are on Zenodo (10.5281/zenodo.21217102);
place the file in ``moldetr/model/`` or pass ``--checkpoint``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from moldetr.inference import build_model, load_checkpoint, run
from moldetr.postprocess import decode_predictions, load_extrema
from moldetr.reproducibility import set_seed
from moldetr.validation import validate_spectrum
from moldetr.visualization import plot_spectrum

ROOT = Path(__file__).resolve().parent.parent


def load_input(path: str) -> tuple[np.ndarray, dict]:
    """Load a real 1D spectrum (+ ppm calibration if present) from .npz or .npy."""
    p = Path(path)
    cal: dict = {}
    if p.suffix == ".npz":
        data = np.load(
            p, allow_pickle=True
        )  # npz metadata is an object array (author's Zenodo file)
        # Prefer the per-point ppm axis (correct for the ROI window); fall back to metadata
        # bounds only if it is absent. The metadata left_ppm/right_ppm span the *full* spectrum,
        # not the ROI, so using them would mis-place every peak.
        if "ppm_axis_padded" in data:
            axis = np.asarray(data["ppm_axis_padded"], dtype=float)
            cal = {"ppm_left": float(axis[0]), "ppm_right": float(axis[-1])}
        elif "metadata" in data:
            md = data["metadata"].item()
            cal = {"ppm_left": md.get("left_ppm"), "ppm_right": md.get("right_ppm")}
        for key in ("spectrum_padded", "spec"):
            if key in data:
                return np.real(data[key]), cal
        return np.real(data[list(data.keys())[0]]), cal
    return np.real(np.load(p)), cal


def demo_spectrum(n: int = 6144) -> np.ndarray:
    """A synthetic spectrum (three Lorentzian multiplets) for a no-data smoke run."""
    x = np.arange(n)

    def lorentz(center, width, amp):
        return amp * width**2 / ((x - center) ** 2 + width**2)

    signal = lorentz(2000, 8, 1.0) + lorentz(2035, 8, 0.9) + lorentz(4000, 6, 1.2)
    return signal + 0.01 * np.random.RandomState(0).randn(n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict multiplets from one 1H NMR spectrum.")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", help="path to a .npz (spectrum_padded/spec) or .npy real spectrum"
    )
    source.add_argument("--demo", action="store_true", help="run on a synthetic spectrum")
    ap.add_argument(
        "--checkpoint",
        default=str(ROOT / "moldetr" / "model" / "model_spin_system_ABCDEFG_exp2.pth"),
    )
    ap.add_argument("--extrema", default=str(ROOT / "moldetr" / "assets" / "extrema.txt"))
    ap.add_argument("--points-per-hz", type=float, default=5.12)
    ap.add_argument(
        "--ppm-left", type=float, default=None, help="ppm at point 0 (enables ppm output)"
    )
    ap.add_argument("--ppm-right", type=float, default=None, help="ppm at the last point")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument(
        "--plot",
        nargs="?",
        const="prediction.png",
        default=None,
        help="save an annotated spectrum plot (default prediction.png)",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = ap.parse_args()
    set_seed(args.seed)

    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Download it from Zenodo (10.5281/zenodo.21217102) into moldetr/model/."
        )

    amplitudes, cal = (demo_spectrum(), {}) if args.demo else load_input(args.input)
    try:
        validate_spectrum(amplitudes, points_per_hz=args.points_per_hz)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model = load_checkpoint(build_model(), args.checkpoint)
    output = run(model, amplitudes)
    ppm_left = args.ppm_left if args.ppm_left is not None else cal.get("ppm_left")
    ppm_right = args.ppm_right if args.ppm_right is not None else cal.get("ppm_right")
    predictions = decode_predictions(
        output,
        load_extrema(args.extrema),
        args.points_per_hz,
        ppm_left=ppm_left,
        ppm_right=ppm_right,
        threshold=args.threshold,
    )

    print(f"Detected {len(predictions)} multiplet(s):")
    for i, pred in enumerate(predictions, 1):
        max_j = f"{pred['coupling_constants_hz'][0]:.2f}" if pred["coupling_constants_hz"] else "-"
        shift = (
            f"{pred['chemical_shift_ppm']:.3f} ppm"
            if pred["chemical_shift_ppm"] is not None
            else f"{pred['chemical_shift_hz']:.1f} Hz"
        )
        print(
            f"  #{i}: {pred['proton_count']}H  shift={shift}  max J={max_j} Hz  "
            f"linewidth={pred['linewidth_hz']:.2f} Hz  (conf {pred['confidence']:.2f})"
        )
    if predictions:
        print("  (max J = the largest coupling constant per multiplet; see docs/SCOPE.md)")

    if args.plot:
        plot_spectrum(
            amplitudes, predictions, ppm_left=ppm_left, ppm_right=ppm_right, save_path=args.plot
        )
        print(f"Saved annotated spectrum plot to {args.plot}")


if __name__ == "__main__":
    main()
