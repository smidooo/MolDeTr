"""Evaluate MolDeTr on the experimental ROI test set (checkpoint + preprocessed ROIs only).

Loads each preprocessed ROI array (``structured_output/roi_S*.npz``), runs the model,
decodes predictions, greedily matches them to the ground-truth spin systems, and reports
median |dd| (Hz), median |dJ| (Hz), and proton-count accuracy. Regenerates predictions from
the weights (unlike ``aggregate_experimental.py``, which reads committed predictions).

Requires the weights and the ROI npz from Zenodo (10.5281/zenodo.21217102). No vendor NMR
reader is needed -- the ROIs are already preprocessed to model-ready arrays.
"""

from __future__ import annotations

import argparse
import glob
import statistics
from pathlib import Path

import numpy as np

from moldetr.inference import build_model, load_checkpoint, run
from moldetr.postprocess import decode_predictions, load_extrema
from moldetr.reproducibility import set_seed

ROOT = Path(__file__).resolve().parent.parent


def coupling_errors(pred_cc: list[float], label_cc: list[float]) -> list[float]:
    true = sorted((abs(c) for c in label_cc if c not in (0, 0.0)), reverse=True)
    pred = sorted((abs(c) for c in pred_cc), reverse=True)[: len(true)]
    return [abs(t - p) for t, p in zip(true, pred)]


def match_and_score(preds: list[dict], gts: list[dict], points_per_hz: float) -> dict:
    """Greedy nearest-chemical-shift matching of predictions to ground-truth spin systems."""
    dshift: list[float] = []
    dj: list[float] = []
    correct = 0
    used: set[int] = set()
    for gt in gts:
        gt_pts = gt["chemical_shift_in_points"]
        best, best_d = None, float("inf")
        for i, pred in enumerate(preds):
            if i in used:
                continue
            d = abs(pred["chemical_shift_in_points"] - gt_pts)
            if d < best_d:
                best, best_d = i, d
        if best is None:
            continue
        used.add(best)
        pred = preds[best]
        dshift.append(abs(pred["chemical_shift_in_points"] - gt_pts) / points_per_hz)
        correct += int(pred["proton_count"] == gt["proton_count"])
        dj.extend(coupling_errors(pred["coupling_constants_hz"], gt.get("coupling_constants", [])))
    return {"dshift": dshift, "dj": dj, "correct": correct, "n": len(gts)}


def load_ground_truth(npz) -> list[dict]:
    gt = npz["ground_truth"]
    items = gt.tolist() if hasattr(gt, "tolist") else list(gt)
    out = []
    for it in items:
        if isinstance(it, dict):
            cc = it.get("coupling_constants", [])
            out.append(
                {
                    "proton_count": it.get("proton_count"),
                    "chemical_shift_in_points": it.get("chemical_shift_in_points"),
                    "coupling_constants": cc if isinstance(cc, (list, tuple)) else [cc],
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate on the experimental ROI test set (npz).")
    ap.add_argument("--structured-output", type=Path, default=ROOT / "structured_output")
    ap.add_argument(
        "--checkpoint",
        default=str(ROOT / "moldetr" / "model" / "model_spin_system_ABCDEFG_exp2.pth"),
    )
    ap.add_argument("--extrema", default=str(ROOT / "moldetr" / "assets" / "extrema.txt"))
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = ap.parse_args()
    set_seed(args.seed)

    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Download it from Zenodo (10.5281/zenodo.21217102) into moldetr/model/."
        )
    extrema = load_extrema(args.extrema)
    model = load_checkpoint(build_model(), args.checkpoint)

    all_dshift: list[float] = []
    all_dj: list[float] = []
    total_correct = total = 0
    npz_files = sorted(glob.glob(str(args.structured_output / "roi_S*.npz")))
    if not npz_files:
        raise SystemExit(
            f"No roi_S*.npz found in {args.structured_output}. "
            "Download the ROI arrays from Zenodo (10.5281/zenodo.21217102)."
        )
    for path in npz_files:
        npz = np.load(path, allow_pickle=True)
        pph = float(npz["metadata"].item().get("points_per_hz", 5.12))
        preds = decode_predictions(
            run(model, np.real(npz["spectrum_padded"])), extrema, pph, threshold=args.threshold
        )
        score = match_and_score(preds, load_ground_truth(npz), pph)
        all_dshift += score["dshift"]
        all_dj += score["dj"]
        total_correct += score["correct"]
        total += score["n"]
        print(
            f"  {Path(path).stem}: {score['n']} spin systems, "
            f"{score['correct']} proton-count correct, {len(preds)} predicted"
        )

    print(f"\nOverall ({total} spin systems):")
    print(f"  median |dd| = {statistics.median(all_dshift):.2f} Hz   (paper ~ 0.89 Hz)")
    print(f"  median |dJ| = {statistics.median(all_dj):.2f} Hz   (paper ~ 0.20 Hz)")
    print(f"  proton-count accuracy = {100 * total_correct / total:.1f} %   (paper ~ 93.5 %)")


if __name__ == "__main__":
    main()
