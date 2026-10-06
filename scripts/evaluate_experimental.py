"""Evaluate MolDeTr on the experimental ROI test set (checkpoint + preprocessed ROIs only).

Loads each Zenodo ROI with ``moldetr.roi.load_roi`` (the real part of ``spectrum_padded``, the
frame the labels are indexed in -- never ``spectrum_raw``), runs the model under several noise
seeds, and scores with ``moldetr.roi.match_hungarian``, the matcher the article's numbers were
produced with. Regenerates predictions from the weights (unlike ``aggregate_experimental.py``,
which reads committed predictions and reproduces the article exactly).

Why several seeds: the article pooled ~5 noisy repeats of the 13 ROIs (215 matched pairs over 220
labels). One run is 44 labels, and its medians move by up to ~0.5 Hz between noise draws.

How this compares with the article (measured 2026-10-06, see docs/DATA_SCHEMA.md):
- coupling: comparable. Every label carries at most one J, and the article scored it against the
  largest predicted J, which is the one J this decode reports.
- decode: all 8 query groups pooled and merged within 20 points. The article read group 0 only,
  which matches more of the closely overlapped multiplets (S2, S7).
- matcher: the primary lines reproduce the article's matcher, including its quirk of discarding
  every match in a spectrum with both an unmatchable prediction and an unmatchable label. A
  second line gives the lenient result that keeps those valid pairs.

Requires the weights and the ROI npz from Zenodo (10.5281/zenodo.21217102).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from moldetr.inference import build_model, load_checkpoint, run
from moldetr.postprocess import decode_predictions, load_extrema
from moldetr.reproducibility import set_seed
from moldetr.roi import MatchResult, coupling_errors, load_roi, match_hungarian, summarize

ROOT = Path(__file__).resolve().parent.parent

#: Article Table 1(d) and the committed matched pairs (aggregate_experimental.py), per denominator.
PAPER_MEDIAN_DSHIFT_HZ = 0.89
PAPER_MEDIAN_DJ_HZ = 0.20
PAPER_PROTON_PER_MATCH = 92.1  # 198/215 matched pairs
PAPER_PROTON_PER_LABEL = 90.0  # 198/220 labels


def match_and_score(preds: list[dict], gts: list[dict], points_per_hz: float) -> dict:
    """Nearest-shift greedy matching. NOT the article's metric; kept as a labelled secondary.

    It pairs co-located labels (one shared point, different proton counts) arbitrarily and has no
    distance gate. On the same predictions it reads ~81 % of labels where the article's matcher
    reads 86.8 % of labels (93.6 % of matched pairs, the denominator Table 1(d) uses).
    """
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


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate on the experimental ROI test set (npz).")
    ap.add_argument("--structured-output", type=Path, default=ROOT / "structured_output")
    ap.add_argument(
        "--checkpoint",
        default=str(ROOT / "moldetr" / "model" / "model_spin_system_ABCDEFG_exp2.pth"),
    )
    ap.add_argument("--extrema", default=str(ROOT / "moldetr" / "assets" / "extrema.txt"))
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42, help="global RNG seed for reproducibility")
    ap.add_argument(
        "--seeds", type=int, default=5, help="noise draws per ROI (run()'s noise_seed 0..N-1)"
    )
    args = ap.parse_args()
    if args.seeds < 1:
        ap.error("--seeds must be at least 1")
    return args


Scores = dict[str, list[MatchResult]]


def _score_roi(model, extrema, path: str, args) -> tuple[Scores, int]:
    """Faithful + lenient Hungarian results for one ROI over all seeds, plus nearest-shift hits."""
    roi = load_roi(path)
    labels_as_dicts = [
        {
            "proton_count": lab.proton_count,
            "chemical_shift_in_points": lab.center_in_points,
            "coupling_constants": lab.coupling_constants_hz,
        }
        for lab in roi.labels
    ]
    scores: Scores = {"faithful": [], "lenient": []}
    greedy_correct = 0
    for noise_seed in range(args.seeds):
        out = run(model, roi.spectrum, noise_seed=noise_seed)
        preds = decode_predictions(out, extrema, roi.points_per_hz, threshold=args.threshold)
        for mode in scores:
            scores[mode].append(
                match_hungarian(
                    preds, roi.labels, roi.points_per_hz, article_faithful=mode == "faithful"
                )
            )
        greedy_correct += match_and_score(preds, labels_as_dicts, roi.points_per_hz)["correct"]
    s = summarize(scores["faithful"])
    print(
        f"  {roi.name}: {len(roi.labels)} spin systems x {args.seeds} seeds -> "
        f"{s.n_matched} matched, {s.n_correct} proton-count correct, "
        f"{s.false_positives} false positives, {s.false_negatives} missed"
    )
    return scores, greedy_correct


def _report(scores: Scores, greedy_correct: int, n_seeds: int) -> None:
    s = summarize(scores["faithful"])
    print(f"\nOverall ({s.n_labels} labels = spin systems x {n_seeds} noise seeds), paper matcher:")
    print(f"  median |dd| = {s.median_dshift_hz:.2f} Hz   (paper {PAPER_MEDIAN_DSHIFT_HZ} Hz)")
    print(
        f"  median |dJ| = {s.median_dj_hz:.2f} Hz   (paper {PAPER_MEDIAN_DJ_HZ} Hz; "
        "largest predicted J vs the label J, as the article scored it)"
    )
    print(
        f"  proton-count accuracy = {100 * s.proton_acc_per_match:.1f} % of matched pairs "
        f"(paper {PAPER_PROTON_PER_MATCH} %), {100 * s.proton_acc_per_label:.1f} % of labels "
        f"(paper {PAPER_PROTON_PER_LABEL} %)"
    )
    print(
        f"  matched {s.n_matched}/{s.n_labels}, false positives {s.false_positives}, "
        f"missed {s.false_negatives}"
    )
    lenient = summarize(scores["lenient"])
    print(
        "  [lenient matcher: keeps valid pairs in spectra the article's code zeroes] matched "
        f"{lenient.n_matched}/{lenient.n_labels}, false positives {lenient.false_positives}, "
        f"proton-count accuracy {100 * lenient.proton_acc_per_match:.1f} % of matched pairs, "
        f"median |dd| {lenient.median_dshift_hz:.2f} Hz"
    )
    print(
        f"  [secondary, NOT the paper's metric] nearest-shift matcher proton-count accuracy = "
        f"{100 * greedy_correct / s.n_labels:.1f} % of labels"
    )


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Download it from Zenodo (10.5281/zenodo.21217102) into moldetr/model/."
        )
    npz_files = sorted(glob.glob(str(args.structured_output / "roi_S*.npz")))
    if not npz_files:
        raise SystemExit(
            f"No roi_S*.npz found in {args.structured_output}. "
            "Download the ROI arrays from Zenodo (10.5281/zenodo.21217102)."
        )
    extrema = load_extrema(args.extrema)
    model = load_checkpoint(build_model(), args.checkpoint)
    scores: Scores = {"faithful": [], "lenient": []}
    greedy_correct = 0
    for path in npz_files:
        roi_scores, roi_greedy = _score_roi(model, extrema, path, args)
        for mode in scores:
            scores[mode] += roi_scores[mode]
        greedy_correct += roi_greedy
    _report(scores, greedy_correct, args.seeds)


if __name__ == "__main__":
    main()
