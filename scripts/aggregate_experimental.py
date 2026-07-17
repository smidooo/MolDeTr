"""Reproduce the paper's headline experimental error medians from committed data.

Reads the Hungarian-matched prediction/label pairs in
``structured_output/experimental_matched_pairs.json`` -- the exact intermediate the article's
evaluation produced -- and reports median |dd| (Hz), median |dJ| (Hz), and proton-count accuracy,
reproducing the article's headline numbers from committed data alone (no checkpoint, GPU, or raw
spectra required).

Methodology mirrors the article's ``extract_predictions_and_targets``: chemical-shift error is
converted to Hz via ``points_per_hz = 5.12``; coupling error pairs the first ``len(label)``
predicted couplings (in their original order) with the label couplings.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POINTS_PER_HZ = 5.12


def coupling_errors(pred_cc, label_cc) -> list[float]:
    """First ``len(label)`` predicted couplings, in order, vs the label couplings (no sort/filter)."""
    label = label_cc if isinstance(label_cc, list) else [label_cc]
    pred = pred_cc if isinstance(pred_cc, list) else [pred_cc]
    return [abs(p - t) for p, t in zip(pred[: len(label)], label)]


def aggregate(matched_pairs: list) -> dict:
    dshift: list[float] = []
    dj: list[float] = []
    correct = 0
    for pred, label in matched_pairs:
        if "chemical_shift_in_points" in pred and "chemical_shift_in_points" in label:
            dshift.append(
                abs(pred["chemical_shift_in_points"] - label["chemical_shift_in_points"])
                / POINTS_PER_HZ
            )
        correct += int(pred.get("proton_count") == label.get("proton_count"))
        if label.get("coupling_constants") is not None:
            dj.extend(
                coupling_errors(pred.get("coupling_constants", []), label["coupling_constants"])
            )
    return {
        "n_pairs": len(matched_pairs),
        "n_couplings": len(dj),
        "median_abs_dshift_hz": statistics.median(dshift) if dshift else float("nan"),
        "median_abs_dJ_hz": statistics.median(dj) if dj else float("nan"),
        "proton_count_accuracy": correct / len(matched_pairs) if matched_pairs else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproduce the experimental error medians from committed matched pairs."
    )
    ap.add_argument(
        "--matched-pairs",
        type=Path,
        default=ROOT / "structured_output" / "experimental_matched_pairs.json",
    )
    ap.add_argument(
        "--total-queries",
        type=int,
        default=13 * 10 * 5,
        help="total model queries across the test set (for the DETR-style overall accuracy)",
    )
    ap.add_argument("--json", type=Path, default=None, help="also write metrics as JSON")
    args = ap.parse_args()
    if not args.matched_pairs.exists():
        raise SystemExit(
            f"Matched-pairs file not found: {args.matched_pairs}\n"
            "Pass --matched-pairs <path>, or run from the repo root where "
            "structured_output/experimental_matched_pairs.json is committed."
        )
    data = json.loads(args.matched_pairs.read_text(encoding="utf-8"))
    pairs = data["matched_pairs_total"] if isinstance(data, dict) else data
    result = aggregate(pairs)

    # Overall (DETR-style) accuracy: matched-correct + correctly-predicted empty ("no spin")
    # queries, over all queries -- the article's headline proton-count accuracy.
    matched_correct = round(result["proton_count_accuracy"] * result["n_pairs"])
    n_unmatched_pred = (
        len(data.get("unmatched_predictions_total", [])) if isinstance(data, dict) else 0
    )
    n_unmatched_label = len(data.get("unmatched_labels_total", [])) if isinstance(data, dict) else 0
    no_spin_correct = args.total_queries - result["n_pairs"] - n_unmatched_pred - n_unmatched_label
    overall_acc = (matched_correct + no_spin_correct) / args.total_queries

    print(f"Matched spin-system pairs: {result['n_pairs']} | couplings: {result['n_couplings']}")
    print(f"  median |dd| = {result['median_abs_dshift_hz']:.2f} Hz   (paper 0.89 Hz)")
    print(f"  median |dJ| = {result['median_abs_dJ_hz']:.2f} Hz   (paper 0.20 Hz)")
    print(f"  proton-count accuracy (overall) = {100 * overall_acc:.1f} %   (paper 93.5 %)")
    print(f"  proton-count accuracy (matched) = {100 * result['proton_count_accuracy']:.1f} %")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "median_abs_dshift_hz": result["median_abs_dshift_hz"],
                    "median_abs_dJ_hz": result["median_abs_dJ_hz"],
                    "proton_count_overall": overall_acc,
                    "proton_count_matched": result["proton_count_accuracy"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  wrote metrics to {args.json}")


if __name__ == "__main__":
    main()
