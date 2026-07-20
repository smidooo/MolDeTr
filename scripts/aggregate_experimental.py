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


def per_class_accuracy(matched_pairs: list, unmatched_labels: list | None = None) -> dict:
    """Per label proton-count class: {class: (n_correct, n_total)}.

    A label that was not matched counts as a miss for its class (the model predicted "no spin"
    there), mirroring the confusion-matrix methodology behind the paper's per-class numbers.
    """
    from collections import defaultdict

    total: dict = defaultdict(int)
    correct: dict = defaultdict(int)
    for pred, label in matched_pairs:
        c = label.get("proton_count")
        if c is None:
            continue
        total[c] += 1
        correct[c] += int(pred.get("proton_count") == c)
    for label in unmatched_labels or []:
        c = label.get("proton_count")
        if c is not None:
            total[c] += 1
    return {c: (correct[c], total[c]) for c in sorted(total)}


def regression_stats(matched_pairs: list) -> dict:
    """MAE (Hz) and R² for chemical shift and coupling from the matched pairs — the paper's Table 4.
    R² is scale-invariant, so shift is evaluated in Hz (points / POINTS_PER_HZ)."""
    ds_p: list[float] = []
    ds_l: list[float] = []
    j_p: list[float] = []
    j_l: list[float] = []
    for pred, label in matched_pairs:
        if "chemical_shift_in_points" in pred and "chemical_shift_in_points" in label:
            ds_p.append(pred["chemical_shift_in_points"] / POINTS_PER_HZ)
            ds_l.append(label["chemical_shift_in_points"] / POINTS_PER_HZ)
        lc = label.get("coupling_constants")
        if lc is not None:
            pc = pred.get("coupling_constants", [])
            lc = lc if isinstance(lc, list) else [lc]
            pc = pc if isinstance(pc, list) else [pc]
            for p, t in zip(pc[: len(lc)], lc):
                j_p.append(p)
                j_l.append(t)

    def _mae(pred: list[float], lab: list[float]) -> float:
        return statistics.fmean(abs(p - t) for p, t in zip(pred, lab)) if pred else float("nan")

    def _r2(pred: list[float], lab: list[float]) -> float:
        if len(lab) < 2:
            return float("nan")
        mean_l = statistics.fmean(lab)
        ss_res = sum((t - p) ** 2 for p, t in zip(pred, lab))
        ss_tot = sum((t - mean_l) ** 2 for t in lab)
        return 1 - ss_res / ss_tot if ss_tot else float("nan")

    return {
        "mae_dshift_hz": _mae(ds_p, ds_l),
        "r2_dshift": _r2(ds_p, ds_l),
        "mae_dJ_hz": _mae(j_p, j_l),
        "r2_dJ": _r2(j_p, j_l),
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
    unmatched_labels = data.get("unmatched_labels_total") if isinstance(data, dict) else None
    for cls, (cor, tot) in per_class_accuracy(pairs, unmatched_labels).items():
        print(f"    {cls}H proton-count accuracy = {100 * cor / tot:.1f} %  ({cor}/{tot})")
    reg = regression_stats(pairs)
    print(
        f"  MAE |dd| = {reg['mae_dshift_hz']:.3f} Hz, R^2 = {reg['r2_dshift']:.3f}   (paper 1.368 Hz, 0.999)"
    )
    print(
        f"  MAE |dJ| = {reg['mae_dJ_hz']:.3f} Hz, R^2 = {reg['r2_dJ']:.3f}   (paper 0.470 Hz, 0.936)"
    )
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
