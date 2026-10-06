"""The experimental benchmark ROIs: load them in the frame their labels use, score like the paper.

**Use ``load_roi``, not ``np.load``, on the Zenodo ``roi_S*.npz`` files.** Each file carries two
copies of the spectrum, and the labels fit only one of them:

- ``spectrum_padded`` -- 6144 points, the model's input frame. ``ground_truth`` is indexed here.
- ``spectrum_raw`` -- the ROI before padding, 500-4190 points for nine of the thirteen ROIs, and for
  seven of them it starts ``metadata["padding_before"]`` (1000-2000) points into the padded frame.

Building the input from ``spectrum_raw`` and padding it at the end shifts every peak 195-390 Hz off
its label. Measured 2026-10-06 on the real checkpoint (13 ROIs x 5 noise seeds, scored with
:func:`match_hungarian`): 110/220 labels matched with 116 false positives, against 204/220 and 17 for
``spectrum_padded``. Taking the magnitude instead of the real part costs 105 false positives.

``match_hungarian`` is the matcher the article's numbers were produced with (transcribed from the
private training repo, ``multiplet_detection_detr/matching_4_experimental_evaluation.py:100-190``).
A nearest-shift matcher on the same predictions reports ~81 % proton accuracy instead of ~94 %,
because it pairs co-located labels (S2, S7) arbitrarily: the two are different metrics.

numpy + scipy only, so scoring never needs the torch extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from moldetr.labels import Multiplet, from_experimental

N_POINTS = 6144
POINTS_PER_HZ = 5.12
#: The article's matcher: pairs further apart than this never match (Hz).
GATE_HZ = 20.0
#: One proton-count mismatch costs as much as this many Hz of shift error.
PROTON_MISMATCH_HZ = 80.0


@dataclass(frozen=True)
class RoiData:
    """One benchmark ROI in the model's input frame. ``labels`` are indexed in that same frame."""

    name: str
    spectrum: NDArray[np.float64]
    labels: list[Multiplet]
    points_per_hz: float
    padding_before: int
    padding_after: int
    ppm_axis: NDArray[np.float64] | None = None
    compound: str | None = None


def load_roi(path: str | Path) -> RoiData:
    """Load a Zenodo benchmark ROI as the model must see it: the real part of ``spectrum_padded``.

    Unpickles ``metadata`` and ``ground_truth`` (object arrays), so this is for the trusted benchmark
    files from Zenodo 10.5281/zenodo.21217102 only -- never for user uploads; the GUI does not call it.
    Raises ``ValueError`` rather than falling back to ``spectrum_raw``, because that fallback IS the trap.
    """
    p = Path(path)
    with np.load(p, allow_pickle=True) as npz:
        if "spectrum_padded" not in npz.files:
            raise ValueError(
                f"{p.name}: no 'spectrum_padded'. The labels are indexed in the padded 6144-point "
                "frame; 'spectrum_raw' is offset by metadata['padding_before'] and must not be used."
            )
        spectrum = np.real(np.asarray(npz["spectrum_padded"])).astype(np.float64)
        if spectrum.shape != (N_POINTS,):
            raise ValueError(f"{p.name}: spectrum_padded has shape {spectrum.shape}, expected (6144,)")
        md: dict[str, Any] = npz["metadata"].item() if "metadata" in npz.files else {}
        gt = npz["ground_truth"].tolist() if "ground_truth" in npz.files else []
        axis = np.asarray(npz["ppm_axis_padded"], dtype=np.float64) if "ppm_axis_padded" in npz.files else None
    return RoiData(
        name=p.stem,
        spectrum=spectrum,
        labels=[from_experimental(g) for g in gt if isinstance(g, dict)],
        points_per_hz=float(md.get("points_per_hz", POINTS_PER_HZ)),
        padding_before=int(md.get("padding_before", 0)),
        padding_after=int(md.get("padding_after", 0)),
        ppm_axis=axis,
        compound=md.get("compound"),
    )


def coupling_errors(pred_cc: list[float], label_cc: list[float]) -> list[float]:
    """|ΔJ| per label coupling: both sides sorted by magnitude, zero labels ignored (Hz)."""
    true = sorted((abs(c) for c in label_cc if c not in (0, 0.0)), reverse=True)
    pred = sorted((abs(c) for c in pred_cc), reverse=True)[: len(true)]
    return [abs(t - p) for t, p in zip(true, pred)]


@dataclass
class MatchResult:
    """Outcome of matching one spectrum's predictions to its labels."""

    pairs: list[tuple[dict, Multiplet]] = field(default_factory=list)
    unmatched_predictions: list[dict] = field(default_factory=list)
    unmatched_labels: list[Multiplet] = field(default_factory=list)
    dshift_hz: list[float] = field(default_factory=list)
    dj_hz: list[float] = field(default_factory=list)
    n_correct: int = 0

    @property
    def n_matched(self) -> int:
        return len(self.pairs)

    @property
    def false_positives(self) -> int:
        return len(self.unmatched_predictions)

    @property
    def false_negatives(self) -> int:
        return len(self.unmatched_labels)


def _cost_matrix(
    preds: list[dict], labels: list[Multiplet], points_per_hz: float, gate_hz: float
) -> NDArray[np.float64]:
    cost = np.full((len(preds), len(labels)), np.inf)
    for i, p in enumerate(preds):
        for j, lab in enumerate(labels):
            d_hz = abs(float(p["chemical_shift_in_points"]) - lab.center_in_points) / points_per_hz
            if d_hz <= gate_hz:
                mismatch = float(int(p["proton_count"]) != lab.proton_count)
                cost[i, j] = mismatch + d_hz / PROTON_MISMATCH_HZ
    return cost


def match_hungarian(
    preds: list[dict],
    labels: list[Multiplet],
    points_per_hz: float = POINTS_PER_HZ,
    gate_hz: float = GATE_HZ,
) -> MatchResult:
    """The article's matcher: Hungarian assignment on proton mismatch + |Δδ| / 80 Hz, 20 Hz gate.

    ``preds`` are ``decode_predictions`` dicts. Inside the gate a correct proton count always wins
    over a closer shift, which is what lets co-located labels (one shared point, different H) pair up.
    """
    res = MatchResult()
    cost = _cost_matrix(preds, labels, points_per_hz, gate_hz)
    used_p: set[int] = set()
    used_l: set[int] = set()
    if cost.size and not np.isinf(cost).all():
        # scipy refuses an infeasible all-inf row/col, so gate with a large finite cost instead.
        rows, cols = linear_sum_assignment(np.where(np.isinf(cost), 1e9, cost))
        for i, j in zip(rows, cols):
            if np.isinf(cost[i, j]):
                continue
            p, lab = preds[i], labels[j]
            used_p.add(int(i))
            used_l.add(int(j))
            res.pairs.append((p, lab))
            shift = abs(float(p["chemical_shift_in_points"]) - lab.center_in_points)
            res.dshift_hz.append(shift / points_per_hz)
            res.n_correct += int(int(p["proton_count"]) == lab.proton_count)
            res.dj_hz.extend(
                coupling_errors(list(p.get("coupling_constants_hz", [])), lab.coupling_constants_hz)
            )
    res.unmatched_predictions = [p for k, p in enumerate(preds) if k not in used_p]
    res.unmatched_labels = [lab for k, lab in enumerate(labels) if k not in used_l]
    return res


@dataclass(frozen=True)
class RoiSummary:
    """Pooled outcome over several spectra (or noise seeds). NaN where nothing matched."""

    n_labels: int
    n_matched: int
    n_correct: int
    false_positives: int
    false_negatives: int
    median_dshift_hz: float
    median_dj_hz: float

    @property
    def proton_acc_per_label(self) -> float:
        """Correct proton counts over ALL labels: the paper's 198/220 = 90.0 % denominator."""
        return self.n_correct / self.n_labels if self.n_labels else float("nan")

    @property
    def proton_acc_per_match(self) -> float:
        """Correct proton counts over matched pairs: the paper's 92.1 % (Table 1(d))."""
        return self.n_correct / self.n_matched if self.n_matched else float("nan")


def summarize(results: list[MatchResult]) -> RoiSummary:
    """Pool ``match_hungarian`` results; medians are over all matched pairs, not per spectrum."""
    ds = [d for r in results for d in r.dshift_hz]
    dj = [d for r in results for d in r.dj_hz]
    n_matched = sum(r.n_matched for r in results)
    return RoiSummary(
        n_labels=n_matched + sum(r.false_negatives for r in results),
        n_matched=n_matched,
        n_correct=sum(r.n_correct for r in results),
        false_positives=sum(r.false_positives for r in results),
        false_negatives=sum(r.false_negatives for r in results),
        median_dshift_hz=float(np.median(ds)) if ds else float("nan"),
        median_dj_hz=float(np.median(dj)) if dj else float("nan"),
    )
