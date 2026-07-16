"""Decode raw model output into physical multiplet parameters.

The model emits, per query, ``num_classes`` class logits followed by ``num_params``
normalized regression parameters. This module turns that into human-readable spin-system
predictions: proton count, chemical shift, coupling constants, and line width.

Coupling note: the four "coupling" regression slots are a permutation-invariant embedding
``[sum, min, max, std]`` of the multiplet's couplings (see
:mod:`moldetr.dataloader.normalization`), not four independent constants. The paper-exact
``structured_output`` decode inverts the full embedding; this live-demo decode reports only the
single quantity the paper emphasizes as most reliable — the largest coupling, ``max(J)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Class index -> proton count, from conf mult_class_indices {1p, 2p, 3p, 4p, 6p}.
PROTON_COUNTS = [1, 2, 3, 4, 6]
# Regression-parameter order (must match the extrema.txt key order). The last four slots are NOT
# four couplings: they are the permutation-invariant embedding [sum, min, max, std] of the
# multiplet's couplings. Index 5 (coupling_constant_3) is the 'max' slot, i.e. max(J).
PARAM_NAMES = [
    "center_position_in_points",
    "line_width_in_points",
    "bounding_box_range_in_points",
    "coupling_constant_1_in_points",
    "coupling_constant_2_in_points",
    "coupling_constant_3_in_points",
    "coupling_constant_4_in_points",
]


def load_extrema(path: str | Path) -> dict:
    """Load the per-parameter ``[min, max]`` normalization extrema."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _untransform(normed, extrema: dict) -> list[float]:
    bounds = list(extrema.values())
    return [float(normed[i]) * (b[1] - b[0]) + b[0] for i, b in enumerate(bounds)]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def decode_predictions(
    output,
    extrema: dict,
    points_per_hz: float,
    ppm_left: float | None = None,
    ppm_right: float | None = None,
    n_points: int = 6144,
    num_classes: int = 5,
    threshold: float = 0.3,
    min_coupling_hz: float = 0.5,
    merge_tol_points: float = 20.0,
) -> list[dict]:
    """Decode a ``(num_queries, num_classes + num_params)`` output block into predictions.

    Queries whose maximum class probability is below ``threshold`` are treated as
    "no object" and dropped. Chemical shift and couplings are converted from model points
    to Hz using ``points_per_hz`` (and to ppm if ``ppm_left``/``ppm_right`` are given).
    """
    output = np.asarray(output.detach().cpu()) if hasattr(output, "detach") else np.asarray(output)
    results = []
    for query in output:
        logits, params = query[:num_classes], query[num_classes:]
        probs = _sigmoid(logits)
        if probs.max() <= threshold:
            continue
        phys = _untransform(params, extrema)
        center_pts, lw_pts = phys[0], phys[1]
        shift_hz = center_pts / points_per_hz
        # phys[3:7] is the [sum, min, max, std] coupling embedding, not four couplings.
        # Report only the physical largest coupling: the 'max' slot (phys[5]).
        max_coupling_hz = phys[5] / points_per_hz
        couplings_hz = [max_coupling_hz] if max_coupling_hz > min_coupling_hz else []
        results.append(
            {
                "proton_count": PROTON_COUNTS[int(np.argmax(logits))],
                "confidence": float(probs.max()),
                "chemical_shift_in_points": center_pts,
                "chemical_shift_hz": shift_hz,
                "chemical_shift_ppm": (
                    ppm_left + (center_pts / (n_points - 1)) * (ppm_right - ppm_left)
                )
                if (ppm_left is not None and ppm_right is not None)
                else None,
                "linewidth_hz": lw_pts / points_per_hz,
                "coupling_constants_hz": sorted(couplings_hz, reverse=True),
            }
        )
    return _merge(results, merge_tol_points)


def _merge(preds: list[dict], tol_points: float) -> list[dict]:
    """Greedy non-max suppression: collapse near-duplicate detections (from the query
    groups) that share a chemical shift within ``tol_points``, keeping the most confident."""
    kept: list[dict] = []
    for p in sorted(preds, key=lambda r: r["confidence"], reverse=True):
        if all(
            abs(p["chemical_shift_in_points"] - k["chemical_shift_in_points"]) > tol_points
            for k in kept
        ):
            kept.append(p)
    return sorted(kept, key=lambda r: r["chemical_shift_in_points"])
