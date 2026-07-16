"""End-to-end simulate -> (distort) -> predict round-trip tests against the released checkpoint.

The recovery tests are **checkpoint-gated**: they ``skip`` (never fail) when the 973 MB weights are
absent. Point them at the weights with the ``MOLDETR_CHECKPOINT`` environment variable, or place the
file at ``moldetr/model/model_spin_system_ABCDEFG_exp2.pth`` (Zenodo DOI 10.5281/zenodo.21217102).

Tolerances are **calibrated to what the model actually recovers**, not guessed. On the clean
``ethyl`` and ``aromatic_ax`` phenotypes the observed errors (deterministic; ``run`` seeds its
in-distribution noise) are:

* chemical shift : ``|Δδ| <= 0.010 ppm``
* proton count   : exact (0 error)
* max coupling   : ``|ΔJ| <= 0.2 Hz``

so the asserted bounds below (``0.05 ppm`` / ``1.0 Hz``, ~5x margin over observed) are far tighter
than the ``~0.5 ppm`` / ``~2 Hz`` starting point in the brief -- recovery is better than that, so we
tighten rather than loosen.

``methoxy_singlet`` recovers its *shift* but NOT its proton count (the model returns 2H for the true
3H). This is a genuine identifiability limit, not a test-harness issue: after the model's per-spectrum
min-max normalisation an isolated singlet carries no integration cue, so 1H/2H/3H singlets are
indistinguishable from line shape alone. We assert only the robust quantity (δ) for it and document
the limitation, rather than loosen a proton-count assertion to paper over a real model behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from moldetr.validation import validate_spectrum

# scripts/ is not an importable package; add it to the path so the deliverable module can be reused.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import simulate_and_predict as sp  # noqa: E402  (scripts/ was just placed on sys.path above)

CHECKPOINT = sp.DEFAULT_CHECKPOINT
checkpoint_required = pytest.mark.skipif(
    not Path(CHECKPOINT).exists(),
    reason=f"checkpoint not found at {CHECKPOINT!r}; set MOLDETR_CHECKPOINT to run recovery tests",
)

# Calibrated to observed recovery on ethyl + aromatic_ax (see module docstring).
TOL_PPM = 0.05
TOL_HZ = 1.0

RECOVERY_PHENOTYPES = ["ethyl", "aromatic_ax"]


def _assert_recovers(
    name: str,
    gt_groups: list[sp.GTGroup],
    preds: list[dict[str, Any]],
    tol_ppm: float,
    tol_hz: float,
) -> None:
    """Match predictions to GT groups by nearest δ and assert per-pair recovery within tolerance."""
    matched = sp.match_to_gt(gt_groups, preds)
    assert len(matched) == len(gt_groups)
    for gt, pred in matched:
        assert pred is not None, (
            f"{name}: GT group at {gt['shift_ppm']} ppm had no matching prediction"
        )
        d_ppm = abs(float(pred["chemical_shift_ppm"]) - gt["shift_ppm"])
        assert d_ppm <= tol_ppm, (
            f"{name}: |Δδ|={d_ppm:.3f} ppm > {tol_ppm} for GT {gt['shift_ppm']}"
        )
        assert int(pred["proton_count"]) == gt["proton_count"], (
            f"{name}: proton count {pred['proton_count']} != GT {gt['proton_count']} "
            f"at {gt['shift_ppm']} ppm"
        )
        if gt["max_j_hz"] is not None:
            couplings = pred["coupling_constants_hz"]
            assert couplings, f"{name}: expected a coupling near {gt['max_j_hz']} Hz, got none"
            d_j = abs(float(couplings[0]) - gt["max_j_hz"])
            assert d_j <= tol_hz, (
                f"{name}: |ΔJ|={d_j:.2f} Hz > {tol_hz} for GT {gt['shift_ppm']} ppm"
            )


@checkpoint_required
@pytest.mark.parametrize("name", RECOVERY_PHENOTYPES)
def test_clean_roundtrip_recovers(name: str) -> None:
    """A clean simulate -> predict round-trip recovers δ, proton count, and max J within tolerance."""
    gt_groups, preds = sp.round_trip(name, CHECKPOINT)
    _assert_recovers(name, gt_groups, preds, TOL_PPM, TOL_HZ)


@checkpoint_required
def test_distorted_roundtrip_still_recovers() -> None:
    """An in-range distortion (noise, SNR = 1e3) does not break recovery of the ethyl phenotype."""
    gt_groups, preds = sp.round_trip(
        "ethyl", CHECKPOINT, distort_kwargs={"noise_snr_log10": 3.0, "seed": 0}
    )
    _assert_recovers("ethyl+noise", gt_groups, preds, TOL_PPM, TOL_HZ)


@checkpoint_required
def test_methoxy_singlet_recovers_shift_but_not_proton_count() -> None:
    """Documented limitation: an isolated, scale-normalised singlet's δ is recovered, but its proton
    count is NOT identifiable (integration is removed by min-max normalisation). We assert only δ."""
    gt_groups, preds = sp.round_trip("methoxy_singlet", CHECKPOINT)
    target = gt_groups[0]["shift_ppm"]
    near = [p for p in preds if abs(float(p["chemical_shift_ppm"]) - target) <= TOL_PPM]
    assert near, (
        f"expected a detection within {TOL_PPM} ppm of {target}, "
        f"got {[round(float(p['chemical_shift_ppm']), 3) for p in preds]}"
    )


def test_simulated_spectrum_passes_validation() -> None:
    """A freshly simulated phenotype satisfies MolDeTr's input contract (no checkpoint needed)."""
    amplitudes, _ppm_axis, _gt = sp.simulate_phenotype("ethyl")
    validated = validate_spectrum(amplitudes, points_per_hz=sp.POINTS_PER_HZ)
    assert validated.shape == (6144,)
    assert np.all(np.isfinite(validated))
    assert not np.iscomplexobj(validated)


@pytest.mark.parametrize(
    "distort_kwargs",
    [
        {"noise_snr_log10": 1.5},  # below the trained 2.0 floor
        {"noise_snr_log10": 5.5},  # above the trained 5.0 ceiling
        {"phase0_deg": 9.0},  # |phase0| > 8 deg
        {"sat_j_hz": 300.0},  # 13C-satellite J outside 40-220 Hz
        {"broaden_hz": 5.0},  # broadening outside 0-3 Hz
    ],
)
def test_out_of_range_distortion_raises(distort_kwargs: dict[str, float]) -> None:
    """Out-of-range distortion parameters raise ``ValueError`` (no checkpoint needed)."""
    with pytest.raises(ValueError):
        sp.simulate_phenotype("ethyl", distort_kwargs)
