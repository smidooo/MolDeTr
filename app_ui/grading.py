"""How close is close enough — graded, and separately for shift and coupling.

`app.py` and `app_ui.plotting` both judged a prediction against a hardcoded ``tol_hz = 2.0``. Three
copies of the same literal, agreeing by coincidence rather than by construction, and answering the
wrong question three ways:

* **binary** — 0.3 Hz out and 40 Hz out both read ``~ off``;
* **conjunctive** — ``dd_hz <= tol and dh == 0`` forced ``~ off`` on a proton-count mismatch *at zero
  shift error*, hiding a perfect δ behind an unrelated defect;
* **one tolerance for two quantities** — 2 Hz is loose for a 2 Hz meta coupling and absurdly tight
  for a 130 Hz ¹³C satellite.

So: `status` grades the **shift** and `ΔH` carries the proton count, reported independently. And
couplings get a hybrid tolerance rather than a fixed one.

Band edges are the maintainer's decision, not derived from the data. For calibration, the paper's
median |ΔJ| is 0.20 Hz, so the 0.5 Hz coupling floor is roughly 2.5x a typical good prediction.
"""

from __future__ import annotations

#: Ordered best-to-worst. The leading glyph keeps the existing table vocabulary readable at a glance
#: — ✓ usable, ~ marginal, ✗ wrong — while the word carries the grade.
EXCELLENT = "✓ excellent"
GOOD = "✓ good"
OK = "✓ ok"
FAIR = "~ fair"
OFF = "✗ off"

#: Unchanged from the binary scheme: these are not grades, they are the absence of a pairing.
MISSED = "✗ missed"
EXTRA = "+ extra"

#: Chemical shift, in **Hz** (field-independent, so the same bands hold at 80 and 600 MHz).
#: Upper edges, inclusive. Anything past the last one is :data:`OFF`.
SHIFT_BANDS_HZ: tuple[tuple[float, str], ...] = (
    (1.0, EXCELLENT),
    (2.0, GOOD),
    (4.0, OK),
    (10.0, FAIR),
)

#: Coupling tolerance is ``max(FLOOR, FRACTION * J)``. The two meet at J = 5 Hz: below that the floor
#: governs, above it the fraction does. 5 Hz is mid-range for ordinary H-H couplings, so small
#: couplings are judged absolutely and satellites proportionally — which is the point of the hybrid.
COUPLING_FLOOR_HZ = 0.5
COUPLING_FRACTION = 0.10

#: Coupling bands, as multiples of that tolerance. Same shape as the shift bands so the two columns
#: read on one scale.
COUPLING_RATIO_BANDS: tuple[tuple[float, str], ...] = (
    (0.5, EXCELLENT),
    (1.0, GOOD),
    (2.0, OK),
    (5.0, FAIR),
)


def _band(value: float, bands: tuple[tuple[float, str], ...]) -> str:
    for edge, label in bands:
        if value <= edge:
            return label
    return OFF


def grade_shift(dd_hz: float) -> str:
    """Grade a chemical-shift error given in **Hz**, not ppm."""
    return _band(abs(dd_hz), SHIFT_BANDS_HZ)


def coupling_tolerance_hz(j_hz: float) -> float:
    """The tolerance a coupling of ``j_hz`` is judged against.

    Hybrid rather than fixed or purely proportional: a fixed tolerance is meaningless across the
    0.5–200 Hz range real couplings span, and a purely proportional one collapses to nothing for a
    small coupling — a 2 Hz meta coupling would get a 0.2 Hz window it could never earn.
    """
    return max(COUPLING_FLOOR_HZ, COUPLING_FRACTION * abs(j_hz))


def grade_coupling(dj_hz: float | None, gt_j_hz: float | None) -> str | None:
    """Grade a coupling error, or ``None`` when there is nothing to grade.

    ``None`` is a real outcome, not a failure: a singlet has no ground-truth coupling, and the model
    emits **0 or 1** couplings by construction (``PARAM_NAMES[3:7]`` are ``[sum, min, max, std]`` of
    the multiset, not four separate J values). Neither case should be scored as a miss.
    """
    if dj_hz is None or gt_j_hz is None:
        return None
    return _band(abs(dj_hz) / coupling_tolerance_hz(gt_j_hz), COUPLING_RATIO_BANDS)


def is_usable(grade: str | None) -> bool:
    """Whether a grade is one a reader would act on — drives the connector colour in the figure."""
    return grade in (EXCELLENT, GOOD, OK)
