"""Graded match status, replacing a single hardcoded 2 Hz threshold.

Three things were wrong with the old rule.

**It was binary.** `dd_hz <= 2.0` collapsed "0.3 Hz out" and "40 Hz out" into the same `~ off`, so
the table could not distinguish a prediction a chemist would accept from one that is simply wrong.

**It was conjunctive.** `"✓ match" if (dd_hz <= tol_hz and dh == 0)` forced `~ off` on a proton-count
mismatch *at zero shift error*, hiding a perfect δ behind an unrelated defect. Shift and proton count
are now reported independently -- `status` grades δ, and `ΔH` carries the count.

**One tolerance cannot serve both δ and J.** A fixed 2 Hz is loose for a 2 Hz meta coupling and
absurdly tight for a 130 Hz ¹³C satellite. Couplings use a hybrid instead,
`tol = max(FLOOR_HZ, FRACTION * J)`, so small couplings are judged on an absolute floor and large
ones proportionally. The crossover sits at 5 Hz, mid-range for ordinary H-H couplings.

The band edges are the maintainer's, not derived: δ ≤1 excellent · ≤2 good · ≤4 ok · ≤10 fair · >10
off (Hz), and `max(0.5 Hz, 0.10 x J)` for couplings. For reference, the paper's median |ΔJ| is
0.20 Hz, so the 0.5 Hz floor is ~2.5x a typical good prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app_ui.grading import (  # noqa: E402
    EXCELLENT,
    FAIR,
    GOOD,
    OFF,
    OK,
    coupling_tolerance_hz,
    grade_coupling,
    grade_shift,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dd_hz,expected",
    [
        (0.0, EXCELLENT),
        (1.0, EXCELLENT),  # inclusive upper edge
        (1.01, GOOD),
        (2.0, GOOD),
        (2.5, OK),
        (4.0, OK),
        (4.1, FAIR),
        (10.0, FAIR),
        (10.1, OFF),
        (504.0, OFF),  # the stubbed model's worst row
    ],
)
def test_shift_bands(dd_hz: float, expected: str):
    assert grade_shift(dd_hz) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "j_hz,expected_tol",
    [
        (0.0, 0.5),  # floor applies all the way down, including a singlet-ish 0
        (2.0, 0.5),  # meta / long-range -> floor
        (5.0, 0.5),  # the crossover: 0.10 * 5 == the 0.5 floor exactly
        (7.0, 0.7),  # typical vicinal -> proportional
        (12.0, 1.2),  # trans-alkene
        (130.0, 13.0),  # 1J C-H satellite
    ],
)
def test_coupling_tolerance_is_hybrid(j_hz: float, expected_tol: float):
    """Below 5 Hz the floor governs; above it the fraction does. That is the whole point."""
    assert coupling_tolerance_hz(j_hz) == pytest.approx(expected_tol)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dj_hz,gt_j_hz,expected",
    [
        # A 2 Hz meta coupling: tol 0.5, so the bands are 0.25 / 0.5 / 1.0 / 2.5 Hz.
        (0.2, 2.0, EXCELLENT),
        (0.5, 2.0, GOOD),
        (1.0, 2.0, OK),
        (2.5, 2.0, FAIR),
        (3.0, 2.0, OFF),
        # A 130 Hz satellite: tol 13.0, so 6.5 Hz out is still excellent -- the same absolute error
        # that is "off" on the meta coupling above. This is the hybrid earning its keep.
        (6.5, 130.0, EXCELLENT),
        (13.0, 130.0, GOOD),
        (70.0, 130.0, OFF),
    ],
)
def test_coupling_bands_scale_with_the_coupling(dj_hz: float, gt_j_hz: float, expected: str):
    assert grade_coupling(dj_hz, gt_j_hz) == expected


@pytest.mark.unit
def test_an_unknown_coupling_is_not_graded():
    """A singlet has no GT coupling, and the model emits 0 or 1 -- neither is a failure to report."""
    assert grade_coupling(None, 7.0) is None
    assert grade_coupling(0.3, None) is None


@pytest.mark.unit
def test_the_grades_are_ordered_and_distinct():
    """Guards the guard: the parametrised cases above are vacuous if two labels collide."""
    grades = [EXCELLENT, GOOD, OK, FAIR, OFF]
    assert len(set(grades)) == 5, f"grade labels are not distinct: {grades}"
