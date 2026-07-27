"""Ground-truth groups must read their coupling from the matrix, not from one global J.

`_build_gt_groups` used to take a scalar `j_hz` and stamp it on every coupled group. That was
survivable only because the UI forced a single J on every pair. With a real coupling matrix there is
no single J, and a stale scalar would put a number in the GT column that has nothing to do with the
simulated spectrum — which then propagates silently into every match status in the comparison table.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_spins_sharing_a_shift_are_one_group_even_across_spin_systems(app_module) -> None:
    """Grouping is by shift, not by coupling block — and that is the deliberate choice.

    A review suggested keying on `coupling_blocks` so that two protons at the same δ in independent
    systems stay two 1H multiplets. Measuring the spectra rejected it. Ground truth here is compared
    against a detector that sees only the spectrum, and co-located protons produce a single peak
    carrying their combined area:

        3 equivalent uncoupled protons at 3.8 (a methoxy)  -> 1 peak, area 3.00
        2 coupled + 1 free, all at 3.5                     -> 1 peak, area 3.00

    Block-based grouping reports three 1H multiplets for the first and 2H + 1H for the second, where
    the spectrum shows one 3H line in both. That is strictly worse, so the shift-based grouping
    stands.

    Known limitation, kept here so it is not rediscovered as a bug: when co-located spins have
    *different* multiplet structure — a doublet overlapping a singlet at the same δ — one merged
    group with a single max J describes them poorly. Blocking does not fix that case well either
    (it would be right there and wrong above), so it is left alone until the UI can produce it.
    """
    methoxy = app_module._build_gt_groups([3.8, 3.8, 3.8], np.zeros((3, 3)))
    assert [(g["shift_ppm"], g["proton_count"]) for g in methoxy] == [(3.8, 3)]

    j = np.zeros((3, 3))
    j[0, 1] = j[1, 0] = 12.0  # spins 0 and 1 are one coupled system; spin 2 is independent
    mixed = app_module._build_gt_groups([3.5, 3.5, 3.5], j)
    assert [(g["shift_ppm"], g["proton_count"]) for g in mixed] == [(3.5, 3)]


@pytest.mark.unit
def test_a_coupling_written_below_the_diagonal_is_not_read(app_module) -> None:
    """GT must read the same half of the matrix as `simulate` and `coupling_blocks` do.

    All three now take the upper triangle as the contract. If GT scanned whole rows instead, a
    matrix editor that fills only above the diagonal would give a ground-truth J for a coupling the
    simulated spectrum does not contain — the GT column would describe a different spin system from
    the one plotted beside it.
    """
    below_only = np.zeros((2, 2))
    below_only[1, 0] = 8.0

    groups = app_module._build_gt_groups([7.5, 6.9], below_only)

    assert [g["max_j_hz"] for g in groups] == [None, None]


@pytest.mark.unit
def test_each_group_reports_its_own_largest_coupling(app_module) -> None:
    """Two groups with different couplings must not report the same J."""
    shifts = [7.5, 6.9, 1.2]
    j = np.zeros((3, 3))
    j[0, 1] = j[1, 0] = 8.0  # aromatic pair
    j[1, 2] = j[2, 1] = 2.0  # smaller coupling to the methyl

    groups = app_module._build_gt_groups(shifts, j)
    by_shift = {g["shift_ppm"]: g for g in groups}

    assert by_shift[7.5]["max_j_hz"] == pytest.approx(8.0)
    assert by_shift[6.9]["max_j_hz"] == pytest.approx(8.0)  # its largest, not its only
    assert by_shift[1.2]["max_j_hz"] == pytest.approx(2.0)


@pytest.mark.unit
def test_coupling_inside_an_equivalent_group_is_not_reported(app_module) -> None:
    """Equivalent protons may carry a mutual J, but it produces no observable splitting.

    Reporting it would claim a multiplet the spectrum does not show.
    """
    shifts = [1.2, 1.2, 1.2]
    j = np.zeros((3, 3))
    j[0, 1] = j[1, 0] = 12.0  # intra-methyl, unobservable
    j[0, 2] = j[2, 0] = 12.0
    j[1, 2] = j[2, 1] = 12.0

    (group,) = app_module._build_gt_groups(shifts, j)
    assert group["proton_count"] == 3
    assert group["max_j_hz"] is None


@pytest.mark.unit
def test_uncoupled_group_reports_no_coupling(app_module) -> None:
    (group,) = app_module._build_gt_groups([3.8, 3.8, 3.8], np.zeros((3, 3)))
    assert group["proton_count"] == 3
    assert group["max_j_hz"] is None


@pytest.mark.unit
def test_groups_are_ordered_downfield_first(app_module) -> None:
    """Matches how a chemist reads a spectrum, and how the comparison table is numbered."""
    j = np.zeros((3, 3))
    groups = app_module._build_gt_groups([1.2, 7.5, 3.5], j)
    assert [g["shift_ppm"] for g in groups] == [7.5, 3.5, 1.2]


@pytest.mark.unit
def test_ethyl_still_reports_seven_hertz_for_both_groups(app_module) -> None:
    """The pre-existing behaviour survives: with one J on every pair, nothing changes."""
    shifts = [1.2, 1.2, 1.2, 3.5, 3.5]
    j = np.zeros((5, 5))
    for i in (0, 1, 2):
        for k in (3, 4):
            j[i, k] = j[k, i] = 7.0

    groups = app_module._build_gt_groups(shifts, j)
    assert [(g["shift_ppm"], g["proton_count"], g["max_j_hz"]) for g in groups] == [
        (3.5, 2, 7.0),
        (1.2, 3, 7.0),
    ]
