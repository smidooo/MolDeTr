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
