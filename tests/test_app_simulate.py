"""Simulate tab — pure helpers + `app.simulate_and_detect` matrix (stubbed model, weight-free)."""

from __future__ import annotations

from pathlib import Path

import pytest


# --- _build_gt_groups ----------------------------------------------------------------------------


@pytest.mark.unit
def test_build_gt_groups_ethyl(app_module):
    pheno = app_module.sp.PHENOTYPES["ethyl"]
    j = app_module.sp.build_coupling_matrix(len(pheno["shifts_ppm"]), pheno["couplings"])
    groups = app_module._build_gt_groups(pheno["shifts_ppm"], j)
    # two equivalence groups (3H @1.2, 2H @3.5), sorted high→low ppm, both coupled
    assert [(g["shift_ppm"], g["proton_count"], g["max_j_hz"]) for g in groups] == [
        (3.5, 2, 7.0),
        (1.2, 3, 7.0),
    ]


@pytest.mark.unit
def test_build_gt_groups_singlet_has_no_coupling(app_module):
    pheno = app_module.sp.PHENOTYPES["methoxy_singlet"]
    j = app_module.sp.build_coupling_matrix(len(pheno["shifts_ppm"]), pheno["couplings"])
    (group,) = app_module._build_gt_groups(pheno["shifts_ppm"], j)
    assert group == {"shift_ppm": 3.8, "proton_count": 3, "max_j_hz": None}


# --- simulate_and_detect (stubbed, matrix-driven) -------------------------------------------------


@pytest.mark.unit
def test_simulate_checkpoint_absent(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "CHECKPOINT", str(Path("nope.pth")))
    grid, widths = app_module._phenotype_grid("ethyl")
    _t, _f, msg = app_module.simulate_and_detect(
        grid, widths, False, 3.0, 0.0, 0.0, 0.0, 0.3, False, 130.0
    )
    assert "Checkpoint not found" in msg


@pytest.mark.unit
def test_simulate_ethyl_roundtrip(patch_model):
    app = patch_model
    grid, widths = app._phenotype_grid("ethyl")
    table, fig, msg = app.simulate_and_detect(
        grid, widths, False, 3.0, 0.0, 0.0, 0.0, 0.3, False, 130.0
    )
    assert "2 ground-truth multiplet(s)" in msg and "detected" in msg
    assert fig is not None
    # new comparison table: match status + explicit error columns; one non-spurious row per GT group,
    # plus any spurious (+ extra) detection rows.
    expected = {
        "status",
        "GT δ (ppm)",
        "GT H",
        "GT J (Hz)",
        "pred δ (ppm)",
        "pred H",
        "Δδ (Hz)",
        "ΔH",
        "conf",
    }
    assert expected <= set(table.columns)
    assert len(table[table["status"] != "+ extra"]) == 2  # one per GT group


@pytest.mark.unit
def test_simulate_reports_a_bad_matrix_cell(patch_model):
    """A typo in the grid names the cell instead of taking the tab down."""
    app = patch_model
    grid, widths = app._phenotype_grid("aromatic_ax")
    grid[0][1] = "foo"
    _t, _f, msg = app.simulate_and_detect(
        grid, widths, False, 3.0, 0.0, 0.0, 0.0, 0.3, False, 130.0
    )
    assert msg.startswith("Invalid spin matrix:") and "row 1" in msg


@pytest.mark.unit
def test_simulate_nonpositive_width(patch_model):
    app = patch_model
    grid, widths = app._phenotype_grid("ethyl")
    widths[0][3] = 0.0  # FWHM is the last column: system | δ | n H | FWHM
    _t, _f, msg = app.simulate_and_detect(
        grid, widths, False, 3.0, 0.0, 0.0, 0.0, 0.3, False, 130.0
    )
    assert msg.startswith("Invalid parameters:") and "line width must be positive" in msg


# --- _comparison_dataframe: the status branches the round-trip test never reaches ----------------
#
# `simulate_and_detect` with the 3-detection fake always lands on ✓/~ rows, so `✗ missed` and
# `+ extra` — the two statuses a user reads as "the model got this wrong" — were never constructed.
# Calling the helper directly is the only way to drive them deterministically.

_GT_AROMATIC = {"shift_ppm": 7.5, "proton_count": 1, "max_j_hz": 8.0}
_GT_METHYL = {"shift_ppm": 1.2, "proton_count": 3, "max_j_hz": 7.0}


def _pred(shift_ppm: float, protons: int = 1, confidence: float = 0.9) -> dict:
    return {
        "chemical_shift_ppm": shift_ppm,
        "proton_count": protons,
        "confidence": confidence,
    }


@pytest.mark.unit
def test_comparison_dataframe_marks_unmatched_gt_as_missed(app_module):
    """More GT groups than predictions → the leftover GT row is `✗ missed` with a dashed pred side."""
    df = app_module._comparison_dataframe([_GT_AROMATIC, _GT_METHYL], [_pred(7.5)])

    assert list(df["status"]) == ["✓ match", "✗ missed"]
    missed = df.iloc[1]
    assert list(missed[["pred δ (ppm)", "pred H", "Δδ (Hz)", "ΔH", "conf"]]) == ["–"] * 5
    assert missed["GT δ (ppm)"] == "1.20" and missed["GT H"] == 3  # GT side stays populated


@pytest.mark.unit
def test_comparison_dataframe_appends_spurious_predictions_as_extra(app_module):
    """More predictions than GT groups → the leftover prediction is `+ extra`, numbering unbroken."""
    df = app_module._comparison_dataframe(
        [_GT_AROMATIC], [_pred(7.5), _pred(2.0, protons=2, confidence=0.42)]
    )

    assert list(df["status"]) == ["✓ match", "+ extra"]
    extra = df.iloc[1]
    assert list(extra[["GT δ (ppm)", "GT H", "GT J (Hz)"]]) == ["–", "–", "–"]
    assert (extra["pred δ (ppm)"], extra["pred H"], extra["conf"]) == ("2.000", 2, "0.42")
    assert list(df["#"]) == [1, 2]  # spurious rows continue the GT numbering, not restart it


@pytest.mark.unit
def test_comparison_dataframe_right_shift_wrong_proton_count_is_off_not_match(app_module):
    """`✓ match` needs BOTH |Δδ| ≤ tol and ΔH == 0 — an exact shift with the wrong H is `~ off`."""
    row = app_module._comparison_dataframe([_GT_AROMATIC], [_pred(7.5, protons=2)]).iloc[0]
    assert (row["status"], row["Δδ (Hz)"], row["ΔH"]) == ("~ off", "0.00", "+1")


@pytest.mark.unit
def test_comparison_dataframe_never_reports_missed_and_extra_together(app_module):
    """`match_to_gt` is a greedy *full* assignment with no distance cutoff, so a GT is paired with
    the nearest remaining prediction however far away it sits.

    Consequence, pinned here because the status vocabulary implies otherwise: a GT whose real
    detection is missing, alongside a spurious detection elsewhere, renders as a single `~ off`
    row — never `✗ missed` + `+ extra`. If the matcher ever grows a cutoff, this is the test that
    should be changed deliberately rather than discovered by surprise.
    """
    df = app_module._comparison_dataframe([_GT_AROMATIC], [_pred(1.0)])  # 520 Hz away, same H
    assert list(df["status"]) == ["~ off"]
    assert df.iloc[0]["Δδ (Hz)"] == "520.00"


@pytest.mark.unit
def test_simulate_singlet_gt_j_dashed(patch_model):
    app = patch_model
    grid, widths = app._phenotype_grid("methoxy_singlet")
    table, _f, msg = app.simulate_and_detect(
        grid, widths, False, 3.0, 0.0, 0.0, 0.0, 0.3, False, 130.0
    )
    assert "1 ground-truth multiplet(s)" in msg
    # the singlet's one GT group has no coupling -> its GT J renders as a dash
    gt_rows = table[table["status"] != "+ extra"]
    assert list(gt_rows["GT J (Hz)"]) == ["–"]
