"""Simulate tab — pure helpers + `app.simulate_and_detect` matrix (stubbed model, weight-free)."""

from __future__ import annotations

from pathlib import Path

import pytest


# --- _phenotype_defaults -------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,expected",
    [
        ("ethyl", ("1.2, 1.2, 1.2, 3.5, 3.5", 7.0, 1.0)),
        ("aromatic_ax", ("7.5, 6.9", 8.0, 1.0)),
        ("methoxy_singlet", ("3.8, 3.8, 3.8", 0.0, 1.0)),
    ],
)
def test_phenotype_defaults(app_module, name, expected):
    assert app_module._phenotype_defaults(name) == expected


# --- _parse_spin_shifts --------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_shifts_valid_and_separators(app_module):
    assert app_module._parse_spin_shifts("1.0, 2.0, 3.0", 3, [0.0, 0.0, 0.0]) == [1.0, 2.0, 3.0]
    assert app_module._parse_spin_shifts("1.0 2.0", 2, [0.0, 0.0]) == [1.0, 2.0]
    assert app_module._parse_spin_shifts("1.0; 2.0", 2, [0.0, 0.0]) == [1.0, 2.0]


@pytest.mark.unit
def test_parse_shifts_blank_uses_default(app_module):
    assert app_module._parse_spin_shifts("   ", 2, [4.0, 5.0]) == [4.0, 5.0]


@pytest.mark.unit
def test_parse_shifts_wrong_count_raises(app_module):
    with pytest.raises(ValueError, match=r"expected 3 shift value\(s\), got 2"):
        app_module._parse_spin_shifts("1.0, 2.0", 3, [0.0, 0.0, 0.0])


@pytest.mark.unit
def test_parse_shifts_non_numeric_raises(app_module):
    with pytest.raises(ValueError):
        app_module._parse_spin_shifts("1.0, foo", 2, [0.0, 0.0])


# --- _simulate_distort_kwargs --------------------------------------------------------------------


@pytest.mark.unit
def test_distort_kwargs_all_off_is_identity(app_module):
    assert app_module._simulate_distort_kwargs(False, 3.0, 0.0, 0.0, 0.0) == {}


@pytest.mark.unit
def test_distort_kwargs_selective(app_module):
    assert app_module._simulate_distort_kwargs(True, 3.0, 0.0, 0.0, 0.0) == {"noise_snr_log10": 3.0}
    assert app_module._simulate_distort_kwargs(False, 3.0, 5.0, 0.0, 0.0) == {"phase0_deg": 5.0}
    assert app_module._simulate_distort_kwargs(False, 3.0, 0.0, 1.0, 0.0) == {"broaden_hz": 1.0}
    assert app_module._simulate_distort_kwargs(False, 3.0, 0.0, 0.0, 0.05) == {"baseline": 0.05}


# --- _build_gt_groups ----------------------------------------------------------------------------


@pytest.mark.unit
def test_build_gt_groups_ethyl(app_module):
    pheno = app_module.sp.PHENOTYPES["ethyl"]
    groups = app_module._build_gt_groups(pheno["shifts_ppm"], pheno["couplings"], 7.0)
    # two equivalence groups (3H @1.2, 2H @3.5), sorted high→low ppm, both coupled
    assert [(g["shift_ppm"], g["proton_count"], g["max_j_hz"]) for g in groups] == [
        (3.5, 2, 7.0),
        (1.2, 3, 7.0),
    ]


@pytest.mark.unit
def test_build_gt_groups_singlet_has_no_coupling(app_module):
    pheno = app_module.sp.PHENOTYPES["methoxy_singlet"]
    (group,) = app_module._build_gt_groups(pheno["shifts_ppm"], pheno["couplings"], 0.0)
    assert group == {"shift_ppm": 3.8, "proton_count": 3, "max_j_hz": None}


# --- simulate_and_detect (stubbed) ---------------------------------------------------------------


@pytest.mark.unit
def test_simulate_checkpoint_absent(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "CHECKPOINT", str(Path("nope.pth")))
    _t, _f, msg = app_module.simulate_and_detect(
        "ethyl", "", 7.0, 1.0, False, 3.0, 0.0, 0.0, 0.0, 0.3
    )
    assert "Checkpoint not found" in msg


@pytest.mark.unit
def test_simulate_ethyl_roundtrip(patch_model):
    app = patch_model
    table, fig, msg = app.simulate_and_detect("ethyl", "", 7.0, 1.0, False, 3.0, 0.0, 0.0, 0.0, 0.3)
    assert "Simulated `ethyl`" in msg and "2 ground-truth multiplet(s)" in msg and "detected" in msg
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
def test_simulate_shift_count_mismatch(patch_model):
    app = patch_model
    _t, _f, msg = app.simulate_and_detect(
        "ethyl", "1.0, 2.0", 7.0, 1.0, False, 3.0, 0.0, 0.0, 0.0, 0.3
    )
    assert msg == "Invalid shifts: expected 5 shift value(s), got 2"


@pytest.mark.unit
def test_simulate_bad_shift_token(patch_model):
    app = patch_model
    _t, _f, msg = app.simulate_and_detect(
        "aromatic_ax", "7.5, foo", 8.0, 1.0, False, 3.0, 0.0, 0.0, 0.0, 0.3
    )
    assert msg.startswith("Invalid shifts:")


@pytest.mark.unit
def test_simulate_nonpositive_width(patch_model):
    app = patch_model
    _t, _f, msg = app.simulate_and_detect("ethyl", "", 7.0, 0.0, False, 3.0, 0.0, 0.0, 0.0, 0.3)
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
    table, _f, msg = app.simulate_and_detect(
        "methoxy_singlet", "", 0.0, 1.0, False, 3.0, 0.0, 0.0, 0.0, 0.3
    )
    assert "Simulated `methoxy_singlet`" in msg and "1 ground-truth multiplet(s)" in msg
    # the singlet's one GT group has no coupling -> its GT J renders as a dash
    gt_rows = table[table["status"] != "+ extra"]
    assert list(gt_rows["GT J (Hz)"]) == ["–"]
