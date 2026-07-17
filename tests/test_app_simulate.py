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
    assert "Simulated `ethyl` (2 GT group(s)); detected 3 multiplet(s)" in msg
    assert fig is not None
    assert len(table) == 2  # one row per GT group
    assert {"GT δ (ppm)", "GT H", "GT J (Hz)", "pred δ (ppm)"} <= set(table.columns)


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


@pytest.mark.unit
def test_simulate_singlet_gt_j_dashed(patch_model):
    app = patch_model
    table, _f, msg = app.simulate_and_detect(
        "methoxy_singlet", "", 0.0, 1.0, False, 3.0, 0.0, 0.0, 0.0, 0.3
    )
    assert "Simulated `methoxy_singlet` (1 GT group(s))" in msg
    assert list(table["GT J (Hz)"]) == ["–"]
