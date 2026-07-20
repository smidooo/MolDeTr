"""Data-contract tests for the committed experimental metadata + example npz — no checkpoint needed.

These encode the *contract* the release makes: the input shape (6144 pts @ 5.12 pts/Hz), the ROI-JSON
schema that downstream `combined_spin_refinement` consumes, and the corrected (2026-04) compound mapping —
so the mapping bug, or a count/schema drift, can never silently return. All assets are committed; runs in CI.
"""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SO = ROOT / "structured_output"
INPUT_LEN = 6144
POINTS_PER_HZ = 5.12

# The corrected roi_id -> compound mapping (verified against Bruker titles + .mol files, 2026-04).
EXPECTED_COMPOUND = {
    "S1": "Ethyl Acetate",
    "S2": "Ethylbenzene",
    "S3": "Ethyl-cis-3-iodacrylate",
    "S4": "Ibuprofen",
    "S5": "Ethyl Vanillin",
    "S5_R2": "Ethyl Vanillin",
    "S6": "Ethyl Acetate",
    "S7": "Ethylbenzene",
    "S8": "Vanillin",
    "S9": "Cinnamic Acid",
    "S10": "Guajazulene",
    "S12": "Cocaine",
    "S13": "Caffeic Acid",
}
ALLOWED_COMPOUNDS = set(EXPECTED_COMPOUND.values())
ROI_FILES = sorted(SO.glob("roi_S*.json"))
NPZ_FILES = sorted((ROOT / "examples").glob("*.npz"))


def _load(p: Path) -> object:
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- ROI JSON contract


def test_thirteen_rois_forty_four_spin_systems():
    """The canonical evaluation set: 13 ROIs, 44 spin systems (README/paper invariant)."""
    assert len(ROI_FILES) == 13
    assert sum(_load(f)["num_spin_systems"] for f in ROI_FILES) == 44


def test_thirteen_rois_across_twelve_spectra():
    """13 ROIs, 12 distinct spectra — the ethyl vanillin spectrum (S5) yields two regions."""
    assert len({_load(f)["base_sid"] for f in ROI_FILES}) == 12
    assert len(ALLOWED_COMPOUNDS) == 10  # 10 unique compounds


def test_roi_set_is_exactly_the_expected_ids():
    """No ROI added/removed without updating the mapping (and its Obsidian note)."""
    assert {_load(f)["roi_id"] for f in ROI_FILES} == set(EXPECTED_COMPOUND)


@pytest.mark.parametrize("path", ROI_FILES, ids=lambda p: p.stem)
def test_roi_json_schema_and_compound_mapping(path):
    d = _load(path)
    for key in ("roi_id", "base_sid", "compound", "num_spin_systems", "matches", "metadata"):
        assert key in d, f"{path.name} is missing '{key}'"
    assert d["compound"] in ALLOWED_COMPOUNDS
    assert d["compound"] == EXPECTED_COMPOUND[d["roi_id"]]  # regression guard for the mapping bug
    assert d["num_spin_systems"] == len(d["matches"])  # counts are consistent
    assert d["metadata"]["points_per_hz"] == POINTS_PER_HZ  # the input contract
    for match in d["matches"]:
        assert {"prediction", "label"} <= set(match)


def test_top_level_metadata_files_shape():
    rd = _load(SO / "roi_definitions.json")
    assert isinstance(rd, list) and len(rd) == 13
    assert {"export_info", "rois", "unmatched"} <= set(_load(SO / "all_rois_combined.json"))
    mp = _load(SO / "experimental_matched_pairs.json")
    assert {"matched_pairs_total", "unmatched_predictions_total", "unmatched_labels_total"} <= set(
        mp
    )


# ---------------------------------------------------------------- example npz contract
# allow_pickle=True is required to read the object arrays (ground_truth/labels/metadata are dicts) and is
# safe here: these are the repo's own committed example fixtures, loaded exactly as the dataloader/app do.


def test_all_examples_carry_a_6144_point_spectrum():
    """Every shipped example is a length-6144 window, whatever the key (spectrum_padded / spec)."""
    assert NPZ_FILES, "no example npz found"
    for path in NPZ_FILES:
        z = np.load(path, allow_pickle=True)
        key = "spectrum_padded" if "spectrum_padded" in z else "spec"
        assert z[key].shape == (INPUT_LEN,), f"{path.name}[{key}] is not ({INPUT_LEN},)"


@pytest.mark.parametrize(
    "path", [p for p in NPZ_FILES if p.name.startswith("roi_")], ids=lambda p: p.stem
)
def test_experimental_example_npz_schema(path):
    z = np.load(path, allow_pickle=True)
    assert z["spectrum_padded"].shape == (INPUT_LEN,)
    assert z["ppm_axis_padded"].shape == (INPUT_LEN,)
    md = z["metadata"].tolist()
    assert md["points_per_hz"] == POINTS_PER_HZ
    assert md["left_ppm"] > md["right_ppm"]  # NMR convention: ppm decreases across the window
    gt = z["ground_truth"].tolist()
    assert len(gt) == md["num_spin_systems"]
    for g in gt:
        assert g["proton_count"] >= 1
        assert 0.0 <= g["chemical_shift_in_points"] <= INPUT_LEN
        assert isinstance(g["coupling_constants"], list)


def test_synthetic_example_npz_schema():
    z = np.load(ROOT / "examples" / "synthetic_example.npz", allow_pickle=True)
    assert z["spec"].shape == (INPUT_LEN,)
    assert np.iscomplexobj(
        z["spec"]
    )  # synthetic spectra are stored complex; the model reads the real part
    assert z["labels"].size >= 1
