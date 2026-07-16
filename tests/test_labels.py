"""Tests for the canonical label adapters (synthetic + experimental -> one schema)."""
from moldetr.labels import from_experimental, from_synthetic


def test_synthetic_couplings_points_to_hz():
    m = from_synthetic({
        "proton_number": 3, "center_position_in_points": 1000.0,
        "line_width_in_points": 3.6, "bounding_box_range_in_points": 37.5,
        "coupling_constants_in_points": [37.55740549],
    })
    assert m.proton_count == 3
    assert m.center_in_points == 1000.0
    assert abs(m.coupling_constants_hz[0] - 37.55740549 / 5.12) < 1e-9  # ~7.34 Hz


def test_experimental_couplings_stay_hz():
    m = from_experimental({
        "proton_count": 3, "chemical_shift_in_points": 3983.0,
        "coupling_constants": [7.13], "chemical_shift_ppm": -3.3,
    })
    assert m.proton_count == 3 and m.center_in_points == 3983.0
    assert m.coupling_constants_hz == [7.13] and m.chemical_shift_ppm == -3.3


def test_both_produce_identical_canonical_schema():
    syn = from_synthetic({"proton_number": 2, "center_position_in_points": 500.0,
                          "coupling_constants_in_points": [36.0]})
    exp = from_experimental({"proton_count": 2, "chemical_shift_in_points": 500.0,
                            "coupling_constants": [36.0 / 5.12]})
    assert set(syn.as_dict()) == set(exp.as_dict())
    assert syn.proton_count == exp.proton_count
    assert abs(syn.coupling_constants_hz[0] - exp.coupling_constants_hz[0]) < 1e-9
