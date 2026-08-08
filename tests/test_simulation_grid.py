"""The Simulate tab's grid must stay on the resolution the model was trained at.

The checkpoint expects **6144 points across a 1200 Hz window** -- 5.12 points/Hz. The simulator is
parameterised in *ppm*, so that resolution is only correct while

    (LEFT_PPM - RIGHT_PPM) * BASE_FREQ_MHZ == 1200

At the shipped 80 MHz with a 15 -> 0 ppm window that holds exactly, which is why nothing has gone
wrong yet. But the three constants are declared independently, so raising `BASE_FREQ_MHZ` to 600 and
leaving the ppm bounds alone yields a 9000 Hz window at **0.68 points/Hz** -- an eighth of the trained
resolution, silently, with no error anywhere. The detector would still return numbers, and they would
be confidently wrong.

`test_the_shipped_grid_matches_the_trained_resolution` is a **regression guard**: it passes today and
is here so a future edit to any one constant cannot quietly break the pair. The test after it is the
one with teeth -- it requires the ppm window to be *derived* from the field rather than declared
beside it, so the invariant cannot be broken by editing a single number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import simulate_and_predict as sp  # noqa: E402

#: What the checkpoint was trained on. Both are properties of the frozen weights, not preferences.
TRAINED_WINDOW_HZ = 1200.0
TRAINED_POINTS_PER_HZ = 5.12


@pytest.mark.unit
def test_the_shipped_grid_matches_the_trained_resolution():
    """Regression guard -- green on arrival, and that is the point."""
    window_hz = (sp.LEFT_PPM - sp.RIGHT_PPM) * sp.BASE_FREQ_MHZ
    assert window_hz == pytest.approx(TRAINED_WINDOW_HZ), (
        f"the simulated window spans {window_hz:g} Hz, not {TRAINED_WINDOW_HZ:g}: "
        f"{sp.LEFT_PPM}-{sp.RIGHT_PPM} ppm at {sp.BASE_FREQ_MHZ} MHz"
    )
    assert sp.N_POINTS / window_hz == pytest.approx(TRAINED_POINTS_PER_HZ), (
        f"the grid is {sp.N_POINTS / window_hz:.3f} points/Hz, not the trained "
        f"{TRAINED_POINTS_PER_HZ}: {sp.N_POINTS} points across {window_hz:g} Hz"
    )


@pytest.mark.unit
@pytest.mark.parametrize("base_freq_mhz", [80.0, 300.0, 400.0, 500.0, 600.0])
def test_the_ppm_window_is_derived_from_the_field(base_freq_mhz: float):
    """The window must follow the field automatically, so one edited number cannot desynchronise it.

    This is what the regression guard above cannot check: that test would still pass if someone
    changed `BASE_FREQ_MHZ` *and* `LEFT_PPM` together and got the arithmetic wrong in the same
    commit. Deriving removes the opportunity.
    """
    left, right = sp.ppm_window(base_freq_mhz)
    span_hz = (left - right) * base_freq_mhz
    assert span_hz == pytest.approx(TRAINED_WINDOW_HZ), (
        f"at {base_freq_mhz:g} MHz the derived window {left:g}-{right:g} ppm spans {span_hz:g} Hz, "
        f"not the trained {TRAINED_WINDOW_HZ:g}"
    )
    assert sp.N_POINTS / span_hz == pytest.approx(TRAINED_POINTS_PER_HZ)


@pytest.mark.unit
def test_the_shipped_constants_are_the_derived_ones():
    """The module-level constants must come from the derivation, not sit beside it out of step."""
    left, right = sp.ppm_window(sp.BASE_FREQ_MHZ)
    assert (sp.LEFT_PPM, sp.RIGHT_PPM) == pytest.approx((left, right)), (
        f"LEFT_PPM/RIGHT_PPM are ({sp.LEFT_PPM}, {sp.RIGHT_PPM}) but the derivation at "
        f"{sp.BASE_FREQ_MHZ} MHz gives ({left}, {right}) -- they have drifted apart"
    )
