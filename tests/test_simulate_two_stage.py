"""The Simulate path is split so distortion changes never re-run the spin dynamics.

`simulate_and_detect` used to do everything on one button press: diagonalise the Hamiltonian,
distort, run the model, build the figure. Moving a distortion slider therefore paid the 2**n
eigendecomposition again, which is why live updates were not viable.

Two guards live here:

* the split is **behaviour-preserving** — composing the stages equals the old monolithic call, so the
  refactor cannot quietly change what users see;
* re-distorting **does not simulate again** — asserted by counting calls, not by timing, so it stays
  meaningful on a slow CI runner.
"""

from __future__ import annotations

import numpy as np
import pytest

ARGS = ("ethyl", "1.2, 1.2, 1.2, 3.5, 3.5", 7.0, 1.0)
DISTORTION = (True, 3.0, 2.0, 0.0, 0.0, 0.3)  # noise on, snr, phase0, broaden, baseline, threshold


@pytest.mark.unit
def test_stages_compose_to_the_monolithic_result(app_module, patch_model) -> None:
    """Cache-invariance: simulate-then-detect is identical to the one-shot call."""
    app = patch_model

    table_a, fig_a, msg_a = app.simulate_and_detect(*ARGS, *DISTORTION)

    cache = app._simulate_stage(*ARGS)
    assert not isinstance(cache, str), cache  # a str is the error channel
    table_b, fig_b, msg_b = app._detect_stage(cache, *DISTORTION)

    assert msg_a == msg_b
    assert (table_a is None) == (table_b is None)
    if table_a is not None:
        assert table_a.equals(table_b)
    assert (fig_a is None) == (fig_b is None)


@pytest.mark.unit
def test_redistorting_never_reruns_the_spin_dynamics(app_module, patch_model, monkeypatch) -> None:
    """Five distortion changes must cost exactly one simulation: the whole point of the split."""
    app = patch_model
    calls = {"n": 0}
    real = app.simulate_systems

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(app, "simulate_systems", counting)

    cache = app._simulate_stage(*ARGS)
    assert calls["n"] == 1

    for phase0 in (0.0, 2.0, 4.0, 6.0, 8.0):
        app._detect_stage(cache, True, 3.0, phase0, 0.0, 0.0, 0.3)

    assert calls["n"] == 1, "a distortion change re-ran the simulation"


@pytest.mark.unit
def test_distortion_actually_changes_the_spectrum(app_module, patch_model) -> None:
    """Guards against a cache that is reused so aggressively the distortion stops applying."""
    app = patch_model
    cache = app._simulate_stage(*ARGS)

    clean = app._distorted_amplitudes(cache, False, 3.0, 0.0, 0.0, 0.0)
    noisy = app._distorted_amplitudes(cache, True, 2.0, 0.0, 0.0, 0.0)
    phased = app._distorted_amplitudes(cache, False, 3.0, 8.0, 0.0, 0.0)

    assert not np.allclose(clean, noisy)
    assert not np.allclose(clean, phased)


@pytest.mark.unit
def test_the_cache_is_not_mutated_by_distorting(app_module, patch_model) -> None:
    """Re-distorting the same cache twice must give the same answer.

    `distort()` copies its input, but the stage could still hand out a view; if it did, distortions
    would accumulate silently as the user dragged a slider.
    """
    app = patch_model
    cache = app._simulate_stage(*ARGS)

    first = app._distorted_amplitudes(cache, False, 3.0, 6.0, 0.0, 0.0)
    second = app._distorted_amplitudes(cache, False, 3.0, 6.0, 0.0, 0.0)
    assert np.allclose(first, second)


@pytest.mark.unit
def test_simulate_stage_reports_bad_input_without_raising(app_module, patch_model) -> None:
    """Errors travel as a message, matching how the monolithic call behaved."""
    app = patch_model
    assert isinstance(app._simulate_stage("ethyl", "1.2, 3.5", 7.0, 1.0), str)
    assert isinstance(app._simulate_stage("ethyl", "1.2, 1.2, 1.2, 3.5, 3.5", 7.0, 0.0), str)
