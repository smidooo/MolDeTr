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

DISTORTION = (True, 3.0, 2.0, 0.0, 0.0, 0.3)  # noise on, snr, phase0, broaden, baseline, threshold


@pytest.fixture
def args(app_module):
    """The ethyl preset as the matrix grid and width table the stages now take."""
    return app_module._phenotype_grid("ethyl")


# Captured from `main`'s monolithic `simulate_and_detect` (pre-split, commit b0241c7) driven by the
# same deterministic fake model. These are the reference: an external record of what the user saw
# before the refactor, which is what makes the assertions below impossible to satisfy by definition.
#
# The label is the one part that legitimately moved: the phenotype stopped being an input to
# simulation when the matrix became the single source of truth, so the message can no longer name
# one. Everything after the label — the counts, the legend — is main's text unchanged, and the
# table below is main's byte for byte, which is what says the spectrum itself did not move.
MAIN_MESSAGE_TAIL = (
    ": 2 ground-truth multiplet(s); the model **detected** 3 "
    "(2 matched, 1 spurious). Teal ▽ = ground truth · clay ● = model detection; a connector turns "
    "**green** within tolerance and **amber** when off. Missed GT and spurious peaks are outlined "
    "in red."
)
ETHYL_LABEL = "**Simulated 5 spin(s) in 1 system(s)**"
MAIN_TABLE_CSV = [
    "#,status,GT δ (ppm),GT H,GT J (Hz),pred δ (ppm),pred H,Δδ (Hz),ΔH,conf",
    "1,~ off,3.50,2,7.0,3.000,3,40.00,+1,1.00",
    "2,~ off,1.20,3,7.0,7.500,2,504.00,-1,1.00",
    "3,+ extra,–,–,–,12.000,1,–,–,1.00",
]


def _csv_lines(table) -> list[str]:
    return [line for line in table.to_csv(index=False).splitlines() if line]


@pytest.mark.unit
def test_both_paths_still_produce_what_main_showed_the_user(app_module, patch_model, args) -> None:
    """The split must not move a single character of the output, one-shot or staged.

    This replaces a test that compared `simulate_and_detect` against
    `_detect_stage(_simulate_stage(...))`. After the refactor those are the *same expression* —
    `simulate_and_detect` is defined as that composition — so it asserted `f(x) == g(h(x))` for
    `f := g∘h` and passed even with `_detect_stage` monkeypatched to prepend "BROKEN" to every
    message. Pinning main's actual output instead means the refactor cannot define agreement into
    existence: the goldens came from code that predates it.
    """
    app = patch_model

    table_a, fig_a, msg_a = app.simulate_and_detect(*args, *DISTORTION)

    cache = app._simulate_stage(*args)
    assert not isinstance(cache, str), cache  # a str is the error channel
    table_b, fig_b, msg_b = app._detect_stage(cache, *DISTORTION)

    assert msg_a == ETHYL_LABEL + MAIN_MESSAGE_TAIL
    assert msg_b == ETHYL_LABEL + MAIN_MESSAGE_TAIL
    assert _csv_lines(table_a) == MAIN_TABLE_CSV
    assert _csv_lines(table_b) == MAIN_TABLE_CSV
    assert fig_a is not None and fig_b is not None


@pytest.mark.unit
def test_detect_stage_passes_a_simulation_error_through(app_module, patch_model, args) -> None:
    """`_simulate_stage` reports failure as a string, so `_detect_stage` must recognise one.

    Only `simulate_and_detect` checks for the string today. The moment a cache lives in `gr.State`
    and a slider re-runs the detect stage on its own, a checkpoint-missing run would store the error
    *string* in state and the next slider move would index a `str` — a Gradio 500 in place of the
    message the user should have seen.
    """
    app = patch_model
    error = "Checkpoint not found at `nowhere.pth`."

    table, fig, msg = app._detect_stage(error, *DISTORTION)

    assert (table, fig) == (None, None)
    assert msg == error


@pytest.mark.unit
def test_redistorting_never_reruns_the_spin_dynamics(
    app_module, patch_model, args, monkeypatch
) -> None:
    """Five distortion changes must cost exactly one simulation: the whole point of the split."""
    app = patch_model
    calls = {"n": 0}
    real = app.simulate_systems

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(app, "simulate_systems", counting)

    cache = app._simulate_stage(*args)
    assert calls["n"] == 1

    for phase0 in (0.0, 2.0, 4.0, 6.0, 8.0):
        app._detect_stage(cache, True, 3.0, phase0, 0.0, 0.0, 0.3)

    assert calls["n"] == 1, "a distortion change re-ran the simulation"


@pytest.mark.unit
def test_distortion_actually_changes_the_spectrum(app_module, patch_model, args) -> None:
    """Guards against a cache that is reused so aggressively the distortion stops applying."""
    app = patch_model
    cache = app._simulate_stage(*args)

    clean = app._distorted_amplitudes(cache, False, 3.0, 0.0, 0.0, 0.0)
    noisy = app._distorted_amplitudes(cache, True, 2.0, 0.0, 0.0, 0.0)
    phased = app._distorted_amplitudes(cache, False, 3.0, 8.0, 0.0, 0.0)

    assert not np.allclose(clean, noisy)
    assert not np.allclose(clean, phased)


@pytest.mark.unit
def test_the_cache_is_not_mutated_by_distorting(app_module, patch_model, args) -> None:
    """Re-distorting the same cache twice must give the same answer.

    `distort()` copies its input, but the stage could still hand out a view; if it did, distortions
    would accumulate silently as the user dragged a slider.
    """
    app = patch_model
    cache = app._simulate_stage(*args)

    first = app._distorted_amplitudes(cache, False, 3.0, 6.0, 0.0, 0.0)
    second = app._distorted_amplitudes(cache, False, 3.0, 6.0, 0.0, 0.0)
    assert np.allclose(first, second)


@pytest.mark.unit
def test_the_no_distortion_path_hands_back_a_copy(app_module, patch_model, args) -> None:
    """With every distortion at its neutral value the stage must still not return the cache itself.

    The test above only proves determinism: it uses `phase0=6.0`, which builds a non-empty kwargs
    dict and goes through `distort()`, and `distort()` copies. With all sliders neutral the dict is
    empty, `distort()` is skipped, and `np.asarray(np.real(x), dtype=float)` is a no-op on a real
    C-contiguous float64 array — it returns *the same object*. The caller then holds a writable view
    of the cached clean spectrum, which is exactly what a long-lived `gr.State` must never hand out:
    one `amplitudes -= baseline` downstream and every later slider move distorts a corrupted cache.
    """
    app = patch_model
    cache = app._simulate_stage(*args)

    out = app._distorted_amplitudes(cache, False, 3.0, 0.0, 0.0, 0.0)  # neutral: no distortion

    assert not np.shares_memory(out, cache["spectrum"])


@pytest.mark.unit
def test_simulate_stage_reports_bad_input_without_raising(app_module, patch_model, args) -> None:
    """Errors travel as a message, matching how the monolithic call behaved."""
    app = patch_model
    grid, widths = args

    typo = [list(row) for row in grid]
    typo[0][1] = "three point five"
    assert isinstance(app._simulate_stage(typo, widths), str)

    zero_width = [list(row) for row in widths]
    zero_width[0][3] = 0.0  # FWHM is the last column: system | δ | n H | FWHM
    assert isinstance(app._simulate_stage(grid, zero_width), str)
