"""Unit tests for the pure helpers behind ``_expect_plotly_or_report``.

Why these are here and not inside ``tests/e2e/test_browser_journeys.py``: that module is
``-m browser`` behind an ``importorskip``, so the classification logic would be exercised only
inside the three-browser job. Classifying a failure needs no page, and a silent regression here is
exactly the #78 failure mode -- the report asserting something the evidence contradicts -- so it is
checked in the lane that runs on every push.

The fault classes these tests pin are documented in ``tests/plot_wait.py``'s docstring: four real
webkit failures, two of them (issues #78 and #92) where the old code reported a misleading cause
because the target had died before the diagnostic could run.
"""

from __future__ import annotations

import pytest

from tests.plot_wait import (
    POLL_INTERVAL_S,
    WAIT_BUDGET_S,
    format_never_appeared,
    format_target_death,
    is_target_death,
    remaining_timeout,
    sliced_wait,
)

HEALTHY_SAMPLE = {
    "box": "1320x420",
    "display": "block",
    "visibility": "visible",
    "children": 3,
    "hasPlotlyDiv": True,
    "plotlyLoaded": True,
}


class TestRemainingTimeout:
    def test_full_budget_before_any_polling(self) -> None:
        assert remaining_timeout(0.0) == WAIT_BUDGET_S

    def test_shrinks_with_elapsed_time(self) -> None:
        assert remaining_timeout(WAIT_BUDGET_S / 2) == pytest.approx(WAIT_BUDGET_S / 2)

    def test_clamps_at_zero_so_a_slow_poll_cannot_extend_the_budget(self) -> None:
        """REGRESSION-shaped: a negative timeout passed to playwright means 'use the default',
        i.e. an unbounded wait. The clamp is what keeps the 30 s contract honest."""
        assert remaining_timeout(WAIT_BUDGET_S + 5.0) == 0.0


class TestIsTargetDeath:
    @pytest.mark.parametrize(
        "message",
        [
            "Locator.click: Target crashed",  # measured: issue #92, run 31585679098
            "Page.evaluate: Target crashed",  # measured: issue #78, run 31483547839
            "Target page, context or browser has been closed",
            "Target closed",
            # Both grepped verbatim from the installed driver bundle (playwright-python 1.61),
            # 2026-08-25; not yet observed in this repository's own logs.
            "Browser closed",
            "Page has been closed.",
        ],
    )
    def test_matches_the_measured_and_upstream_death_phrases(self, message: str) -> None:
        assert is_target_death(RuntimeError(message)) is True

    def test_matching_is_case_insensitive(self) -> None:
        assert is_target_death(RuntimeError("page.evaluate: TARGET CRASHED")) is True

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("Timeout 30000ms exceeded"),  # a genuine wait timeout, page alive
            RuntimeError("my_field is not defined"),  # a bug in the probe JS itself
            AssertionError("the Plotly canvas never appeared within 30 s"),
            KeyError("box"),
        ],
    )
    def test_rejects_failures_that_are_not_target_death(self, exc: BaseException) -> None:
        assert is_target_death(exc) is False


class TestFormatTargetDeath:
    def test_names_the_elapsed_time(self) -> None:
        report = format_target_death([], 17.8, RuntimeError("Page.evaluate: Target crashed"))
        assert "17.8" in report

    def test_empty_samples_says_so_instead_of_printing_none(self) -> None:
        report = format_target_death([], 4.0, RuntimeError("Target crashed"))
        assert "no successful sample" in report

    def test_includes_the_last_known_state(self) -> None:
        earlier = {"box": "0x0", "children": 0}
        last = HEALTHY_SAMPLE
        report = format_target_death([earlier, last], 12.3, RuntimeError("Target crashed"))
        assert '"plotlyLoaded": True' in report or "'plotlyLoaded': True" in report
        assert "'box': '0x0'" not in report and '"box": "0x0"' not in report, (
            "only the LAST sample belongs in the report"
        )

    def test_does_not_claim_the_canvas_never_appeared(self) -> None:
        """REGRESSION: this exact false claim shipped in #78's failure message, because the
        post-hoc diagnostic ran against a dead target and the wrapper blamed Plotly."""
        report = format_target_death([HEALTHY_SAMPLE], 19.7, RuntimeError("Target crashed"))
        assert "never appeared" not in report


class TestFormatNeverAppeared:
    def test_reports_the_last_sampled_state(self) -> None:
        report = format_never_appeared({**HEALTHY_SAMPLE, "hasPlotlyDiv": False})
        assert "never appeared" in report
        assert "30" in report
        assert "hasPlotlyDiv" in report

    def test_handles_the_no_sample_case_readably(self) -> None:
        report = format_never_appeared(None)
        assert "never appeared" in report
        assert "no successful sample" in report


def test_poll_interval_is_strictly_inside_the_budget() -> None:
    """A poll interval >= the budget would degenerate the loop back into a single blind wait --
    precisely the shape that made #78's diagnostic unable to fire."""
    assert 0 < POLL_INTERVAL_S < WAIT_BUDGET_S


class _TickClock:
    """Deterministic clock: read k returns (k-1)*step_s, so each loop iteration costs one step."""

    def __init__(self, step_s: float = 1.0) -> None:
        self._now = 0.0
        self._step = step_s

    def __call__(self) -> float:
        value = self._now
        self._now += self._step
        return value


def _always_timeout(_slice_s: float) -> None:
    raise AssertionError("Locator expected to be visible")


class TestSlicedWait:
    """The routing decisions of the polling loop, pinned in the always-on lane.

    These exist because every decision here used to live inline in the ``-m browser`` module,
    exercised only by the three-browser job -- the exact coverage gap that let #78's wrapper
    report a false cause for two occurrences before anyone noticed.
    """

    def test_returns_after_the_first_successful_slice(self) -> None:
        waits: list[float] = []
        sliced_wait(lambda: {"children": 3}, waits.append)
        assert waits == [POLL_INTERVAL_S]

    def test_budget_exhaustion_names_the_last_sample_and_chains_the_last_wait_error(self) -> None:
        with pytest.raises(AssertionError) as ei:
            sliced_wait(lambda: {"children": 3}, _always_timeout, clock=_TickClock())
        message = str(ei.value)
        assert "never appeared" in message
        assert "'children': 3" in message, "the freshest DOM evidence must survive"
        cause = ei.value.__cause__
        assert cause is not None and "visible" in str(cause), (
            "the playwright call log of the final slice must stay chained, not `from None`d away "
            "(review finding: it is the only record of what happened inside the last 2 s)"
        )

    def test_probe_death_after_successful_samples_reports_elapsed_and_last_state(self) -> None:
        outcomes = iter([{"a": 1}, {"a": 2}, RuntimeError("Page.evaluate: Target crashed")])

        def probe() -> dict:
            item = next(outcomes)
            if isinstance(item, Exception):
                raise item
            return item

        with pytest.raises(RuntimeError) as ei:
            sliced_wait(probe, _always_timeout, clock=_TickClock(step_s=4.0))
        message = str(ei.value)
        assert "died 12.0 s" in message, (
            "elapsed must be reported from the clock, not guessed (the initial `started` "
            "read consumes the first tick of a 4 s step: 0*start, then 4, 8, 12)"
        )
        assert "'a': 2" in message, "the last sample taken while alive must be reported"
        assert "never appeared" not in message
        assert isinstance(ei.value.__cause__, RuntimeError)

    def test_a_probe_bug_propagates_unchanged_not_relabelled_as_target_death(self) -> None:
        def boom() -> dict:
            raise ValueError("my_field is not defined")

        with pytest.raises(ValueError, match="my_field"):
            sliced_wait(boom, _always_timeout)

    def test_a_rejected_slice_is_classified_like_the_probe_is(self) -> None:
        """REGRESSION-shaped (review): death mid-slice surfaces as AssertionError and completes,
        but a strict-mode violation or a destroyed execution context REJECTS as a raw playwright
        error; unclassified, it would escape without elapsed time or last-known-state framing."""

        def reject_with_death(_slice_s: float) -> None:
            raise RuntimeError("Target page, context or browser has been closed")

        with pytest.raises(RuntimeError) as ei:
            sliced_wait(lambda: {}, reject_with_death)
        assert "died" in str(ei.value)

    def test_success_on_a_later_slice_stops_the_loop(self) -> None:
        calls = {"n": 0}

        def flaky_wait(_slice_s: float) -> None:
            calls["n"] += 1
            if calls["n"] < 3:
                raise AssertionError("Locator expected to be visible")

        sliced_wait(lambda: {}, flaky_wait, clock=_TickClock(step_s=2.0))
        assert calls["n"] == 3
