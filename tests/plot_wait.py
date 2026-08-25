"""Pure helpers behind ``_expect_plotly_or_report`` in ``tests/e2e/test_browser_journeys.py``.

Kept free of any playwright import so the unit lane can exercise them (same reasoning as
``tests/css_at_rules.py``): classifying and wording a failure needs no page, and a silent
regression here reproduces exactly what issues #78 and #92 suffered -- a failure report whose
stated cause contradicted its own evidence.

Two fault classes are documented against four real webkit failures:

- **Plotly never initialises** (#41 and #49 merge runs): the page lives out the full wait,
  ``window.Plotly`` never appears, and the post-hoc diagnostic answers truthfully.
- **The browser target dies** (#78, run 31483547839; #92, run 31585679098): the webkit process
  dies seconds into the journey -- measured once inside ``Page.evaluate``, once inside
  ``Locator.click``. Against a dead target, the old post-hoc diagnostic could only ever raise
  ``Target crashed`` itself, so the wrapper reported "the Plotly canvas never appeared within
  30 s" about a page that had stopped existing at ~1.9 s. These helpers exist so the report says
  *when* the target died and what the DOM looked like beforehand.

The death markers: two are measured in this repository's logs (``Target crashed`` in both the
evaluate and click forms); two cover upstream phrasings for the same condition; two more were
grepped verbatim from the installed driver bundle (playwright-python 1.61) during review of this
change and are insurance against phrasings not yet observed here. Matching is deliberately
substring-based and case-insensitive: the exception type is ``playwright...Error`` across all of
these, so only the message separates death from an ordinary timeout or a bug in the probe script
itself. A scan of the driver's benign error paths (timeouts, strict-mode violations, locator
texts) found no message containing any marker substring.
"""

from __future__ import annotations

import time
from collections.abc import Callable

WAIT_BUDGET_S = 30.0
POLL_INTERVAL_S = 2.0

_TARGET_DEATH_MARKERS = (
    "target crashed",
    "target page, context or browser has been closed",
    "target closed",
    "browser closed",
    "page has been closed",
)


def remaining_timeout(elapsed_s: float) -> float:
    """How much of the total wait budget is left; clamped at zero.

    The clamp matters more than it looks: playwright treats a negative ``timeout=`` as "use the
    default", which would silently turn one slow poll into an unbounded wait. In ``sliced_wait``
    the budget guard raises before this is consulted with a non-positive remainder, so today the
    clamp is pure defense.
    """
    return max(0.0, WAIT_BUDGET_S - elapsed_s)


def is_target_death(exc: BaseException) -> bool:
    """Whether this exception means the browser target died, as opposed to timing out or a probe bug."""
    message = str(exc).lower()
    return any(marker in message for marker in _TARGET_DEATH_MARKERS)


def format_target_death(samples: list[dict], elapsed_s: float, exc: BaseException) -> str:
    """Report a mid-wait target death with the elapsed time and the last state seen alive.

    Never claims anything about whether Plotly appeared: with the process dead, that question has
    no answer, and asserting "never appeared" was the exact falsehood shipped in #78's message.
    """
    if samples:
        last = repr(samples[-1])
    else:
        last = "no successful sample was taken before death"
    return (
        f"the browser target died {elapsed_s:.1f} s into the wait for the Plotly canvas "
        f"({type(exc).__name__}: {exc}). Last known #md-plot state while alive: {last}"
    )


def format_never_appeared(last_sample: dict | None) -> str:
    """Report a genuine full-budget timeout, with the freshest DOM evidence available."""
    state = repr(last_sample) if last_sample is not None else "no successful sample"
    return (
        f"the Plotly canvas never appeared within {WAIT_BUDGET_S:.0f} s. "
        f"#md-plot last sampled before timeout: {state}"
    )


def sliced_wait(
    probe: Callable[[], dict],
    wait_slice: Callable[[float], None],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Poll until the caller's ``wait_slice`` succeeds or ``WAIT_BUDGET_S`` elapses.

    The routing decisions of the browser journeys' Plotly wait, extracted behind callables so
    they are pinned by the unit lane instead of only by the three-browser job:

    - Each round samples ``probe()`` *before* waiting. A probe raising a target-death error
      raises ``RuntimeError(format_target_death(...))`` -- with elapsed time and the last state
      seen while alive. Any other probe exception propagates unchanged: a bug in the probe must
      surface as itself, never be relabelled as target death.
    - ``wait_slice(seconds)`` gets the remaining slice of the budget. An ``AssertionError``
      means the canvas was not visible yet; the loop continues, keeping the error so budget
      exhaustion can chain it (its playwright call log is the only record of what happened
      inside the final slice). Any other wait_slice exception is classified exactly like a probe
      exception: death mid-slice surfaces as a rejection (strict-mode violation, destroyed
      execution context), not as a completed false check.
    - Success returns normally; the caller owns whatever "visible" then means.

    This is not the retry that issues #51/#68 rejected: nothing is re-attempted, no timeout was
    raised, and total wall time stays at ``WAIT_BUDGET_S`` plus at most one partial final slice.
    Residual blind spot, stated rather than hidden: if the target dies after a slice's probe
    succeeded but during its wait, the completed-false check swallows into ``continue`` and death
    is named by the NEXT round's probe, roughly one interval late.
    """
    started = clock()
    samples: list[dict] = []
    last_wait_error: AssertionError | None = None
    while True:
        elapsed = clock() - started
        if elapsed >= WAIT_BUDGET_S:
            raise AssertionError(format_never_appeared(samples[-1] if samples else None)) from (
                last_wait_error
            )
        try:
            samples.append(probe())
        except Exception as exc:
            if is_target_death(exc):
                raise RuntimeError(format_target_death(samples, elapsed, exc)) from exc
            raise
        try:
            wait_slice(min(POLL_INTERVAL_S, remaining_timeout(elapsed)))
        except AssertionError as exc:
            last_wait_error = exc
            continue
        except Exception as exc:
            if is_target_death(exc):
                raise RuntimeError(format_target_death(samples, elapsed, exc)) from exc
            raise
        return
