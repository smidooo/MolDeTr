"""The dependency audit must be able to produce a result at all.

`security.yml`'s audit step is deliberately report-only -- `continue-on-error: true` -- while the
noise floor of a torch-sized transitive tree is unknown. That is a defensible choice, and it is also
what hid this: the lane has run **once** in the repository's history (the only non-`pull_request`
`security.yml` run; the job is gated `if: github.event_name != 'pull_request'`), and in that run
`pip-audit` produced **no audit at all**. Measured 2026-08-08 from the job log::

    Run pip-audit --strict --desc --format columns          08:04:29
    ERROR:pip_audit._cli:moldetr: Dependency not found on   08:04:30
      PyPI and could not be audited: moldetr (1.2.0)

`--strict` promotes "could not be audited" to a fatal error, so the run ended 1.7 s in and never
printed a vulnerability table. `continue-on-error` then swallowed the non-zero exit and the job
reported **success**. The project installs itself with `pip install -e`, so the local distribution
can never be on PyPI and this is not an intermittent condition -- it is every run, forever.

That is the same defect class the rot watcher in #43 had: a guard that reports green while performing
none of its work.

**`--skip-editable` does not fix it, and that near-miss is why this test asserts what it does.**
Checked against pip-audit's source rather than assumed: the flag does not filter an editable
distribution out, it yields it as a ``SkippedDependency`` (``_dependency_source/pip.py``)::

    if dist.editable and self._skip_editable:
        dep = SkippedDependency(name=dist.name, skip_reason="distribution marked as editable")

and ``--strict`` fatals on *any* skipped dependency, whatever the reason (``_cli.py``)::

    if spec.is_skipped():
        if args.strict:
            _fatal(f"{spec.name}: {spec.skip_reason}")

So ``--strict --skip-editable`` still exits before the table, only with a different message. A test
that asserted "``--skip-editable`` is present" would have gone green on a fix that audits exactly as
little as before -- the very defect class it was written to catch.

The invariant is therefore about ``--strict`` itself: this project always has at least one skipped
dependency by construction, so ``--strict`` guarantees no audit output rather than a stricter audit.

Reads only the committed workflow, so it needs no network and cannot skip itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SECURITY_WORKFLOW = REPO / ".github" / "workflows" / "security.yml"


def _pip_audit_invocations(workflow: str) -> list[str]:
    """Every `pip-audit` command line in the workflow, whitespace-normalised."""
    return [
        " ".join(line.split())
        for line in workflow.splitlines()
        if re.search(r"(^|\s|`)pip-audit\s+-", line)
    ]


@pytest.mark.unit
def test_pip_audit_is_invoked_at_all():
    """Guards the guard: every assertion below is vacuous if the command vanishes."""
    calls = _pip_audit_invocations(SECURITY_WORKFLOW.read_text(encoding="utf-8"))
    assert calls, (
        f"no `pip-audit` invocation found in {SECURITY_WORKFLOW.name}; the dependency audit this "
        f"file exists to run is gone, and the tests below would pass on its absence"
    )


@pytest.mark.unit
def test_pip_audit_does_not_run_strict():
    """`--strict` cannot coexist with an editable self-install: it aborts before any advisory."""
    offenders = [
        call
        for call in _pip_audit_invocations(SECURITY_WORKFLOW.read_text(encoding="utf-8"))
        if "--strict" in call
    ]
    assert not offenders, (
        "pip-audit runs with `--strict`:\n  "
        + "\n  ".join(offenders)
        + "\nThis project installs itself with `pip install -e`, so `moldetr` is always a skipped "
        "dependency -- whether because it is not on PyPI, or, with `--skip-editable`, because it is "
        "'marked as editable'. `--strict` calls `_fatal()` on *any* skip, so the audit exits before "
        "printing a single advisory. Measured 2026-08-08: the one run this lane has ever had died "
        "1.7 s in and still reported success, because the step is `continue-on-error: true`. "
        "Adding `--skip-editable` does not rescue `--strict`; dropping `--strict` is the fix."
    )


@pytest.mark.unit
def test_pip_audit_skips_the_editable_self_install():
    """Without this, every run prints a 'could not be audited' warning for the project itself.

    Not a correctness requirement the way the `--strict` rule is -- the audit completes either way --
    but a report-only lane whose output opens with a warning about a non-problem is a lane people
    learn to skim.
    """
    calls = _pip_audit_invocations(SECURITY_WORKFLOW.read_text(encoding="utf-8"))
    missing = [call for call in calls if "--skip-editable" not in call]
    assert not missing, (
        "pip-audit does not pass `--skip-editable`:\n  "
        + "\n  ".join(missing)
        + "\n`moldetr` is installed editable and can never be resolved on PyPI, so every run reports "
        "it as unauditable. Skipping it explicitly keeps the advisory output about dependencies."
    )
