"""Every workflow with a `schedule:` trigger must ping an external dead-man's switch on success.

GitHub disables a scheduled workflow after 60 days without a *commit* to the repository (issues,
PRs, releases and tags do not count) -- `nightly.yml` and `integrations.yml` both document this, and
`security.yml`'s Wednesday cron carries the identical exposure even though it says nothing about it.
None of the three could ever notice the other two going quiet: they die under the same rule, so any
watcher living *inside* GitHub dies with what it watches. `.github/actions/heartbeat-ping` pings an
external URL (a repo secret) after a scheduled run succeeds; a service outside GitHub (see
`docs/MONITORING.md`) alerts when a ping fails to arrive on schedule.

This is the guard on the guard: it catches a future cron lane added without a ping, the way
`tests/test_integrations_isolation.py` catches a pytest invocation added to `integrations.yml`
without being read by the isolation check. Reads only the committed workflow files -- no network,
cannot skip itself -- mirroring `tests/test_security_workflow.py`.

Two things this file checks beyond "the string is present somewhere in the file", both found in
review: a heartbeat step whose own `if:` overrides the implicit `success()` (e.g. `always()`) would
ping on a failed run and defeat the entire mechanism, so the check parses the step's own condition,
not just its presence. And the seeded-defect test exercises the REAL discovery + assertion functions
end to end against scratch copies of the three actual workflows, not a synthetic string handed
straight to a helper -- a test that only proves a helper behaves on input it was written to accept
is not evidence the check would catch a real regression.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO / ".github" / "workflows"
HEARTBEAT_ACTION = ".github/actions/heartbeat-ping"

#: Status-check functions that would override the implicit `success()` a bare `if:` carries. Any of
#: these on a heartbeat-ping step's own `if:` could make it fire on a failed or cancelled run.
OVERRIDING_FUNCTIONS = ("always(", "cancelled(", "failure(")


def _scheduled_workflow_paths(directory: Path) -> list[Path]:
    """Every `.github/workflows/*.yml`/`*.yaml` in `directory` whose top-level `on:` block has a
    `schedule:` trigger.

    Line-based, not a YAML parse -- this project has no `pyyaml` dependency declared (only
    available transitively), and `tests/test_integrations_isolation.py` exists precisely because a
    lane can install `pytest` alone; adding a real parser to a `unit`-marked test would be a new,
    undeclared risk of the same kind. `schedule:` may carry a trailing comment (`schedule:  # weekly`).
    """
    paths = []
    for path in sorted(directory.glob("*.yml")) + sorted(directory.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*schedule:\s*(#.*)?$", text, re.MULTILINE):
            paths.append(path)
    return paths


def _heartbeat_step_blocks(workflow_text: str) -> list[str]:
    """The text of each step block that `uses:` the heartbeat action.

    A step block runs from its `- name:`/`- uses:` line to the next line at the same indentation
    that starts a new list item (`- `) or drops to a shallower indentation. Matching on `uses:`
    rather than a bare substring search means a workflow that only *mentions* the action in a
    comment does not count as having wired it.
    """
    lines = workflow_text.splitlines()
    blocks = []
    for i, line in enumerate(lines):
        stripped = line.lstrip(" ")
        if not (stripped.startswith("- uses:") or stripped.startswith("uses:")):
            continue
        if HEARTBEAT_ACTION not in line:
            continue
        indent = len(line) - len(stripped)
        # Walk back to the start of this step (a `- ` at this step's list-item indentation may be
        # a few lines above `uses:`, e.g. after a `name:` line).
        start = i
        while start > 0 and not lines[start].lstrip(" ").startswith("- "):
            start -= 1
        end = len(lines)
        for j in range(i + 1, len(lines)):
            candidate = lines[j]
            if not candidate.strip():
                continue
            candidate_indent = len(candidate) - len(candidate.lstrip(" "))
            if candidate_indent <= indent and candidate.lstrip(" ").startswith("- "):
                end = j
                break
            if candidate_indent < indent - 2:
                end = j
                break
        blocks.append("\n".join(lines[start:end]))
    return blocks


def _heartbeat_steps_are_success_only(workflow_text: str) -> list[str]:
    """Problems found with this workflow's heartbeat-ping step condition(s); empty if none.

    A missing `heartbeat-ping` step entirely is reported by the caller, not here -- this function's
    job is only to check that a step that DOES exist cannot fire on a non-success run.
    """
    problems = []
    blocks = _heartbeat_step_blocks(workflow_text)
    if not blocks:
        return ["no heartbeat-ping step found"]
    for block in blocks:
        if_lines = [line for line in block.splitlines() if re.search(r"\bif\s*:", line)]
        for if_line in if_lines:
            for fn in OVERRIDING_FUNCTIONS:
                if fn in if_line:
                    problems.append(
                        f"a heartbeat-ping step's `if:` contains `{fn}`, which can make it fire on "
                        f"a failed or cancelled run and defeats the switch: {if_line.strip()!r}"
                    )
    return problems


def _assert_scheduled_workflows_ping_safely(directory: Path) -> dict[str, list[str]]:
    """`{workflow filename: [problems]}` for every scheduled workflow in `directory` with issues."""
    failures: dict[str, list[str]] = {}
    for path in _scheduled_workflow_paths(directory):
        problems = _heartbeat_steps_are_success_only(path.read_text(encoding="utf-8"))
        if problems:
            failures[path.name] = problems
    return failures


@pytest.mark.unit
def test_at_least_one_workflow_is_actually_scheduled():
    """Guards the guard: every assertion below is vacuous if no workflow has `schedule:` at all."""
    assert _scheduled_workflow_paths(WORKFLOWS_DIR), (
        "no workflow under .github/workflows/ has a `schedule:` trigger -- either the discovery "
        "regex broke, or every cron lane in this repository was removed"
    )


@pytest.mark.unit
def test_every_scheduled_workflow_pings_the_heartbeat_action_on_success_only():
    failures = _assert_scheduled_workflows_ping_safely(WORKFLOWS_DIR)
    assert not failures, (
        f"{failures}. GitHub silently disables a scheduled workflow after 60 days without a "
        "commit, and nothing inside GitHub can notice that about itself -- see "
        "docs/MONITORING.md for the external switch every cron lane must ping on success, and "
        "only on success."
    )


@pytest.mark.unit
def test_heartbeat_action_exists_and_guards_an_empty_url():
    action = REPO / HEARTBEAT_ACTION / "action.yml"
    assert action.exists(), f"{HEARTBEAT_ACTION}/action.yml is missing"
    text = action.read_text(encoding="utf-8")
    assert '[ -z "$HEALTHCHECK_URL" ]' in text, (
        "the heartbeat action must visibly report an unset URL (e.g. a fork with no secret "
        "configured), not silently no-op -- an invisible skip is the defect class this repo names "
        "in CLAUDE.md's 'Things not to do'"
    )
    assert "GITHUB_STEP_SUMMARY" in text, "the empty-URL case must write to the step summary"


@pytest.mark.unit
def test_the_three_documented_secret_names_appear_in_both_the_workflows_and_the_doc():
    """Renaming a secret in one workflow without updating `docs/MONITORING.md` (or vice versa)
    should not pass silently -- this ties the two together."""
    doc = (REPO / "docs" / "MONITORING.md").read_text(encoding="utf-8")
    secrets_in_doc = set(re.findall(r"HEALTHCHECK_URL_\w+", doc))
    secrets_in_workflows: set[str] = set()
    for path in _scheduled_workflow_paths(WORKFLOWS_DIR):
        secrets_in_workflows |= set(
            re.findall(r"HEALTHCHECK_URL_\w+", path.read_text(encoding="utf-8"))
        )
    assert secrets_in_doc, "docs/MONITORING.md mentions no HEALTHCHECK_URL_* secret by name"
    assert secrets_in_workflows == secrets_in_doc, (
        f"secret names drifted between the workflows {secrets_in_workflows} and "
        f"docs/MONITORING.md {secrets_in_doc}"
    )


@pytest.mark.unit
def test_seeded_defect_a_scheduled_workflow_without_the_ping_is_caught(tmp_path):
    """Copy the three REAL workflows into a scratch directory, strip the heartbeat step from one,
    and confirm the actual discovery + assertion functions catch exactly that file -- not a
    synthetic stand-in string handed straight to a helper. This is the difference between "the
    helper behaves as written" and "the check would catch a real regression"."""
    scratch = tmp_path / "workflows"
    scratch.mkdir()
    for path in WORKFLOWS_DIR.glob("*.yml"):
        shutil.copy(path, scratch / path.name)

    scheduled = _scheduled_workflow_paths(scratch)
    assert {p.name for p in scheduled} >= {"nightly.yml", "integrations.yml", "security.yml"}, (
        "the scratch copy did not reproduce the three known-scheduled workflows -- the seeded "
        "defect below would prove nothing"
    )

    baseline_failures = _assert_scheduled_workflows_ping_safely(scratch)
    assert not baseline_failures, (
        f"the unmodified scratch copies already fail: {baseline_failures} -- fix the real "
        "workflows or this test's expectations before trusting the seeded defect below"
    )

    victim = scratch / "nightly.yml"
    text = victim.read_text(encoding="utf-8")
    stripped = re.sub(
        r"\n\s*- name: Report success to the freshness watch.*?(?=\n {6}\S|\Z)",
        "",
        text,
        flags=re.DOTALL,
    )
    assert stripped != text, (
        "the regex removing the heartbeat step matched nothing -- fix the regex"
    )
    victim.write_text(stripped, encoding="utf-8")

    failures = _assert_scheduled_workflows_ping_safely(scratch)
    assert set(failures) == {"nightly.yml"}, (
        f"expected only the seeded victim to fail, got {failures}"
    )
