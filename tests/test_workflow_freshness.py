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

The seeded-defect test at the bottom removes the ping step from a scratch copy of one workflow and
confirms the check fails: a guard nobody has seen fire is not a guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO / ".github" / "workflows"
HEARTBEAT_ACTION = ".github/actions/heartbeat-ping"


def _scheduled_workflow_paths() -> list[Path]:
    """Every `.github/workflows/*.yml` whose top-level `on:` block has a `schedule:` trigger.

    Line-based, not a YAML parse -- see `tests/test_security_workflow.py`'s established convention
    in this repo: no `pyyaml` dependency is declared, only available transitively.
    """
    paths = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*schedule:\s*$", text, re.MULTILINE):
            paths.append(path)
    return paths


def _has_heartbeat_step(workflow_text: str) -> bool:
    return HEARTBEAT_ACTION in workflow_text


@pytest.mark.unit
def test_at_least_one_workflow_is_actually_scheduled():
    """Guards the guard: every assertion below is vacuous if no workflow has `schedule:` at all."""
    assert _scheduled_workflow_paths(), (
        "no workflow under .github/workflows/ has a `schedule:` trigger -- either the discovery "
        "regex broke, or every cron lane in this repository was removed"
    )


@pytest.mark.unit
def test_every_scheduled_workflow_pings_the_heartbeat_action():
    missing = [
        path.name for path in _scheduled_workflow_paths()
        if not _has_heartbeat_step(path.read_text(encoding="utf-8"))
    ]
    assert not missing, (
        f"these scheduled workflows have no {HEARTBEAT_ACTION} step: {missing}. GitHub silently "
        "disables a scheduled workflow after 60 days without a commit, and nothing inside GitHub "
        "can notice that about itself -- see docs/MONITORING.md for the external switch every cron "
        "lane must ping on success."
    )


@pytest.mark.unit
def test_heartbeat_action_exists_and_guards_an_empty_url():
    action = REPO / HEARTBEAT_ACTION / "action.yml"
    assert action.exists(), f"{HEARTBEAT_ACTION}/action.yml is missing"
    text = action.read_text(encoding="utf-8")
    assert "GITHUB_STEP_SUMMARY" in text, (
        "the heartbeat action must visibly report an unset URL (e.g. a fork with no secret "
        "configured), not silently no-op -- an invisible skip is the defect class this repo names "
        "in CLAUDE.md's 'Things not to do'"
    )


@pytest.mark.unit
def test_seeded_defect_a_scheduled_workflow_without_the_ping_is_caught(tmp_path):
    """Remove the heartbeat step from a scratch copy and confirm the check would have failed."""
    scratch = tmp_path / "nightly.yml"
    scratch.write_text(
        "name: fake\non:\n  schedule:\n    - cron: \"0 3 * * *\"\njobs:\n  x:\n    steps: []\n",
        encoding="utf-8",
    )
    assert not _has_heartbeat_step(scratch.read_text(encoding="utf-8")), (
        "the seeded scratch workflow unexpectedly contains a heartbeat reference -- the seeded "
        "defect did not actually remove what this test exists to catch"
    )
