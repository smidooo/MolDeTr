"""The integrations lane must collect in an environment that has no third-party stack.

`.github/workflows/integrations.yml` installs `pytest` alone, on the reasoning that
`tests/test_integrations.py` imports only the standard library. That reasoning is correct about
*that file* and wrong about the run: pytest loads `tests/conftest.py` before collecting anything
under `tests/`, and the conftest imported numpy and torch at module scope. Collection dies before a
single check runs, and the job's own failure handler then files a `ModuleNotFoundError` issue that
looks like a broken dependency rather than a job that never worked. The rot watcher was rotten.

It was caught before it ever fired — the lane was added in #40 and its first scheduled run is Monday
2026-08-10, so this is what *would* have happened rather than what did.

Nothing inside the job could notice: it reports the ImportError faithfully. Only a test that
*recreates the job's environment* can.

Two things here are deliberately stricter than the bug that prompted them:

**The shim is an allowlist, not a denylist.** Blocking `{numpy, torch}` would guard the two modules
that happened to break it and green-light the next one — `httpx`, say, which `tests/e2e/conftest.py`
already imports and which arrives transitively with gradio. The job installs *only pytest*, so the
faithful question is "would a `pip install pytest` venv have this?", and anything else must be
refused. `PYTEST_ONLY_CLOSURE` was read off such a venv rather than guessed.

**The invocation is read from the workflow, not restated here.** A guard that hard-codes
`tests/test_integrations.py` stops covering the job the moment a second file is added to that step —
which is the same shape as the defect. `tests/test_marker_hygiene.py` makes this argument about
itself and already parses these workflows; this follows it.

The assertion compares collected node IDs against an unblocked baseline rather than a literal count:
`test_every_declared_doi_still_resolves` is parametrized over `CITATION.cff`, so the count moves
whenever a DOI is added, and a hardcoded number would be right today and quietly wrong later.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "integrations.yml"

#: Every top-level module a `pip install pytest` venv actually provides, read off one built for the
#: purpose (pytest 9.1.1) rather than assumed. Widen this ONLY after checking that `pip install
#: pytest` really ships the addition — every name here is a hole in the guard.
PYTEST_ONLY_CLOSURE = frozenset(
    {"_pytest", "colorama", "iniconfig", "packaging", "pip", "pluggy", "py", "pygments", "pytest"}
)

#: This repository's own importable roots. Present in the job because the checkout is.
FIRST_PARTY = frozenset({"app", "app_ui", "conftest", "moldetr", "scripts", "tests"})

#: Substring the shim puts in its own ImportError, so the self-check can prove the refusal came from
#: the shim rather than from the module being absent anyway.
SHIM_MARKER = "installs only pytest"

SUBPROCESS_TIMEOUT = 300


def _shim_source() -> str:
    """A `sitecustomize.py` that permits the standard library and nothing else the job lacks."""
    extra = sorted(PYTEST_ONLY_CLOSURE | FIRST_PARTY)
    return (
        "import sys\n"
        f"EXTRA = {extra!r}\n"
        "ALLOWED = set(sys.stdlib_module_names) | set(EXTRA)\n"
        f"MESSAGE = 'the integrations job {SHIM_MARKER}, so this is absent there'\n"
        "\n"
        "\n"
        "class _OnlyWhatTheJobInstalls:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] not in ALLOWED:\n"
        "            raise ImportError(f'No module named {name!r} — ' + MESSAGE)\n"
        "        return None\n"
        "\n"
        "\n"
        "sys.meta_path.insert(0, _OnlyWhatTheJobInstalls())\n"
    )


def _workflow_pytest_argv() -> list[str]:
    """The integrations job's own pytest invocation, read from the workflow rather than restated."""
    for raw in WORKFLOW.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("run: pytest ") and "test_integrations" in line:
            return shlex.split(line.removeprefix("run: "))
    raise AssertionError(
        f"no `run: pytest ... test_integrations ...` step found in {WORKFLOW.name}. The guard can "
        "no longer tell what the job runs, so it fails rather than silently checking a stale "
        "invocation — re-point it at whatever the step became."
    )


def _run(args: list[str], shim: Path | None) -> subprocess.CompletedProcess[str]:
    """Run `python <args>` from the repo root, optionally with the block shim on `PYTHONPATH`.

    Plugin autoload is off in both directions: the job's venv has no plugins, while a developer
    machine has nbmake, hypothesis, pytest-cov and playwright. Leaving autoload on would let a
    conftest that only works *because* a plugin is installed pass here and fail in CI.
    """
    env = dict(os.environ)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if shim is not None:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join([str(shim), existing]) if existing else str(shim)
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )


def _collect(shim: Path | None = None) -> tuple[int, set[str], str]:
    """Collect the integrations lane exactly as the workflow invokes it."""
    argv = _workflow_pytest_argv()
    proc = _run(["-m", "pytest", *argv[1:], "--collect-only"], shim)
    node_ids = {
        line.strip().replace("\\", "/")
        for line in proc.stdout.splitlines()
        if "test_integrations.py::" in line
    }
    return proc.returncode, node_ids, proc.stdout + proc.stderr


@pytest.fixture
def block_shim(tmp_path: Path) -> Path:
    """The shim, proven to be doing the blocking before anything is built on it.

    `returncode != 0` alone would be satisfied by a module that is simply not installed, so a shim
    that failed to load altogether would look identical to one working perfectly. Requiring the
    shim's own marker in the output is what separates those two.
    """
    shim = tmp_path / "blockshim"
    shim.mkdir()
    (shim / "sitecustomize.py").write_text(_shim_source(), encoding="utf-8")

    probe = _run(["-c", "import numpy"], shim)
    combined = probe.stdout + probe.stderr
    assert probe.returncode != 0 and SHIM_MARKER in combined, (
        "the shim did not refuse numpy for its own stated reason, so it is probably not loading at "
        f"all and any result built on it would be vacuous:\n{combined}"
    )
    return shim


@pytest.mark.unit
def test_integrations_lane_collects_without_the_third_party_stack(block_shim: Path) -> None:
    """The weekly job's own invocation must collect the same tests a full environment collects."""
    baseline_rc, baseline_ids, baseline_out = _collect()
    assert baseline_rc == 0, (
        f"baseline collection failed; this test proves nothing:\n{baseline_out}"
    )
    assert baseline_ids, f"baseline collected no tests, so set equality is vacuous:\n{baseline_out}"

    blocked_rc, blocked_ids, blocked_out = _collect(block_shim)

    assert blocked_rc == 0, (
        "collection failed in an environment holding only the standard library and pytest — which "
        "is what `integrations.yml` runs every Monday, so none of its checks ran. Whatever "
        f"`tests/conftest.py` now imports at module scope has to move into a function:\n"
        f"{blocked_out}"
    )
    assert blocked_ids == baseline_ids, (
        "the integrations lane collects a different set of tests without the third-party stack.\n"
        f"  missing when blocked: {sorted(baseline_ids - blocked_ids)}\n"
        f"  extra when blocked:   {sorted(blocked_ids - baseline_ids)}"
    )
