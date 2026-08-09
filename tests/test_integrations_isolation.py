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
refused. That closure is **measured in the running interpreter**, not hardcoded — pytest needs
`tomli` and `exceptiongroup` only below Python 3.11, so a list read off one version is wrong on
another. See `_pytest_closure`.

**The invocation is read from the workflow, not restated here.** A guard that hard-codes
`tests/test_integrations.py` stops covering the job the moment a second file is added to that step —
which is the same shape as the defect. `tests/test_marker_hygiene.py` makes this argument about
itself and already parses these workflows; this follows it.

The assertion compares collected node IDs against an unblocked baseline rather than a literal count:
`test_every_declared_doi_still_resolves` is parametrized over `CITATION.cff`, so the count moves
whenever a DOI is added, and a hardcoded number would be right today and quietly wrong later.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "integrations.yml"

#: Probe reporting what a `pip install pytest` environment provides on the interpreter running it.
#:
#: It walks pytest's *declared* requirements with environment markers evaluated, and does not rely
#: on what `import pytest` happens to pull in. That distinction is the whole point: pytest imports
#: `tomli` lazily, only when it parses `pyproject.toml`, so an import-time snapshot misses it on
#: Python 3.10 and the shim then refuses a module pytest genuinely needs.
_CLOSURE_PROBE = r"""
import json, sys, pytest
from importlib.metadata import distribution

names = {m.split(".")[0] for m in sys.modules}
try:
    from packaging.requirements import Requirement
except Exception:
    Requirement = None

if Requirement is not None:
    seen, queue = set(), ["pytest"]
    while queue:
        raw = queue.pop()
        key = raw.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        try:
            dist = distribution(raw)
        except Exception:
            continue
        names.add(raw.replace("-", "_"))
        names.update((dist.read_text("top_level.txt") or "").split())
        for req in dist.requires or []:
            try:
                parsed = Requirement(req)
            except Exception:
                continue
            # Markers carry the version conditions -- `tomli>=1; python_version < "3.11"`.
            # Evaluating them here is what makes this correct per interpreter.
            if parsed.marker is None or parsed.marker.evaluate():
                queue.append(parsed.name)

print(json.dumps(sorted(n for n in names if n)))
"""

#: Python's own `site` machinery. `site` imports these by name after the shim is installed, and
#: blocking them makes the interpreter report an import error that has nothing to do with the lane.
SITE_MACHINERY = frozenset({"sitecustomize", "usercustomize"})

#: This repository's own importable roots. Present in the job because the checkout is.
FIRST_PARTY = frozenset({"app", "app_ui", "conftest", "moldetr", "scripts", "tests"})

#: Substring the shim puts in its own ImportError, so the self-check can prove the refusal came from
#: the shim rather than from the module being absent anyway.
SHIM_MARKER = "installs only pytest"

SUBPROCESS_TIMEOUT = 300


def _pytest_closure() -> set[str]:
    """Top-level modules a `pip install pytest` environment provides — measured in *this*
    interpreter, never hardcoded.

    A literal list was wrong, and wrong in a way that only showed up in CI. pytest's dependencies
    are version-conditional: `pytest` requires `tomli>=1` and `exceptiongroup>=1` **only** on
    `python_version < "3.11"`. A closure read off a 3.12 venv therefore blocks both on 3.10, the
    blocked collection fails for a reason that has nothing to do with the conftest, and the guard
    goes RED while reporting the wrong cause. That is exactly what happened: all three py3.10 legs
    failed while 3.11 and 3.12 passed. Measuring beats guessing, but only if you measure on the
    thing that actually runs it.

    Slight over-permission is possible (a richer dev environment could import something a bare
    pytest venv would not), which is why `FIRST_PARTY` stays explicit and the floor below asserts
    the probe measured a pytest install at all rather than silently returning junk.
    """
    proc = _run(["-c", _CLOSURE_PROBE], None)
    assert proc.returncode == 0, (
        f"could not measure the pytest closure:\n{proc.stdout}{proc.stderr}"
    )
    closure = set(json.loads(proc.stdout))

    floor = {"pytest", "_pytest", "pluggy", "iniconfig", "packaging"}
    if sys.version_info < (3, 11):
        # pytest declares these only for `python_version < "3.11"`, and `tomli` in particular is
        # imported lazily when pyproject.toml is parsed — so an import-time snapshot misses it and
        # the shim then blocks it. That cost three red py3.10 legs; assert it here, where the
        # message says what is wrong, instead of downstream as an opaque collection error.
        floor |= {"tomli", "exceptiongroup"}

    missing = floor - closure
    assert not missing, (
        f"the closure probe did not measure a usable pytest install on Python "
        f"{sys.version_info.major}.{sys.version_info.minor}; missing {sorted(missing)}"
    )
    return closure


def _shim_source() -> str:
    """A `sitecustomize.py` that permits the standard library and nothing else the job lacks."""
    extra = sorted(_pytest_closure() | FIRST_PARTY | SITE_MACHINERY)
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


def _run(
    args: list[str], shim: Path | None, executable: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `<executable> <args>` from the repo root, optionally with the block shim on `PYTHONPATH`.

    `executable` defaults to this interpreter; `_collect` overrides it with the `pytest` console
    script so the guard runs the job's actual entry point. The shim travels on `PYTHONPATH` either
    way — `sitecustomize` is imported by the interpreter at startup regardless of entry point.

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
        [executable or sys.executable, *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )


def _pytest_entrypoint() -> list[str]:
    """The `pytest` console script, which is what the workflow actually types.

    `python -m pytest` is NOT equivalent, and the difference is the whole reason this matters:
    `-m` prepends the working directory to `sys.path`, the console script does not. That was inert
    until `tests/test_integrations.py` began importing `scripts.zenodo_add_paper_doi` at module
    scope, which resolves only if the repo root is importable. It is today — `tests/__init__.py`
    makes pytest's prepend mode walk up to the repo root — but a guard that runs a *more permissive*
    invocation than the job cannot see that break. Remove `tests/__init__.py` or switch to
    `importmode=importlib` and the job dies at collection while this file stays green, which is
    precisely the shape of the defect this module exists to catch.

    Falls back to `-m pytest` where no console script exists, rather than skipping: a weaker check
    beats no check, and the fallback is announced in the failure message via `_collect`.
    """
    scripts_dir = Path(sys.executable).parent
    for name in ("pytest.exe", "pytest"):
        candidate = scripts_dir / name
        if candidate.is_file():
            return [str(candidate)]
    return [sys.executable, "-m", "pytest"]


def _collect(shim: Path | None = None) -> tuple[int, set[str], str]:
    """Collect the integrations lane exactly as the workflow invokes it."""
    argv = _workflow_pytest_argv()
    executable, *prefix = _pytest_entrypoint()
    proc = _run([*prefix, *argv[1:], "--collect-only"], shim, executable)
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
