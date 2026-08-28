"""`.githooks/pre-commit` refuses a commit that breaks the paper-median reproduction.

Tier-High mandates a pre-commit hook pinning an empirical-claim output; this project had none. The
test that pins the claim already exists --
`tests/test_scripts.py::test_aggregate_reproduces_paper_medians` -- and it is stdlib-only end to
end: `scripts/aggregate_experimental.py` imports only `argparse`/`json`/`statistics`/`pathlib`, and
loading `tests/conftest.py` (which pytest does before collecting anything under `tests/`) needs
nothing beyond `pytest` itself. Measured directly: this one test node id passes in a bare venv that
has `pytest` installed and nothing else -- no `moldetr`, no `numpy`, no `torch`. So the hook needs
only `python` and `pytest` on `PATH`, and this file's fixtures build a scratch repo carrying exactly
that minimal slice rather than a full clone.

`.githooks/` (tracked) rather than `.git/hooks/` (never checked out by `git clone`) is the same fix
already applied once in this repo: an untracked `.git/hooks/commit-msg` silently vanished for every
fresh clone. `scripts/install_hooks.py` points `core.hooksPath` at `.githooks/` so both hooks -- the
commit-msg trailer-stripper and this one -- survive a clone.

"A hook nobody has seen refuse a commit is not a hook": every scenario below actually runs `git
commit` against a real scratch repository and reads its actual exit code, mirroring how
`tests/test_diagram_svgs.py`'s docstring frames the same discipline for a `--check` script nobody
had run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / ".githooks" / "pre-commit"
AGGREGATE_SCRIPT = REPO / "scripts" / "aggregate_experimental.py"
MATCHED_PAIRS = REPO / "structured_output" / "experimental_matched_pairs.json"

_MINIMAL_TEST = '''
"""A trimmed, self-contained copy of the one real assertion this hook exists to run."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_aggregate_reproduces_paper_medians():
    env = {**os.environ, "MPLBACKEND": "Agg"}
    r = subprocess.run(
        [sys.executable, "scripts/aggregate_experimental.py"],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert "median |dd| = 0.90 Hz" in r.stdout
    assert "median |dJ| = 0.20 Hz" in r.stdout
    assert "proton-count accuracy (overall) = 93.5 %" in r.stdout
'''


def _env_with_this_interpreter_on_path() -> dict[str, str]:
    """`os.environ` with `sys.executable`'s directory prepended to PATH.

    The hook shells out to bare `python`/`pytest`, exactly as a developer's activated shell would.
    This test suite may itself run from a Python whose directory is not on the *test runner's own*
    PATH (true of this project's `.venv` under the Bash tool used to develop it), so without this,
    `git commit` spawns the hook with a PATH that cannot find either -- and the hook correctly, but
    unhelpfully, reports "pytest is not available" for what is really a test-fixture gap.
    """
    import os
    import sys

    interpreter_dir = str(Path(sys.executable).resolve().parent)
    env = dict(os.environ)
    env["PATH"] = interpreter_dir + os.pathsep + env.get("PATH", "")
    return env


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=60,
        env=_env_with_this_interpreter_on_path(),
    )


@pytest.fixture
def scratch_repo(tmp_path: Path) -> Path:
    """A minimal git repo carrying only what the hook needs: itself, the two guarded inputs, and a
    trimmed test asserting the one thing the hook checks. No `moldetr` install, no `conftest.py`
    fixtures -- see the module docstring for why the real test doesn't need them either."""
    repo = tmp_path / "scratch"
    repo.mkdir()
    (repo / "scripts").mkdir()
    (repo / "structured_output").mkdir()
    (repo / "tests").mkdir()
    (repo / ".githooks").mkdir()

    shutil.copy(AGGREGATE_SCRIPT, repo / "scripts" / "aggregate_experimental.py")
    shutil.copy(MATCHED_PAIRS, repo / "structured_output" / "experimental_matched_pairs.json")
    (repo / "tests" / "test_scripts.py").write_text(_MINIMAL_TEST, encoding="utf-8")
    shutil.copy(HOOK, repo / ".githooks" / "pre-commit")
    (repo / ".githooks" / "pre-commit").chmod(0o755)

    _run(["git", "init", "-q"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test"], repo)
    _run(["git", "config", "core.hooksPath", ".githooks"], repo)
    _run(["git", "add", "-A"], repo)
    commit = _run(["git", "commit", "-q", "-m", "initial"], repo)
    assert commit.returncode == 0, f"scratch repo's own initial commit failed: {commit.stderr}"
    return repo


@pytest.mark.unit
def test_hook_file_exists_and_is_executable_on_posix():
    assert HOOK.exists(), ".githooks/pre-commit is missing"
    if sys.platform != "win32":
        assert HOOK.stat().st_mode & 0o111, ".githooks/pre-commit is not executable"


@pytest.mark.unit
def test_commit_touching_an_unrelated_file_is_not_slowed_down(scratch_repo: Path):
    """The hook must not run pytest at all for a change that doesn't touch a guarded path -- see
    its own early-exit check.

    Asserts on the hook's own output, not just the exit code: a bare `returncode == 0` would still
    pass if the early exit were deleted and pytest ran (and passed) on every commit, which is
    exactly the "not slowed down" claim this test is supposed to hold the hook to.
    """
    (scratch_repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(["git", "add", "README.md"], scratch_repo)
    result = _run(["git", "commit", "-q", "-m", "unrelated"], scratch_repo)
    assert result.returncode == 0, (
        f"an unrelated commit was refused: {result.stdout}{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "pre-commit:" not in combined, (
        f"the hook produced output for an unrelated commit, meaning it ran (or attempted to run) "
        f"pytest instead of exiting early: {combined!r}"
    )


@pytest.mark.unit
def test_commit_with_an_unchanged_matched_pairs_file_succeeds(scratch_repo: Path):
    """Touching a guarded path with content that still reproduces the paper medians must pass.

    Appends a trailing newline rather than writing back identical bytes: git stages nothing (and
    then refuses to commit for an unrelated reason -- "nothing to commit") for a byte-identical
    rewrite, which would make this test pass or fail for the wrong reason either way.
    """
    matched_pairs = scratch_repo / "structured_output" / "experimental_matched_pairs.json"
    matched_pairs.write_text(matched_pairs.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _run(["git", "add", "structured_output/experimental_matched_pairs.json"], scratch_repo)
    result = _run(["git", "commit", "-q", "-m", "touch, no change"], scratch_repo)
    assert result.returncode == 0, (
        f"a commit that reproduces the paper medians was refused: {result.stdout}{result.stderr}"
    )


@pytest.mark.unit
def test_seeded_defect_a_broken_matched_pairs_file_is_refused(scratch_repo: Path):
    """The point of this file: replace the guarded data with something that reproduces a different
    (wrong) set of numbers, stage it, and confirm `git commit` actually refuses -- not merely that
    some script would have.

    Each `matched_pairs_total` entry is a `[pred, label]` pair (schema read from
    `scripts/aggregate_experimental.py::aggregate`, not guessed); with 215 real pairs, perturbing
    one value would not move the MEDIAN at all, so this replaces the whole array with a single pair
    whose chemical-shift and coupling errors are nowhere near 0.90 Hz / 0.20 Hz -- guaranteed to
    change every one of the three pinned strings, not just nudge one.
    """
    import json

    matched_pairs_path = scratch_repo / "structured_output" / "experimental_matched_pairs.json"
    matched_pairs_path.write_text(
        json.dumps(
            {
                "matched_pairs_total": [
                    [
                        {
                            "chemical_shift_in_points": 0,
                            "proton_count": 1,
                            "coupling_constants": 0.0,
                        },
                        {
                            "chemical_shift_in_points": 5000,
                            "proton_count": 2,
                            "coupling_constants": 999.0,
                        },
                    ]
                ],
                "unmatched_predictions_total": [],
                "unmatched_labels_total": [],
            }
        ),
        encoding="utf-8",
    )

    _run(["git", "add", "structured_output/experimental_matched_pairs.json"], scratch_repo)
    result = _run(["git", "commit", "-q", "-m", "corrupt the paper numbers"], scratch_repo)
    assert result.returncode != 0, (
        "a commit that breaks the paper-median reproduction was NOT refused -- the hook nobody has "
        "seen refuse a commit is not a hook"
    )
    log = _run(["git", "log", "--oneline"], scratch_repo).stdout
    assert "corrupt the paper numbers" not in log, "the bad commit landed despite a nonzero exit"
