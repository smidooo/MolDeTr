"""`scripts/install_hooks.py` points `core.hooksPath` at the tracked `.githooks/` directory.

Runs against a real scratch git repo (mirroring `tests/test_pre_commit_hook.py`'s discipline): a
claim about `git config` behavior is worth checking against real git, not just against the script's
own return value.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
INSTALL_SCRIPT = REPO / "scripts" / "install_hooks.py"


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, timeout=30)


@pytest.fixture
def scratch_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "scratch"
    (repo / ".githooks").mkdir(parents=True)
    (repo / ".githooks" / "pre-commit").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (repo / ".githooks" / "commit-msg").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    _run(["git", "init", "-q"], repo)
    return repo


@pytest.mark.unit
def test_sets_core_hooks_path(scratch_repo: Path):
    result = _run([sys.executable, str(INSTALL_SCRIPT)], scratch_repo)
    assert result.returncode == 0, result.stdout + result.stderr

    check = _run(["git", "config", "--get", "core.hooksPath"], scratch_repo)
    assert check.stdout.strip() == ".githooks"


@pytest.mark.unit
def test_is_idempotent(scratch_repo: Path):
    _run([sys.executable, str(INSTALL_SCRIPT)], scratch_repo)
    second = _run([sys.executable, str(INSTALL_SCRIPT)], scratch_repo)
    assert second.returncode == 0, second.stdout + second.stderr


@pytest.mark.unit
def test_refuses_to_clobber_a_different_hooks_path_without_force(scratch_repo: Path):
    _run(["git", "config", "core.hooksPath", "some-other-hooks-dir"], scratch_repo)
    result = _run([sys.executable, str(INSTALL_SCRIPT)], scratch_repo)
    assert result.returncode != 0

    check = _run(["git", "config", "--get", "core.hooksPath"], scratch_repo)
    assert check.stdout.strip() == "some-other-hooks-dir", "the existing setting was overwritten"


@pytest.mark.unit
def test_force_overwrites_a_different_hooks_path(scratch_repo: Path):
    _run(["git", "config", "core.hooksPath", "some-other-hooks-dir"], scratch_repo)
    result = _run([sys.executable, str(INSTALL_SCRIPT), "--force"], scratch_repo)
    assert result.returncode == 0, result.stdout + result.stderr

    check = _run(["git", "config", "--get", "core.hooksPath"], scratch_repo)
    assert check.stdout.strip() == ".githooks"
