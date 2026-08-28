"""Install this repository's tracked git hooks by pointing `core.hooksPath` at `.githooks/`.

    python scripts/install_hooks.py

Why a tracked hooks directory rather than `.git/hooks/` directly: `.git/hooks/` is never checked
out by `git clone` -- it is machine state, not repository content -- so a hook that lives only
there is lost on every fresh clone. This project already had exactly that problem: an untracked
`.git/hooks/commit-msg` stripped Claude/Anthropic co-author trailers, silently absent for anyone
who cloned after it was added. `.githooks/` is tracked; this script's only job is to tell git to
read hooks from there instead.

Idempotent: safe to run again. Refuses to silently overwrite a DIFFERENT, already-configured
`core.hooksPath` -- a contributor may have their own hook setup, and clobbering it without asking
is the wrong default. Pass `--force` to overwrite anyway.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TARGET = ".githooks"


def _repo_root() -> Path:
    """The top level of the git repo the caller is standing in, via `git`, not `__file__`.

    Resolving from `__file__` would hardcode this project's own checkout regardless of where the
    script is invoked from or against -- harmless for the documented `python scripts/install_hooks.py`
    usage, but wrong the moment this script is vendored elsewhere or exercised against a scratch
    repo in a test. `git rev-parse --show-toplevel` answers "which repo is the CURRENT DIRECTORY
    actually in", which is the question that matters.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        print(
            "error: not inside a git repository (or git is not on PATH) -- "
            "nothing to install hooks into",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    return Path(result.stdout.strip())


def _current_hooks_path(repo: Path) -> str | None:
    # `--local`, not a bare `--get`: a global or system-scope `core.hooksPath` (husky, a company
    # template) would otherwise make this report "already set" for a value that isn't even
    # repo-local, and the write below is already local-only -- the read should match.
    result = subprocess.run(
        ["git", "config", "--local", "--get", "core.hooksPath"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def install(force: bool = False) -> int:
    repo = _repo_root()
    hooks_dir = repo / ".githooks"
    if not hooks_dir.is_dir():
        print(f"error: {hooks_dir} does not exist -- nothing to install", file=sys.stderr)
        return 1

    current = _current_hooks_path(repo)
    if current is not None and current != TARGET and not force:
        print(
            f"core.hooksPath is already set to {current!r}, not {TARGET!r}. Not overwriting a "
            "hook setup that may be deliberate -- pass --force to overwrite anyway.",
            file=sys.stderr,
        )
        return 1

    subprocess.run(["git", "config", "core.hooksPath", TARGET], cwd=repo, check=True)

    for hook in sorted(hooks_dir.iterdir()):
        if hook.is_file():
            hook.chmod(hook.stat().st_mode | 0o111)

    print(
        f"core.hooksPath set to {TARGET!r}. Installed hooks: "
        f"{', '.join(p.name for p in sorted(hooks_dir.iterdir()) if p.is_file())}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing, different core.hooksPath"
    )
    args = parser.parse_args(argv)
    return install(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
