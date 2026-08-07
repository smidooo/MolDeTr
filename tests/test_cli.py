"""The ``moldetr`` console dispatcher: help lists commands, unknown errors, args forward to the sub-main."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_EVAL_DEPS = ("pandas", "seaborn", "sklearn", "cmcrameri")
needs_eval_extra = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in _EVAL_DEPS),
    reason="evaluate_synthetic needs the [eval] extra (pandas/seaborn/scikit-learn/cmcrameri)",
)


#: Subcommands whose module needs an extra the default CI install (`.[dev,app]`) does not provide.
#: Explicit rather than derived: which extra a script needs is a packaging fact the dispatcher does
#: not model — `cli._missing_extra` maps every non-`app` command to the `model` extra, so it cannot
#: be asked. The first draft of the sweep below omitted this and went red on every CI leg while
#: passing locally, because the development venv happens to carry the `eval` packages.
_NEEDS_UNINSTALLED_EXTRA = {"evaluate-synthetic": needs_eval_extra}


def _dispatcher_command_params() -> list:
    """Every subcommand the dispatcher advertises, read from the dispatcher itself.

    Read rather than hard-coded so a new subcommand is covered the moment it is added — a literal
    list here would silently stop covering the thing it was written for.
    """
    from moldetr.cli import COMMANDS

    return [
        pytest.param(cmd, marks=[mark]) if (mark := _NEEDS_UNINSTALLED_EXTRA.get(cmd)) else cmd
        for cmd in sorted(set(COMMANDS) | {"app"})
    ]


@pytest.mark.unit
@pytest.mark.parametrize("command", _dispatcher_command_params())
def test_every_subcommand_answers_help_in_a_real_subprocess(command):
    """`--help` is the one contract every subcommand owes, and it rots silently.

    Honest status: this is a **regression guard, not a bug fix** — all nine subcommands were probed
    before it was written and all nine already exit 0. It is green on arrival and is kept anyway,
    because the failure it guards against (someone edits an argparse block, `--help` starts raising,
    nothing notices) has no other detector.

    Subprocess rather than calling `main()` in-process: `test_cli_routes_each_command_to_its_module`
    already covers routing, but it monkeypatches the target, so it passes even if the sub-script's
    own parser is broken. Only a real invocation exercises that parser. `app` is included because
    `--help` there must print usage rather than launch a blocking server — a regression that once
    shipped.
    """
    result = subprocess.run(
        [sys.executable, "-m", "moldetr.cli", command, "--help"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, (
        f"`moldetr {command} --help` exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-800:]}\n--- stderr ---\n{result.stderr[-800:]}"
    )
    assert result.stdout.strip(), f"`moldetr {command} --help` printed nothing to stdout"


@pytest.mark.unit
def test_cli_help_lists_commands(capsys):
    from moldetr.cli import main

    main(["--help"])
    out = capsys.readouterr().out
    for c in ("predict", "app", "reproduce", "download-weights"):
        assert c in out


@pytest.mark.unit
def test_cli_unknown_command_exits():
    from moldetr.cli import main

    with pytest.raises(SystemExit):
        main(["definitely-not-a-command"])


@pytest.mark.unit
def test_cli_forwards_args_to_subcommand_main(monkeypatch):
    """'reproduce --total-queries 5' calls scripts.aggregate_experimental.main with the right argv."""
    import scripts.aggregate_experimental as agg
    from moldetr import cli

    seen: dict = {}
    monkeypatch.setattr(agg, "main", lambda: seen.setdefault("argv", list(sys.argv)))
    cli.main(["reproduce", "--total-queries", "5"])
    assert seen["argv"][0] == "moldetr reproduce"
    assert seen["argv"][1:] == ["--total-queries", "5"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "cmd,module",
    [
        ("predict", "scripts.predict"),
        ("download-weights", "scripts.download_weights"),
        ("evaluate-experimental", "scripts.evaluate_experimental"),
        ("quick-validation", "scripts.quick_validation"),
    ],
)
def test_cli_routes_each_command_to_its_module(cmd, module, monkeypatch):
    """Every declared subcommand dispatches to the right script's main()."""
    import importlib

    from moldetr.cli import main

    mod = importlib.import_module(module)
    ran = {}
    monkeypatch.setattr(mod, "main", lambda: ran.setdefault("ok", True))
    main([cmd])
    assert ran.get("ok")


@pytest.mark.unit
def test_cli_no_args_prints_usage(capsys):
    from moldetr.cli import main

    main([])
    assert "usage: moldetr" in capsys.readouterr().out


@pytest.mark.unit
def test_cli_app_help_prints_usage_instead_of_launching(capsys, monkeypatch):
    """`moldetr app --help` must describe the command, not start a blocking server.

    The dispatcher forwarded nothing to the app branch, so `--help` fell through to `launch_app()`
    and the terminal hung on a running server, contradicting this module's own docstring ("so
    ``moldetr <cmd> --help`` shows that command's own options").
    """
    import app as app_module

    from moldetr.cli import main

    launched: dict = {}
    monkeypatch.setattr(app_module, "launch_app", lambda *a, **k: launched.setdefault("ran", True))
    main(["app", "--help"])
    assert not launched, "`moldetr app --help` launched the server instead of printing help"
    assert "moldetr app" in capsys.readouterr().out


@pytest.mark.unit
@needs_eval_extra
def test_cli_hydra_subcommand_resolves_its_config():
    """`moldetr evaluate-synthetic` must find `conf/`, exactly as running the script does.

    `@hydra.main(config_path="../conf")` resolves relative to the *file* only when the task
    function's module is `__main__`. Importing it as `scripts.evaluate_synthetic` sent Hydra down
    its config-*module* path instead, so every `moldetr evaluate-synthetic ...` invocation died
    with "Primary config module 'conf' not found" while `python scripts/evaluate_synthetic.py`
    worked.
    """
    r = subprocess.run(
        [sys.executable, "-m", "moldetr.cli", "evaluate-synthetic", "--help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert "Powered by Hydra" in r.stdout


@pytest.mark.unit
def test_cli_restores_sys_argv_after_dispatch(monkeypatch):
    """The dispatcher must not leak its rewritten sys.argv into the caller (test isolation)."""
    import scripts.aggregate_experimental as agg

    from moldetr.cli import main

    before = list(sys.argv)
    monkeypatch.setattr(agg, "main", lambda: None)
    main(["reproduce", "--total-queries", "5"])
    assert sys.argv == before
