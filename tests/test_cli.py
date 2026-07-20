"""The ``moldetr`` console dispatcher: help lists commands, unknown errors, args forward to the sub-main."""

import sys

import pytest


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
def test_cli_restores_sys_argv_after_dispatch(monkeypatch):
    """The dispatcher must not leak its rewritten sys.argv into the caller (test isolation)."""
    import scripts.aggregate_experimental as agg

    from moldetr.cli import main

    before = list(sys.argv)
    monkeypatch.setattr(agg, "main", lambda: None)
    main(["reproduce", "--total-queries", "5"])
    assert sys.argv == before
