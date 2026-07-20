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
