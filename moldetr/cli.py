"""The ``moldetr`` console entry point — a thin dispatcher over the repo's scripts.

Installed by ``pip install -e .`` as the ``moldetr`` command, so a clone can run::

    moldetr predict --demo
    moldetr app
    moldetr reproduce
    moldetr download-weights

Each subcommand forwards its remaining arguments to the matching script's ``main()`` (or, for ``app``,
launches the Gradio UI), so ``moldetr <cmd> --help`` shows that command's own options.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Make the repo root importable (scripts/ + app.py live there) — robust for an editable install.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# subcommand -> module exposing main()
COMMANDS: dict[str, str] = {
    "predict": "scripts.predict",
    "detect": "scripts.predict",  # alias
    "reproduce": "scripts.aggregate_experimental",
    "download-weights": "scripts.download_weights",
    "evaluate-experimental": "scripts.evaluate_experimental",
    "evaluate-synthetic": "scripts.evaluate_synthetic",
    "quick-validation": "scripts.quick_validation",
    "simulate-predict": "scripts.simulate_and_predict",
}


def _usage() -> str:
    cmds = "\n  ".join(sorted(set(COMMANDS) | {"app"}))
    return (
        f"usage: moldetr <command> [args...]\n\ncommands:\n  {cmds}\n\n"
        "Run 'moldetr <command> --help' for a command's own options."
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "app":  # the Gradio app has no main(); it launches build_ui()
        app = importlib.import_module("app")
        app.build_ui().launch(theme=app.MOLDETR_THEME, css=app.CUSTOM_CSS)
        return
    if cmd not in COMMANDS:
        print(f"moldetr: unknown command '{cmd}'\n\n{_usage()}", file=sys.stderr)
        raise SystemExit(2)
    module = importlib.import_module(COMMANDS[cmd])
    saved_argv = sys.argv
    try:
        sys.argv = [f"moldetr {cmd}", *rest]  # let the sub-main's argparse see only its own args
        module.main()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    main()
