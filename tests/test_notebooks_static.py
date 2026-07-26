"""Static import guard for the shipped notebooks (fast lane, weight-free).

Regression guard for the e816d64 failure class: ``theme.py`` moved to ``app_ui/theme.py``
while ``notebooks/MolDeTr_colab_demo.ipynb`` kept ``from theme import ...``, and no CI lane
executes notebooks (running them needs the 974 MB checkpoint), so the break shipped silently.
These tests parse every code cell of every shipped notebook (IPython magics stripped) and
assert, without executing any cell:

1. every absolute import resolves on the CI ``sys.path`` (``tests/conftest.py`` puts the
   repo root there), unless the module exists only inside Colab;
2. for modules that live in this repo, every imported name exists, so a renamed symbol is
   caught as well as a renamed module.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((REPO / "notebooks").glob("*.ipynb"))

# Modules that exist only inside the Colab runtime, never on CI.
COLAB_ONLY = frozenset({"google"})


def _code_sources(nb_path: Path) -> list[str]:
    """Return each code cell's source with IPython magic/shell lines removed."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    sources: list[str] = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        raw = cell["source"]
        text = "".join(raw) if isinstance(raw, list) else raw
        kept = [ln for ln in text.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        sources.append("\n".join(kept))
    return sources


def _imports(nb_path: Path) -> list[tuple[str, list[str]]]:
    """All ``(module, imported_names)`` pairs across the notebook's code cells."""
    found: list[tuple[str, list[str]]] = []
    for source in _code_sources(nb_path):
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - only on a broken notebook
            pytest.fail(f"{nb_path.name}: cell does not parse after magic-stripping: {exc}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.extend((alias.name, []) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.append((node.module, [alias.name for alias in node.names]))
    return found


def _resolves(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # missing parent package, or an oddball spec
        return False


def _repo_local(module: str) -> bool:
    """True when the module's source file lives inside this repository."""
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return False
    return Path(spec.origin).resolve().is_relative_to(REPO)


@pytest.mark.unit
def test_notebooks_discovered() -> None:
    assert len(NOTEBOOKS) >= 2, f"notebook glob found only {NOTEBOOKS}; path broken?"


@pytest.mark.unit
@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_imports_resolve(nb_path: Path) -> None:
    missing = sorted(
        {
            module
            for module, _ in _imports(nb_path)
            if module.split(".")[0] not in COLAB_ONLY and not _resolves(module)
        }
    )
    assert not missing, (
        f"{nb_path.name} imports unresolvable module(s) {missing}: "
        "a repo module was moved or renamed without updating the notebook"
    )


@pytest.mark.unit
@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_repo_local_symbols_exist(nb_path: Path) -> None:
    for module, names in _imports(nb_path):
        if not names or not _resolves(module) or not _repo_local(module):
            continue
        mod = importlib.import_module(module)
        gone = [name for name in names if name != "*" and not hasattr(mod, name)]
        assert not gone, (
            f"{nb_path.name}: `from {module} import {', '.join(gone)}`: symbol gone or renamed"
        )
