"""The committed diagrams must still be what ``scripts/build_diagram_svgs.py`` produces.

The generator has shipped a ``--check`` mode since the diagrams were introduced, and **nothing ran
it** — not CI, not any test. Grepped the tree: the only references were prose. So a change to the
generator that was never re-run, or an SVG hand-edited after the fact, left the fourteen committed
files silently disagreeing with the source that claims to produce them, and the existing figure
tests could not notice: they assert a viewBox exists, that light and dark share a geometry, and that
the banner's trace plots its NPZ — all of which a stale file satisfies perfectly well.

That matters most for the light/dark pairs. They are ONE geometry with two palettes precisely so
they cannot drift, and a half-regenerated pair is the drift the shared geometry was meant to make
impossible.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _run(*args, timeout: int = 300):
    env = {**os.environ, "MPLBACKEND": "Agg", "GRADIO_ANALYTICS_ENABLED": "False"}
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.unit
def test_the_committed_svgs_match_the_generator() -> None:
    """The wiring this file exists for: run the mode that was shipped and never called."""
    from scripts.build_diagram_svgs import DIAGRAMS

    r = _run("scripts/build_diagram_svgs.py", "--check")
    assert r.returncode == 0, (
        "committed SVG(s) disagree with scripts/build_diagram_svgs.py. Regenerate with "
        f"`python scripts/build_diagram_svgs.py` and commit the result.\n{r.stdout}\n{r.stderr}"
    )
    # The count, not just the phrase: `--check` iterates DIAGRAMS, so a diagram removed from that
    # mapping stops being verified while this line still reads "match this source". Pinning the
    # number makes the coverage of this check visible in its own assertion.
    assert f"{2 * len(DIAGRAMS)} committed SVG(s) match this source" in r.stdout, r.stdout


@pytest.mark.unit
def test_check_mode_actually_detects_a_stale_file(tmp_path: Path, monkeypatch) -> None:
    """Otherwise the test above is a green tick that would survive `--check` doing nothing.

    Runs entirely in ``tmp_path``: ``main()`` reads ``OUT_DIR`` as its ``--out-dir`` default *at
    call time*, and its own guard compares the resolved value against the same global — so patching
    the module attribute redirects both, and the guard is satisfied rather than bypassed. The repo's
    own SVGs are never written to, which is why this can corrupt a file to prove the point.

    ``--only mark`` because one diagram is enough to state the property and the banner solves a few
    thousand Lorentzian points per panel.
    """
    import scripts.build_diagram_svgs as gen

    monkeypatch.setattr(gen, "OUT_DIR", tmp_path)
    # `main()` ASSIGNS the module's `TRACE`, so an in-process call leaks its choice into every later
    # one. Restoring it keeps this test from becoming the thing that breaks the next.
    monkeypatch.setattr(gen, "TRACE", gen.TRACE)
    monkeypatch.setattr(sys, "argv", ["build_diagram_svgs.py", "--only", "mark"])
    assert gen.main() == 0, "generating into a fresh directory should succeed"

    monkeypatch.setattr(sys, "argv", ["build_diagram_svgs.py", "--only", "mark", "--check"])
    assert gen.main() == 0, "freshly generated files must satisfy --check"

    target = tmp_path / "mark.svg"
    target.write_text(
        target.read_text(encoding="utf-8").replace("</svg>", "<!--x--></svg>"), encoding="utf-8"
    )
    assert gen.main() == 1, "--check returned 0 for a file that no longer matches its source"

    target.unlink()
    assert gen.main() == 1, "--check returned 0 for a file that is missing entirely"
