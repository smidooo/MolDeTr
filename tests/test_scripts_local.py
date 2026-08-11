"""Checkpoint-gated *success* paths for the scripts/ entry points (local only).

The CI-safe *failure* paths live in ``test_scripts.py``. These need the 974 MB checkpoint
(``MOLDETR_CHECKPOINT``) and — for the evaluation — the Zenodo ROI arrays (``roi_S*.npz`` in
``structured_output/`` or a dir in ``MOLDETR_ROI_NPZ_DIR``); each gate ``skip``s (never fails) when
its asset is absent, so CI stays green.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.model

REPO = Path(__file__).resolve().parent.parent
CKPT = os.environ.get("MOLDETR_CHECKPOINT", "")
ROI_DIR = os.environ.get("MOLDETR_ROI_NPZ_DIR", str(REPO / "structured_output"))
FIGURE_NUMBERS = REPO / "docs" / "figure_predictions.json"


def _md5(path: Path) -> str:
    """Streamed, because the checkpoint is 974 MB."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


checkpoint_required = pytest.mark.skipif(
    not (CKPT and Path(CKPT).exists()),
    reason="real checkpoint absent (set MOLDETR_CHECKPOINT) — local-only success-path test",
)


def _run(*args, timeout: int = 600) -> subprocess.CompletedProcess:
    env = {**os.environ, "MPLBACKEND": "Agg", "GRADIO_ANALYTICS_ENABLED": "False"}
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@checkpoint_required
def test_predict_on_example_emits_physical_detections() -> None:
    """predict.py on the committed vanillin ROI yields a non-empty table with physical δ / J."""
    r = _run("scripts/predict.py", "--input", "examples/roi_S8_example.npz", "--checkpoint", CKPT)
    assert r.returncode == 0, r.stderr
    n = re.search(r"Detected (\d+) multiplet", r.stdout)
    assert n and int(n.group(1)) > 0, r.stdout  # a real detection table, not empty
    shifts = [float(s) for s in re.findall(r"shift=(-?[\d.]+) ppm", r.stdout)]
    assert shifts and all(0.0 <= s <= 12.0 for s in shifts), r.stdout  # physical ¹H δ window
    js = [float(j) for j in re.findall(r"max J=([\d.]+) Hz", r.stdout)]
    assert all(0.0 <= j <= 30.0 for j in js), r.stdout  # physical H–H coupling magnitudes


@checkpoint_required
@pytest.mark.parametrize("figure", sorted(json.loads(FIGURE_NUMBERS.read_text())["figures"]))
def test_the_checkpoint_still_produces_the_numbers_the_figures_publish(figure: str) -> None:
    """`docs/figure_predictions.json` is what the README figures print; this ties it to the weights.

    The half of the contract CI cannot hold. `tests/test_figure_numbers.py` ties the committed SVG
    to that file and runs everywhere; only this one can ask whether the file is still *true*, and it
    skips wherever the 974 MB checkpoint is absent -- which is every CI lane. **A green suite is not
    evidence about this test.** Read the skip count.

    Why it exists at all: before the file, those numbers lived only as pixels inside a PNG, and
    nothing in the repo would have noticed the checkpoint drifting away from the front page. The
    test above is not that guard -- it asserts only that shifts are 0-12 ppm and couplings 0-30 Hz,
    which the superseded vanillin figure satisfied while transposing two of its three couplings.

    Tolerances are loose against float noise and tight against the failure actually seen: the
    transposition swapped 8.25 and 8.74 Hz, a 0.49 Hz move, ten times the bar here.
    """
    spec = json.loads(FIGURE_NUMBERS.read_text())
    assert spec["provenance"]["checkpoint_md5"] == _md5(Path(CKPT)), (
        "this is not the checkpoint the figures were measured from, so a mismatch below would say "
        "nothing about drift. Fetch it with scripts/download_weights.py, or point MOLDETR_CHECKPOINT "
        "at the published file."
    )

    r = _run("scripts/predict.py", "--input", spec["figures"][figure]["npz"], "--checkpoint", CKPT)
    assert r.returncode == 0, r.stderr
    live = [
        (float(s), float(j), float(w))
        for s, j, w in re.findall(
            r"shift=(-?[\d.]+) ppm\s+max J=([\d.]+) Hz\s+linewidth=([\d.]+) Hz", r.stdout
        )
    ]
    rows = spec["figures"][figure]["rows"]
    assert len(live) == len(rows), (
        f"{figure}: the checkpoint now returns {len(live)} multiplet(s), the figure prints "
        f"{len(rows)}\n{r.stdout}"
    )
    for (shift, max_j, width), row in zip(live, rows):
        for got, want, tol, field in (
            (shift, row["shift_ppm"], 0.01, "shift_ppm"),
            (max_j, row["max_j_hz"], 0.05, "max_j_hz"),
            (width, row["linewidth_hz"], 0.05, "linewidth_hz"),
        ):
            assert abs(got - want) <= tol, (
                f"{figure} row {row['n']}: {field} is {got}, but docs/figure_predictions.json "
                f"publishes {want}. Either the checkpoint changed, or the file is stale -- "
                f"regenerate it and rebuild the figure, do not edit one without the other."
            )


@checkpoint_required
@pytest.mark.skipif(
    not glob.glob(str(Path(ROI_DIR) / "roi_S*.npz")),
    reason="Zenodo ROI npz absent (drop roi_S*.npz in structured_output/ or set MOLDETR_ROI_NPZ_DIR)",
)
def test_evaluate_experimental_reproduces_paper_median() -> None:
    """evaluate_experimental.py regenerates predictions and lands near the paper's ~0.89 Hz median."""
    r = _run(
        "scripts/evaluate_experimental.py", "--checkpoint", CKPT, "--structured-output", ROI_DIR
    )
    assert r.returncode == 0, r.stderr
    m = re.search(r"median \|dd\| = ([\d.]+) Hz", r.stdout)
    assert m, r.stdout
    # paper ~0.89 Hz; this script's live decode approximates it rather than reproducing the
    # aggregate figure exactly — see docs/SCOPE.md:88 for why the two paths differ.
    assert 0.5 <= float(m.group(1)) <= 1.3, r.stdout
    assert "proton-count accuracy" in r.stdout
