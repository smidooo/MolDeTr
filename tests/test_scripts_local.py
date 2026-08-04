"""Checkpoint-gated *success* paths for the scripts/ entry points (local only).

The CI-safe *failure* paths live in ``test_scripts.py``. These need the 974 MB checkpoint
(``MOLDETR_CHECKPOINT``) and — for the evaluation — the Zenodo ROI arrays (``roi_S*.npz`` in
``structured_output/`` or a dir in ``MOLDETR_ROI_NPZ_DIR``); each gate ``skip``s (never fails) when
its asset is absent, so CI stays green.
"""

from __future__ import annotations

import glob
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
