"""Run every CI-safe script/entry point as a subprocess and assert on its actual output.

These need no checkpoint and no Zenodo data (committed `examples/*.npz` + `structured_output/*.json`
only), so they run in CI. Checkpoint/Zenodo-gated *success* paths are covered locally (see
`test_scripts_local.py`); here we assert those scripts' clean *failure* messages instead.
"""

from __future__ import annotations

import json
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
def test_quick_validation_passes():
    r = _run("scripts/quick_validation.py")
    assert r.returncode == 0, r.stderr
    assert "[PASS] structured_output" in r.stdout
    assert "gating checks passed" in r.stdout


@pytest.mark.unit
def test_aggregate_reproduces_paper_medians():
    r = _run("scripts/aggregate_experimental.py")
    assert r.returncode == 0, r.stderr
    assert "median |dd| = 0.90 Hz" in r.stdout  # the paper-number regression anchor
    assert "median |dJ| = 0.20 Hz" in r.stdout
    assert "proton-count accuracy (overall) = 93.5 %" in r.stdout


@pytest.mark.unit
def test_aggregate_json_is_valid(tmp_path):
    out = tmp_path / "metrics.json"
    r = _run("scripts/aggregate_experimental.py", "--json", str(out))
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data  # non-empty metrics object


@pytest.mark.unit
def test_plot_deposit_spectrum_writes_png(tmp_path):
    out = tmp_path / "roi.png"
    r = _run(
        "scripts/plot_deposit_spectrum.py",
        "--input",
        "examples/roi_S8_example.npz",
        "--out",
        str(out),
    )
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_predict_demo_without_checkpoint_fails_cleanly():
    r = _run("scripts/predict.py", "--demo", "--checkpoint", "no_such.pth")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "Checkpoint not found" in combined and "10.5281/zenodo.21217102" in combined


@pytest.mark.unit
def test_predict_reads_moldetr_checkpoint_env():
    """With no --checkpoint, predict.py falls back to $MOLDETR_CHECKPOINT (the documented convention)."""
    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "GRADIO_ANALYTICS_ENABLED": "False",
        "MOLDETR_CHECKPOINT": "env_no_such.pth",
    }
    r = subprocess.run(
        [sys.executable, "scripts/predict.py", "--demo"],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "env_no_such.pth" in combined  # the env var was used as the checkpoint default


@pytest.mark.unit
def test_aggregate_missing_matched_pairs_fails_cleanly():
    """A bad --matched-pairs path gives a friendly message, not a raw traceback."""
    r = _run("scripts/aggregate_experimental.py", "--matched-pairs", "no_such.json")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "not found" in combined.lower()
    assert "Traceback" not in combined


@pytest.mark.unit
def test_evaluate_experimental_clean_clone_fails_cleanly():
    r = _run("scripts/evaluate_experimental.py")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    # On a clean clone the checkpoint gate fires first with the Zenodo hint (data would be next).
    assert "10.5281/zenodo.21217102" in combined


@pytest.mark.unit
def test_app_imports_and_builds_ui():
    r = _run("-c", "import app; assert type(app.build_ui()).__name__ == 'Blocks'")
    assert r.returncode == 0, r.stderr
