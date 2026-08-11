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


#: Makes ``moldetr.config`` unimportable, then runs the script. Note this also breaks the
#: ``module_imports`` gate, so the assertion below has to name ``config_imports`` specifically --
#: a bare "exit code is non-zero" would pass for the wrong reason.
_CONFIG_IMPORT_SABOTAGE = """
import importlib.abc, runpy, sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "moldetr.config":
            raise ModuleNotFoundError("No module named 'moldetr.config'", name=fullname)
        return None

sys.meta_path.insert(0, Blocker())
runpy.run_path("scripts/quick_validation.py", run_name="__main__")
"""


@pytest.mark.unit
def test_quick_validation_config_gate_can_actually_fail():
    """A gate that cannot fail is not a gate, and CI runs this script as one.

    ``check_config_imports`` used to wrap a bare ``print`` in ``try``/``except``: nothing was
    imported, so the handler was unreachable and the gate reported PASS unconditionally -- while
    its docstring said it checked that the config loads.
    """
    env = {**os.environ, "MPLBACKEND": "Agg"}
    r = subprocess.run(
        [sys.executable, "-c", _CONFIG_IMPORT_SABOTAGE],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert "[FAIL] config_imports" in r.stdout, (
        "with moldetr.config unimportable the config gate must report FAIL; got:\n" + r.stdout
    )
    assert r.returncode != 0


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
def test_load_input_returns_ground_truth_from_roi_npz():
    """load_input surfaces the ground truth stored in a ROI npz (feeds the dashed overlay)."""
    from scripts.predict import load_input

    amplitudes, cal, ground_truth = load_input(str(REPO / "examples" / "roi_S8_example.npz"))
    assert amplitudes.ndim == 1 and amplitudes.size > 0
    assert cal.get("ppm_left") is not None
    assert ground_truth, "the bundled ROI example carries ground_truth"
    assert all("chemical_shift_in_points" in g for g in ground_truth)


@pytest.mark.unit
def test_predict_plot_hands_ground_truth_to_the_renderer(tmp_path, monkeypatch):
    """--plot on a ROI file passes the file's ground truth to plot_spectrum (dashed overlay)."""
    import scripts.predict as predict

    seen = {}

    def fake_plot(*args, **kwargs):
        seen.update(kwargs)
        return None, []

    ckpt = tmp_path / "ckpt.pth"
    ckpt.write_bytes(b"stub")
    monkeypatch.setattr(predict, "plot_spectrum", fake_plot)
    monkeypatch.setattr(predict, "build_model", lambda: object())
    monkeypatch.setattr(predict, "load_checkpoint", lambda model, path: model)
    # **kwargs, not a fixed signature: this double stands in for moldetr.inference.run, and pinning
    # its parameter list here means any new keyword on the real function fails as a TypeError in an
    # unrelated plotting test rather than where it belongs.
    monkeypatch.setattr(predict, "run", lambda model, amplitudes, **kwargs: object())
    monkeypatch.setattr(predict, "decode_predictions", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "--input",
            str(REPO / "examples" / "roi_S8_example.npz"),
            "--checkpoint",
            str(ckpt),
            "--plot",
            str(tmp_path / "out.png"),
        ],
    )
    predict.main()
    ground_truth = seen.get("ground_truth")
    assert ground_truth, "plot_spectrum never received the file's ground truth"
    assert all("chemical_shift_in_points" in g for g in ground_truth)


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


@pytest.mark.unit
def test_download_weights_verifies_and_pins_the_version_doi(tmp_path):
    """The download helper checksums correctly and points at the immutable v1.0.0 record (not the
    concept DOI), so a fresh clone always fetches the exact published checkpoint."""
    import hashlib

    from scripts.download_weights import EXPECTED_MD5, ZENODO_URL, _md5

    p = tmp_path / "blob.bin"
    p.write_bytes(b"molde-tr")
    assert _md5(p) == hashlib.md5(b"molde-tr").hexdigest()
    assert "21217102" in ZENODO_URL  # pinned to the immutable v1.0.0 version record
    assert len(EXPECTED_MD5) == 32


@pytest.mark.unit
def test_zenodo_add_paper_doi_resolves_the_concept_and_never_reads_the_env():
    """The release-relation fixer, checked on the three things that would silently break it.

    It imports with no network — `tests/test_integrations.py` depends on that, and so does the
    weekly job, which installs pytest and nothing else.

    The concept id is the *resolve* target, never the edit target: `21214876` has no independently
    editable metadata and merely mirrors whichever version is newest. A hardcoded record id is the
    other half of the same mistake — the one this replaced was obsolete within two releases.

    And it must not grow a `ZENODO_TOKEN` environment fallback. One exists on the maintainer's
    machine, it is stale, and it 403s even on a read; honouring it would swap a working credential
    for a broken one and report the failure as Zenodo's. That is a plausible future "fix", so it is
    pinned here rather than left to a comment.
    """
    source = (REPO / "scripts" / "zenodo_add_paper_doi.py").read_text(encoding="utf-8")
    probes = ("os.environ", "getenv", "import os", "from os import", "environ[")
    reads_env = [probe for probe in probes if probe in source]
    assert not reads_env, (
        f"the script reads the environment ({reads_env}). The credential must come from "
        f"--token-file only: a ZENODO_TOKEN variable exists on the maintainer's machine, it is "
        f"stale, and it 403s even on a read, so an env fallback silently swaps a working token "
        f"for a broken one."
    )

    r = _run(
        "-c",
        "import scripts.zenodo_add_paper_doi as z; "
        "print(z.ZENODO_CONCEPT_ID, z.PAPER_DOI, z.PAPER_RELATION, "
        "z.paper_relation_present({}))",
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [
        "21214876",
        "10.1021/acs.analchem.5c03465",
        "isSupplementTo",
        "False",
    ]


@pytest.mark.unit
def test_zenodo_add_paper_doi_notices_a_put_that_ate_the_metadata():
    """The preservation check, which only ever executes on the `--confirm` path.

    That path cannot be exercised here — it needs a credential and a record actually missing the
    relation, and every live record now carries it. So the pure half is tested directly, because
    the alternative is shipping untested code whose entire job is to notice that an irreversible
    write to an archival record went wrong.

    A replacing PUT that dropped the creator list or reopened a restricted record would otherwise
    print a perfectly healthy VERIFY block: the block lists `related_identifiers`, and those would
    be exactly right.
    """
    from scripts.zenodo_add_paper_doi import _fingerprint, _report_preserved

    record = {
        "title": "MolDeTr",
        "access_right": "restricted",
        "license": {"id": "apache-2.0"},
        "doi": "10.5281/zenodo.21856870",
        "version": "v1.3.0",
        "creators": [{"name": f"Author {i}"} for i in range(11)],
        "related_identifiers": [{"relation": "isSupplementTo", "identifier": "10.1021/x"}],
    }
    before = _fingerprint(record)
    assert before["n_creators"] == 11
    assert _report_preserved(before, _fingerprint(record)), "an unchanged record must report intact"

    # Each of these is a way a PUT has been observed, or is documented, to go wrong.
    for field, damage in (
        ("creators", record["creators"][:1]),  # 10 of 11 authors silently dropped
        ("access_right", "open"),  # a restricted deposit reopened
        ("license", None),  # licence lost
        ("title", ""),  # title blanked
    ):
        broken = {**record, field: damage}
        assert not _report_preserved(before, _fingerprint(broken)), (
            f"a PUT that changed {field!r} must be reported, not passed over"
        )


@pytest.mark.unit
def test_the_version_guard_refuses_a_record_the_release_did_not_mint():
    """The fixer must not edit whatever happens to be newest when a specific release was meant.

    This closes a real path rather than a hypothetical one. `integrations.yml` gates the repair on
    the paper-relation check, and that check asserts the record's version against the newest
    published GitHub release **before** it looks at the relation — so its failure often means
    "Zenodo has not minted yet". The wait step ahead of it deliberately gives up after six minutes.
    Without a version guard the fixer would then resolve the concept id to the **previous** release
    and unlock, PUT and republish a settled archival record that was never in question.

    Tested here rather than end to end because the live path needs a credential *and* a release
    Zenodo has not yet minted — a race that cannot be arranged on demand. The pure half is the half
    that decides, so it is the half worth pinning.
    """
    from scripts.zenodo_add_paper_doi import target_version_matches

    assert target_version_matches({"version": "v1.4.0"}, "v1.4.0")

    assert not target_version_matches({"version": "v1.3.0"}, "v1.4.0"), (
        "the previous release is exactly what the concept id resolves to before the webhook mints; "
        "editing it is the failure this guard exists to prevent"
    )
    assert not target_version_matches({}, "v1.4.0"), (
        "a record with no version at all must not pass as the expected one"
    )
    assert not target_version_matches({"version": "1.4.0"}, "v1.4.0"), (
        "Zenodo stores the tag verbatim with its `v`; a bare number is a different string and "
        "silently normalising it here would let a mismatch through"
    )

    assert target_version_matches({"version": "v1.3.0"}, None), (
        "with no --expect-version the newest record is the intended target, which is what the "
        "weekly cron relies on"
    )
    assert target_version_matches({}, ""), "an empty expectation must behave as no expectation"
