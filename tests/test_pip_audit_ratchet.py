"""`scripts/pip_audit_ratchet.py` fails the audit lane only on a genuinely new finding.

`security.yml`'s dependency-audit step was `continue-on-error: true` while the noise floor was
unmeasured (see `tests/test_security_workflow.py`). It is measured now: four non-`pull_request`
runs since #49 fixed the pip-audit flags (`82d39fb`, 2026-08-08) all report the identical
`setuptools 79.0.1 PYSEC-2026-3447`. This module is the comparator that lets the lane go red on
anything NOT already in `.github/pip-audit-baseline.json`, mirroring the `coverage (ratcheted)`
idiom in `ci.yml` rather than the all-or-nothing choice between permanently advisory and an
allowlist nobody revisits.

The seeded-defect tests are the point of this file: a comparator nobody has seen return non-zero is
not a comparator, and this repo has shipped exactly that guard once already (#49). The
dependency-count floor tests close a second instance of the same defect class found in review: the
original discriminator asked only "is `dependencies` a key", which a broken audit that resolves
nothing still satisfies -- and an empty result set makes every baseline entry look STALE (fixed) by
construction, not obviously broken.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pip_audit_ratchet import compare

REPO = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO / ".github" / "pip-audit-baseline.json"


def _audit(*deps: dict) -> dict:
    return {"dependencies": list(deps), "fixes": []}


def _resolved(name: str, version: str = "1.0.0", vuln_ids: tuple[str, ...] = ()) -> dict:
    return {
        "name": name,
        "version": version,
        "vulns": [{"id": vid, "fix_versions": []} for vid in vuln_ids],
    }


def _skipped(name: str, reason: str = "distribution marked as editable") -> dict:
    return {"name": name, "skip_reason": reason}


def _baseline(*entries: dict) -> dict:
    return {"entries": list(entries)}


def _padding(n: int = 25) -> list[dict]:
    """Enough clean, unrelated dependencies to clear the resolved-dependency floor.

    The real audited environment is torch plus `[dev,app,eval]` -- hundreds of distributions -- so
    any test exercising ordinary pair logic (as opposed to the floor itself) needs a payload that
    does not accidentally trip the floor check for the wrong reason.
    """
    return [_resolved(f"padding-pkg-{i}") for i in range(n)]


@pytest.mark.unit
def test_finding_covered_by_baseline_passes(capsys):
    audit = _audit(_resolved("setuptools", "79.0.1", ("PYSEC-2026-3447",)), *_padding())
    baseline = _baseline({"package": "setuptools", "id": "PYSEC-2026-3447", "note": "known"})
    assert compare(audit, baseline) == 0
    out = capsys.readouterr().out
    assert "BASELINE  setuptools PYSEC-2026-3447" in out


@pytest.mark.unit
def test_a_pair_not_in_the_baseline_fails():
    """The seeded defect: a new advisory the baseline has never seen must fail the run."""
    audit = _audit(_resolved("setuptools", "79.0.1", ("PYSEC-2026-3447",)), *_padding())
    assert compare(audit, _baseline()) == 1


@pytest.mark.unit
def test_new_pair_is_reported_even_alongside_a_known_one(capsys):
    audit = _audit(
        _resolved("setuptools", "79.0.1", ("PYSEC-2026-3447",)),
        _resolved("urllib3", "2.0.0", ("GHSA-fake-0000",)),
        *_padding(),
    )
    baseline = _baseline({"package": "setuptools", "id": "PYSEC-2026-3447", "note": "known"})
    assert compare(audit, baseline) == 1
    out = capsys.readouterr().out
    assert "NEW       urllib3 GHSA-fake-0000" in out


@pytest.mark.unit
def test_stale_baseline_entry_does_not_fail_but_is_reported(capsys):
    """A vulnerability that stopped appearing is good news, not a build break -- as long as the
    audit that no longer reports it plainly still ran against a real tree (see the floor tests)."""
    baseline = _baseline({"package": "setuptools", "id": "PYSEC-2026-3447", "note": "known"})
    assert compare(_audit(*_padding()), baseline) == 0
    out = capsys.readouterr().out
    assert "STALE     setuptools PYSEC-2026-3447" in out


@pytest.mark.unit
def test_skipped_dependency_has_no_vulns_key_and_is_ignored():
    """`--skip-editable` entries carry no `vulns` key at all (pip-audit 2.10.1 schema, verified
    against `pip_audit/_format/json.py`) -- `.get("vulns", [])` must not KeyError on them."""
    audit = _audit(_skipped("moldetr"), *_padding())
    assert compare(audit, _baseline()) == 0


@pytest.mark.unit
def test_resolved_dependency_with_zero_vulns_is_not_a_pair():
    audit = _audit(_resolved("requests", "2.31.0", ()), *_padding())
    assert compare(audit, _baseline()) == 0


@pytest.mark.unit
def test_committed_baseline_parses_and_every_entry_is_shaped_correctly():
    """`.github/pip-audit-baseline.json` must exist and be well-formed for the CI step to use it."""
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["entries"], "baseline has no entries -- nothing to ratchet against yet"
    for entry in baseline["entries"]:
        assert entry["package"], "a baseline entry is missing 'package'"
        assert entry["id"], "a baseline entry is missing 'id'"
        assert entry["note"], "a baseline entry is missing a dated rationale in 'note'"


@pytest.mark.unit
def test_committed_baseline_covers_the_measured_setuptools_finding():
    """This is the specific floor measured 2026-08-27 (see CLAUDE.md / the security decision note);
    a baseline that has drifted away from it would make the ratchet fail on a known-accepted pair."""
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    pairs = {(e["package"], e["id"]) for e in baseline["entries"]}
    assert ("setuptools", "PYSEC-2026-3447") in pairs


@pytest.mark.unit
def test_an_empty_dependency_list_fails_rather_than_reporting_a_clean_run():
    """The output-shape discriminator in security.yml only checks that `dependencies` is a KEY; it
    says nothing about whether the audit actually resolved the real tree (torch + [dev,app,eval],
    hundreds of distributions). Without this floor, a broken install that produces
    `{"dependencies": []}` reports every baseline entry as STALE and exits 0 -- a false green
    indistinguishable from a genuinely fixed vulnerability. This is the exact defect class (`#43`)
    the ratchet exists to remove; a ratchet that reintroduces it for the empty case is not a fix.
    """
    baseline = _baseline({"package": "setuptools", "id": "PYSEC-2026-3447", "note": "known"})
    assert compare(_audit(), baseline) == 1


@pytest.mark.unit
def test_missing_dependencies_key_also_fails(capsys):
    """`.get("dependencies", [])` must not treat an absent key the same as "nothing to report"."""
    assert compare({"fixes": []}, _baseline()) == 1
    assert "did not resolve" in capsys.readouterr().out


@pytest.mark.unit
def test_a_real_run_with_many_resolved_dependencies_is_not_penalised():
    """The floor must not fire on a normal, healthy audit -- only on a suspiciously empty one."""
    audit = _audit(_resolved("setuptools", "79.0.1", ("PYSEC-2026-3447",)), *_padding())
    baseline = _baseline({"package": "setuptools", "id": "PYSEC-2026-3447", "note": "known"})
    assert compare(audit, baseline) == 0


@pytest.mark.unit
def test_baseline_package_name_is_canonicalised_against_a_non_canonical_entry(capsys):
    """pip-audit reports PEP 503 canonical names (lowercase, `-`-separated) -- verified against
    `pip_audit/_format/json.py` 2.10.1. A human hand-editing the baseline after a red run is exactly
    who is told to write `PyYAML` or `zope.interface` verbatim; the comparator must still match."""
    audit = _audit(_resolved("zope.interface", "5.0", ("GHSA-fake-1111",)), *_padding())
    baseline = _baseline({"package": "Zope.Interface", "id": "GHSA-fake-1111", "note": "known"})
    assert compare(audit, baseline) == 0
    assert "BASELINE" in capsys.readouterr().out
