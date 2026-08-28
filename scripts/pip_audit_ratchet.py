"""Diff a `pip-audit` JSON run against a committed baseline; fail only on a genuinely new finding.

    python scripts/pip_audit_ratchet.py <audit.json> [--baseline PATH]

Why a ratchet rather than leaving `security.yml`'s dependency-audit step permanently
`continue-on-error: true`, and why not `--strict`: `--strict` cannot coexist with this project's
editable self-install (`tests/test_security_workflow.py`), and leaving the step advisory forever is
the cargo-cult outcome CLAUDE.md names for it. The noise floor was measured 2026-08-27 from four
non-`pull_request` Security runs after #49 fixed the pip-audit flags: all four report the identical
`setuptools 79.0.1 PYSEC-2026-3447`. This is the middle path -- mirroring the `coverage (ratcheted)`
job in `ci.yml` -- fail on anything the baseline has not already recorded and reviewed.

Baseline schema, `.github/pip-audit-baseline.json`::

    {"entries": [{"package": "<name>", "id": "<vuln id>", "note": "<dated rationale>"}]}

`package` must be the name exactly as `pip-audit --format json` reports it: pip-audit canonicalises
per PEP 503 (lowercase, `-` separators) -- verified against `pip_audit/_format/json.py` at the
currently-shipping 2.10.1, `_format_dep` writes `dep.canonical_name`.

Exit codes: 0 if every `(package, id)` pair in the audit is covered by the baseline; 1 if the audit
contains a pair the baseline does not. A baseline entry that no longer appears in the audit is
printed as STALE but does NOT fail the run -- a fixed vulnerability is good news, not a build break;
pruning a stale entry is left to a human reviewing the printed line.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

Pair = tuple[str, str]

#: The audited environment is torch plus `[dev,app,eval]` -- hundreds of distributions. Below this
#: many resolved entries, the audit did not run against the real tree (a broken install, a
#: resolver that silently found nothing), and reporting "0 findings" for that as a clean pass would
#: recreate defect #43 one layer down: a guard reporting green while performing none of its work.
#: Picked well under the real count so it never fires on a healthy run; not tied to an exact
#: dependency count because that count drifts with every new pinned dependency.
MIN_EXPECTED_DEPENDENCIES = 20


def _canonical(name: str) -> str:
    """PEP 503 canonicalisation: lowercase, runs of `-`/`_`/`.` collapsed to a single `-`.

    pip-audit reports dependency names this way -- verified against `pip_audit/_format/json.py`
    2.10.1, `_format_dep` writes `dep.canonical_name`. Applying it to the baseline side too means a
    human hand-adding an entry after a red run (told to write the name pip-audit printed) does not
    silently fail to match because they wrote `PyYAML` or `zope.interface` instead of the
    canonical form.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _pairs_from_audit(audit: dict) -> set[Pair]:
    """`(package, vuln id)` pairs from a `pip-audit --format json` payload.

    A skipped dependency (e.g. `--skip-editable`) has no `vulns` key at all -- confirmed against
    `pip_audit/_format/json.py` 2.10.1, where a skipped entry is `{"name", "skip_reason"}` only.
    `.get("vulns", [])` treats that the same as a resolved dependency with zero vulnerabilities,
    which is the correct behaviour either way: no pair to report.
    """
    pairs: set[Pair] = set()
    for dep in audit.get("dependencies", []):
        for vuln in dep.get("vulns", []):
            pairs.add((_canonical(dep["name"]), vuln["id"]))
    return pairs


def _pairs_from_baseline(baseline: dict) -> set[Pair]:
    return {(_canonical(entry["package"]), entry["id"]) for entry in baseline.get("entries", [])}


def compare(audit: dict, baseline: dict) -> int:
    """Print every pair's status and return the process exit code.

    Per-pair printing, not one aggregate line: a single "N covered / M new" summary would hide
    exactly which advisory is new, which is the number that matters when this fails.
    """
    resolved = audit.get("dependencies", [])
    if len(resolved) < MIN_EXPECTED_DEPENDENCIES:
        print(
            f"the audit did not resolve a real dependency tree: only {len(resolved)} "
            f"distribution(s) reported, expected at least {MIN_EXPECTED_DEPENDENCIES}. Treating "
            "this as a broken audit, not a clean one -- see MIN_EXPECTED_DEPENDENCIES in "
            "scripts/pip_audit_ratchet.py."
        )
        return 1

    found = _pairs_from_audit(audit)
    known = _pairs_from_baseline(baseline)

    new = sorted(found - known)
    carried = sorted(found & known)
    stale = sorted(known - found)

    for package, vuln_id in carried:
        print(f"BASELINE  {package} {vuln_id}")
    for package, vuln_id in stale:
        print(f"STALE     {package} {vuln_id}  (in baseline, not in this run -- consider pruning)")
    for package, vuln_id in new:
        print(f"NEW       {package} {vuln_id}  (not in .github/pip-audit-baseline.json)")

    if new:
        print(
            f"\n{len(new)} advisory pair(s) not covered by the baseline. Review and either fix "
            "the dependency or add a reviewed, dated entry to .github/pip-audit-baseline.json."
        )
        return 1

    print(f"\nall {len(found)} advisory pair(s) from this run are covered by the baseline.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_json", type=Path, help="path to `pip-audit --format json` output")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).resolve().parent.parent / ".github" / "pip-audit-baseline.json",
        help="the reviewed baseline to ratchet against (default: .github/pip-audit-baseline.json)",
    )
    args = parser.parse_args(argv)

    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    return compare(audit, baseline)


if __name__ == "__main__":
    sys.exit(main())
