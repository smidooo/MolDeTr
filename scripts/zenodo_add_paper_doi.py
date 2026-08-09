#!/usr/bin/env python
"""Relate the newest MolDeTr software deposit to the article it accompanies.

Adds `isSupplementTo` → `10.1021/acs.analchem.5c03465` to the latest Zenodo software record.

    python scripts/zenodo_add_paper_doi.py             # DRY RUN — prints, changes nothing
    python scripts/zenodo_add_paper_doi.py --confirm   # apply

**Why this exists.** Every deposit should point at the paper it accompanies, and the relation does
not carry forward. It was on v0.1.0 and absent from every release since — **four for four**,
including v1.3.0, minted four days after this tool was first written to fix the problem. Zenodo is
not seeding it from the previous record: v1.2.0 came out without it even though v1.1.1 had already
been hand-edited to carry it. So this is not a lapse of attention that a checklist can catch, and
`docs/RELEASING.md` no longer pretends otherwise — `tests/test_integrations.py` watches for it on
every release and weekly thereafter, and this is what closes it when that guard fires.

Assume a fresh release LACKS the relation until checked.

**Do not target the concept id as the edit target.** `21214876` has no independently editable
metadata; it mirrors whichever version is newest. Reading it is exactly how the newest record is
found below, but editing it is meaningless. Every early handoff named it and was wrong.

**Restricted records are safe.** v1.0.0 is `access_right=restricted`, and because the PUT re-sends
the whole metadata object it re-asserts that restriction rather than reopening the files. Verified
2026-08-05 on record 21364202: files stayed unexposed and the GPL/SHIMpanzee notice survived.

**A metadata edit mints no new DOI.** The record id and its DOI are unchanged by this flow; do not
expect, or fear, a new version record.

Token: `~/.secrets/.zenodo_token`, scopes `deposit:write` + `deposit:actions`. Never printed.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

#: The article this software accompanies.
PAPER_DOI = "10.1021/acs.analchem.5c03465"

#: The relation verb. Checked as well as the DOI: the right identifier under `references` or
#: `isCitedBy` is *not* this relation, and a fixer that matched on the identifier alone would
#: report such a record as already correct while the guard kept failing it.
PAPER_RELATION = "isSupplementTo"

#: The software *concept* record. GETting it returns the newest version record, so the target
#: resolves itself and cannot go stale — the hardcoded record id this replaced was obsolete within
#: two releases.
ZENODO_CONCEPT_ID = "21214876"

RECORD_API = "https://zenodo.org/api/records/{record_id}"
DEPOSIT_API = "https://zenodo.org/api/deposit/depositions/{record_id}"

#: Outside the repository on purpose, and outside any project-local `secrets/` deny rule.
DEFAULT_TOKEN_FILE = Path.home() / ".secrets" / ".zenodo_token"

TIMEOUT = 60


def paper_relation_present(metadata: dict, doi: str = PAPER_DOI) -> bool:
    """Does this deposit already relate `doi` correctly?

    The single definition of "present", shared with `tests/test_integrations.py` so the guard that
    detects the gap and the tool that closes it cannot disagree about what they are looking for.

    `doi` is a parameter rather than a constant read directly so that the idempotency check below
    asks about the DOI actually being added. Hardcoding `PAPER_DOI` here would make
    `--paper-doi <other>` append unconditionally and duplicate an entry that was already there.

    Tolerates a deposit with no `related_identifiers` key at all — Zenodo omits it entirely on a
    record that has never had one.
    """
    return any(
        entry.get("relation") == PAPER_RELATION and entry.get("identifier") == doi
        for entry in metadata.get("related_identifiers") or []
    )


def _api(url: str, *, token: str | None = None, method: str = "GET", payload: dict | None = None):
    """One JSON round trip. Raises `HTTPError` on any non-2xx, which the write path relies on."""
    data = None
    headers = {"Accept": "application/json", "User-Agent": "MolDeTr-release/1.0"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        body = response.read()
    return json.loads(body) if body else None


def _read_token(path: Path) -> str:
    """Read the token from a file, reporting only its length.

    Deliberately does **not** fall back to a `ZENODO_TOKEN` environment variable. One exists on the
    maintainer's machine, it is stale, and it 403s even on a plain read — honouring it would swap a
    working credential for a broken one and report the failure as a Zenodo problem. A file argument
    also keeps the value out of shell history and process listings, which is why there is no
    `--token` flag either.
    """
    if not path.is_file():
        raise SystemExit(
            f"No token file at {path}. Create it with a Zenodo personal access token carrying "
            f"scopes deposit:write + deposit:actions, or pass --token-file."
        )
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(f"Token file {path} is empty.")
    print(f"token: read from {path} ({len(token)} chars)")
    return token


def _resolve_record_id(concept_id: str) -> tuple[str, str]:
    """Return `(record_id, version)` for the newest release under `concept_id`."""
    latest = _api(RECORD_API.format(record_id=concept_id))
    record_id, version = str(latest["id"]), latest["metadata"].get("version")
    print(f"target: auto-resolved to newest release {version} -> record {record_id}")
    return record_id, version


def _show(heading: str, metadata: dict) -> None:
    print(f"--- {heading} ---")
    for entry in metadata.get("related_identifiers") or []:
        print(f"  {entry.get('relation')} -> {entry.get('identifier')} [{entry.get('scheme')}]")


def _apply(record_id: str, token: str, metadata: dict) -> None:
    """Unlock, replace the metadata wholesale, publish — discarding if anything goes wrong.

    The PUT *replaces* the metadata object, so it is handed the full mutated copy. A partial
    payload wipes every field it omits.
    """
    deposit = DEPOSIT_API.format(record_id=record_id)
    _api(f"{deposit}/actions/edit", token=token, method="POST")
    print("  unlocked")
    try:
        _api(deposit, token=token, method="PUT", payload={"metadata": metadata})
        _api(f"{deposit}/actions/publish", token=token, method="POST")
        print("  published")
    except Exception:
        # An unlocked record left unpublished is stranded in edit state and invisible to citation
        # tools, which is worse than the missing relation this came to fix.
        print("  FAILED after unlock -- discarding so the record is not left stranded")
        try:
            _api(f"{deposit}/actions/discard", token=token, method="POST")
            print("  discarded")
        except Exception as discard_error:
            print(f"  DISCARD FAILED ({discard_error}) -- check https://zenodo.org/me/uploads")
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    ap.add_argument("--record-id", help="override the auto-resolved target (rarely needed)")
    ap.add_argument("--concept-id", default=ZENODO_CONCEPT_ID)
    ap.add_argument("--paper-doi", default=PAPER_DOI)
    ap.add_argument("--confirm", action="store_true", help="apply; omit for a dry run")
    args = ap.parse_args()

    token = _read_token(args.token_file)
    record_id = args.record_id or _resolve_record_id(args.concept_id)[0]

    deposit = _api(DEPOSIT_API.format(record_id=record_id), token=token)
    metadata = deposit["metadata"]

    if "access_right" not in metadata:
        print("Not the legacy deposit shape -- aborting rather than guessing field names.")
        return 2

    _show(f"CURRENT ({metadata.get('version')})", metadata)

    if paper_relation_present(metadata, args.paper_doi):
        print("\nPaper DOI already present. Nothing to do.")
        return 0

    metadata["related_identifiers"] = [*(metadata.get("related_identifiers") or [])] + [
        {"relation": PAPER_RELATION, "identifier": args.paper_doi, "scheme": "doi"}
    ]
    _show("PROPOSED", metadata)
    print("  (title / creators / licence / description untouched)")

    if not args.confirm:
        print("\nDRY RUN -- nothing changed. Re-run with --confirm.")
        return 0

    _apply(record_id, token, metadata)

    # Read back from the public API, not the edit form: this is what a citation tool sees.
    print("\n--- VERIFY ---")
    _show("PUBLIC", _api(RECORD_API.format(record_id=record_id))["metadata"])
    concept = _api(RECORD_API.format(record_id=args.concept_id))
    print(f"  concept -> {concept['metadata'].get('version')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
