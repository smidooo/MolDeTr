#!/usr/bin/env python
"""Relate the newest MolDeTr software deposit to the article it accompanies.

Adds `isSupplementTo` → `10.1021/acs.analchem.5c03465` to the latest Zenodo software record.

    python scripts/zenodo_add_paper_doi.py             # DRY RUN — prints, changes nothing
    python scripts/zenodo_add_paper_doi.py --confirm   # apply

**Why this exists.** Every deposit should point at the paper it accompanies, and the relation does
not carry forward. It was on v0.1.0 and absent from every release since — **five for five**:
v1.0.0, v1.1.0, v1.1.1, v1.2.0 and v1.3.0, the last of them minted four days after this tool was
first written to fix the problem. Zenodo is
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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class ZenodoError(RuntimeError):
    """An API call Zenodo refused, carrying the response body that says why."""


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


def target_version_matches(metadata: dict[str, Any], expected: str | None) -> bool:
    """Is this record the one `expected` names? True when nothing is expected.

    Pure, and separate from `main`, for the same reason `paper_relation_present` is: it decides
    whether an irreversible-feeling write happens, and the live path that would exercise it needs a
    credential plus a release Zenodo has not minted yet — a state that cannot be arranged on demand.

    The comparison is exact. Zenodo carries the tag verbatim including the `v` prefix (`v1.3.0`,
    verified across all six records), so there is no normalisation here to get subtly wrong.
    """
    return not expected or metadata.get("version") == expected


def _api(
    url: str, *, token: str | None = None, method: str = "GET", payload: dict | None = None
) -> Any:
    """One JSON round trip, raising `ZenodoError` with the response body on any non-2xx.

    Reading `exc.read()` is the whole point. Zenodo answers a rejected PUT with **400 and a
    per-field `errors` array naming the metadata key it refused** — precisely the diagnostic you
    need when a replacing PUT fails, and precisely what a bare `HTTP Error 400: BAD REQUEST`
    throws away. The same applies to the very first GET: a stale token 403s there, and without the
    body the operator sees a traceback that looks like Zenodo being down rather than a credential
    problem, which is the misattribution `_read_token` exists to prevent.
    """
    data = None
    headers = {"Accept": "application/json", "User-Agent": "MolDeTr-release/1.0"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        hint = " (a 403 here is usually a stale token, not Zenodo)" if exc.code == 403 else ""
        raise ZenodoError(
            f"{method} {url}\n  HTTP {exc.code} {exc.reason}{hint}\n  {detail}"
        ) from exc
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


#: What a replacing PUT must carry through untouched. The docstring above promises restricted
#: records stay restricted and that creators/licence/title are not disturbed; without a read-back
#: that checks it, a PUT which silently dropped 11 creators or reopened a restricted record would
#: print a perfectly healthy VERIFY block listing two correct relations.
PRESERVED_FIELDS = ("title", "access_right", "license", "doi", "version")


def _fingerprint(metadata: dict) -> dict:
    """The preserved fields, plus creator count rather than the list itself (order is noise)."""
    snapshot = {field: metadata.get(field) for field in PRESERVED_FIELDS}
    snapshot["n_creators"] = len(metadata.get("creators") or [])
    return snapshot


def _report_preserved(before: dict, after: dict) -> bool:
    """Compare two *public* snapshots taken either side of the edit.

    Deliberately public-to-public. The deposit API and the records API disagree on shape for the
    same field — `license` is a bare string (`apache2.0`) on a deposit and an object (`{"id": ...}`)
    on a record — so comparing a deposit read against a public read would report a difference on
    every single run and train the operator to ignore this block.
    """
    drift = {k: (v, after.get(k)) for k, v in before.items() if v != after.get(k)}
    for field, (was, now) in drift.items():
        print(f"  !! {field}: {was!r} -> {now!r}")
    if not drift:
        print(f"  preserved: {', '.join(f'{k}={v!r}'[:46] for k, v in before.items())}")
    return not drift


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
    except BaseException as error:
        # An unlocked record left unpublished is stranded in edit state and invisible to citation
        # tools, which is worse than the missing relation this came to fix.
        #
        # BaseException, not Exception: Ctrl-C during a 60 s PUT raises KeyboardInterrupt, which is
        # not an Exception. Catching only Exception would let the one interruption a human is most
        # likely to cause produce the exact outcome this handler exists to prevent.
        print(f"  FAILED after unlock ({type(error).__name__}: {error})")
        print("  discarding so the record is not left stranded")
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
    ap.add_argument(
        "--expect-version",
        help="refuse unless the target record carries this version (e.g. v1.4.0). Use it whenever "
        "a specific release is meant: the concept id resolves to whatever is newest, which is the "
        "PREVIOUS release until Zenodo's webhook mints.",
    )
    ap.add_argument("--confirm", action="store_true", help="apply; omit for a dry run")
    args = ap.parse_args()

    if args.record_id and str(args.record_id) == str(args.concept_id):
        # Named in the docstring as the mistake every early handoff made. The concept record has no
        # independently editable metadata, so editing it is meaningless -- and silently so.
        print(f"--record-id {args.record_id} is the CONCEPT id, which has no editable metadata.")
        print("Omit --record-id to auto-resolve the newest version record instead.")
        return 2

    token = _read_token(args.token_file)
    if args.record_id:
        record_id = str(args.record_id)
        print(f"target: record {record_id} (from --record-id, not auto-resolved)")
    else:
        record_id = _resolve_record_id(args.concept_id)[0]

    deposit = _api(DEPOSIT_API.format(record_id=record_id), token=token)
    metadata = deposit["metadata"]

    if "access_right" not in metadata:
        print("Not the legacy deposit shape -- aborting rather than guessing field names.")
        return 2

    # Checked here, after the fetch, rather than at resolve time: this is the one place both the
    # auto-resolved and the `--record-id` path pass through, so neither can skip it. The read is
    # harmless; the write it guards is not.
    if not target_version_matches(metadata, args.expect_version):
        print(
            f"\nRefusing: record {record_id} carries version {metadata.get('version')!r}, "
            f"but {args.expect_version!r} was expected."
        )
        print(
            "On a release this means Zenodo has not minted the new version yet -- the concept id "
            "still resolves to the previous release. Editing that record would modify a published "
            "deposit nobody asked about. Wait for the webhook and re-run."
        )
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

    # Taken before the edit so the read-back below has something to compare against. Public, not
    # the deposit copy, because the two APIs render the same fields differently.
    before = _fingerprint(_api(RECORD_API.format(record_id=record_id))["metadata"])

    _apply(record_id, token, metadata)

    # Read back from the public API, not the edit form: this is what a citation tool sees.
    print("\n--- VERIFY ---")
    public = _api(RECORD_API.format(record_id=record_id))["metadata"]
    _show("PUBLIC", public)
    concept = _api(RECORD_API.format(record_id=args.concept_id))
    print(f"  concept -> {concept['metadata'].get('version')}")

    intact = _report_preserved(before, _fingerprint(public))
    if not paper_relation_present(public, args.paper_doi):
        print("  !! the relation is NOT on the published record despite a successful publish")
        return 1
    if not intact:
        print("\nThe relation landed but the PUT disturbed fields it should not have.")
        print("Inspect https://zenodo.org/records/" + str(record_id) + " before releasing again.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
