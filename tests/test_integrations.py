"""External references this repo depends on, and cannot fix once they break.

The DOI badge on the README broke in August 2026 and *a user reported it*. No CI lane could have
caught it: every check here runs against the repository's own contents, and nothing looked outward.
This module is the outward-looking half, and it is why the `network` marker exists again — Phase 1
deleted that marker precisely because it was declared and applied to nothing.

Run weekly from `.github/workflows/integrations.yml`, never on a PR. A publisher WAF having a bad
afternoon must not block unrelated work; a check that blocks unrelated work gets ignored, and an
ignored check is worse than none.

Marked per-test rather than with a module-level `pytestmark`: the Colab test needs no network at all
(it resolves a URL against the working tree) and belongs in the fast lane, where it can catch a
renamed notebook on the PR that renames it rather than up to a week later.

Imports are the standard library plus one first-party module, `scripts.zenodo_add_paper_doi`. That
matters more than it looks: `integrations.yml` installs pytest and nothing else, so a third-party
import here kills collection and the job files an issue blaming a missing dependency. The script is
stdlib-only and arrives with the checkout, and `tests/test_integrations_isolation.py` holds that end
of the contract. It is imported rather than reimplemented so the guard and the fixer cannot drift
apart about what "the relation is present" means.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scripts.zenodo_add_paper_doi import PAPER_DOI, ZENODO_CONCEPT_ID, paper_relation_present

REPO = Path(__file__).resolve().parent.parent
CITATION = REPO / "CITATION.cff"
README = REPO / "README.md"

#: Zenodo's record API. GETting the *concept* id returns the newest version record, which is how
#: both this module and `scripts/zenodo_add_paper_doi.py` find the current release without a
#: hardcoded record number — the previous hardcoded one was obsolete within two releases.
ZENODO_RECORD_API = "https://zenodo.org/api/records/{record_id}"

#: Releases newest-first. Deliberately not `/releases/latest`, which hides prereleases; see
#: `test_latest_software_record_supplements_the_article`.
RELEASES_API = "https://api.github.com/repos/smidooo/MolDeTr/releases?per_page=10"

#: The Handle System's own resolver API. Deliberately not a GET of `https://doi.org/<doi>`: that
#: follows the redirect to the publisher, and publishers block generic clients. `pubs.acs.org`
#: returns 403 to anything that looks automated, so a plain HTTP check reports the article DOI as
#: dead every single week — the classic false positive that trains a maintainer to ignore the job.
#: The Handle API answers the question actually being asked ("is this DOI registered and pointing
#: somewhere?") without ever contacting the publisher.
HANDLE_API = "https://doi.org/api/handles/{doi}"

#: Handle System response codes. 1 = resolved; 100 = handle not found (the DOI does not exist);
#: 200 = values not found. Only 1 is success, and 100 vs a transport error is the distinction a
#: bare status code cannot make: unregistered is our problem, unreachable is the internet's.
HANDLE_SUCCESS = 1

TIMEOUT = 30


def _get(url: str, *, method: str = "GET") -> tuple[int, bytes]:
    """Fetch with a browser-ish UA, returning `(status, body)` and never raising on HTTP status."""
    request = urllib.request.Request(
        url, method=method, headers={"User-Agent": "Mozilla/5.0 (compatible; MolDeTr-CI/1.0)"}
    )
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:  # a status, not a failure to reach anything
        return exc.code, exc.read()


def _get_rendered_readme() -> str:
    """The README as GitHub renders it, which is the only form containing camo URLs."""
    request = urllib.request.Request(
        "https://api.github.com/repos/smidooo/MolDeTr/readme",
        headers={"Accept": "application/vnd.github.html", "User-Agent": "MolDeTr-CI/1.0"},
    )
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read().decode("utf-8", "replace")


def _declared_dois() -> list[str]:
    """Every DOI in `CITATION.cff`, read rather than hard-coded.

    Reading the source of truth means a DOI added to the citation metadata is covered the moment it
    is added. A literal list here would quietly stop covering exactly the newest thing.

    Parsed with a regex rather than PyYAML: PyYAML is not a dependency of this project, and
    `tests/test_deploy_manifest.py` already sets that precedent.
    """
    text = CITATION.read_text(encoding="utf-8")
    return re.findall(r"^\s*value:\s*(10\.\d{4,}/\S+)\s*$", text, re.M)


def _newest_published_release_tag() -> str:
    """The tag Zenodo most recently archived, as GitHub reports it.

    Drafts are excluded because they fire no webhook and mint nothing. Prereleases are *kept* for
    the opposite reason: Zenodo archives one exactly like a full release, so skipping them would
    leave the deposit and this check naming different tags.

    `metadata.version` on the Zenodo side carries the `v` prefix verbatim (`v1.3.0`, verified
    across all six records), so the comparison is a plain equality with no normalisation to get
    subtly wrong.
    """
    status, body = _get(RELEASES_API)
    assert status == 200, f"could not list releases (HTTP {status}); the cross-check cannot run"

    published = [release for release in json.loads(body) if not release.get("draft")]
    assert published, "no published releases found — the scrape shape changed or the repo has none"
    return published[0]["tag_name"]


@pytest.mark.network
def test_citation_declares_the_dois_we_expect():
    """Guards this module's own premise: the checks below are only as good as what they enumerate.

    If `CITATION.cff` is restructured and the regex stops matching, every DOI test below would pass
    vacuously against an empty list. That is the failure this whole effort keeps finding, so the
    enumeration gets its own assertion rather than a comment.
    """
    dois = _declared_dois()
    assert len(dois) >= 4, f"expected at least 4 DOIs in CITATION.cff, parsed {dois}"
    assert any(d.startswith("10.1021/") for d in dois), f"the article DOI is missing: {dois}"
    assert sum(d.startswith("10.5281/zenodo.") for d in dois) >= 3, f"Zenodo DOIs missing: {dois}"


@pytest.mark.network
@pytest.mark.parametrize("doi", _declared_dois())
def test_every_declared_doi_still_resolves(doi):
    """A DOI is a promise that outlives the repo. This is the only check that can catch it breaking."""
    status, body = _get(HANDLE_API.format(doi=doi))
    assert status == 200, f"the Handle API itself was unreachable for {doi} (HTTP {status})"

    payload = json.loads(body)
    code = payload.get("responseCode")
    assert code == HANDLE_SUCCESS, (
        f"DOI {doi} did not resolve: Handle responseCode={code} "
        f"({'handle not found — the DOI does not exist' if code == 100 else 'see handle.net docs'})"
    )


@pytest.mark.unit
def test_paper_relation_predicate_bites():
    """The network test below cannot prove itself, so this proves the part that decides.

    Every one of the six software records carries the relation today — only v0.1.0 was born with
    it and the other five were backfilled. There is no live record left that *lacks* it, so a purely network-based
    guard would have gone green on its first run and stayed green whether or not it checked
    anything. This repository has already shipped three guards that never guarded; the crafted
    payloads below are what keep this from being the fourth.

    The third case is the one with teeth. `zenodo_add_paper_doi.ps1` decided "already present" on
    the *identifier* alone, so a record relating the article under any other verb — `references`,
    `isCitedBy` — would have read as correct to the fixer while being wrong in the metadata. A
    guard and a fixer that disagree about what "present" means cannot close a loop between them,
    which is why both now call this one function.
    """
    tree_url = "https://github.com/smidooo/MolDeTr/tree/v1.3.0"

    assert not paper_relation_present({"related_identifiers": []})
    assert not paper_relation_present(
        {"related_identifiers": [{"relation": "isSupplementTo", "identifier": tree_url}]}
    ), (
        "a record carrying only the GitHub-tree relation is exactly the state this guard exists to catch"
    )
    assert not paper_relation_present(
        {"related_identifiers": [{"relation": "references", "identifier": PAPER_DOI}]}
    ), "the article must be related as isSupplementTo; the right DOI under the wrong verb is not it"
    assert not paper_relation_present({}), (
        "a deposit with no related_identifiers key at all must answer False, not raise — Zenodo "
        "omits the key entirely on a record that has never had one"
    )

    published = {
        "related_identifiers": [
            {"relation": "isSupplementTo", "identifier": tree_url},
            {"relation": "isSupplementTo", "identifier": PAPER_DOI},
        ]
    }
    assert paper_relation_present(published), "the shape every published record is supposed to have"

    # The DOI is a parameter, and the fixer's idempotency check passes `--paper-doi` through it.
    # Reading the constant instead would make `--paper-doi <something already present>` append a
    # duplicate, because the check would be asking about a different DOI than the one being added.
    assert not paper_relation_present(published, "10.9999/not.in.this.record"), (
        "asked about a DOI the record does not carry, the answer must be False even though the "
        "record is otherwise perfectly related"
    )
    other = "10.5281/zenodo.21217101"
    assert paper_relation_present(
        {"related_identifiers": [{"relation": "isSupplementTo", "identifier": other}]}, other
    ), "asked about a DOI that IS present, the answer must be True regardless of PAPER_DOI"


@pytest.mark.network
def test_latest_software_record_supplements_the_article():
    """The newest software deposit must point at the paper it accompanies.

    It never does. The relation has been absent from **every release since v0.1.0 — five for
    five** (v1.0.0, v1.1.0, v1.1.1, v1.2.0, v1.3.0) — the last minted four days after a script was
    written to fix the problem.
    Zenodo does not carry it forward: v1.2.0 was published without it even though v1.1.1 had
    already been hand-edited to carry it, which rules out "the previous record seeds the next one"
    and rules out diligence as the cause. Nothing in the repository could see it, and nothing
    about a release without it looks wrong.

    **The version cross-check is load-bearing, not decoration.** `GET /records/<concept id>`
    returns the newest *version* record, so a run that fires before Zenodo's webhook has minted
    resolves to the *previous* release — which does carry the relation — and passes while the new
    release carries nothing. Comparing against the newest published GitHub release is what turns
    that silent pass into a readable failure.

    Releases are enumerated rather than read from `/releases/latest`, which excludes prereleases.
    Zenodo archives a prerelease like any other release, so `latest` would name one tag while the
    deposit named another and the guard would cry drift over the one release shape it cannot see.
    """
    tag = _newest_published_release_tag()

    status, body = _get(ZENODO_RECORD_API.format(record_id=ZENODO_CONCEPT_ID))
    assert status == 200, (
        f"the Zenodo concept record {ZENODO_CONCEPT_ID} was unreachable (HTTP {status}), so "
        f"nothing below was checked"
    )
    metadata = json.loads(body)["metadata"]
    version = metadata.get("version")

    assert version == tag, (
        f"Zenodo's newest software record is {version!r} but the newest published GitHub release "
        f"is {tag!r}. Either the release webhook has not minted yet (wait and re-run) or it "
        f"failed — until they agree, this guard would be checking the wrong record."
    )

    assert paper_relation_present(metadata), (
        f"the {version} software record does not relate the article as isSupplementTo → "
        f"{PAPER_DOI}. Fix it with `python scripts/zenodo_add_paper_doi.py` (dry run) then "
        f"`--confirm`; see docs/RELEASING.md. Present relations: "
        f"{[(r.get('relation'), r.get('identifier')) for r in metadata.get('related_identifiers', [])]}"
    )


@pytest.mark.network
def test_every_readme_badge_still_renders_through_camo():
    """The check that would have caught the original incident.

    GitHub never serves a README image directly — it proxies through `camo.githubusercontent.com`,
    fetching upstream server-side from an IP pool shared by the whole site. A badge can therefore be
    perfectly healthy from a laptop and broken for every visitor, which is exactly what happened:
    Zenodo served badges `no-cache` under a per-IP rate limit, camo blew through it, and the badge
    rendered as `502 Invalid upstream response (429)`.

    So this must fetch the *camo* URLs, not the canonical ones. Testing the canonical URLs would
    pass while the page is visibly broken — the precise mistake that made the original bug
    user-reported instead of CI-reported.
    """
    # `Accept: application/vnd.github.html` is the whole trick: the default JSON representation
    # returns raw markdown, in which the badges are still their canonical URLs and camo does not
    # appear at all. Only the rendered form carries the proxied `src` a visitor's browser requests.
    rendered = _get_rendered_readme()

    pairs = re.findall(
        r'src="(https://camo\.githubusercontent\.com/[A-Za-z0-9/]+)"[^>]*data-canonical-src="([^"]+)"',
        rendered,
    )
    assert pairs, "no camo-proxied images found in the rendered README — the scrape shape changed"

    broken = []
    for camo_url, canonical in pairs:
        code, payload = _get(camo_url)
        if code != 200:
            broken.append(f"{canonical} → HTTP {code} {payload[:60].decode('utf-8', 'replace')!r}")

    assert not broken, (
        f"{len(broken)} of {len(pairs)} README badges do not render for visitors:\n  "
        + "\n  ".join(broken)
    )


@pytest.mark.network
@pytest.mark.parametrize(
    "label,url",
    [
        (
            "Zenodo checkpoint",
            "https://zenodo.org/api/records/21217102/files/model_spin_system_ABCDEFG_exp2.pth/content",
        ),
        ("Hugging Face model page", "https://huggingface.co/smidooo/moldetr"),
    ],
)
def test_weight_sources_are_reachable(label, url):
    """`HEAD` only — never a 974 MB GET to answer a liveness question.

    Both are load-bearing: `scripts/download_weights.py` fetches the Zenodo copy, and
    `.github/workflows/nightly.yml` fetches the HF mirror. If either disappears, a documented
    install step and the nightly lane break together, and nothing else would notice.
    """
    status, _ = _get(url, method="HEAD")
    assert status in (200, 302), f"{label} is not reachable: HTTP {status} for {url}"


@pytest.mark.unit
def test_colab_links_point_at_notebooks_that_exist():
    """Deliberately NOT network-marked — this resolves against the working tree, so it belongs in
    the fast lane where it catches a renamed notebook on the PR that renames it, rather than up to a
    week later.

    A Colab URL embeds a repo path (`/github/OWNER/REPO/blob/BRANCH/<path>`). Rename or move a
    notebook and the badge still renders perfectly while the link 404s on click — a broken link that
    looks healthy is the worst kind.
    """
    readme = README.read_text(encoding="utf-8")
    paths = re.findall(
        r"colab\.research\.google\.com/github/[^/]+/[^/]+/blob/[^/]+/([^\")\s]+)", readme
    )
    assert paths, "no Colab links found in README.md — the badge row or this pattern changed"

    missing = [p for p in paths if not (REPO / p).is_file()]
    assert not missing, f"Colab links point at files that do not exist in the repo: {missing}"
