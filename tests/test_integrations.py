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
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CITATION = REPO / "CITATION.cff"
README = REPO / "README.md"

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
