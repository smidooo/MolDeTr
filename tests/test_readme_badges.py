"""README badge contract — the front page is the first thing a reader sees, and it can break
without a single line of code changing.

GitHub never loads a README image from the browser. It rewrites every `<img src>` to
`camo.githubusercontent.com/<hmac>/<hex-url>` and fetches the upstream **server-side**, from an IP
pool shared by every repository on the site. Two consequences follow, and both are invisible
locally because `curl` from a laptop is a different rate-limit bucket entirely:

* An upstream that forbids caching is re-fetched on *every* render of *every* repo that embeds it.
* An upstream that rate-limits per IP therefore sees GitHub's whole pool as one very busy client.

`zenodo.org` does both — it serves badges `cache-control: no-cache, max-age=0` under an
`x-ratelimit-limit: 120` per-IP-per-minute cap. The badge that resulted was measured returning
`502 Invalid upstream response (429)` on 4 of 5 fetches through camo while all nine `img.shields.io`
badges on the same page returned `200`. The repo-id form `zenodo.org/badge/<id>.svg` is the worst
case: it costs two requests, a 302 and then the SVG.

The second test guards a different failure. `zenodo.org/badge/latestdoi/<id>` tracks the newest
*version* DOI, but `docs/RELEASING.md` states the rule for this project — citation surfaces pin the
**concept** DOI — which is also why `CITATION.cff` deliberately carries no `version:`. The README's
own Availability section already cites the concept DOI, so a `latestdoi` badge made the front page
disagree with itself.

Both tests read only committed text, so they need no network and cannot skip themselves.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"

#: The software *concept* DOI — stable across every release, unlike the per-version DOI.
CONCEPT_DOI = "10.5281/zenodo.21214876"

#: Hosts that combine a no-cache badge response with a per-IP rate limit. Serving a README image
#: from one of these is a slow-motion outage: it renders until GitHub's shared pool exhausts the
#: upstream's quota, then fails for everyone at once.
RATE_LIMITED_IMAGE_HOSTS = {"zenodo.org", "www.zenodo.org"}


def _image_urls(markdown: str) -> list[str]:
    """Every image the README asks a browser to load, in HTML and Markdown syntax alike."""
    html = re.findall(r"<img\b[^>]*?\bsrc=[\"']([^\"']+)[\"']", markdown, re.I)
    inline = re.findall(r"!\[[^\]]*\]\(\s*([^)\s]+)", markdown)
    return [*html, *inline]


@pytest.mark.unit
def test_readme_images_avoid_rate_limited_hosts():
    """Relative paths are exempt: GitHub serves same-repo images directly, without camo."""
    urls = _image_urls(README.read_text(encoding="utf-8"))
    remote = [url for url in urls if url.startswith(("http://", "https://"))]
    offenders = [url for url in remote if urlsplit(url).hostname in RATE_LIMITED_IMAGE_HOSTS]
    assert not offenders, (
        f"{len(offenders)} README image(s) are served from a rate-limited, no-cache host and will "
        f"intermittently render as `Invalid upstream response (429)`: {offenders}. "
        f"Use an equivalent img.shields.io badge, which is CDN-cached (max-age=432000)."
    )


@pytest.mark.unit
def test_readme_doi_badge_pins_the_concept_doi():
    """`latestdoi` is the mechanism that makes the badge track a moving version DOI, so it is the
    thing to forbid — asserting merely that the concept DOI appears *somewhere* would pass against
    the broken README too, since the Availability section already cites it.
    """
    text = README.read_text(encoding="utf-8")
    assert "latestdoi" not in text, (
        "the README resolves a DOI through `latestdoi`, which names the newest version DOI; "
        f"docs/RELEASING.md requires citation surfaces to pin the concept DOI {CONCEPT_DOI}"
    )

    badge = re.search(r"<img\b[^>]*?\bsrc=\"([^\"]+)\"[^>]*?\balt=\"DOI[^\"]*\"", text)
    assert badge, "the README has no DOI badge (expected an <img> whose alt starts with 'DOI')"
    assert CONCEPT_DOI in unquote(badge.group(1)), (
        f"the DOI badge renders {unquote(badge.group(1))!r}, which does not name the concept DOI "
        f"{CONCEPT_DOI}"
    )
