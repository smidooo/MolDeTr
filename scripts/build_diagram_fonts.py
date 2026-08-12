"""Build `docs/fonts/*.woff2` from upstream, with the glyph set derived from the diagrams.

Run:

    python scripts/build_diagram_fonts.py            # fetches upstream, writes docs/fonts/*.woff2
    python scripts/build_diagram_fonts.py --tables   # also prints the GLYPHS / FONT_SHA256 literals
    python scripts/build_diagram_fonts.py --dry-run  # build and report, write nothing

then re-run `python scripts/build_diagram_svgs.py`, because every SVG carries the font bytes inline.

WHY THIS SCRIPT EXISTS
----------------------
It did not, and that is what issue #84 was. The recipe lived as prose in `docs/fonts/README.md`
describing a manual `fontTools` session, and its character list was **hand-written** -- so it aged
behind the diagrams it was supposed to describe. Five figures ended up printing eight characters no
vendored face carried -- `Δ` in two figures, plus `✓ ✕ • ↕ ∅ −`. (A ninth, `₂`, was found first
and composed away rather than subsetted.) Those did not render
as blank boxes; they fell through to whatever face the reader's OS supplied, which is precisely the
reader-dependent rendering that vendoring exists to prevent.

Re-subsetting by hand would have fixed those eight and left the *class* of defect in place. So the
glyph set here is **read out of the committed SVGs** rather than typed: whatever the diagrams print
is what gets subsetted, and adding a character to a figure can no longer outrun the fonts by more
than one rebuild.

WHAT IS PINNED, AND WHY EACH
----------------------------
* **The upstream commit.** `google/fonts` revises its fonts; an unpinned fetch would let outlines
  drift into published figures silently, under a diff that shows only base64.
* **The SHA-256 of each source TTF**, checked after download. The commit pin alone trusts the
  transport; this does not.
* **The output hashes** land in `tests/test_diagram_fonts.py::FONT_SHA256`, which is what stops a
  re-subset that silently *dropped* a glyph -- `GLYPHS` constrains the SVGs only from below, so a
  missing `δ` would leave `GLYPHS` claiming it and the suite green. That test deliberately reads
  only committed bytes and pinned tables, never fontTools, so it runs in CI instead of skipping
  there. Keep it that way: this script is the only place fontTools belongs.

THE ONE ASYMMETRY WORTH KNOWING
-------------------------------
One shared character set is requested from every face, but subsetting can only keep glyphs the
source actually has -- Space Grotesk carries no Greek, so `δ` survives in the IBM Plex Sans faces
alone. That is why `GLYPHS` is per family and must not be collapsed into a union: a `δ` in a `MONO`
run would pass a unioned check and render in the reader's system face.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "docs" / "fonts"
OUT_DIR = ROOT / "docs" / "img"

#: `google/fonts` at a fixed commit. Bump deliberately, and re-run the render diff in
#: `docs/fonts/README.md` when you do -- a newer upstream can change outlines, not just coverage.
UPSTREAM_COMMIT = "038b637da7b3fd956a4ed93ffc607c3d5e4ce172"
_RAW = f"https://raw.githubusercontent.com/google/fonts/{UPSTREAM_COMMIT}/ofl"

#: Printable ASCII, subsetted in full rather than by usage. The diagrams are edited often and their
#: prose is ASCII; deriving *this* range from the SVGs too would make an innocuous wording change
#: able to drop a letter from the font.
ASCII = "".join(chr(c) for c in range(0x20, 0x7F))

#: Characters the vendored faces already carried before the set was derived, kept whether or not a
#: diagram currently prints them.
#:
#: Deriving the set purely from today's SVGs would *shrink* coverage -- `°±²³Å÷‘’“”…≥` are all
#: carried today and printed by nothing -- and shrinking is a regression, not a saving. (That
#: list is measured, not typed: an earlier draft of this comment named `×`, which this same
#: change made `coupling_rule` print, and omitted `≥`. Re-measure it, do not edit it by eye.) The
#: subsetting is shared across diagrams precisely so that "which figure did you edit?" cannot decide
#: whether a character renders; a set that tracks current usage exactly reintroduces that, one
#: rebuild behind instead of one release behind. Union, therefore: usage can only ever add.
LEGACY_SUPERSET = "°±²³·¹Å×÷δ–—‘’“”…→≤≥"

#: Unix timestamp stamped into every built face's `head.modified`, so the build is reproducible.
#: An arbitrary fixed instant (2025-01-01T00:00:00Z) -- what matters is that it is not `now`. See
#: :func:`build_face` for why a wall-clock stamp here would quietly disarm `FONT_SHA256`.
BUILD_TIMESTAMP = 1_735_689_600


class Face(NamedTuple):
    """One output `.woff2`, and how to get there from an upstream file."""

    stem: str
    family: str
    weight: int
    source: str
    #: SHA-256 of the upstream file, checked after download.
    source_sha256: str
    #: Variation axes to pin. Empty for a source that is already static -- IBM Plex Mono ships as
    #: static instances upstream, so there is nothing to instance, and the earlier prose recipe
    #: describing all three as variable was wrong about it.
    axes: dict[str, float]


FACES: tuple[Face, ...] = (
    Face(
        "sg700",
        "Space Grotesk",
        700,
        f"{_RAW}/spacegrotesk/SpaceGrotesk%5Bwght%5D.ttf",
        "acad6de1fc93436f5c0f1f4137751ef04f1aea3063e7036535970ffcfbd79f72",
        {"wght": 700},
    ),
    Face(
        "plex400",
        "IBM Plex Sans",
        400,
        f"{_RAW}/ibmplexsans/IBMPlexSans%5Bwdth,wght%5D.ttf",
        "3b031aa4216174205bd8471f88a49b91f093169e9e87bd5262242bc5967fe2e3",
        {"wght": 400, "wdth": 100},
    ),
    Face(
        "plex600",
        "IBM Plex Sans",
        600,
        f"{_RAW}/ibmplexsans/IBMPlexSans%5Bwdth,wght%5D.ttf",
        "3b031aa4216174205bd8471f88a49b91f093169e9e87bd5262242bc5967fe2e3",
        {"wght": 600, "wdth": 100},
    ),
    Face(
        "mono400",
        "IBM Plex Mono",
        400,
        f"{_RAW}/ibmplexmono/IBMPlexMono-Regular.ttf",
        "6a3412f058c7d8dfd9170c41e85ade48e5156ecb89356110ca57a0a27734af46",
        {},
    ),
)

_SVG_NS = "{http://www.w3.org/2000/svg}"


def printed_characters(svg_dir: Path = OUT_DIR) -> set[str]:
    """Every character the committed diagrams paint, plus printable ASCII.

    Walks `<text>` trees rather than regexing them, for the same reason
    `tests/test_diagram_fonts.py` does: a `<tspan>`'s tail text belongs to its parent, and getting
    that wrong misattributes exactly the runs that matter.
    """
    # stdlib ElementTree rather than defusedxml: the inputs are repo-local files this repository's
    # own generator wrote, committed, and re-verifies byte-for-byte with `--check`. They carry no
    # DOCTYPE, so there are no entities to expand. Same reasoning as `tests/test_diagram_fonts.py`
    # and `tests/test_figure_numbers.py`, which walk the same files.
    found = set(ASCII) | set(LEGACY_SUPERSET)
    for path in sorted(svg_dir.glob("*.svg")):
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        for text in root.iter(f"{_SVG_NS}text"):
            for node in text.iter():
                found.update(node.text or "")
                found.update(node.tail or "")
    # Mirror what the guard checks, exactly. `_uncovered` skips a run only when it is *entirely*
    # blank, then skips characters below 0x80; so an NBSP or thin space sitting inside a label IS
    # demanded by the test. `isprintable()` alone returns False for those -- verified, not assumed --
    # which would leave the build unable to supply a character the test requires, a red no rebuild
    # could clear. Hence the explicit `>= 0x80` arm; the ASCII half is already covered in full.
    return {c for c in found if ord(c) >= 0x80 or c.isprintable()}


def _fetch(url: str, expect_sha256: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - pinned https URL
        blob: bytes = response.read()
    got = hashlib.sha256(blob).hexdigest()
    if expect_sha256 and got != expect_sha256:
        raise SystemExit(
            f"upstream file changed under a pinned commit: {url}\n"
            f"  expected {expect_sha256}\n  got      {got}\n"
            "Either the pin is wrong or the transport is. Do not proceed."
        )
    if not expect_sha256:
        print(f"  (no source pin recorded; measured {got})", file=sys.stderr)
    return blob


def build_face(face: Face, characters: set[str], blob: bytes) -> bytes:
    """Instance the variable axes, subset to `characters`, and return woff2 bytes.

    Byte-reproducible **for a given fontTools and brotli**: identical inputs give an identical
    file. The toolchain itself is not pinned (`fonttools[woff]>=4.40`, brotli unconstrained), and
    woff2 is brotli-compressed, so a rebuild on a newer stack may legitimately differ -- if
    `FONT_SHA256` moves without an input change, check the versions before assuming a defect.
    Within one toolchain it is exact, and that is not
    free -- `head.modified` is a wall-clock stamp, so two builds a second apart differ, and it was
    measured here rather than assumed (two calls in one process produced two different SHA-256s).
    Left alone it would defeat `FONT_SHA256`: the pin exists to make a re-subset fail until a human
    re-derives the coverage table, and a hash that changes on every rebuild trains that human to
    paste the new value without reading it. Which is how the hand-written glyph list rotted in the
    first place.
    """
    from fontTools import subset
    from fontTools.misc.timeTools import timestampSinceEpoch
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    font = TTFont(io.BytesIO(blob))
    if face.axes:
        font = instancer.instantiateVariableFont(
            font, face.axes, inplace=False, updateFontNames=True
        )

    options = subset.Options()
    options.layout_features = ["*"]
    options.name_IDs = ["*"]
    options.notdef_outline = True
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(text="".join(sorted(characters)))
    subsetter.subset(font)

    # Both lines are load-bearing, and the assignment alone is not enough: `head.compile()`
    # overwrites `modified` with `timestampNow()` whenever `recalcTimestamp` is set, which it is by
    # default -- and `instantiateVariableFont(inplace=False)` hands back a fresh `TTFont` carrying
    # that default again. Setting only the field looked like it worked and did not.
    font.recalcTimestamp = False
    font["head"].modified = timestampSinceEpoch(BUILD_TIMESTAMP)
    font.flavor = "woff2"
    out = io.BytesIO()
    font.save(out)
    return out.getvalue()


def coverage(blob: bytes) -> str:
    """The non-ASCII characters a built face actually carries, sorted -- i.e. a `GLYPHS` row."""
    from fontTools.ttLib import TTFont

    cmap = TTFont(io.BytesIO(blob)).getBestCmap()
    return "".join(chr(c) for c in sorted(cmap) if c > 0x7F)


def main() -> int:
    # The first thing this prints is the derived charset, which is by definition non-ASCII. On a
    # cp1252 console (Git Bash, cmd) that raises UnicodeEncodeError *before* the fetch, so the
    # script dies having built nothing -- while the maintainer's PowerShell 7 gives utf-8 and
    # works. A recipe that runs only in one shell is barely better than the prose it replaced.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", action="store_true", help="print the GLYPHS / FONT_SHA256 literals")
    ap.add_argument("--dry-run", action="store_true", help="build but do not write docs/fonts/")
    args = ap.parse_args()

    characters = printed_characters()
    extra = sorted(c for c in characters if ord(c) > 0x7F)
    print(
        f"charset from {len(sorted(OUT_DIR.glob('*.svg')))} committed SVGs: "
        f"{len(characters)} characters, {len(extra)} non-ASCII: {''.join(extra)}"
    )

    sources: dict[str, bytes] = {}
    built: dict[str, bytes] = {}
    families: dict[str, str] = {}
    for face in FACES:
        if face.source not in sources:
            print(f"fetching {face.source.rsplit('/', 1)[-1]}")
            sources[face.source] = _fetch(face.source, face.source_sha256)
        blob = build_face(face, characters, sources[face.source])
        built[f"{face.stem}.woff2"] = blob
        carried = coverage(blob)
        # Two weights of one family must agree -- they are instances of one variable font, and
        # `GLYPHS` has a single row per family. Without this the second silently overwrites the
        # first and the table publishes one face's coverage as the family's.
        assert families.setdefault(face.family, carried) == carried, (
            f"{face.stem} carries {carried!r} but another {face.family} face carries "
            f"{families[face.family]!r}; GLYPHS cannot describe both."
        )
        print(f"  {face.stem}.woff2  {len(blob):>7,} bytes")

    if not args.dry_run:
        for name, blob in built.items():
            (FONT_DIR / name).write_bytes(blob)
        print(f"wrote {len(built)} file(s) to {FONT_DIR.relative_to(ROOT)}")

    if args.tables:
        print("\n# scripts/build_diagram_svgs.py")
        print("GLYPHS = {")
        for family in ("Space Grotesk", "IBM Plex Sans", "IBM Plex Mono"):
            print(f'    "{family}": "{families[family]}",')
        print("}")
        print("\n# tests/test_diagram_fonts.py")
        print("FONT_SHA256 = {")
        for name in sorted(built):
            print(f'    "{name}": "{hashlib.sha256(built[name]).hexdigest()}",')
        print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
