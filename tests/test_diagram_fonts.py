"""Every character a diagram prints must exist in a font that diagram embeds.

The diagrams are rendered inside a sandboxed ``<img>``, where external resource loads are blocked --
that is why the faces are base64-embedded rather than referenced. A character outside them does
**not** produce a blank box: measured 2026-08-11, the browser falls through per glyph to whatever
the reader's OS supplies, because a system font is not a resource load. See ``docs/fonts/README.md``
§ How these were produced for the probe.

That makes the failure mode harder to catch, not milder. A missing box gets reported; a ``✓`` in the
reader's default face at the wrong weight does not -- and the same figure loses the glyph outright
for a reader whose OS lacks it. Reader-dependent rendering is precisely what vendoring exists to
prevent, and every other check here is blind to it: the file exists, the SVG parses, the viewBox is
present, the light/dark geometry matches, the banner's trace still plots its NPZ.

Found by measurement rather than review. ``banner.svg`` shipped ``T₂`` as a literal U+2082 on the
README hero while the same generator composed the identical label correctly forty lines away; a
guard written for that one defect returned eight more across five figures (issue #84).

Reads only committed bytes plus two pinned tables -- no fontTools at runtime, so this runs in CI
rather than skipping there, which is the whole point of pinning the coverage as data.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.build_diagram_svgs import DIAGRAMS, FACES, FONT_DIR, GLYPHS, OUT_DIR

SVGS = sorted(OUT_DIR.glob("*.svg"))
_SVG_NS = "{http://www.w3.org/2000/svg}"

#: Exactly which characters each figure prints that no font it embeds can carry, as code points.
#:
#: A *set* rather than a prose note, and compared for equality rather than used as an xfail reason.
#: An xfail gives one bit per figure -- "still has at least one" -- so `architecture` could have all
#: three of its characters fixed and a brand-new uncoverable one introduced, and it would still
#: xfail and the suite would still be green. Equality catches both directions: a character that
#: appears and one that is repaired both fail here until this table is updated to match.
#:
#: **Empty, and that is the assertion.** Issue #84 -- eight characters across five figures
#: -- is closed: six by re-subsetting (`scripts/build_diagram_fonts.py`) and two by the
#: generator, because no upstream face carries U+2715 and only Space Grotesk carries
#: U+2205. Kept rather than deleted so a newly-introduced uncoverable character fails
#: against `{}`. History: https://github.com/smidooo/MolDeTr/issues/84
KNOWN_UNCOVERED: dict[str, set[int]] = {}

#: SHA-256 of each vendored face, so `GLYPHS` cannot drift from the binaries it describes.
#:
#: `GLYPHS` is hand-pinned from the cmaps and constrains the SVGs only from *below* -- the check
#: below proves every printed character is covered, never that a claimed character exists. A
#: re-subset that silently dropped `δ` would leave `GLYPHS` claiming it, `--check` green (you
#: regenerated), and `pipeline.svg`'s "δ · J" resolving to the reader's system face. Hashing is what
#: closes that: re-subsetting fails here until a human re-derives the table.
#:
#: This used to add "a fontTools cmap test would be stronger, but fontTools is not a declared
#: dependency and would *skip* in CI". That stopped being true when `fonttools[woff]` joined
#: the `dev` extra, which every CI job installs -- so the cmap check at the bottom of this file
#: runs there, and it is the only thing verifying `GLYPHS`'s *content* rather than its bytes.
FONT_SHA256 = {
    "mono400.woff2": "2888c1ecc1b8394052c06ba18e1b87ce16798b1b835f066b1d42fab2e9d8d668",
    "plex400.woff2": "64871baba0f990153c90b4207339860ba93d95766b491e95241667e8854098be",
    "plex600.woff2": "f3f2d0e77db68d03a18153bb1c922fbfa49998e9e5c789be2c078f50a8d5530d",
    "sg700.woff2": "2b09d6676c40a739eee11bd555194e8130808ed289b9566e1d2c875a4f3e3c05",
}


def _embedded(svg: str) -> set[str]:
    """Families this file actually carries an ``@font-face`` for.

    Not every diagram embeds every face -- ``_faces(wanted)`` takes a subset -- so naming a family
    in a ``font-family`` stack is not the same as shipping it. ``mark.svg`` already names
    ``'IBM Plex Sans'`` while embedding only Space Grotesk; that is harmless today only because
    Space Grotesk is first and covers the ASCII it prints.
    """
    return set(re.findall(r"@font-face\{font-family:'([^']+)'", svg))


def _text_runs(root: ET.Element) -> list[tuple[str, str]]:
    """``(text, font-family stack)`` for every painted run, honouring ``<tspan>`` inheritance.

    A tree walk rather than a regex sweep, because inheritance is the point: a ``<tspan>`` with no
    ``font-family`` of its own takes the parent ``<text>``'s, and a ``<tspan>``'s *tail* text
    belongs to the parent, not to the tspan. Matching ``<text>`` blocks with a regex would attribute
    both to the wrong stack, and the ``δ``-in-a-``MONO``-run case is exactly where that matters.
    """
    runs: list[tuple[str, str]] = []

    def walk(node: ET.Element, inherited: str) -> None:
        family = node.get("font-family", inherited)
        if node.text:
            runs.append((node.text, family))
        for child in node:
            walk(child, family)
            if child.tail:  # tail text is the PARENT's, so it keeps the parent's family
                runs.append((child.tail, family))

    for text in root.iter(f"{_SVG_NS}text"):
        walk(text, text.get("font-family", ""))
    return runs


def _named_families(stack: str) -> list[str]:
    """The real families in a CSS stack, in order, dropping generics like ``sans-serif``."""
    return [f.strip().strip("'\"") for f in stack.split(",") if f.strip().strip("'\"") in GLYPHS]


def _uncovered(path: Path) -> tuple[set[int], list[str]]:
    """``(code points with no embedded face, runs that resolve to no embedded family at all)``."""
    svg = path.read_text(encoding="utf-8")
    embedded = _embedded(svg)
    missing: set[int] = set()
    homeless: list[str] = []

    for text, stack in _text_runs(ET.fromstring(svg)):
        if not text.strip():
            continue
        usable = [f for f in _named_families(stack) if f in embedded]
        if not usable:
            # Checked separately from the per-character loop because ASCII is skipped there: a run
            # naming only families this file does not embed renders entirely in the reader's default
            # face, and would otherwise pass unnoticed for as long as it stays ASCII-only.
            homeless.append(f"{text.strip()[:40]!r} (stack {stack!r})")
            continue
        for char in text:
            if ord(char) < 0x80:  # every subset covers printable ASCII in full (measured)
                continue
            if not any(char in GLYPHS[f] for f in usable):
                missing.add(ord(char))
    return missing, homeless


@pytest.mark.unit
@pytest.mark.parametrize("path", SVGS, ids=lambda p: p.name)
def test_printed_characters_match_the_fonts_each_figure_embeds(path: Path) -> None:
    """Per-glyph fallback is per *character*, so this is asserted per character, not per run."""
    missing, homeless = _uncovered(path)
    expected = KNOWN_UNCOVERED.get(path.stem.removesuffix("-dark"), set())

    assert not homeless, (
        f"{path.name} paints {len(homeless)} run(s) in a family it does not embed, so they render "
        f"in the reader's default face:\n  " + "\n  ".join(homeless) + "\n\nEither add the face to "
        "this figure's `faces=` tuple in scripts/build_diagram_svgs.py, or use a family it already "
        "embeds."
    )

    appeared = sorted(missing - expected)
    repaired = sorted(expected - missing)
    assert missing == expected, (
        f"{path.name}'s uncoverable characters no longer match the inventory in KNOWN_UNCOVERED.\n"
        + (
            "  NEW, and a real defect: "
            + ", ".join(f"U+{c:04X} {chr(c)!r}" for c in appeared)
            + "\n    Compose it instead of writing the code point — `Canvas.sub` and the `dy` run "
            "in `Canvas.runs` exist for this, and that is how the banner's subscript was fixed.\n"
            "    (2) Re-subset: `python scripts/build_diagram_fonts.py --tables`, paste "
            "GLYPHS + FONT_SHA256, then rerun `build_diagram_svgs.py`. Mind the ordering "
            "-- the font build reads the COMMITTED SVGs, so a newly added character needs "
            "svgs then fonts then svgs.\n"
            "    (3) If no upstream face carries it at all, neither route works: substitute a "
            "covered character, or emit it in a family that has one, as `U+00D7` and "
            "`U+2205` were.\n"
            if appeared
            else ""
        )
        + (
            "  REPAIRED, so delete it from KNOWN_UNCOVERED: "
            + ", ".join(f"U+{c:04X} {chr(c)!r}" for c in repaired)
            + "\n"
            if repaired
            else ""
        )
    )


@pytest.mark.unit
def test_no_text_carries_a_style_attribute_the_walk_would_not_see() -> None:
    """The walk reads the ``font-family`` *presentation attribute*; CSS ``style=`` would outrank it.

    No generator path emits ``style=`` on text, so this is latent rather than live -- which is
    exactly when it is cheap to close. Without it, a future
    ``<text font-family="…Plex Sans" style="font-family:'IBM Plex Mono'">δ</text>`` would pass this
    file while rendering in a face that cannot supply the glyph.
    """
    offenders = []
    for path in SVGS:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        for text in root.iter(f"{_SVG_NS}text"):
            for node in text.iter():
                if node.get("style"):
                    offenders.append(f"{path.name}: <{node.tag.split('}')[-1]} style=…>")
    assert not offenders, (
        "text nodes carry a `style` attribute, which overrides the `font-family` this file reads, "
        "so their real font is not what is checked here:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.unit
def test_the_vendored_faces_are_the_ones_the_glyph_table_describes() -> None:
    """`GLYPHS` is data about specific binaries; this pins it to those exact bytes.

    Named for what it does. Its predecessor asserted only that `FACES` and `GLYPHS` -- two literals
    thirteen lines apart in one file -- agreed on family names, and that committed files exist:
    both tautologies on any checkout, and blind to the strings that carry the actual claim.
    """
    for name, digest in FONT_SHA256.items():
        blob = (FONT_DIR / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == digest, (
            f"{name} is not the file GLYPHS was measured from. Re-derive the coverage table (the "
            f"command is in docs/fonts/README.md § How these were produced), update GLYPHS and this "
            f"hash together, and regenerate the SVGs."
        )
    assert {f"{stem}.woff2" for stem, _, _ in FACES} <= set(FONT_SHA256), (
        "a face is embedded whose bytes are not pinned here, so GLYPHS could drift for it silently"
    )
    assert {family for _, family, _ in FACES} <= set(GLYPHS), (
        "a face is embedded whose glyph coverage GLYPHS does not declare; a run in that family "
        "would be checked against nothing"
    )


@pytest.mark.unit
def test_the_published_coverage_list_matches_the_glyph_table() -> None:
    """`docs/fonts/README.md` publishes the coverage as prose, and prose is what rotted last time.

    The list it replaced was wrong in both directions -- it claimed `⁻` no face carries, and had
    aged behind the figures. Re-publishing a hand-typed list without checking it is how that
    happens again; this makes the two disagree loudly instead.
    """
    doc = (FONT_DIR / "README.md").read_text(encoding="utf-8")
    published = re.search(r"plus `([^`]+)` in every face", doc)
    assert published, "docs/fonts/README.md no longer publishes a coverage list in the pinned form"

    every_face = set.intersection(*(set(v) for v in GLYPHS.values()))
    assert set(published.group(1)) == every_face, (
        "docs/fonts/README.md's published coverage disagrees with GLYPHS.\n"
        f"  in the doc, not in GLYPHS: {sorted(set(published.group(1)) - every_face)}\n"
        f"  in GLYPHS, not in the doc: {sorted(every_face - set(published.group(1)))}"
    )


@pytest.mark.unit
def test_every_committed_svg_belongs_to_a_diagram_that_still_exists() -> None:
    """An orphan is checked by nothing: `--check` iterates `DIAGRAMS`, never the output directory.

    Drop a name from `DIAGRAMS` and its two committed SVGs stay on disk, unreferenced by the
    generator and unverified by `--check`, while every test here stays green.
    """
    expected = {f"{name}{suffix}.svg" for name in DIAGRAMS for suffix in ("", "-dark")}
    on_disk = {p.name for p in SVGS}
    assert on_disk == expected, (
        f"docs/img/ and DIAGRAMS disagree.\n"
        f"  on disk but not generated (orphaned, verified by nothing): {sorted(on_disk - expected)}\n"
        f"  generated but not committed: {sorted(expected - on_disk)}"
    )


@pytest.mark.unit
def test_the_glyph_table_matches_what_the_vendored_binaries_actually_carry() -> None:
    """``GLYPHS`` is hand-transcribed from ``--tables``; this checks it against the binaries.

    The one remaining human link in the chain. ``FONT_SHA256`` pins *bytes*, not claims, so a
    character added to ``GLYPHS`` that the font does not carry passes everything else here -- the
    hash is unchanged and the per-figure check believes the table -- and ships as a silent fallback
    in a published figure. Both directions are asserted, because they fail differently:
    over-claiming hides a defect, under-claiming makes a correct figure look broken.

    It also pins the assumption ``_uncovered`` rests on but never states. That function skips every
    code point below 0x80 on the strength of "every subset covers printable ASCII in full
    (measured)" -- measured once, by hand. Now measured on every run.

    Needs fontTools, which reaches CI through the ``dev`` extra; see ``FONT_SHA256`` above for why
    that is deliberate rather than incidental. It degrades to a skip elsewhere, and nothing else in
    this file ever does.
    """
    ttlib = pytest.importorskip("fontTools.ttLib", reason="fontTools ships in the dev extra")

    for stem, family, _weight in FACES:
        cmap = set(ttlib.TTFont(FONT_DIR / f"{stem}.woff2").getBestCmap())

        carried = {c for c in cmap if c > 0x7F}
        claimed = {ord(ch) for ch in GLYPHS[family]}
        assert carried == claimed, (
            f"{stem}.woff2 disagrees with GLYPHS[{family!r}].\n"
            "  carried but not claimed: "
            + (", ".join(f"U+{c:04X} {chr(c)!r}" for c in sorted(carried - claimed)) or "none")
            + "\n  claimed but not carried: "
            + (", ".join(f"U+{c:04X} {chr(c)!r}" for c in sorted(claimed - carried)) or "none")
            + "\nRe-derive with `python scripts/build_diagram_fonts.py --tables`; never edit "
            "GLYPHS by hand."
        )

        ascii_gap = set(range(0x20, 0x7F)) - cmap
        assert not ascii_gap, (
            f"{stem}.woff2 is missing printable ASCII: "
            + ", ".join(f"U+{c:04X} {chr(c)!r}" for c in sorted(ascii_gap))
            + ". `_uncovered` skips every character below 0x80 assuming all of it is covered, so a "
            "gap here is invisible to every per-figure check in this file."
        )
