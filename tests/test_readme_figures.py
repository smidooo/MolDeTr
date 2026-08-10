"""README figure resolution — nine images follow a 2x convention that nothing enforced.

A README figure is rendered at whatever width the page gives it, not at its native size. On a HiDPI
display each CSS pixel is backed by two device pixels, so an image shown at 820 CSS px needs 1640
native px to look sharp; at native 820 it is upscaled by the browser and reads as soft. This is
invisible to every check the repo already had -- the file exists, the link resolves, the alt text is
present, and it still looks bad.

Measured 2026-08-08: nine of the eleven local figures were already >=2.00x, each declared as
``<img ... width="N">`` inside a ``<picture>`` block. Exactly two were not, and they were exactly the
two written in Markdown ``![...](...)`` syntax, which cannot carry a width at all:

* ``docs/img/gui.png``  -- 1425x1182 native, rendered at the container width (~1.4x)
* ``docs/img/demo.gif`` --  960x867  native, rendered at the container width (~1.0x)

So the convention was real and the exceptions were the ones that skipped the HTML form. That is a
thing a test can hold, and the reason to hold it here rather than in review: the failure is silent,
cosmetic, and only visible to a reader on a display the author may not own.

Reads only committed bytes -- image headers are parsed directly, so there is no Pillow dependency and
nothing to skip.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"

#: Device-pixel ratio a figure must satisfy. 2 is the floor for a HiDPI panel; going higher costs
#: bytes on the front page for no visible gain on the displays that exist.
MIN_SCALE = 2.0


#: An ``<svg>`` root carrying a four-number ``viewBox``, read off the head of the file. One pattern
#: serves both the "is it scalable" and the "what shape is it" questions -- a second copy of this
#: regex was briefly written by a different route, silently acquired a literal backspace where its
#: ``\b`` should have been, and answered ``None`` for the very files the first copy accepted.
_VIEWBOX = re.compile(
    rb"<svg\b[^>]*\bviewBox\s*=\s*[\"']\s*([-\d.]+)[\s,]+([-\d.]+)[\s,]+"
    rb"([-\d.]+)[\s,]+([-\d.]+)\s*[\"']",
    re.I | re.S,
)


def _is_scalable_svg(path: Path) -> bool:
    """True only for an SVG that actually scales: an ``<svg>`` root carrying a four-number viewBox.

    The viewBox is the whole point. An SVG that declares only ``width``/``height`` in pixels has no
    intrinsic coordinate system to map onto a larger box, so it does not gain the resolution
    independence this test grants it. Checking the suffix alone would hand out that exemption on
    the strength of a filename.

    Matched against bytes rather than parsed, for the reason in the module docstring -- this file
    reads committed bytes and pulls in no parser. It also sidesteps stdlib XML entity expansion,
    which would be a needless attack surface for a repo-local asset.
    """
    if path.suffix.lower() != ".svg":
        return False
    return _VIEWBOX.search(path.read_bytes()[:4096]) is not None


def _png_or_gif_size(path: Path) -> tuple[int, int] | None:
    """Native pixel dimensions from the file header, or None if it is not a PNG/GIF."""
    header = path.read_bytes()[:33]
    if header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
        width, height = struct.unpack(">II", header[16:24])
        return int(width), int(height)
    if header[:3] == b"GIF":
        width, height = struct.unpack("<HH", header[6:10])
        return int(width), int(height)
    return None


def _figure_shape(path: Path) -> tuple[float, float] | None:
    """Intrinsic width and height of a figure -- pixels for a raster, viewBox units for an SVG.

    Returning ``None`` for SVGs here would make the light/dark geometry check pass by *skipping*
    vector pairs, which is precisely the silent-exemption bug that
    ``test_every_vector_figure_really_is_scalable`` exists to stop. Converting a drifted pair to
    SVG would then turn its failure green without fixing anything. A viewBox is directly
    comparable to a pixel size for aspect-ratio purposes, so both are measured the same way.
    """
    raster = _png_or_gif_size(path)
    if raster is not None:
        return float(raster[0]), float(raster[1])
    match = _VIEWBOX.search(path.read_bytes()[:4096])
    if match is None:
        return None
    _, _, width, height = (float(g) for g in match.groups())
    return (width, height) if width > 0 and height > 0 else None


def _local_figures(markdown: str) -> list[tuple[str, int | None]]:
    """Every repo-local image the README renders, paired with its declared CSS width.

    Both syntaxes are collected on purpose. Markdown images are the failure mode this test exists
    to catch -- excluding them because "they have no width attribute" would exempt precisely the
    broken case. ``<source srcset=...>`` is skipped: a ``<picture>`` source inherits the layout of
    the ``<img>`` beside it, so it carries no width of its own and is covered by the sibling.
    """
    figures: list[tuple[str, int | None]] = []
    for tag in re.findall(r"<img\b[^>]*>", markdown, re.I):
        src = re.search(r"\bsrc=[\"']([^\"']+)[\"']", tag, re.I)
        if not src or src.group(1).startswith(("http://", "https://")):
            continue
        width = re.search(r"\bwidth=[\"']?(\d+)", tag, re.I)
        figures.append((src.group(1), int(width.group(1)) if width else None))
    for src in re.findall(r"!\[[^\]]*\]\(\s*([^)\s]+)", markdown):
        if not src.startswith(("http://", "https://")):
            figures.append((src, None))
    return figures


@pytest.mark.unit
def test_every_readme_figure_declares_a_width():
    """Without a width the browser paints the image at native size, so 2x is unachievable."""
    naked = [src for src, width in _local_figures(README.read_text(encoding="utf-8")) if not width]
    assert not naked, (
        f"{len(naked)} README figure(s) render at native size because they declare no width: "
        f"{naked}. Markdown `![alt](path)` cannot carry one -- rewrite as "
        f'`<img src="..." alt="..." width="N">` and supply an asset at least '
        f"{MIN_SCALE:g}x N pixels wide."
    )


@pytest.mark.unit
def test_every_readme_figure_is_at_least_2x_its_rendered_width():
    """Asserts the ratio, not the absolute size: a 1720 px asset is generous at 820 and thin at 900."""
    too_small: list[str] = []
    for src, width in _local_figures(README.read_text(encoding="utf-8")):
        path = REPO / src
        if not path.is_file():
            pytest.fail(f"README references a figure that does not exist: {src}")
        if _is_scalable_svg(path):
            continue  # resolution-independent; owned by the vector test below
        size = _png_or_gif_size(path)
        if size is None or width is None:
            continue  # non-raster or width-less; the companion test owns those
        if size[0] < MIN_SCALE * width:
            too_small.append(
                f"{src} is {size[0]}x{size[1]} native but rendered at {width} px "
                f"({size[0] / width:.2f}x, needs {MIN_SCALE:g}x = {int(MIN_SCALE * width)} px)"
            )
    assert not too_small, (
        "README figure(s) below the "
        f"{MIN_SCALE:g}x device-pixel floor, so they will look soft on a HiDPI display:\n  "
        + "\n  ".join(too_small)
    )


@pytest.mark.unit
def test_every_vector_figure_really_is_scalable():
    """A ``.svg`` in the README is exempt from the pixel floor -- but only if it can cash the cheque.

    This test exists because of how the exemption is granted. ``_png_or_gif_size`` returns ``None``
    for anything that is not a PNG or GIF, and the ratio test above skips on ``None``. So before
    this test, swapping a PNG for an SVG did not *satisfy* the resolution guard -- it silently
    *removed* that figure from it, and the suite stayed green either way. A named file that is
    empty, truncated, half-written by a failed generator, or an HTML error page saved with the
    wrong extension would all have sailed through.

    The check is therefore positive rather than by-suffix: the bytes must contain an ``<svg>`` root
    with a real four-number ``viewBox``, which is the thing that actually makes it scale.
    """
    bad = [
        src
        for src, _ in _local_figures(README.read_text(encoding="utf-8"))
        if src.lower().endswith(".svg") and not _is_scalable_svg(REPO / src)
    ]
    assert not bad, (
        "README figure(s) named .svg but not a scalable SVG (no <svg> root with a four-number "
        f"viewBox), so they are exempt from the {MIN_SCALE:g}x floor while not being resolution-"
        f"independent: {bad}"
    )


@pytest.mark.unit
def test_a_light_figure_and_its_dark_twin_share_one_geometry():
    """``<picture>`` swaps the source but not the layout, so the twins must be the same shape.

    The browser sizes the block from the ``<img>``, then paints whichever source the colour scheme
    selects into that same box. A twin with a different aspect ratio is therefore not shown
    side-by-side with its sibling -- it is *stretched* to the sibling's box, and only dark-mode
    readers see it. Measured 2026-08-10, two pairs had drifted: ``pipeline`` was 1820x388 light
    against 1720x370 dark, and ``banner`` 2560x1283 against 2560x1280.

    Compares the aspect ratio rather than the pixel size on purpose: differing native resolutions
    are fine and sometimes deliberate, a differing *shape* never is.
    """
    drift: list[str] = []
    for src, _ in _local_figures(README.read_text(encoding="utf-8")):
        if "-dark." in src:
            continue
        light = REPO / src
        stem, _, suffix = src.rpartition(".")
        dark = REPO / f"{stem}-dark.{suffix}"
        if not light.is_file() or not dark.is_file():
            continue
        pair = [_figure_shape(p) for p in (light, dark)]
        if any(s is None for s in pair):
            continue  # neither raster nor SVG; nothing to compare
        (lw, lh), (dw, dh) = pair
        if abs(lw / lh - dw / dh) > 0.005:
            drift.append(
                f"{src} is {lw}x{lh} (aspect {lw / lh:.3f}) but its dark twin is "
                f"{dw}x{dh} (aspect {dw / dh:.3f})"
            )
    assert not drift, (
        "light/dark figure pair(s) with different aspect ratios; <picture> paints both into the "
        "box sized from the <img>, so the twin is stretched and only dark-mode readers see it:\n  "
        + "\n  ".join(drift)
    )
