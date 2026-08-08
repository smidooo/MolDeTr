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
