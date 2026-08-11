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


#: The two figures `scripts/capture_gui_media.py` owns. They are screenshots of the running app, so
#: unlike every other figure here they cannot become vector -- and they can be regenerated on demand
#: at any scale, which makes them the only rasters for which a floor above `MIN_SCALE` is both
#: meaningful and satisfiable. That is why the stricter bar is scoped here rather than applied
#: globally.
#:
#: This used to add "unlike the prediction plots, whose generators were deliberately removed in
#: 2827cdb". That stopped being true when `example_prediction` and `example_prediction_vanillin`
#: were re-authored as SVG in `scripts/build_diagram_svgs.py`: they have a generator again, and
#: having left the raster floor entirely through the viewBox exemption, they are no longer rasters
#: at all. `vanillin_spin_systems.png` is the last figure still in that state.
APP_CAPTURES = ("docs/img/gui.png", "docs/img/demo.gif")
CAPTURE_MIN_SCALE = 3.0


@pytest.mark.unit
def test_the_app_captures_clear_a_3x_floor():
    """`gui.png` and `demo.gif` are regenerable, so they are held to more than the global floor.

    The GIF sat at exactly 2.00x because `capture_gui_media.py` believed a `device_scale_factor`
    could not reach a GIF frame, and therefore never passed one. It can: a 344x256 viewport at
    dsf=1.5 screenshots to 516x384, and Pillow writes that straight through to the GIF. The comment
    was describing its own omission. This test pins the outcome rather than the mechanism -- what
    matters to a reader is the device-pixel ratio, not how the capture achieved it.
    """
    soft: list[str] = []
    for src in APP_CAPTURES:
        path = REPO / src
        assert path.is_file(), f"{src} is missing; regenerate with scripts/capture_gui_media.py"
        width = dict(_local_figures(README.read_text(encoding="utf-8"))).get(src)
        assert width, f"{src} must declare a width in the README to have a ratio at all"
        size = _png_or_gif_size(path)
        assert size is not None, f"{src} is neither a PNG nor a GIF"
        if size[0] < CAPTURE_MIN_SCALE * width:
            soft.append(
                f"{src} is {size[0]}x{size[1]} native but rendered at {width} px "
                f"({size[0] / width:.2f}x, needs {CAPTURE_MIN_SCALE:g}x = "
                f"{int(CAPTURE_MIN_SCALE * width)} px)"
            )
    assert not soft, (
        "app capture(s) below the "
        f"{CAPTURE_MIN_SCALE:g}x floor. Both are regenerable -- "
        "`python scripts/capture_gui_media.py [--gif]`:\n  " + "\n  ".join(soft)
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
    readers see it. Both pairs that were drifting when this was written have since been re-authored
    as one shared geometry, which is why nothing here fails: ``pipeline`` was 1820x388 light against
    1720x370 dark, and ``banner`` 2560x1283 against 2560x1280 -- the latter clearing the tolerance
    below by 0.0003, so the guard passed while the stretching was real.

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


@pytest.mark.unit
def test_the_docs_favicon_points_at_files_that_exist():
    """`docs/_includes/head-custom.html` is the only place the mark is referenced, so nothing else
    would notice it rotting.

    Two distinct ways this goes silently wrong. The obvious one: a renamed or deleted asset leaves
    the site serving a 404 for its icon, which no page test would catch because the page still
    renders. The subtler one: the file only does anything at all because
    `jekyll-theme-cayman`'s `_layouts/default.html` ends its `<head>` with
    `{% include head-custom.html %}` -- drop the theme or switch it and this include becomes inert
    while still looking correct. That half cannot be asserted from here, so it is written down in
    the include itself; this test holds the half that can.
    """
    include = REPO / "docs" / "_includes" / "head-custom.html"
    assert include.is_file(), "the favicon include is gone; docs/_config.yml still names a theme"
    hrefs = re.findall(
        r"""href=["']?\{\{\s*'([^']+)'\s*\|\s*relative_url\s*\}\}""",
        include.read_text(encoding="utf-8"),
    )
    assert hrefs, "the include declares no icon href; a favicon that names nothing is not a favicon"
    missing = [h for h in hrefs if not (REPO / "docs" / h.lstrip("/")).is_file()]
    assert not missing, (
        f"docs/_includes/head-custom.html points at {missing}, which do not exist under docs/. "
        "Paths there are site-root-relative, so they resolve against docs/, not the repo root."
    )


#: Every `x y` pair in an `M`/`L`-only path. The banner's traces are emitted as polylines, so this
#: needs no curve support -- and it deliberately does not gain any, because a `C`/`Q` control point
#: is not a point the curve passes through and would be read here as if it were.
_PATH_XY = re.compile(r"[-+]?\d*\.?\d+")

#: `<text x=".." y="..">content</text>`, with the attributes in the order `Canvas.text` writes them.
_TEXT = re.compile(r"<text\b[^>]*\bx=\"([-\d.]+)\"[^>]*\by=\"([-\d.]+)\"[^>]*>(.*?)</text>", re.S)


def _polyline(d: str) -> list[tuple[float, float]]:
    nums = [float(n) for n in _PATH_XY.findall(d)]
    return list(zip(nums[::2], nums[1::2]))


@pytest.mark.unit
def test_the_banner_traces_plot_the_committed_spectrum():
    """The hero's two curves must be the spectrum in ``examples/roi_S8_example.npz``, not a drawing.

    The banner this replaced was a design-tool export, so its curve was a *redrawing* of that
    spectrum -- recognisable (its tallest peak sat at 7.388 ppm against a ground truth of 7.385)
    but only 0.58-0.68 correlated with the array. Nothing could have told you that from the repo;
    a picture of data and a picture of something else are the same kind of file.

    What makes this falsifiable rather than circular: the three quantities compared are produced by
    three independent things. The **tick labels** are text the generator lays out, and are what a
    reader actually uses to interpret the figure. The **curve** is a polyline built from the array.
    The **shifts** come from the NPZ's own ``ground_truth`` record. Inventing a curve, flipping the
    axis (ppm descends left to right, and nothing in SVG knows that), shifting the window, or
    swapping the NPZ each breaks the agreement between a different pair of them.

    Every recorded shift is checked, not just the tallest peak. Checking one was tried and is too
    weak to state the property: mirroring the resolved trace about its own centre -- a flipped axis,
    exactly what this claims to catch -- moved the tallest peak from 6.972 ppm to 7.429, which lands
    within 0.009 of H_B and passes. Two of the three multiplets have to disagree before a
    one-peak test notices, and there is no reason to expect them to.
    """
    np = pytest.importorskip("numpy")

    svg = (REPO / "docs" / "img" / "banner.svg").read_text(encoding="utf-8")
    # Attributes are pulled from the whole tag rather than in one ordered pattern. An `id` before
    # `d` pattern matched nothing against a generator that emits `d` first, and the resulting
    # "found 0" read exactly like a missing-attribute failure.
    traces = [
        (
            re.search(r'\bid="(trace-[a-z]+)"', tag).group(1),
            re.search(r'\bd="([^"]+)"', tag).group(1),
        )
        for tag in re.findall(r"<path\b[^>]*>", svg)
        if re.search(r'\bid="trace-[a-z]+"', tag) and re.search(r'\bd="', tag)
    ]
    assert len(traces) == 2, (
        f'expected the raw and resolved traces to carry id="trace-*", found {len(traces)}'
    )

    labels = [
        (float(x), float(y), body.strip())
        for x, y, body in _TEXT.findall(svg)
        if re.fullmatch(r"\d\.\d", body.strip())
    ]

    # allow_pickle for `ground_truth`, which is an object array of dicts. The file is committed to
    # this repo and is the same one `scripts/plot_deposit_spectrum.py` reads; there is no untrusted
    # input path here.
    npz = np.load(REPO / "examples" / "roi_S8_example.npz", allow_pickle=True)
    shifts = np.array([g["chemical_shift_ppm"] for g in npz["ground_truth"]])

    for name, d in traces:
        points = _polyline(d)
        assert len(points) > 200, f"{name} has {len(points)} vertices; that is not a real spectrum"
        xs = [p[0] for p in points]
        base = max(p[1] for p in points)
        # A tick belongs to this panel if it sits under this trace's baseline and within its span.
        # Matching on the "d.d" shape alone also collects the assignment table's coupling constants,
        # which are the same shape, in the same file, and would silently redefine the axis.
        panel = sorted(t for t in labels if min(xs) <= t[0] <= max(xs) and 0 < t[1] - base < 80)
        assert len(panel) == 2, f"{name} spans {min(xs):.0f}..{max(xs):.0f} with ticks {panel}"

        (x_left, _, left), (x_right, _, right) = panel
        assert float(left) > float(right), (
            f"{name}: the tick at x={x_left:.0f} reads {left} and the one at x={x_right:.0f} reads "
            f"{right}. A 1H axis descends left to right; this one ascends."
        )
        per_px = (float(left) - float(right)) / (x_right - x_left)
        full = base - min(p[1] for p in points)  # this panel's full-scale height, in units

        # Read every vertex back into ppm through the panel's own ticks, keep what rises clear of
        # the noise, and group by nearest recorded shift. A multiplet's centre of mass IS its
        # chemical shift -- for the two doublets here the shift falls in the *gap* between lines,
        # which is why "is there a peak within N units of the shift" cannot state this property.
        signal = [
            (float(left) - (x - x_left) * per_px, base - y)
            for x, y in points
            if base - y > full * 0.10
        ]
        assert signal, f"{name} has no signal above 10% of its own full scale; it is a flat line"

        for shift in shifts:
            mine = [(p, w) for p, w in signal if min(shifts, key=lambda s: abs(p - s)) == shift]
            assert mine, (
                f"{name} puts nothing at all near {shift} ppm, though the spectrum it claims to "
                f"plot records a multiplet there."
            )
            # Height as well as position. Without this, erasing a multiplet outright still passes:
            # its neighbour's line tails leak across the partition boundary and their centre of
            # mass lands close enough to the shift that was removed. Measured, the weakest of the
            # three reaches 0.79 of full scale in both panels, so 0.45 is ~1.75x of headroom.
            assert max(w for _, w in mine) > full * 0.45, (
                f"{name}'s multiplet at {shift} ppm reaches only "
                f"{max(w for _, w in mine) / full:.2f} of the panel's full scale. The assignment "
                f"table names it, so it has to be visible in the curve beside it."
            )
            centre = sum(p * w for p, w in mine) / sum(w for _, w in mine)
            assert abs(centre - shift) < 0.02, (
                f"{name}'s multiplet nearest {shift} ppm has its centre of mass at {centre:.3f} "
                f"ppm, read off the figure's own tick labels. The curve and the spectrum it claims "
                f"to plot disagree, so at least one of them is not the data."
            )
