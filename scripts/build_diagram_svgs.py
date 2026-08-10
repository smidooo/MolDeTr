"""Build the README's vector diagrams from the `docs/BRAND.md` tokens.

Run:

    python scripts/build_diagram_svgs.py            # writes docs/img/<name>{,-dark}.svg
    python scripts/build_diagram_svgs.py --check    # non-zero exit if a committed file is stale

WHY THESE ARE VECTOR
--------------------
A raster figure carries a fixed number of pixels. Displayed larger than that -- a 3x phone, a
zoomed browser, a projector, print -- the browser can only interpolate, so it goes soft. Every
committed diagram already satisfied `tests/test_readme_figures.py`'s 2x floor and still read as
low-resolution, because 2x *is* the floor and the displays kept getting denser. Vector removes the
axis instead of chasing it: the same art is re-rasterised at whatever size the reader asks for.

WHY THERE IS A GENERATOR RATHER THAN HAND-WRITTEN SVG
-----------------------------------------------------
Light and dark are ONE geometry with two palettes. The PNGs they replace were two independent
files, and they had drifted: `pipeline.png` was 1820x388 while `pipeline-dark.png` was 1720x370, a
different aspect ratio, so `<picture>` stretched the dark twin into the box sized from the light
one and only dark-mode readers ever saw it. Sharing the geometry makes that class of bug
impossible rather than merely fixed. `tests/test_readme_figures.py` now also asserts it.

FONTS
-----
`docs/fonts/*.woff2` are weight-instanced, glyph-subset builds of Space Grotesk and IBM Plex Sans,
both SIL OFL 1.1 (licences alongside them; see `docs/fonts/README.md` for provenance). They are
embedded as base64 rather than referenced, because GitHub renders a README image inside a
sandboxed `<img>` where external resources are blocked -- a referenced font would silently fall
back to whatever the reader happens to have.

Note `SG`: Space Grotesk has NO Greek coverage at all, so the delta in "δ · J" cannot come from it.
CSS font fallback is per-glyph, not per-run, so naming IBM Plex Sans second lets the Latin resolve
to Space Grotesk while delta alone drops through. Without that, delta renders as tofu -- invisible
in the PNG this replaces, because its generator fell back silently.
"""

from __future__ import annotations

import argparse
import base64
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "docs" / "fonts"
OUT_DIR = ROOT / "docs" / "img"

#: The figure palette. Ten keys are `docs/BRAND.md` § Palette verbatim -- `panel`, `border`, `navy`,
#: `ink`, `mute`, `latent`, `eyebrow`, `brick`, and the blue/orange/teal tricolor -- and a colour
#: census of the PNGs these replace returned them exactly, confirming the figures were generated
#: from that table.
#:
#: The rest are figure-only roles BRAND.md does not name: `card`, `connector`, `arrow`, `onSolid`,
#: `warnBg`, and `page`. `page` is the one to watch, because BRAND.md *does* publish that token --
#: as `#eef2f7`, for the **app background**. A figure's frame is a different surface, and it was
#: measured at `#f8fafd` (458_413 px of the original `pipeline.png`). BRAND.md carries a
#: `figure-page` row so the two roles stay distinguishable; do not collapse them.
LIGHT = {
    "page": "#f8fafd",
    "panel": "#f1f5fa",
    "border": "#d5dfeb",
    "card": "#ffffff",
    # `navy` fills a solid box; `display` sets the title text. They are the same value in light,
    # and they must not be: dark maps the fill to #31517d and the text to #e2e9f4. Keeping one
    # token for both is what made the first dark pass paint a near-white box with dark text where
    # the original had a blue box with light text -- a regression only dark-mode readers could see.
    "navy": "#1f3a5f",
    "display": "#1f3a5f",
    "ink": "#20242b",
    "mute": "#5b6675",
    "latent": "#7d92b0",
    "connector": "#cdd7e4",
    "arrow": "#9db0c6",
    "onSolid": "#ffffff",
    # The lifted interior of a region-of-interest window. `card` in light, but NOT `card` in dark:
    # there it goes *below* the ground rather than above it.
    "windowFill": "#ffffff",
    # `eyebrow` is BRAND.md's REPLACEMENT token. Every figure this generator
    # supersedes painted small text in the retired #74808f, which measures 4.01:1 on
    # white -- under the 4.5:1 WCAG AA threshold that applies to exactly the small
    # labels it was defined for. BRAND.md replaced it with #666f7d (5.08:1).
    "eyebrow": "#666f7d",
    "warnBg": "#fdf5ea",
    "blue": "#2566b0",
    "orange": "#e08a1f",
    "teal": "#1f9e8c",
    # BRAND.md's error token -- "the brand has no pure red, this is an oklch-harmonised brick".
    # `coupling_rule` is the only figure that needs it, and the two washes are its panel and its
    # window outline, measured off the PNG rather than derived: neither is a straight alpha blend
    # of `brick` over white, so they cannot be computed from it.
    "brick": "#9b3128",
    "brickWash": "#fdf7f6",
    "brickEdge": "#e4b7b0",
    "tealWash": "#f3faf8",
    "tealEdge": "#bfe0d9",
    # `teal` at 3.16:1 is fine for a 3 px rule and fails for a heading, so the heading gets a
    # darkened teal (7.03:1). Same reason BRAND.md retired `#74808f`, applied before it bites.
    "tealText": "#15776a",
    "rule": "#e6ebf2",
    "track": "#eef2f7",
}
#: The dark-figure palette, EXTRACTED rather than invented. `docs/BRAND.md` § Dark-figure palette
#: names only five roles, which is not enough to render these figures, and the gap was previously
#: filled by guessing -- six of the values below were wrong in the first vector pass.
#:
#: How these were derived: `input_contract`, `coupling_rule` and `benchmark` each shipped a light
#: and a dark PNG *of identical dimensions*, so the pair is a pixel-for-pixel colour map. Masking
#: the light image by each token and taking the modal colour under that mask in the dark image
#: recovers the intended mapping exactly -- every row below came back at 100 % agreement except the
#: two that legitimately split (`navy`, and `card` where the window interiors diverge).
#:
#: The tricolor is unchanged, as BRAND.md says it should be -- and the extraction confirms it at
#: 100 %. So is `arrow` (`#9db0c6`), which reads on both grounds and was needlessly darkened before.
DARK = {
    "page": "#141821",
    "panel": "#232c3c",
    "border": "#334054",
    "card": "#1b2130",
    "navy": "#31517d",
    "display": "#e2e9f4",
    "ink": "#e8eef6",
    "mute": "#9fb0c4",
    "latent": "#7d92b0",
    "connector": "#3a4557",
    "arrow": "#9db0c6",
    "onSolid": "#e2e9f4",
    "windowFill": "#10161f",
    "eyebrow": "#8f9db0",
    "warnBg": "#241f14",
    "blue": "#2566b0",
    "orange": "#e08a1f",
    "teal": "#1f9e8c",
    "brick": "#e08575",
    "brickWash": "#241a18",
    "brickEdge": "#5c3a34",
    "tealWash": "#16241f",
    "tealEdge": "#2f5850",
    "tealText": "#5cc2b0",
    "rule": "#28303f",
    "track": "#262f3d",
}

SG = "'Space Grotesk','IBM Plex Sans',sans-serif"
PLEX = "'IBM Plex Sans',sans-serif"
MONO = "'IBM Plex Mono',ui-monospace,monospace"
FACES = (
    ("sg700", "Space Grotesk", 700),
    ("plex400", "IBM Plex Sans", 400),
    ("plex600", "IBM Plex Sans", 600),
    ("mono400", "IBM Plex Mono", 400),
)


def _faces(wanted: tuple[str, ...]) -> str:
    out = []
    for stem, family, weight in FACES:
        if stem not in wanted:
            continue
        blob = (FONT_DIR / f"{stem}.woff2").read_bytes()
        b64 = base64.b64encode(blob).decode("ascii")
        out.append(
            f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}"
        )
    return "".join(out)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _lorentz_d(x0: float, x1: float, base: float, peaks, step: float = 2.0) -> str:
    """Path data for a summed-Lorentzian trace, so a dashed variant can reuse one curve."""
    pts = []
    x = float(x0)
    while x <= x1:
        y = base - sum(h / (1.0 + ((x - c) / w) ** 2) for c, h, w in peaks)
        pts.append(f"{x:g} {y:.1f}")
        x += step
    return "M " + " L ".join(pts)


class Canvas:
    """Minimal SVG emitter. Coordinates are viewBox units, which for these figures are the pixel
    dimensions of the PNGs they replace -- so measurements taken off the old assets transfer 1:1."""

    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        desc: str,
        faces: tuple[str, ...] = ("sg700", "plex400", "plex600"),
        ground: str | None = None,
    ) -> None:
        self.w, self.h = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-labelledby="t d">',
            f'<title id="t">{_esc(title)}</title><desc id="d">{_esc(desc)}</desc>',
            f"<defs><style>{_faces(faces)}</style></defs>",
        ]
        # An SVG with no ground is transparent, and a transparent figure takes the host page's
        # background. In light mode that is invisibly correct -- GitHub's page is white and so were
        # these figures. In dark mode it is not: the PNGs were fully opaque #1b2130 and GitHub's
        # dark page is #0d1117, so the figure silently lost its own ground and only dark-mode
        # readers saw it. `pipeline` is the exception and paints a rounded frame instead, because
        # its PNG really was transparent outside that frame.
        if ground is not None:
            self.parts.append(f'<rect width="{width}" height="{height}" fill="{ground}"/>')

    def rect(self, x, y, w, h, rx, fill, stroke=None, sw=1.5, dash=None) -> None:
        edge = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        if dash:
            edge += f' stroke-dasharray="{dash}"'
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{edge}/>'
        )

    def circle(self, cx, cy, r, fill) -> None:
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>')

    def line(self, x0, y0, x1, y1, stroke, sw=1.6) -> None:
        self.parts.append(
            f'<path d="M {x0} {y0} L {x1} {y1}" stroke="{stroke}" stroke-width="{sw}" fill="none"/>'
        )

    def arrow(self, x0, x1, y, colour, head=15.0) -> None:
        """Shaft plus a solid head. Measured off the originals: 3px shaft, 18px head."""
        back = x1 - head
        self.parts.append(
            f'<path d="M {x0} {y} H {back - 1}" stroke="{colour}" stroke-width="3" '
            f'stroke-linecap="round" fill="none"/>'
            f'<path d="M {x1} {y} L {back} {y - 9} L {back} {y + 9} Z" fill="{colour}"/>'
        )

    def path(self, d, stroke=None, fill="none", sw=1.5, dash=None, cap=None, opacity=None) -> None:
        """Escape hatch for geometry the named helpers do not cover."""
        bits = [f'<path d="{d}" fill="{fill}"']
        if stroke:
            bits.append(f' stroke="{stroke}" stroke-width="{sw}"')
        if dash:
            bits.append(f' stroke-dasharray="{dash}"')
        if cap:
            bits.append(f' stroke-linecap="{cap}" stroke-linejoin="{cap}"')
        if opacity is not None:
            bits.append(f' opacity="{opacity}"')
        self.parts.append("".join(bits) + "/>")

    def spectrum(self, x0, x1, base, peaks, colour, sw=2.6, step=2.0) -> None:
        """A 1-D NMR trace: a flat baseline plus a sum of Lorentzians.

        Lorentzian rather than Gaussian because that is the line shape the model itself predicts
        (``moldetr`` parameterises a multiplet by line width), so the figure and the code agree on
        what a peak is. ``peaks`` are ``(centre, height, half-width at half-maximum)`` in canvas
        units; they sum, which is what makes overlapping peaks look right where they meet.
        """
        self.path(_lorentz_d(x0, x1, base, peaks, step), stroke=colour, sw=sw, cap="round")

    def bracket(self, x0, x1, y, arm, colour, sw=3.0) -> None:
        """A measurement span: one rule with a tick at each end, as on a dimension line."""
        self.path(
            f"M {x0} {y - arm} V {y + arm} M {x0} {y} H {x1} M {x1} {y - arm} V {y + arm}",
            stroke=colour,
            sw=sw,
        )

    def curve_arrow(self, x0, y0, x1, y1, lift, colour, sw=3.0, head=True) -> None:
        """A quadratic arc from one peak to another -- the J-coupling connector.

        The head is drawn from the curve's own end tangent (for a quadratic that is simply the
        control point to the endpoint), so it stays glued to the arc when the lift changes.
        """
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - lift
        self.path(f"M {x0} {y0} Q {cx} {cy} {x1} {y1}", stroke=colour, sw=sw, cap="round")
        if not head:
            return
        dx, dy = x1 - cx, y1 - cy
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / norm, dy / norm
        size, spread = 17.0, 0.42
        pts = []
        for sign in (1, -1):
            ax = ux * math.cos(sign * spread) - uy * math.sin(sign * spread)
            ay = ux * math.sin(sign * spread) + uy * math.cos(sign * spread)
            pts.append(f"{x1 - size * ax:.1f} {y1 - size * ay:.1f}")
        self.path(f"M {x1} {y1} L {pts[0]} L {pts[1]} Z", fill=colour)

    def text(
        self, x, y, s, size, family, weight, fill, anchor="middle", spacing=None, italic=False
    ) -> None:
        extra = f' letter-spacing="{spacing}"' if spacing else ""
        if italic:
            extra += ' font-style="italic"'
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}"{extra}>{_esc(s)}</text>'
        )

    def runs(self, x, y, runs: list[tuple[str, str, float, str]]) -> None:
        """One left-anchored line built from several fonts, e.g. prose followed by a code span.

        Emitted as `<tspan>`s inside a single `<text>` so the runs flow from one advance position.
        Positioning them as separate `<text>` elements would require knowing each run's rendered
        width, which depends on the font -- and would drift the moment any label changed.
        """
        body = "".join(
            f'<tspan font-family="{fam}" font-size="{size}" fill="{fill}">{_esc(s)}</tspan>'
            for s, fam, size, fill in runs
        )
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="start">{body}</text>')

    def done(self) -> str:
        return "".join(self.parts) + "</svg>"


def pipeline(t: dict[str, str]) -> str:
    """The front-page pipeline figure.

    Geometry measured off the committed `pipeline.png` rather than eyeballed. One deliberate
    departure: that PNG's fourth latent node was r=9.5 against its siblings' 11.5 and sat at y=255,
    breaking their 47-unit rhythm. It is regularised here.
    """
    c = Canvas(
        1820,
        388,
        "MolDeTr pipeline",
        "A proton NMR window of up to 1200 Hz is resampled to 6144 points at 5.12 points per Hz, "
        "passed through MolDeTr, a Deformable-DETR with an FPN backbone and transformer, which "
        "emits one box per multiplet, decoded to chemical shift, coupling, proton count and line "
        "width.",
    )
    c.rect(0.75, 0.75, 1818.5, 386.5, 20, t["page"], t["border"])
    for x, w in ((55, 302), (453, 302), (1300, 217)):
        c.rect(x, 120, w, 148, 18, t["panel"], t["border"])
    c.rect(1608, 120, 161, 148, 22, t["navy"])
    c.rect(850, 87, 355, 214, 22, t["card"], t["border"])

    inputs = [(906, 147.5), (906, 194.0), (906, 240.5)]
    latent = [(1027.5, 133.0), (1027.5, 180.0), (1027.5, 227.0), (1027.5, 274.0)]
    output = [(1149.5, 147.5), (1149.5, 194.0), (1149.5, 240.5)]
    for i, j in ((0, 0), (0, 1), (1, 2), (2, 3)):
        c.line(*inputs[i], *latent[j], t["connector"])
    for j, k in ((0, 0), (1, 1), (2, 2), (3, 2)):
        c.line(*latent[j], *output[k], t["connector"])
    for cx, cy in inputs:
        c.circle(cx, cy, 11.5, t["ink"])
    for cx, cy in latent:
        c.circle(cx, cy, 11.5, t["latent"])
    for (cx, cy), col in zip(output, (t["blue"], t["orange"], t["teal"])):
        c.circle(cx, cy, 11.5, col)

    for x0, x1 in ((367, 443), (764, 840), (1527, 1594)):
        c.arrow(x0, x1, 194.0, t["arrow"])
    for y, col in ((147.5, t["blue"]), (194.0, t["orange"]), (240.5, t["teal"])):
        c.arrow(1215, 1282, y, col)

    c.text(206, 180, "¹H NMR window", 27.3, SG, 700, t["display"])
    c.text(206, 220, "≤ 1200 Hz", 23, PLEX, 400, t["mute"])
    c.text(604, 180, "Resample", 27.3, SG, 700, t["display"])
    c.text(604, 220, "6144 pts @ 5.12 pts/Hz", 23, PLEX, 400, t["mute"])
    c.text(1028, 74, "MolDeTr · Deformable-DETR", 25.3, SG, 700, t["display"])
    c.text(1027.5, 332, "FPN backbone + transformer, set prediction", 22.2, PLEX, 400, t["mute"])
    c.text(1409, 180, "One box per", 27.3, SG, 700, t["display"])
    c.text(1409, 216, "multiplet", 27.3, SG, 700, t["display"])
    c.text(1688.5, 167.5, "δ · J", 26, SG, 700, t["onSolid"])
    c.text(1688.5, 204, "protons", 26, SG, 700, t["onSolid"])
    c.text(1688.5, 240.5, "line width", 26, SG, 700, t["onSolid"])
    return c.done()


def architecture(t: dict[str, str]) -> str:
    """The "1-D Deformable-DETR" architecture figure.

    Geometry measured off the committed `architecture.png`. Two things change beyond the format.
    The eyebrow and the secondary labels were `#74808f`, the token `docs/BRAND.md` retired for
    measuring 4.01:1 on white against the 4.5:1 that small text requires; they now use the
    replacement `#666f7d` (5.08:1). And the figure had 82px of empty canvas below its content
    against 52 above, so the viewBox is trimmed to sit evenly.
    """
    c = Canvas(
        1800,
        678,
        "MolDeTr architecture: a 1-D Deformable-DETR",
        "A 6144-point window enters an FPN backbone of eleven residual convolutional blocks, then "
        "a deformable transformer encoder and decoder with N object queries, trained with "
        "Hungarian matching. Per-query heads emit multiplet-or-empty, chemical shift, a coupling "
        "embedding, proton count and line width, giving spin systems in a single forward pass.",
        faces=("sg700", "plex400", "plex600", "mono400"),
        ground=t["card"],
    )
    _masthead(c, t, "Architecture — a 1-D Deformable-DETR")
    c.text(
        1745,
        118,
        "set prediction · single forward pass",
        24.1,
        MONO,
        400,
        t["eyebrow"],
        anchor="end",
    )

    c.rect(58, 350, 220, 112, 18, t["panel"], t["border"], sw=3)
    c.text(168, 396, "6144-pt", 25, SG, 700, t["display"])
    c.text(168, 432, "window", 21, PLEX, 400, t["mute"])

    c.rect(358, 322, 260, 168, 18, t["panel"], t["border"], sw=3)
    c.text(488, 373, "FPN backbone", 25, SG, 700, t["display"])
    c.text(488, 408, "conv, multi-scale", 21, PLEX, 400, t["mute"])
    c.text(488, 440, "11 residual blocks", 19, MONO, 400, t["mute"])

    c.text(846, 277, "Deformable transformer", 25, SG, 700, t["display"])
    c.rect(698, 298, 296, 216, 22, t["card"], t["border"], sw=3)
    c.text(846, 351, "encoder ↕ decoder", 23, PLEX, 400, t["mute"])
    for i, col in enumerate((t["blue"], t["orange"], t["teal"], t["latent"])):
        c.circle(768 + 48 * i, 396, 14, col)
    c.text(846, 459, "N object queries", 23, PLEX, 400, t["mute"])

    # Training-only path: dashed, because Hungarian matching is not part of a forward pass.
    c.path("M 846 514 V 558", stroke=t["orange"], sw=2, dash="6 6", opacity=0.75)
    c.rect(696, 556, 300, 68, 16, t["warnBg"], t["orange"], sw=2.5, dash="7 7")
    c.text(846, 597, "Hungarian matching · training", 21, PLEX, 400, t["orange"])

    c.rect(1074, 226, 396, 360, 22, t["page"], t["border"], sw=3)
    c.text(1272, 264, "Per-query heads", 25, SG, 700, t["display"])
    bullets = [
        (320, "• multiplet / ∅", None),
        (368, "• δ chemical shift", None),
        (416, "• J embedding ", "[sum,min,max,std]"),
        (464, "• proton count (class)", None),
        (512, "• line width", None),
    ]
    for y, label, mono in bullets:
        if mono is None:
            c.text(1105, y, label, 23, PLEX, 400, t["ink"], anchor="start")
        else:
            c.runs(1105, y, [(label, PLEX, 23, t["ink"]), (mono, MONO, 19, t["mute"])])

    for x0, x1 in ((280, 357), (620, 697), (1004, 1073), (1470, 1544)):
        c.arrow(x0, x1, 407, t["arrow"])

    c.rect(1544, 320, 192, 116, 22, t["navy"])
    c.text(1640, 368, "spin", 25, SG, 700, t["onSolid"])
    c.text(1640, 403, "systems", 22, PLEX, 400, t["onSolid"])
    return c.done()


def _masthead(c: Canvas, t: dict[str, str], title: str, size: float = 39.4) -> None:
    """The tricolor dash triple and the display title every figure but `pipeline` opens with.

    `docs/BRAND.md`: "the tricolor dash triple is the recurring mark -- header, figures, banner.
    Reuse it; don't invent new logos." Measured identically on all four PNGs, hence one helper.
    """
    for i, col in enumerate((t["blue"], t["orange"], t["teal"])):
        c.rect(56 + 40 * i, 52, 32, 8, 4, col)
    c.text(55, 112, title, size, SG, 700, t["display"], anchor="start")


def input_contract(t: dict[str, str]) -> str:
    """What MolDeTr accepts: a <=1200 Hz window, resampled to 6144 points, min-max normalised.

    Two changes beyond the format. The caption in the PNG this replaces reads "ependent -- 80-600
    MHz all map to 1200 Hz" -- the leading "Field-ind" is **missing from the committed asset**, and
    the README's own alt text says "field-independent", so the sentence is restored here rather
    than re-rendered as-is. And the figure had 84 px of empty canvas below its content against 52
    above, so the viewBox is trimmed to sit evenly; content coordinates are unchanged.
    """
    c = Canvas(
        1720,
        476,
        "The MolDeTr input contract",
        "A proton NMR window of at most 1200 Hz is resampled at 5.12 points per Hz to 6144 "
        "real-valued finite points, then min-max normalised. The contract is field-independent: "
        "80 to 600 MHz spectra all map to the same 1200 Hz window, because MolDeTr works in hertz.",
        faces=("sg700", "plex400", "plex600", "mono400"),
        ground=t["card"],
    )
    _masthead(c, t, "The input contract")

    c.text(356, 192, "1200 Hz window", 26.1, SG, 700, t["display"])
    c.bracket(96.5, 615.5, 215.5, 8, t["blue"])
    # The axis runs the full span; the trace does not. Each multiplet is its own segment returning
    # to baseline, with bare axis between -- measured off the PNG, and the same construction
    # `coupling_rule` uses.
    c.path("M 96 363.5 H 616", stroke=t["arrow"], sw=2.6)
    for apex, height, half in ((245.5, 53, 43.5), (403.5, 61, 42), (567.5, 51, 43)):
        c.spectrum(apex - half, apex + half, 363.3, ((apex, height, 10.8),), t["ink"], sw=2.6)

    c.text(705, 250, "resample", 23.5, PLEX, 400, t["eyebrow"])
    c.arrow(656, 760, 290, t["arrow"])
    c.text(705, 338, "5.12 pts/Hz", 21.9, MONO, 400, t["eyebrow"])

    c.rect(788.5, 216.5, 359, 147, 18, t["panel"], t["border"], sw=3)
    c.text(968, 276, "6144 points", 30, SG, 700, t["display"])
    c.text(968, 316, "real-valued · finite", 23, PLEX, 400, t["mute"])

    c.arrow(1176, 1280, 290, t["arrow"])

    c.rect(1308, 216, 320, 148, 22, t["navy"])
    c.text(1468, 277, "min−max", 30, SG, 700, t["onSolid"])
    c.text(1468, 316, "normalised", 23, PLEX, 400, t["onSolid"])

    c.text(
        56,
        421,
        "Field-independent — 80–600 MHz all map to 1200 Hz (MolDeTr works in Hz)",
        23,
        PLEX,
        400,
        t["eyebrow"],
        anchor="start",
    )
    return c.done()


def coupling_rule(t: dict[str, str]) -> str:
    """The one rule the README repeats: a window must contain every partner of every proton in it.

    Two panels of the same spectrum, differing only in where the window is drawn -- which is the
    whole argument, so the geometry is shared and only the window rect and the arc's endpoint move.
    """
    c = Canvas(
        1720,
        584,
        "Keep coupling partners together",
        "Two windows over the same spectrum. In the first, both partners of a J coupling lie "
        "inside the window, so the spin system is complete and the prediction is valid. In the "
        "second the coupling partner falls outside the window, so that multiplet is out of "
        "distribution and its prediction is wrong.",
        ground=t["card"],
    )
    _masthead(c, t, "Keep coupling partners together")

    for x, wash, edge in (
        (58, t["tealWash"], t["tealEdge"]),
        (880, t["brickWash"], t["brickEdge"]),
    ):
        c.rect(x, 158, 782, 310, 24, wash, edge, sw=2)

    c.text(112, 220, "✓", 30, PLEX, 600, t["teal"])
    c.text(142, 220, "Complete spin system", 28.3, SG, 700, t["tealText"], anchor="start")
    c.text(934, 220, "✕", 26, PLEX, 600, t["brick"])
    c.text(964, 220, "Partner outside → wrong", 28.3, SG, 700, t["brick"], anchor="start")

    # The windows. Dashed, because a region of interest is a choice the user draws, not a feature
    # of the spectrum -- the same visual language the GUI uses for an unconfirmed selection. The
    # fill is `card`, not transparent: lifting the window off its panel wash is what makes "inside"
    # and "outside" legible at a glance, and it is the whole subject of the figure.
    c.rect(125.5, 265, 646, 130, 18, t["windowFill"], t["teal"], sw=3, dash="9 8.6")
    c.rect(947.5, 265, 465, 130, 18, t["windowFill"], t["brick"], sw=3, dash="9 8.6")

    # Five separate 86 px segments, not one continuous trace. The PNG draws each multiplet as its
    # own curve returning to baseline, with bare ground between them -- measured, and it is the
    # right reading anyway: what the figure argues about is individual multiplets, not a spectrum.
    for apex in (255.5, 373.5, 630.0, 1077.5, 1195.5):
        c.spectrum(apex - 43, apex + 43, 384.3, ((apex, 45, 9.25),), t["ink"], sw=2.6)
    # The partner that fell outside: same peak, drawn dashed and greyed because the model never
    # sees it. It is the only thing that differs between the two panels' spectra.
    c.path(
        _lorentz_d(1505, 1591, 386.3, ((1548, 41, 9.25),)),
        stroke=t["connector"],
        sw=2.6,
        dash="9 8",
        cap="round",
    )

    # Lifts fitted by least squares against the arcs traced off the PNG (max residual 1.6 px and
    # 2.7 px), not eyeballed -- a quadratic control point is not where the apex looks like it is.
    # Italic J because it names a physical quantity, the same rule that keeps δ lowercase in
    # `docs/BRAND.md`. The original is italic too; a roman J here would read as a letter.
    c.curve_arrow(337, 296, 604, 296, 40.3, t["teal"], head=False)
    c.text(466, 284, "J", 22, PLEX, 400, t["teal"], italic=True)
    c.curve_arrow(1153, 296.5, 1508, 332, 72.2, t["brick"])
    c.text(1288, 284, "J", 22, PLEX, 400, t["brick"], italic=True)

    c.text(449, 427, "both partners inside the window", 20.8, PLEX, 400, t["mute"])
    c.text(1508, 427, "out of window", 20.8, PLEX, 400, t["brick"])

    c.text(
        56,
        527,
        "A window need not hold the whole molecule, but every proton coupling to a proton inside "
        "it must also be inside (≤ 1200 Hz).",
        23,
        PLEX,
        400,
        t["eyebrow"],
        anchor="start",
    )
    return c.done()


def benchmark(t: dict[str, str]) -> str:
    """The paper's experimental headline: three medians and the per-class proton-count accuracy.

    The numbers are the article's and are not recomputed here -- `tests/test_paper_medians.py`
    upstream is what holds them. `0.89 / 0.20 / 93.5` and `97 / 89 / 75` must match the README
    prose beside this figure; `tests/test_readme_claims.py` is what would catch a drift.
    """
    c = Canvas(
        1720,
        756,
        "MolDeTr experimental benchmark",
        "Across 12 experimental spectra and 13 regions of interest from 80 to 600 MHz, measured "
        "against ground truth: median absolute chemical-shift error 0.89 Hz, median absolute "
        "coupling error 0.20 Hz, and 93.5 percent overall proton-count accuracy, which breaks "
        "down as 97 percent for 1H, 89 percent for 2H and 75 percent for 3H.",
        faces=("sg700", "plex400", "plex600", "mono400"),
        ground=t["card"],
    )
    _masthead(c, t, "Experimental benchmark")
    c.text(
        1663,
        118,
        "12 spectra · 13 ROIs · 80–600 MHz · vs ground truth",
        24,
        MONO,
        400,
        t["eyebrow"],
        anchor="end",
    )

    cards = (
        (56, t["blue"], "0.89", "Hz", "median |Δδ|", "chemical shift"),
        (602, t["teal"], "0.20", "Hz", "median |ΔJ|", "coupling (structured_output)"),
        (1149, t["orange"], "93.5", "%", "proton-count accuracy", "overall (DETR-style)"),
    )
    for x, accent, value, unit, label, sub in cards:
        # The accent is the card's top edge, so it is a second rounded rect showing through by 8 px
        # rather than a bar laid on top: at the corners it follows the same curve the card does.
        c.rect(x, 160, 515, 242, 21, accent)
        c.rect(x, 168, 515, 234, 21, t["card"], t["border"], sw=3)
        # The display numeral is tracked in. Matching the original on cap height alone left it 7 %
        # too wide, and matching on width alone left it too short -- the two only reconcile with
        # negative tracking, which is the usual treatment for large numerals anyway.
        c.text(x + 42, 268, value, 75.4, SG, 700, accent, "start", "-2")
        # x + 208 clears all three values because all three are four glyphs wide ("0.89", "0.20",
        # "93.5") -- these are the article's frozen medians, so the offset is safe to hardcode.
        c.text(x + 208, 267, unit, 38.5, PLEX, 400, t["mute"], anchor="start")
        c.text(x + 41, 316, label, 27.1, PLEX, 600, t["ink"], anchor="start")
        c.text(x + 42, 358, sub, 25, PLEX, 400, t["eyebrow"], anchor="start")

    c.path("M 56 438.5 H 1664", stroke=t["rule"], sw=2)
    c.text(56, 494, "PROTON-COUNT ACCURACY BY CLASS", 25.7, SG, 700, t["eyebrow"], "start", "2.2")

    # blue -> orange -> teal, the BRAND.md marker cycle for multiplets 1 - 2 - 3, reused here for
    # proton classes 1H - 2H - 3H so the same class keeps the same colour across every figure.
    for i, (row, accent, pct) in enumerate(
        (("1 H", t["blue"], 97), ("2 H", t["orange"], 89), ("3 H", t["teal"], 75))
    ):
        y = 528 + 50 * i
        c.text(56, y + 20, row, 28.3, SG, 700, t["display"], anchor="start")
        c.rect(149, y, 1403, 20, 10, t["track"])
        c.rect(149, y, round(1403 * pct / 100), 20, 10, accent)
        # Number and sign are two runs, in two different faces. The original sets them to the same
        # cap height, which Space Grotesk cannot do -- its percent sign is both taller and narrower
        # than its digits, and matching one dimension always misses the other. IBM Plex's is not.
        c.text(1631, y + 20, str(pct), 27, SG, 700, t["ink"], anchor="end")
        c.text(1663, y + 20, "%", 27, PLEX, 400, t["ink"], anchor="end")

    c.text(
        56,
        698,
        "Medians from the article's Hungarian-matched pairs "
        "(structured_output/experimental_matched_pairs.json). 4H/6H not exercised on real data.",
        23,
        PLEX,
        400,
        t["eyebrow"],
        anchor="start",
    )
    return c.done()


DIAGRAMS = {
    "pipeline": pipeline,
    "architecture": architecture,
    "input_contract": input_contract,
    "coupling_rule": coupling_rule,
    "benchmark": benchmark,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the committed SVGs match this source; write nothing",
    )
    args = ap.parse_args()

    stale: list[str] = []
    for name, fn in DIAGRAMS.items():
        for suffix, palette in (("", LIGHT), ("-dark", DARK)):
            dest = OUT_DIR / f"{name}{suffix}.svg"
            svg = fn(palette)
            if args.check:
                current = dest.read_text(encoding="utf-8") if dest.is_file() else ""
                if current != svg:
                    stale.append(dest.name)
            else:
                dest.write_text(svg, encoding="utf-8")
                print(f"{dest.relative_to(ROOT)}  {len(svg.encode()) / 1024:.1f} KB")

    if args.check:
        if stale:
            print(f"stale, re-run without --check: {', '.join(stale)}", file=sys.stderr)
            return 1
        print(f"{2 * len(DIAGRAMS)} committed SVG(s) match this source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
