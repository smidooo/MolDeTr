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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "docs" / "fonts"
OUT_DIR = ROOT / "docs" / "img"

#: `docs/BRAND.md` § Palette, verbatim. A colour census of the PNGs these replace matched these
#: tokens exactly, so the figures were already generated from this table.
LIGHT = {
    "page": "#f8fafd",
    "panel": "#f1f5fa",
    "border": "#d5dfeb",
    "card": "#ffffff",
    "navy": "#1f3a5f",
    "ink": "#20242b",
    "mute": "#5b6675",
    "latent": "#7d92b0",
    "connector": "#cdd7e4",
    "arrow": "#9db0c6",
    "onSolid": "#ffffff",
    # `eyebrow` is BRAND.md's REPLACEMENT token. Every figure this generator
    # supersedes painted small text in the retired #74808f, which measures 4.01:1 on
    # white -- under the 4.5:1 WCAG AA threshold that applies to exactly the small
    # labels it was defined for. BRAND.md replaced it with #666f7d (5.08:1).
    "eyebrow": "#666f7d",
    "warnBg": "#fdf5ea",
    "blue": "#2566b0",
    "orange": "#e08a1f",
    "teal": "#1f9e8c",
}
#: `docs/BRAND.md` § Dark-figure palette. The tricolor is deliberately unchanged -- BRAND.md:
#: "the tricolor stays identical in dark mode -- blue/orange/teal read well on both grounds".
DARK = {
    "page": "#141821",
    "panel": "#1b2130",
    "border": "#334054",
    "card": "#1b2130",
    "navy": "#e8eef6",
    "ink": "#e8eef6",
    "mute": "#9fb0c4",
    "latent": "#7d92b0",
    "connector": "#334054",
    "arrow": "#5f708a",
    "onSolid": "#141821",
    "eyebrow": "#8f9db0",
    "warnBg": "#241f16",
    "blue": "#2566b0",
    "orange": "#e08a1f",
    "teal": "#1f9e8c",
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
    ) -> None:
        self.w, self.h = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-labelledby="t d">',
            f'<title id="t">{_esc(title)}</title><desc id="d">{_esc(desc)}</desc>',
            f"<defs><style>{_faces(faces)}</style></defs>",
        ]

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

    def text(self, x, y, s, size, family, weight, fill, anchor="middle") -> None:
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}">{_esc(s)}</text>'
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

    c.text(206, 180, "¹H NMR window", 27.3, SG, 700, t["navy"])
    c.text(206, 220, "≤ 1200 Hz", 23, PLEX, 400, t["mute"])
    c.text(604, 180, "Resample", 27.3, SG, 700, t["navy"])
    c.text(604, 220, "6144 pts @ 5.12 pts/Hz", 23, PLEX, 400, t["mute"])
    c.text(1028, 74, "MolDeTr · Deformable-DETR", 25.3, SG, 700, t["navy"])
    c.text(1027.5, 332, "FPN backbone + transformer, set prediction", 22.2, PLEX, 400, t["mute"])
    c.text(1409, 180, "One box per", 27.3, SG, 700, t["navy"])
    c.text(1409, 216, "multiplet", 27.3, SG, 700, t["navy"])
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
    )
    for i, col in enumerate((t["blue"], t["orange"], t["teal"])):
        c.rect(56 + 40 * i, 52, 32, 8, 4, col)

    c.text(56, 106, "Architecture — a 1-D Deformable-DETR", 33, SG, 700, t["navy"], anchor="start")
    c.text(
        1742, 118, "set prediction · single forward pass", 23, MONO, 400, t["eyebrow"], anchor="end"
    )

    c.rect(58, 350, 220, 112, 18, t["panel"], t["border"])
    c.text(168, 396, "6144-pt", 25, SG, 700, t["navy"])
    c.text(168, 432, "window", 21, PLEX, 400, t["mute"])

    c.rect(358, 322, 260, 168, 18, t["panel"], t["border"])
    c.text(488, 373, "FPN backbone", 25, SG, 700, t["navy"])
    c.text(488, 408, "conv, multi-scale", 21, PLEX, 400, t["mute"])
    c.text(488, 440, "11 residual blocks", 19, MONO, 400, t["mute"])

    c.text(846, 277, "Deformable transformer", 25, SG, 700, t["navy"])
    c.rect(698, 298, 296, 216, 22, t["card"], t["border"])
    c.text(846, 351, "encoder ↕ decoder", 23, PLEX, 400, t["mute"])
    for i, col in enumerate((t["blue"], t["orange"], t["teal"], t["latent"])):
        c.circle(768 + 48 * i, 396, 14, col)
    c.text(846, 459, "N object queries", 23, PLEX, 400, t["mute"])

    # Training-only path: dashed, because Hungarian matching is not part of a forward pass.
    c.parts.append(
        f'<path d="M 846 514 V 558" stroke="{t["orange"]}" stroke-width="2" '
        f'stroke-dasharray="6 6" fill="none" opacity="0.75"/>'
    )
    c.rect(697, 558, 298, 64, 16, t["warnBg"], t["orange"], sw=2, dash="7 7")
    c.text(846, 597, "Hungarian matching · training", 21, PLEX, 400, t["orange"])

    c.rect(1074, 226, 396, 360, 22, t["panel"], t["border"])
    c.text(1272, 264, "Per-query heads", 25, SG, 700, t["navy"])
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

    for x0, x1 in ((288, 349), (628, 689), (1004, 1065), (1480, 1537)):
        c.arrow(x0, x1, 406, t["arrow"])

    c.rect(1544, 320, 192, 116, 22, t["navy"])
    c.text(1640, 368, "spin", 25, SG, 700, t["onSolid"])
    c.text(1640, 403, "systems", 22, PLEX, 400, t["onSolid"])
    return c.done()


DIAGRAMS = {"pipeline": pipeline, "architecture": architecture}


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
