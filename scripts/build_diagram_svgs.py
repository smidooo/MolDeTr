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
to Space Grotesk while delta alone drops through to a face that has it.

Without that second name, delta does not render as tofu -- this docstring used to say it did, and
`docs/fonts/README.md` said the same. Measured 2026-08-11: it falls through to whatever the
reader's OS supplies. Secure static mode blocks external *resource loads*, which is why these faces
are embedded rather than referenced, but a system font is not a resource load. Reader-dependent
rendering at the wrong weight is the actual failure, and it is worse to detect than a blank box
because nobody reports it. `tests/test_diagram_fonts.py` holds every committed SVG against
`GLYPHS`; `docs/fonts/README.md` carries the probe.
"""

from __future__ import annotations

import argparse
import base64
import functools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "docs" / "fonts"
OUT_DIR = ROOT / "docs" / "img"

#: The spectrum the hero banner plots. It is the same file the deleted `scripts/gen_banner.py` read
#: before the design-tool banner replaced it, and its `ground_truth` shifts (6.959 / 7.385 / 7.42)
#: are the assignment table's 6.96 / 7.39 / 7.42. `tests/test_readme_figures.py` holds the link.
BANNER_NPZ = ROOT / "examples" / "roi_S8_example.npz"

#: The ppm window both banner panels show, solved from the design-tool banner's own tick labels
#: (`7.4`/`7.0` centred at x 248.5/620.0 left and 1448.0/2207.0 right). The two panels agreed on
#: 7.50 -> 6.90 to within 0.004 ppm, so this is one window drawn at two widths, not two crops.
PPM_LEFT, PPM_RIGHT = 7.50, 6.90

#: The committed treatment, and the only one `--check` may compare against. A separate constant from
#: `TRACE` because `main()` ASSIGNS `TRACE`, so reading the default back out of it gives "whatever
#: the last call in this process left behind" rather than the shipped value. Harmless while the only
#: caller was `python build_diagram_svgs.py`, which exits; `tests/test_diagram_svgs.py` now calls
#: `main()` in-process, where one `--trace ideal` run would leak into every later `--check` and make
#: it compare the committed banner against a variant nobody committed -- the exact outcome the
#: guard below exists to prevent, arrived at through the guard passing.
DEFAULT_TRACE = "faithful"

#: Which treatment `banner()` draws its spectra in; `--trace` overrides it. Module state rather than
#: an argument because `DIAGRAMS` maps a name to a one-argument palette function, and widening that
#: signature for the one figure that needs it would touch all seven.
TRACE = DEFAULT_TRACE

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
    # The line-width column's ink: quieter than `mute` without going as pale as `latent`.
    # Measured off both prediction PNGs (#6a7686 light, #93a2b5 dark, 100% of the masked
    # pixels in each) rather than chosen -- BRAND.md publishes the dark value as `dim`.
    "dim": "#6a7686",
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
    # The mark's own ink, on the navy tile. Not `onSolid`: it is a touch cooler than white, and the
    # extraction shows it does NOT flip in dark -- the tile lightens under it instead.
    "tileInk": "#eaf1fb",
    # The banner's ground is the one non-flat background in the set: a soft diagonal wash. These
    # two stops are a least-squares fit over 20_000 ground pixels of the asset they replace, not
    # its corner samples -- the corners are the least representative points on a wash. The fit
    # leaves 3.7 levels of residual and adding radial or quadratic terms does not reduce it, so
    # that residual is texture in the export rather than a shape a gradient could take.
    "groundTop": "#fafcfd",
    "groundBottom": "#eef3f8",
    # A decomposed multiplet, filled under the resolved trace. These are not alpha blends of the
    # tricolor -- measured off the banner and mapped through its dark twin, which is where they
    # stop being derivable (`orangeFill` lands on #604325, nothing a formula would produce).
    "blueFill": "#a8c3e0",
    "orangeFill": "#e7c7a1",
    "tealFill": "#a1ccc5",
    # What a raised panel casts. A shadow is a *darkening*, so this cannot be one value for both
    # themes: the light figure's blue-grey painted on the dark ground reads as a glow around
    # every card, which is the opposite of the depth cue it is there for.
    "shadowInk": "#8494ab",
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
    "dim": "#93a2b5",
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
    "tileInk": "#eaf1fb",
    "groundTop": "#131923",
    "groundBottom": "#0d1219",
    "blueFill": "#1f3d63",
    "orangeFill": "#604325",
    "tealFill": "#18474a",
    "shadowInk": "#01030a",
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

#: What each embedded family carries BEYOND printable ASCII. Measured from the `.woff2` cmaps on
#: 2026-08-11, not copied from the subsetting recipe -- the recipe in `docs/fonts/README.md` listed
#: `⁻` as included while its own last bullet said no family has it, and the bullet was the correct
#: half. All five subsets cover printable ASCII in full, so only this tail differs. Re-derive after
#: any re-subsetting with:
#:
#:     python -c "from fontTools.ttLib import TTFont; f=TTFont('docs/fonts/plex400.woff2'); \
#:     print(''.join(chr(c) for c in sorted(f.getBestCmap()) if c > 0x7F))"
#:
#: Keyed by family rather than unioned, because the asymmetry is load-bearing: `δ` exists in IBM
#: Plex Sans ONLY. In an `SG` or `PLEX` run CSS's per-glyph fallback supplies it from the Plex face
#: named second; in a `MONO` run it has nowhere to go, since `MONO` names no Plex face at all. A
#: single union would call that safe. `tests/test_diagram_fonts.py` holds every committed SVG to it.
GLYPHS = {
    "Space Grotesk": "°±²³·¹Å×÷Δ–—‘’“”•…→↕∅−≤≥",
    "IBM Plex Sans": "°±²³·¹Å×÷Δδ–—‘’“”•…→↕−≤≥✓",
    "IBM Plex Mono": "°±²³·¹Å×÷–—‘’“”•…→↕−≤≥✓",
}


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


# maxsize covers the distinct (npz, window) triples the figure set asks for -- the banner's window,
# and the two prediction figures' full spans -- so each file is still read once per build. It was 1
# while there was one window; leaving it at 1 would silently re-read and re-normalise on every
# alternating call, which is slow rather than wrong, and therefore the kind of thing nobody notices.
@functools.lru_cache(maxsize=8)
def _spectrum(
    npz: Path = BANNER_NPZ, left: float = PPM_LEFT, right: float = PPM_RIGHT
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """One NPZ over one ppm window: (ppm descending, amplitude normalised 0-1, shifts).

    Defaults are the banner's, so every existing call site is unchanged -- `--check` proves it, and
    that proof is the reason the arguments were added this way rather than by rewriting the callers.

    Amplitude is normalised here rather than at each call site so the trace and the multiplet fills
    below it cannot end up on two different scales -- which is exactly the kind of disagreement a
    reader would take for a statement about the data.

    numpy is imported lazily. It is a core dependency of the package, but a docs generator that
    cannot even print its own `--help` without the scientific stack is a worse thing to hand
    someone than one that loads it when asked to draw.
    """
    import numpy as np

    # allow_pickle for `ground_truth`, an object array of dicts. Safe because `npz` is never user
    # input: every caller passes a module-level constant naming a file committed to this repository,
    # and this script takes no path argument that could reach here. That was self-evident while the
    # path was hardcoded; it is a constraint worth stating now that it is a parameter, because the
    # justification lives in the call sites rather than in this line.
    data = np.load(npz, allow_pickle=True)
    ppm = np.asarray(data["ppm_axis_padded"], dtype=float)
    amp = np.real(data["spectrum_padded"]).astype(float)
    window = (ppm >= right) & (ppm <= left)
    order = np.argsort(-ppm[window])  # high ppm first: a 1H axis reads right-to-left
    ppm, amp = ppm[window][order], amp[window][order]
    amp = (amp - float(np.median(amp))) / float(amp.max() - np.median(amp))
    shifts = sorted(float(g["chemical_shift_ppm"]) for g in data["ground_truth"])
    # Cached, so the three call sites per palette read the file once. Returned as tuples for
    # the same reason: a cached list is one caller away from being mutated for everyone.
    return tuple(ppm.tolist()), tuple(amp.tolist()), tuple(shifts)


def _multiplets(
    x0: float,
    x1: float,
    height: float,
    npz: Path = BANNER_NPZ,
    left: float = PPM_LEFT,
    right: float = PPM_RIGHT,
) -> list[list[tuple[float, float, float]]]:
    """Per multiplet, one `(centre_x, height_px, half-width_px)` Lorentzian per line it contains.

    This is the shape MolDeTr itself predicts -- a multiplet is a set of lines sharing a shift and
    a coupling, not a single broad hump -- and drawing it that way is what makes a fill sit *on*
    the trace. One Lorentzian per multiplet was tried first and floats: with real 300 MHz data the
    lines are 2 px wide, so a hump spanning the whole group towers over the gaps between them.

    Each sample goes to its *nearest* recorded shift. A fixed +-0.045 ppm window was tried too and
    is wider than the 0.035 ppm between this spectrum's two aromatic doublets, so H_B's window
    reached across and claimed H_A's taller peak as its own.
    """
    ppm, amp, shifts = _spectrum(npz, left, right)
    span = (x1 - x0) / (left - right)
    step = abs(ppm[1] - ppm[0])
    out = []
    for shift in shifts:
        idx = [
            i
            for i, p in enumerate(ppm)
            if min(shifts, key=lambda s: abs(p - s)) == shift and abs(p - shift) <= 0.06
        ]
        tallest = max(amp[i] for i in idx)
        lines = []
        for i in idx:
            if amp[i] < tallest * 0.12 or amp[i] < max(amp[max(i - 3, 0) : i + 4]):
                continue
            # Half-width from the data: walk out until the line drops below half, and cap at the
            # gap to its neighbour so an unresolved shoulder cannot inflate it.
            j = next((k for k in range(1, 40) if amp[min(i + k, len(amp) - 1)] < amp[i] * 0.5), 6)
            lines.append((x0 + (left - ppm[i]) * span, amp[i] * height, max(j * step * span, 2.4)))
        out.append(lines)
    return out


def _trace_d(
    x0: float,
    x1: float,
    base: float,
    height: float,
    mode: str,
    npz: Path = BANNER_NPZ,
    left: float = PPM_LEFT,
    right: float = PPM_RIGHT,
) -> str:
    """Path data for a banner panel's spectrum, in whichever treatment `mode` names.

    `faithful` plots the array. The other two exist because the asset this replaces did not: it was
    a design-tool redrawing of the same spectrum, only 0.58-0.68 correlated with it. Both of them
    still take their peak positions, heights and widths from `_multiplets`, so the choice is one of
    *rendering*, not of what the figure claims -- an idealised trace here is a smoothed spectrum,
    never a different one.
    """
    if mode != "faithful":
        peaks = [line for group in _multiplets(x0, x1, height, npz, left, right) for line in group]
        if mode == "ideal":
            return _lorentz_d(x0, x1, base, peaks, step=1.5)
        return _hybrid_d(x0, x1, base, peaks, height)

    ppm, amp, _ = _spectrum(npz, left, right)
    span = (x1 - x0) / (left - right)
    pts = [(x0 + (left - p) * span, base - a * height) for p, a in zip(ppm, amp)]
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in _thin(pts, int(x1 - x0)))


def _thin(pts: list[tuple[float, float]], columns: int) -> list[tuple[float, float]]:
    """Reduce to at most one min/max pair per output column.

    Plain decimation would *discard* noise rather than shrink it: dropping every other sample of a
    band-limited-looking wiggle lowers its apparent amplitude, so the trace would flatten as the
    panel narrows. Keeping each column's extremes preserves the envelope, which is the property a
    reader judges a spectrum's noise floor by.
    """
    if len(pts) <= columns:
        return pts
    per, out = len(pts) / columns, []
    for i in range(columns):
        chunk = pts[int(i * per) : max(int((i + 1) * per), int(i * per) + 1)]
        lo, hi = min(chunk, key=lambda p: p[1]), max(chunk, key=lambda p: p[1])
        out.extend([lo, hi] if lo[0] <= hi[0] else [hi, lo])
    return out


def _hybrid_d(x0: float, x1: float, base: float, peaks, height: float) -> str:
    """The idealised envelope plus reproducible noise.

    The noise is a fixed sum of incommensurate sinusoids, not a PRNG: `--check` compares committed
    bytes, so anything reseeded per interpreter run would report the figure stale on every machine.
    """
    pts, x = [], float(x0)
    while x <= x1:
        y = base - sum(h / (1.0 + ((x - c) / w) ** 2) for c, h, w in peaks)
        wobble = sum(
            math.sin(x * f + p) for f, p in ((1.7, 0.0), (4.31, 1.7), (9.13, 3.9), (17.7, 2.2))
        )
        pts.append(f"{x:g} {y + wobble * height * 0.011:.1f}")
        x += 1.5
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

    def rect(self, x, y, w, h, rx, fill, stroke=None, sw=1.5, dash=None, opacity=None) -> None:
        """`opacity` for the multiplet washes, which are the tricolor at 0.16 rather than their own
        colours. Extracting the dark twin confirms it: 0.16 over the dark ground reproduces all
        three measured dark hexes exactly, so a translucent fill is one token where three opaque
        ones would be three that can drift."""
        edge = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        if opacity is not None:
            edge += f' fill-opacity="{opacity}"'
        if dash:
            edge += f' stroke-dasharray="{dash}"'
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{edge}/>'
        )

    def circle(self, cx, cy, r, fill) -> None:
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>')

    def line(self, x0, y0, x1, y1, stroke, sw=1.6, opacity=None) -> None:
        """`opacity` rather than a pre-blended hex: the lollipop stems sit at 0.65, and blending
        them against the light ground by hand would bake the light palette into a shared geometry.
        """
        alpha = f' stroke-opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(
            f'<path d="M {x0} {y0} L {x1} {y1}" stroke="{stroke}" stroke-width="{sw}"'
            f'{alpha} fill="none"/>'
        )

    def arrow(self, x0, x1, y, colour, head=15.0) -> None:
        """Shaft plus a solid head. Measured off the originals: 3px shaft, 18px head."""
        back = x1 - head
        self.parts.append(
            f'<path d="M {x0} {y} H {back - 1}" stroke="{colour}" stroke-width="3" '
            f'stroke-linecap="round" fill="none"/>'
            f'<path d="M {x1} {y} L {back} {y - 9} L {back} {y + 9} Z" fill="{colour}"/>'
        )

    def path(
        self, d, stroke=None, fill="none", sw=1.5, dash=None, cap=None, opacity=None, ident=None
    ) -> None:
        """Escape hatch for geometry the named helpers do not cover.

        `ident` is for the few paths a test needs to find again. Naming them beats matching on
        shape: `tests/test_readme_figures.py` reads the banner's traces back out to check they
        still plot the committed spectrum, and "the two longest paths" is not a contract.
        """
        bits = [f'<path d="{d}"' + (f' id="{ident}"' if ident else "") + f' fill="{fill}"']
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
        self,
        x,
        y,
        s,
        size,
        family,
        weight,
        fill,
        anchor="middle",
        spacing=None,
        italic=False,
        ident=None,
    ) -> None:
        """`ident` is for tests, exactly as `path`'s is.

        A figure that prints `8.74` *somewhere* is a different claim from one that prints it in row
        1's `max J` column, and a value in the wrong row is the defect that actually shipped here.
        Scraping numbers out of the document cannot tell those apart; a named cell can.
        """
        extra = f' letter-spacing="{spacing}"' if spacing else ""
        if italic:
            extra += ' font-style="italic"'
        if ident:
            extra += f' id="{ident}"'
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}"{extra}>{_esc(s)}</text>'
        )

    def gradient(self, ident: str, top: str, bottom: str) -> str:
        """A corner-to-corner linear wash. Returns the `url(#..)` so it can be used as any fill."""
        self.parts.append(
            f'<defs><linearGradient id="{ident}" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="{top}"/>'
            f'<stop offset="1" stop-color="{bottom}"/></linearGradient></defs>'
        )
        return f"url(#{ident})"

    def shadow(self, ident: str, dy: float, blur: float, colour: str, opacity: float) -> str:
        """A drop shadow, returned as the `filter` value.

        The region is deliberately far larger than the blur radius. A `feDropShadow` is clipped to
        its filter region, and a soft wide shadow clipped mid-falloff shows as a straight edge --
        which is what a default `-10%/120%` region does to these.
        """
        self.parts.append(
            f'<defs><filter id="{ident}" x="-50%" y="-50%" width="200%" height="220%">'
            f'<feDropShadow dx="0" dy="{dy}" stdDeviation="{blur}" flood-color="{colour}" '
            f'flood-opacity="{opacity}"/></filter></defs>'
        )
        return f"url(#{ident})"

    def panel(self, x, y, w, h, rx, fill, filt) -> None:
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'filter="{filt}"/>'
        )

    def hexagon(self, cx, cy, r, fill, stroke=None, sw=3.0, dash=None) -> None:
        """A pointy-top hexagon -- vertical left and right edges, vertices at top and bottom.

        This is the orientation both hexagons in the set use, and the one benzene is conventionally
        drawn in, so `r` is the ring's circumradius and the flat sides are the ones bonds leave.
        """
        pts = [
            (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
            for a in (-90, -30, 30, 90, 150, 210)
        ]
        d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts) + " Z"
        self.path(d, stroke=stroke, fill=fill, sw=sw, dash=dash)

    def sub(self, x, y, base, subscript, size, family, weight, fill, anchor="start") -> None:
        """A label with a subscript, e.g. H_A or T_2, as one `<text>` so the runs share an advance.

        `baseline-shift` is avoided on purpose: Chrome honours it, librsvg and several static
        rasterisers do not, and a subscript that silently sits on the baseline reads as a second
        letter. `dy` is universally supported and is undone by the matching negative shift.
        """
        drop = size * 0.26
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}">{_esc(base)}'
            f'<tspan dy="{drop:.1f}" font-size="{size * 0.62:.1f}">{_esc(subscript)}</tspan></text>'
        )

    def runs(self, x, y, runs: list[tuple], anchor: str = "start") -> None:
        """One line built from several fonts, e.g. prose followed by a code span.

        Emitted as `<tspan>`s inside a single `<text>` so the runs flow from one advance position.
        Positioning them as separate `<text>` elements would require knowing each run's rendered
        width, which depends on the font -- and would drift the moment any label changed. That is
        also why `anchor="end"` is worth having: right-aligning a mixed-font line by computing where
        to start it is the same measurement problem, and the renderer already knows the answer.

        Each run is `(text, family, size, fill)`, optionally extended with `weight`, `italic` and a
        baseline `dy`. The `dy` is how a subscript joins a mixed line -- `<tspan>` shifts accumulate,
        so a shifted run must be followed by one carrying the negation if the line continues.
        """
        body = ""
        for run in runs:
            s, fam, size, fill = run[:4]
            weight = run[4] if len(run) > 4 else 400
            italic = run[5] if len(run) > 5 else False
            shift = run[6] if len(run) > 6 else 0
            # Emitted only when it differs from the initial value. The parent <text> sets no
            # weight, so `font-weight="400"` is a no-op -- and writing it anyway would rewrite
            # every committed SVG that already uses `runs`, reporting them stale for no change.
            bold = f' font-weight="{weight}"' if weight != 400 else ""
            style = ' font-style="italic"' if italic else ""
            dy = f' dy="{shift}"' if shift else ""
            body += (
                f'<tspan font-family="{fam}" font-size="{size}"{bold} '
                f'fill="{fill}"{style}{dy}>{_esc(s)}</tspan>'
            )
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}">{body}</text>')

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
        # `∅` is carried by Space Grotesk ONLY, and this run is Plex-Sans-only -- so it is split
        # across two families rather than reworded. DETR's no-object symbol is worth keeping.
        (320, ("• multiplet / ", "∅"), None),
        (368, "• δ chemical shift", None),
        (416, "• J embedding ", "[sum,min,max,std]"),
        (464, "• proton count (class)", None),
        (512, "• line width", None),
    ]
    for y, label, mono in bullets:
        if isinstance(label, tuple):
            # Prose in Plex Sans, then one symbol in the Space Grotesk stack. A deliberate,
            # single-glyph face change beats the alternative, which is the browser making the same
            # substitution invisibly from whatever the reader's OS happens to hold.
            # Weight 700 stated explicitly, because it is what will paint either way: the only
            # Space Grotesk face any diagram embeds is `sg700`. Left at the inherited 400 the
            # markup would claim a weight that is not present, render bold regardless, and change
            # behaviour the day a 400 face is vendored.
            prose, symbol = label
            c.runs(1105, y, [(prose, PLEX, 23, t["ink"]), (symbol, SG, 23, t["ink"], 700)])
        elif mono is None:
            c.text(1105, y, label, 23, PLEX, 400, t["ink"], anchor="start")
        else:
            c.runs(1105, y, [(label, PLEX, 23, t["ink"]), (mono, MONO, 19, t["mute"])])

    for x0, x1 in ((280, 357), (620, 697), (1004, 1073), (1470, 1544)):
        c.arrow(x0, x1, 407, t["arrow"])

    c.rect(1544, 320, 192, 116, 22, t["navy"])
    c.text(1640, 368, "spin", 25, SG, 700, t["onSolid"])
    c.text(1640, 403, "systems", 22, PLEX, 400, t["onSolid"])
    return c.done()


def _masthead(
    c: Canvas,
    t: dict[str, str],
    title: str,
    size: float = 39.4,
    x0: float = 56,
    y0: float = 52,
    baseline: float = 112,
    spacing: str | None = None,
) -> None:
    """The tricolor dash triple and the display title every figure but `pipeline` opens with.

    `docs/BRAND.md`: "the tricolor dash triple is the recurring mark -- header, figures, banner.
    Reuse it; don't invent new logos." Measured identically on all four PNGs, hence one helper.

    The origin is a parameter because the prediction figures set it 8 px further left and up
    (dashes from x=48, y=48) -- the same mark, not a different one, so it stays one helper rather
    than becoming two that can drift. Defaults reproduce the existing four exactly; `--check` holds
    that.
    """
    for i, col in enumerate((t["blue"], t["orange"], t["teal"])):
        c.rect(x0 + 40 * i, y0, 32, 8, 4, col)
    c.text(x0 - 1, baseline, title, size, SG, 700, t["display"], "start", spacing)


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
    # U+00D7, not U+2715. No vendored face carries U+2715 -- measured against all three upstream
    # cmaps, not merely absent from the subset -- so it would resolve to the reader's system font,
    # in the one figure whose entire semantic content is these two marks. U+00D7 is covered by all
    # three families and is the conventional rejection marker beside a checkmark. Sized to the
    # checkmark rather than to the glyph it replaces: at 30 the multiplication sign measured 13.2
    # units of ink height against the checkmark's 20.7, so it read as an afterthought beside it.
    # 47 = 30 x (20.7 / 13.2), i.e. the size at which the two marks stand the same height -- taken
    # off the render, not guessed, because "looks about right" is what put a 1.57x mismatch on a
    # figure whose entire content is these two marks.
    c.text(934, 220, "×", 47, PLEX, 600, t["brick"])
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


def mark(t: dict[str, str]) -> str:
    """The app-icon mark: navy tile, tricolor dashes, two peaks, wordmark.

    Referenced by nothing until now, which is why it is here at all -- a finished asset the repo
    never showed. As vector it can serve as the docs-site favicon, where a 672 px raster is both
    too big and the wrong shape for the job.

    Its colour mapping is a third independent confirmation of the `navy` fill split: `#1f3a5f`
    maps to `#31517d` at 100 % over 230_084 px, the same value `input_contract` and `architecture`
    gave, and nothing like the `#e8eef6` the display text takes.
    """
    c = Canvas(
        672,
        672,
        "MolDeTr mark",
        "The MolDeTr app icon: a rounded navy tile carrying the blue, orange and teal dash triple, "
        "a two-peak NMR trace, and the MolDeTr wordmark.",
        faces=("sg700",),
        ground=t["card"],
    )
    # Shadow parameters swept against the reference rather than guessed: dy/stdDeviation/opacity
    # over a 3x3x3 grid, scored by mean absolute difference. 26/40/0.34 wins at 5.00/255 against
    # 6.65 for the first guess. The filter region has to be generous -- the reference shadow is
    # still 18 levels deep 62 px below the tile, and a tight region clips it into a visible edge.
    c.parts.append(
        '<defs><filter id="sh" x="-60%" y="-60%" width="220%" height="240%">'
        '<feDropShadow dx="0" dy="26" stdDeviation="40" flood-color="#3d4a5e" '
        'flood-opacity="0.34"/></filter></defs>'
    )
    c.parts.append(
        f'<rect x="84" y="84" width="504" height="504" rx="102" fill="{t["navy"]}" '
        f'filter="url(#sh)"/>'
    )
    for i, col in enumerate((t["blue"], t["orange"], t["teal"])):
        c.rect(198 + 99 * i, 174, 78, 21, 10.5, col)
    c.spectrum(175, 496, 357.5, ((262, 63, 17), (394, 51, 15)), t["tileInk"], sw=6)
    c.text(338, 483, "MolDeTr", 63.7, SG, 700, t["tileInk"])
    return c.done()


def _banner_masthead(c: Canvas, t: dict[str, str]) -> None:
    """Wordmark and tagline on the left, the claim and the citation right-aligned on the right."""
    for i, col in enumerate((t["blue"], t["orange"], t["teal"])):
        c.rect(94 + 40 * i, 83, 29, 7, 3.5, col)
    c.text(
        222,
        96,
        "CHEMISTRY-INFORMED DEEP LEARNING · ¹H NMR",
        27.2,
        SG,
        700,
        t["eyebrow"],
        "start",
        "1.21",
    )
    c.text(101, 219, "MolDeTr", 114.8, SG, 700, t["display"], "start", "-2.2")
    c.text(
        94,
        295,
        "The spin system, straight from the spectrum.",
        44.3,
        SG,
        700,
        t["display"],
        "start",
        "-3.4",
    )

    for i, line in enumerate(("Reads spectra beyond", "manual interpretation")):
        c.text(2461, 119.5 + 52.4 * i, line, 44.5, SG, 700, t["display"], anchor="end")
    c.text(
        2462,
        226.5,
        "& outperforms leading analysis software on difficult, strongly-coupled spectra",
        29.5,
        PLEX,
        400,
        t["mute"],
        anchor="end",
    )
    # The journal name is the only italic in the set. It is a title, and the alternative -- setting
    # it upright and hoping the reader infers it -- is what the citation surfaces elsewhere avoid.
    c.runs(
        2461,
        282,
        [
            ("Published in ", PLEX, 30, t["mute"]),
            ("Analytical Chemistry", PLEX, 30, t["display"], 600, True),
            (" · 2026", PLEX, 30, t["mute"]),
        ],
        anchor="end",
    )


def _banner_axis(c: Canvas, t: dict[str, str], ticks, mid: float) -> None:
    """The shared ppm scale: two labelled ticks and the axis title, one per panel."""
    for x, label in ticks:
        c.text(x, 756, label, 26, PLEX, 400, t["ink"])
    c.runs(
        mid,
        791,
        [("δ", PLEX, 27.1, t["mute"], 400, True), (" [ppm]", PLEX, 26, t["mute"])],
        anchor="middle",
    )


def _banner_raw_card(c: Canvas, t: dict[str, str], shade: str, mode: str) -> None:
    """Left panel: the spectrum as it arrives, and the structure you do not have."""
    c.panel(97, 333, 675, 806, 32, t["card"], shade)
    c.text(137, 388, "RAW ¹H NMR", 30.4, SG, 700, t["display"], "start", "1.9")
    c.text(136, 424, "overlapping · strongly coupled", 26.9, PLEX, 400, t["mute"], anchor="start")
    c.path("M 133 715.5 H 735", stroke=t["rule"], sw=2)
    c.path(
        _trace_d(153, 714, 715.5, 202, mode), stroke=t["ink"], sw=3, cap="round", ident="trace-raw"
    )
    _banner_axis(c, t, ((248.5, "7.4"), (620, "7.0")), 434)

    c.path("M 133 808 H 735", stroke=t["border"], sw=2.5, dash="12 10")
    # The ghost hexagon is `panel`, not the near-white literal the design tool used. That literal
    # was invisible on white by luck and a solid bright blob on the dark card -- 36 % of the region.
    c.hexagon(300.8, 966.8, 65.6, t["panel"], t["latent"], sw=3, dash="11 9")
    for x, y in ((357.5, 934.0), (204.0, 1021.0), (397.5, 1022.5)):
        c.path(
            f"M {244 if x < 300 else 357.5} {999.5 if x < 300 else (934 if y < 1000 else 999.5)}"
            f" L {x} {y}",
            stroke=t["latent"],
            sw=3,
            dash="11 9",
        )
    c.text(300.8, 985, "?", 52, SG, 700, t["latent"])
    c.text(497, 957, "STRUCTURE", 23.2, SG, 700, t["eyebrow"], "start", "2.2")
    c.text(498, 999, "unknown?", 30.9, SG, 700, t["display"], "start", "-1.1")


def _banner_bridge(c: Canvas, t: dict[str, str], shade: str) -> None:
    """The model between the panels: one grey arrow in, three coloured ones out."""
    c.arrow(774, 843, 629, t["arrow"])
    c.panel(882, 528, 200, 202, 55, t["card"], shade)
    layers = (
        ([572, 613, 654, 686], 926, 10, t["ink"]),
        ([560, 596, 634, 670, 703], 983, 8, t["latent"]),
        ([592, 634, 674], 1040, 12.5, None),
    )
    for (ys0, x0, _, _), (ys1, x1, _, _) in zip(layers, layers[1:]):
        for y0 in ys0:
            for y1 in ys1:
                c.line(x0, y0, x1, y1, t["connector"], sw=1)
    for ys, x, r, fill in layers:
        for i, y in enumerate(ys):
            c.circle(x, y, r, fill or (t["blue"], t["orange"], t["teal"])[i])
    for y, col in ((579, t["blue"]), (628, t["orange"]), (680, t["teal"])):
        c.arrow(1107, 1172, y, col)
    c.text(983, 852, "MolDeTr", 37.2, SG, 700, t["display"], spacing="-0.94")
    c.text(982, 881, "detection transformer", 25.2, PLEX, 400, t["mute"])


def _banner_molecule(c: Canvas, t: dict[str, str]) -> None:
    """Ethyl vanillin, drawn around the ring the three assigned protons sit on.

    The ring is solved rather than eyeballed: the orange, teal and blue dots in the asset this
    replaces measure (1362.1, 962.3), (1362.0, 1004.2) and (1434.2, 962.4), which for a pointy-top
    hexagon fixes the centre and circumradius to 0.24 px. Every other vertex follows.
    """
    cx, cy, r = 1398.2, 983.3, 41.9
    top, ur, lr = (cx, cy - r), (cx + 0.866 * r, cy - r / 2), (cx + 0.866 * r, cy + r / 2)
    bot, ll, ul = (cx, cy + r), (cx - 0.866 * r, cy + r / 2), (cx - 0.866 * r, cy - r / 2)
    c.hexagon(cx, cy, r, "none", t["ink"], sw=4.5)
    # Kekulé: alternate edges carry the inner line. Inset 8 units toward the centre and shortened
    # at both ends, which is how a second bond is drawn rather than a doubled outline.
    for a, b in ((top, ur), (ul, ll), (bot, lr)):
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx, dy = (cx - mx) / r * 9, (cy - my) / r * 9
        ax, ay = a[0] + (b[0] - a[0]) * 0.18 + dx, a[1] + (b[1] - a[1]) * 0.18 + dy
        bx, by = a[0] + (b[0] - a[0]) * 0.82 + dx, a[1] + (b[1] - a[1]) * 0.82 + dy
        c.path(f"M {ax:.1f} {ay:.1f} L {bx:.1f} {by:.1f}", stroke=t["ink"], sw=4.5, cap="round")

    c.path(
        f"M {top[0]} {top[1]} L 1398.2 910 L 1369 890 M 1398.2 910 L 1432 890",
        stroke=t["ink"],
        sw=4.5,
        cap="round",
    )
    c.path("M 1404 906 L 1436 886", stroke=t["ink"], sw=4.5, cap="round")  # the carbonyl's second
    c.text(1353, 886, "H", 28.6, SG, 700, t["ink"])
    c.text(1443, 886, "O", 28.6, SG, 700, t["ink"])
    c.path(f"M {bot[0]} {bot[1]} L 1398.2 1056", stroke=t["ink"], sw=4.5, cap="round")
    c.text(1398.5, 1086, "OH", 28.6, SG, 700, t["ink"])
    c.path(f"M {lr[0]:.1f} {lr[1]:.1f} L 1452 1000", stroke=t["ink"], sw=4.5, cap="round")
    c.text(1463.5, 1011, "O", 28.6, SG, 700, t["ink"])
    c.path("M 1476 1006 L 1489 1024 L 1508 1014", stroke=t["ink"], sw=4.5, cap="round")
    c.sub(1512, 1020, "CH", "3", 28.6, SG, 700, t["ink"])

    # Each label is start-anchored at its own measured left edge. Mirroring the two on the left to
    # `anchor="end"` looks symmetric and is wrong: it hangs them off the far side of the bond.
    for (vx, vy), (lx, ly), col, letter, tx, ty in (
        (ul, (1332, 948), t["orange"], "B", 1299, 950),
        (ll, (1332, 1014), t["teal"], "C", 1299, 1024),
        (ur, (1466, 948), t["blue"], "A", 1469, 948),
    ):
        c.path(f"M {vx:.1f} {vy:.1f} L {lx} {ly}", stroke=col, sw=4.5, cap="round")
        c.circle(vx, vy, 6, col)
        c.sub(tx, ty, "H", letter, 28.6, SG, 700, col)
    c.text(1411, 1107, "ASSIGNMENT", 21.7, SG, 700, t["eyebrow"], spacing="2.9")


def _banner_table(c: Canvas, t: dict[str, str]) -> None:
    """The per-multiplet answer: one row per spin system, the four quantities MolDeTr returns."""
    c.text(1626, 863, "SPIN", 23, SG, 700, t["eyebrow"], "start", "1.6")
    # The `2` of `T2` is a composed run, not U+2082. No vendored subset carries a subscript digit,
    # so the literal dropped out of the font stack and rendered in whatever face the reader's OS
    # supplied -- at a fixed design size that ignores the run around it. `_banner_resolved_card`
    # below composes the identical label; the 0.62 / 0.26 ratios here are `Canvas.sub`'s, but that
    # site hand-types its 16.7 / 7, so the three agree to ~1% BY EYE, not by construction. Deriving
    # them here rather than typing them is still worth it: `Canvas.runs` does not round, so a
    # hand-typed `27.1 * 0.62` would have emitted `16.802000000000003` into the file.
    # `<tspan>` shifts accumulate, so the unit run carries the negation or the line steps down.
    drop = 23 * 0.26
    for x, base, sub, unit in (
        (1896, "δ", "", " [PPM]"),
        (2091, "J", "", " [HZ]"),
        (2419, "T", "2", " [MS]"),
    ):
        runs = [(base, SG, 23, t["eyebrow"], 700, True)]
        if sub:
            runs.append((sub, SG, 23 * 0.62, t["eyebrow"], 700, False, drop))
        runs.append((unit, SG, 23, t["eyebrow"], 700, False, -drop if sub else 0))
        c.runs(x, 863, runs, anchor="end")
    c.text(2269, 863, "PROTONS", 23, SG, 700, t["eyebrow"], "end", "1.6")
    c.path("M 1625 888.5 H 2421", stroke=t["border"], sw=2)

    rows = (
        ("A", t["blue"], "7.39", "1.5", "557"),
        ("B", t["orange"], "7.42", "8.7, 1.5", "637"),
        ("C", t["teal"], "6.96", "8.7", "707"),
    )
    for i, (letter, col, shift, coupling, t2) in enumerate(rows):
        y = 938 + 72 * i
        if i:
            c.path(f"M 1625 {y - 49.5} H 2421", stroke=t["rule"], sw=1.5)
        c.circle(1634.5, y - 12, 10, col)
        c.text(1663, y - 1, "H", 37, SG, 700, col, anchor="start")
        c.text(1701, y - 5, letter, 23, SG, 700, col, anchor="start")
        for x, value in ((1897, shift), (2092, coupling), (2270, "1")):
            c.text(x, y, value, 34.4, SG, 700, t["ink"], anchor="end")
        c.text(2419, y, t2, 34.4, SG, 700, t["mute"], anchor="end")


def _banner_resolved_card(c: Canvas, t: dict[str, str], shade: str, mode: str) -> None:
    """Right panel: the same window, decomposed into the three multiplets and their parameters."""
    c.panel(1193, 333, 1268, 806, 32, t["card"], shade)
    c.text(1234, 388, "RESOLVED SPIN SYSTEM", 30.4, SG, 700, t["display"], "start", "1.9")
    c.runs(
        1234,
        424,
        [
            ("per multiplet — ", PLEX, 27.1, t["mute"]),
            ("δ", PLEX, 27.1, t["mute"], 400, True),
            (" · ", PLEX, 27.1, t["mute"]),
            ("J", PLEX, 27.1, t["mute"], 400, True),
            (" · protons · ", PLEX, 27.1, t["mute"]),
            ("T", PLEX, 27.1, t["mute"], 400, True),
            ("2", PLEX, 16.7, t["mute"], 400, False, 7),
        ],
    )
    c.path("M 1240 717 H 2415", stroke=t["rule"], sw=2)

    # The decomposition, drawn under the trace. Ordered `shifts` ascending, so the fills pair with
    # C, A, B down the ppm axis -- the colours follow the proton, not the drawing order.
    fills = (t["tealFill"], t["blueFill"], t["orangeFill"])
    for lines, fill in zip(_multiplets(1256, 2397, 212), fills):
        lo = min(cx for cx, _, _ in lines) - 46
        hi = max(cx for cx, _, _ in lines) + 46
        d = _lorentz_d(lo, hi, 717, lines, step=1.5)
        c.path(d + f" L {hi:.0f} 717 Z", fill=fill)
    c.path(
        _trace_d(1256, 2397, 717, 212, mode),
        stroke=t["ink"],
        sw=2.6,
        cap="round",
        ident="trace-resolved",
    )

    for x, y, letter, col in (
        (1459, 502, "A", t["blue"]),
        (1322, 552, "B", t["orange"]),
        (2264, 572, "C", t["teal"]),
    ):
        c.sub(x, y, "H", letter, 34, SG, 700, col)
    _banner_axis(c, t, ((1448, "7.4"), (2207, "7.0")), 1826.5)
    c.path("M 1240 808 H 2420", stroke=t["rule"], sw=2)
    _banner_molecule(c, t)
    _banner_table(c, t)


def banner(t: dict[str, str]) -> str:
    """The README and docs-site hero.

    Unlike the six diagrams beside it this is a data figure, and its curve is the spectrum in
    `examples/roi_S8_example.npz` -- vanillin in DMSO-d6 at 300.13 MHz, whose `ground_truth` shifts
    are the assignment table's three rows. The asset it replaces was a design-tool *redrawing* of
    that array, recognisable but only 0.58-0.68 correlated with it; here the array is plotted.

    Sharing one geometry between light and dark also retires a defect the tests could not see: the
    two PNGs were 2560x1283 and 2560x1280, an aspect drift that cleared the 0.005 tolerance by
    0.0003 while `<picture>` was stretching the dark twin to the light one's box.
    """
    c = Canvas(
        2560,
        1283,
        "MolDeTr",
        "MolDeTr reads a raw 1H NMR spectrum and returns the spin system in it. Left, the "
        "overlapping, strongly coupled 7.5 to 6.9 ppm window of vanillin at 300 MHz with the "
        "structure unknown; right, the same window resolved into three multiplets with their "
        "chemical shifts 7.39, 7.42 and 6.96 ppm, couplings 1.5, 8.7 and 8.7 Hz, one proton each, "
        "and T2 of 557, 637 and 707 ms, assigned onto ethyl vanillin. Across 12 experimental "
        "spectra from 80 to 600 MHz the median errors are 0.89 Hz in shift and 0.20 Hz in "
        "coupling, with 93.5 percent proton-count accuracy.",
        faces=("sg700", "plex400", "plex600"),
    )
    ground = c.gradient("wash", t["groundTop"], t["groundBottom"])
    c.parts.append(f'<rect width="{c.w}" height="{c.h}" fill="{ground}"/>')
    shade = c.shadow("lift", 10, 18, t["shadowInk"], 0.30)

    _banner_masthead(c, t)
    _banner_raw_card(c, t, shade, TRACE)
    _banner_bridge(c, t, shade)
    _banner_resolved_card(c, t, shade, TRACE)

    c.runs(
        210,
        1195,
        [
            (
                "Experimental benchmark · 12 spectra · 80–600 MHz · vs. ground truth:  ",
                PLEX,
                32.7,
                t["mute"],
            ),
            ("0.89 Hz", SG, 32.7, t["blue"], 700),
            (" median |Δδ|   ·   ", PLEX, 32.7, t["mute"]),
            ("0.20 Hz", SG, 32.7, t["teal"], 700),
            (" median |ΔJ|   ·   ", PLEX, 32.7, t["mute"]),
            ("93.5%", SG, 32.7, t["orange"], 700),
            (" proton-count accuracy", PLEX, 32.7, t["mute"]),
        ],
    )
    return c.done()


# --- the two prediction figures ------------------------------------------------------------------
#
# Both replace a 1520x996 PNG (exactly 2x a 760x498 design) that sat at the 2.00x device-pixel floor
# and read soft. There was nothing to re-export from -- no design source is committed or on disk, and
# the two matplotlib generators that once existed were deleted in `2827cdb` precisely because they
# were off-brand and would clobber these assets -- so "re-raster" would have meant writing a new
# generator to emit PNGs. This writes SVG instead, for the reason at the top of this file.
#
# Three deliberate departures from the superseded artwork, each because the artwork was wrong:
#
# 1. THE TRACE IS THE ARRAY, at full resolution and proportional. The old figures drew the
#    guajazulene set with C (delta 7.314, amplitude 0.881) TALLER than A (delta 8.221, amplitude
#    1.000) -- the ordering inverted, most likely by point-sampling that clipped A's narrower apex.
#    Relative intensity is something a reader takes from a spectrum, so this is a misstatement of the
#    data, and reproducing it would mean copying a sampling artefact on purpose.
# 2. NO DECORATIVE BASELINE NOISE. The old wobble is about 7x the array's real noise (max |amp|
#    outside the peaks is 4.2e-04; it is not identically zero, as a first measurement pass reported).
#    Plotting the array gives the true noise floor, which is what the trace claims to show.
# 3. THE TICKS ARE LABELLED. The originals draw ticks every 0.5 ppm and label none of them, so the
#    scale is unstated and the figure is not interpretable on its own. Labelling them also makes
#    `tests/test_figure_numbers.py` able to read the curve back into ppm through the figure's own
#    labels, which is what turns "it plots the right spectrum" into a falsifiable claim.
#
# One departure that is not a correction: the eyebrow ships BRAND's `#666f7d`, not the retired
# `#74808f` these PNGs used, which measures 4.01:1 on white and fails WCAG AA for the small text it
# was defined for.

#: What the figures print. Read, never hardcoded: `tests/test_figure_numbers.py` holds the committed
#: SVG to this file and `tests/test_scripts_local.py` holds the file to the published weights, so the
#: figure cannot drift from the model without one of them failing. Before it existed, these numbers
#: lived only as pixels -- and one of the two figures was transposing a pair of couplings.
FIGURE_NUMBERS = ROOT / "docs" / "figure_predictions.json"

#: Measured off the superseded PNGs. `PLOT_TOP` is where normalised amplitude 1.0 lands, so the
#: tallest peak in the array reaches it by construction whichever spectrum is drawn.
PLOT_X0, PLOT_X1 = 68.0, 1452.0
PLOT_BASE, PLOT_TOP = 576.5, 223.0
AXIS_Y, TICK_Y = 580.0, 590.0

#: `(anchor x, header, anchor side, json field, format, fill key)`. One table drives both figures;
#: the numerals are tabular, which is why `8.221` and `7.420` occupy identical extents.
TABLE_COLUMNS = (
    (530.9, "δ [PPM]", "end", "shift_ppm", "{:.3f}", "ink"),
    (818.9, "MAX J [HZ]", "end", "max_j_hz", "{:.1f}", "ink"),
    (1057.0, "PROTONS", "end", "protons", "{} H", "ink"),
    (1472.9, "LINE WIDTH [HZ]", "end", "linewidth_hz", "{:.2f}", "dim"),
)


@functools.lru_cache(maxsize=1)
def _figure_numbers() -> dict:
    """`FIGURE_NUMBERS`, parsed once. Plain JSON -- no pickle, no numpy, importable at --help time."""
    import json

    return json.loads(FIGURE_NUMBERS.read_text(encoding="utf-8"))["figures"]


def _prediction_eyebrow(
    c: Canvas, t: dict[str, str], head: str, tail: tuple[str, ...], anchor_x: float = 1470.4
) -> None:
    """The right-aligned mono caption, with its subscripts composed rather than written.

    `C₆D₆` and `DMSO-d₆` need U+2086, which no vendored subset carries -- see
    `tests/test_diagram_fonts.py`. The subscript is 0.6x size dropped 3 px, matching the PNGs
    rather than `Canvas.sub`'s 0.62/0.26, whose drop would be twice as far here.

    The padding run is the part that is easy to miss: in the original the subscript still occupies a
    full 14.4 px monospace cell, while a 14.4 px glyph advances only 8.64. Without the pad the `D`
    of `C₆D₆` lands 5.76 px early and the caption stops being monospaced.
    """
    size, drop = 24.0, 3.0
    runs: list[tuple] = [(head, MONO, size, t["eyebrow"])]
    for piece in tail:
        runs.append(("6", MONO, size * 0.6, t["eyebrow"], 400, False, drop))
        runs.append((" ", MONO, size * 0.4, t["eyebrow"], 400, False, -drop))
        if piece:
            runs.append((piece, MONO, size, t["eyebrow"]))
    # 1470.4, not the 1474.4 the PNGs measure as their advance end. That figure assumes the final
    # subscript occupies a full monospace cell, which it does in a raster and does NOT here: XML
    # collapses TRAILING whitespace, so the pad above is silently dropped on the last run while the
    # mid-string ones survive (the C-to-D pitch measures 29 px, one cell, correctly). Padding the
    # last one unconditionally was tried first and changed nothing, which is what identified this.
    c.runs(anchor_x, 110.0, runs, anchor="end")


def _prediction_plot(
    c: Canvas, t: dict[str, str], npz: Path, left: float, right: float, rows
) -> None:
    """The spectrum, its ppm axis, and one lollipop per detected multiplet."""
    height = PLOT_BASE - PLOT_TOP
    c.path(
        _trace_d(PLOT_X0, PLOT_X1, PLOT_BASE, height, "faithful", npz, left, right),
        stroke=t["ink"],
        sw=2.2,
        ident="trace",
    )

    span = (PLOT_X1 - PLOT_X0) / (left - right)
    c.path(f"M {PLOT_X0} {AXIS_Y} H {PLOT_X1}", stroke=t["arrow"], sw=2.4)
    n, tick = 0, math.ceil(right * 2) / 2
    while tick <= left + 1e-9:
        x = PLOT_X0 + (left - tick) * span
        c.path(f"M {x:.1f} {AXIS_Y} V {TICK_Y}", stroke=t["arrow"], sw=2.4)
        c.text(round(x, 1), 616, f"{tick:.1f}", 22.0, PLEX, 400, t["mute"], ident=f"tick-{n}")
        n, tick = n + 1, tick + 0.5
    c.runs(762.5, 656.0, [("δ [ppm]", PLEX, 25.2, t["mute"], 400, True)], anchor="middle")

    # Lollipop height follows the trace, so a marker can never float above or sink into its own peak.
    #
    # Markers closer than a diameter are stacked instead of overlapped. The superseded vanillin PNG
    # is why: its two ortho protons sit 0.031 ppm apart, which is 11 px in a 4 ppm window against
    # r=21 circles, so it painted the orange marker straight over the blue one and the figure shipped
    # with one of its three multiplets carrying no visible marker at all. The colour-to-multiplet
    # link is the whole point of the lollipops, so a hidden one is not a cosmetic loss.
    ppm, amp, _ = _spectrum(npz, left, right)
    placed: list[tuple[float, float]] = []
    for row, colour in zip(rows, (t["blue"], t["orange"], t["teal"])):
        shift = row["shift_ppm"]
        near = [a for p, a in zip(ppm, amp) if abs(p - shift) < 0.05]
        x = PLOT_X0 + (left - shift) * span
        apex = PLOT_BASE - max(near) * height
        cy = apex - 52.8
        # Walk up past anything already occupying this column. One pass over `placed` is enough
        # because each lift clears a full diameter, so a third marker in the same column lands
        # above the second rather than back on the first.
        for px, py in placed:
            if abs(px - x) < 46 and abs(py - cy) < 46:
                cy = py - 46
        placed.append((x, cy))
        c.line(round(x, 1), round(apex, 1), round(x, 1), round(cy + 21, 1), colour, 2.4, 0.65)
        c.circle(round(x, 1), round(cy, 1), 21, colour)


def _prediction_table(c: Canvas, t: dict[str, str], rows) -> None:
    """The same four quantities `predict.py` prints, one row per detected multiplet."""
    c.text(47.5, 706.0, "MULTIPLET", 22.86, SG, 700, t["eyebrow"], "start", "1.3")
    for x, header, anchor, *_ in TABLE_COLUMNS:
        c.text(x, 706.0, header, 22.86, SG, 700, t["eyebrow"], anchor, "1.3")
    c.path("M 48 729 H 1472", stroke=t["connector"], sw=2)

    for i, (row, colour) in enumerate(zip(rows, (t["blue"], t["orange"], t["teal"]))):
        y = 778.0 + 76.0 * i
        if i:
            c.path(f"M 48 {y - 49} H 1472", stroke=t["rule"], sw=2)
        c.circle(58, y - 10, 10, colour)
        c.text(83.85, y, str(row["n"]), 28.65, PLEX, 600, t["display"], "start")
        for x, _, anchor, field, fmt, fill in TABLE_COLUMNS:
            c.text(
                x,
                y,
                fmt.format(row[field]),
                28.65,
                PLEX,
                600,
                t[fill],
                anchor,
                ident=f"cell-{row['n']}-{field}",
            )


def _prediction_figure(t: dict[str, str], name: str, head: str, tail: tuple[str, ...]) -> str:
    spec = _figure_numbers()[name]
    npz = ROOT / spec["npz"]
    import numpy as np  # only to read the window the file itself declares

    meta = np.load(npz, allow_pickle=True)["metadata"].item()
    left, right = float(meta["left_ppm"]), float(meta["right_ppm"])

    c = Canvas(
        1520,
        996,
        "MolDeTr — detected multiplets",
        spec["alt"],
        faces=("sg700", "plex400", "plex600", "mono400"),
        ground=t["card"],
    )
    _masthead(c, t, "MolDeTr — detected multiplets", 38.0, 48, 48, 104, "-0.37")
    _prediction_eyebrow(c, t, head, tail)
    _prediction_plot(c, t, npz, left, right, spec["rows"])
    _prediction_table(c, t, spec["rows"])
    return c.done()


def example_prediction(t: dict[str, str]) -> str:
    """Guajazulene at 500 MHz: three well-separated aromatic protons."""
    return _prediction_figure(t, "example_prediction", "guajazulene · 500 MHz · C", ("D", ""))


def example_prediction_vanillin(t: dict[str, str]) -> str:
    """Vanillin at 300 MHz: the 1,2,4-trisubstituted ABX, two ortho doublets and a meta doublet."""
    return _prediction_figure(
        t, "example_prediction_vanillin", "vanillin · 300 MHz · DMSO-d", ("",)
    )


# --- vanillin, drawn beside its own spectrum ------------------------------------------------------

#: Ring geometry, all measured. Pointy-top, so the flat sides are the ones bonds leave -- the
#: orientation `Canvas.hexagon` already draws and the one benzene is conventionally drawn in.
RING_C, RING_R = (333.0, 438.5), 117.8
#: Which three edges carry the inner Kekulé bond, as (vertex angle, vertex angle) pairs.
#: **The opposite alternation from `_banner_molecule`** -- upper-left/top, the right vertical, and
#: lower-left/bottom, leaving the LEFT vertical bare. Both are valid Kekulé structures of benzene,
#: but they are different drawings, and copying the banner's would silently redraw this molecule.
KEKULE = ((210, -90), (-30, 30), (150, 90))
SPIN_X0, SPIN_X1 = 639.0, 1565.0
SPIN_LEFT, SPIN_RIGHT = 7.80, 6.55
SPIN_BASE, SPIN_HEIGHT = 612.51, 350.0


def _ring_vertex(angle: float) -> tuple[float, float]:
    return (
        RING_C[0] + RING_R * math.cos(math.radians(angle)),
        RING_C[1] + RING_R * math.sin(math.radians(angle)),
    )


def _vanillin_molecule(c: Canvas, t: dict[str, str], rows) -> None:
    """The ring, its substituents, and one numbered badge per aromatic proton.

    Not `_banner_molecule`: that one is smaller, labels its protons `H_A/H_B/H_C`, and draws the
    other Kekulé alternation. Same technique, different molecule drawing.
    """
    c.hexagon(*RING_C, RING_R, fill="none", stroke=t["ink"], sw=3.8)
    for a, b in KEKULE:
        (x0, y0), (x1, y1) = _ring_vertex(a), _ring_vertex(b)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        # Inset toward the ring centre, then take the middle 92 px of the edge.
        ix, iy = RING_C[0] - mx, RING_C[1] - my
        inorm = math.hypot(ix, iy)
        cx, cy = mx + 22.99 * ix / inorm, my + 22.99 * iy / inorm
        ex, ey = x1 - x0, y1 - y0
        enorm = math.hypot(ex, ey)
        ux, uy = 46.0 * ex / enorm, 46.0 * ey / enorm
        c.line(
            round(cx - ux, 2),
            round(cy - uy, 2),
            round(cx + ux, 2),
            round(cy + uy, 2),
            t["ink"],
            3.8,
        )

    for angle, (tx, ty) in ((-90, (333.0, 271.24)), (90, (333.0, 606.76)), (-30, (484.6, 351.0))):
        vx, vy = _ring_vertex(angle)
        c.line(round(vx, 2), round(vy, 2), tx, ty, t["ink"], 3.8)
    c.text(333.0, 253.0, "OH", 28.6, PLEX, 600, t["ink"])
    c.runs(
        496.5,
        359.0,
        [("OCH", PLEX, 28.6, t["ink"], 600), ("3", PLEX, 21.0, t["ink"], 600, False, 7.0)],
    )
    c.text(333.0, 642.0, "CHO", 28.6, PLEX, 600, t["ink"])

    # Badge, its proton label, and the colour that ties both to the multiplet below. The badge
    # numerals stay pure white in BOTH themes -- `onSolid` flips to #e2e9f4 and `card` to #1b2130,
    # so neither token expresses this and a literal is the honest way to say "always white".
    for row, colour, badge, label, label_x in zip(
        rows,
        (t["blue"], t["orange"], t["teal"]),
        ((198.07, 520.76), (467.93, 520.76), (198.07, 357.24)),
        ("H6", "H2", "H5"),
        (123.5, 511.5, 123.5),
    ):
        c.circle(badge[0], badge[1], 20.9, colour)
        c.text(badge[0], badge[1] + 8.0, str(row["n"]), 23.0, SG, 700, "#ffffff")
        c.text(label_x, badge[1] + 7.24, label, 23.5, SG, 700, t["display"], "start")
    c.text(261.5, 740.0, "Vanillin", 30.0, SG, 700, t["display"], "start")


def _spin_system_plot(c: Canvas, t: dict[str, str], npz: Path, rows) -> None:
    """The 7.80-6.55 ppm window, each multiplet washed in its own colour and marked."""
    span = (SPIN_X1 - SPIN_X0) / (SPIN_LEFT - SPIN_RIGHT)
    ppm, amp, _ = _spectrum(npz, SPIN_LEFT, SPIN_RIGHT)

    # Washes first, so the trace draws over them. 0.16 of the tricolor rather than three opaque
    # tokens -- extracting the dark twin reproduces all three measured dark hexes at that opacity.
    # `_multiplets` orders its groups by ASCENDING ppm; `rows` run descending, as `predict.py`
    # prints them. Zipping the two directly swaps blue and teal -- and the swap is only visible
    # because the markers below take their colour from `rows`, so the wash and the marker for one
    # proton end up different colours. Re-key by shift rather than trusting either order.
    by_shift = [
        colour
        for _, colour in sorted(
            zip(rows, (t["blue"], t["orange"], t["teal"])), key=lambda rc: rc[0]["shift_ppm"]
        )
    ]
    for lines, colour in zip(
        _multiplets(SPIN_X0, SPIN_X1, SPIN_HEIGHT, npz, SPIN_LEFT, SPIN_RIGHT), by_shift
    ):
        lo = min(cx for cx, _, _ in lines) - 10
        hi = max(cx for cx, _, _ in lines) + 10
        c.rect(
            round(lo, 1),
            185.0,
            round(hi - lo, 1),
            round(SPIN_BASE - 185.0, 1),
            0,
            colour,
            opacity=0.16,
        )

    c.path(
        _trace_d(SPIN_X0, SPIN_X1, SPIN_BASE, SPIN_HEIGHT, "faithful", npz, SPIN_LEFT, SPIN_RIGHT),
        stroke=t["ink"],
        sw=2.45,
        ident="trace",
    )
    c.path(f"M {SPIN_X0} 616.25 H {SPIN_X1}", stroke=t["arrow"], sw=2.32)
    n, tick = 0, 6.6
    while tick <= SPIN_LEFT - 0.1:
        x = SPIN_X0 + (SPIN_LEFT - tick) * span
        c.path(f"M {x:.1f} 616.25 V 625.84", stroke=t["arrow"], sw=2.32)
        c.text(round(x, 1), 651.0, f"{tick:.1f}", 21.0, PLEX, 400, t["mute"], ident=f"tick-{n}")
        n, tick = n + 1, tick + 0.2
    c.runs(1104.0, 697.0, [("δ [ppm]", PLEX, 24.0, t["mute"], 400, True)], anchor="middle")

    # No stems on this figure's markers -- measured, the gap to each peak is pure ground.
    for row, colour in zip(rows, (t["blue"], t["orange"], t["teal"])):
        near = [a for p, a in zip(ppm, amp) if abs(p - row["shift_ppm"]) < 0.03]
        apex = SPIN_BASE - max(near) * SPIN_HEIGHT
        x = SPIN_X0 + (SPIN_LEFT - row["shift_ppm"]) * span
        c.circle(round(x, 1), round(apex - 49.5, 1), 20.17, colour)


def _spin_system_legend(c: Canvas, t: dict[str, str], rows) -> None:
    """Three chips instead of a table. The couplings come from the data file, not from the artwork.

    The superseded PNG printed `J 8.2` on H6 and `J 8.7` on H5 -- the same transposition its sibling
    figure carried, from the same source. H6's `dd` is its ortho coupling to H5 (row 1's `max J`)
    and its meta coupling to H2, which IS row 2's `max J`: one physical coupling, predicted at both
    protons, so both numbers are derived here rather than typed.
    """
    ortho_h6, meta, ortho_h5 = (r["max_j_hz"] for r in rows)
    for (x, w), row, colour, proton, coupling in zip(
        ((621, 352), (989, 288), (1293, 290)),
        rows,
        (t["blue"], t["orange"], t["teal"]),
        ("H6", "H2", "H5"),
        (f"dd, J {ortho_h6:.1f}, {meta:.1f} Hz", f"d, J {meta:.1f} Hz", f"d, J {ortho_h5:.1f} Hz"),
    ):
        c.rect(x, 720, w, 50, 25, t["card"], stroke=colour, sw=2)
        c.circle(x + 32, 745, 9, colour)
        c.text(
            x + 53,
            754,
            f"{row['n']} · {proton} — {coupling}",
            24.0,
            PLEX,
            600,
            t["display"],
            "start",
            ident=f"legend-{row['n']}",
        )


def vanillin_spin_systems(t: dict[str, str]) -> str:
    """The molecule and its spectrum side by side, one colour per aromatic proton."""
    spec = _figure_numbers()["example_prediction_vanillin"]
    c = Canvas(
        1640,
        810,
        "Three aromatic protons → three spin systems",
        "Vanillin drawn from its structure with the three aromatic protons colour-coded by spin "
        "system, H2 a meta doublet, H5 an ortho doublet, H6 a doublet-of-doublets, each linked by "
        "colour to its multiplet in the 300 MHz 1H NMR spectrum beside it",
        faces=("sg700", "plex400", "plex600", "mono400"),
        ground=t["card"],
    )
    _masthead(c, t, "Three aromatic protons → three spin systems", 38.0, 48, 48, 104, "-0.37")
    _prediction_eyebrow(
        c, t, "¹H NMR · aromatic region · 300 MHz · DMSO-d", ("",), anchor_x=1587.16
    )
    _vanillin_molecule(c, t, spec["rows"])
    _spin_system_plot(c, t, ROOT / spec["npz"], spec["rows"])
    _spin_system_legend(c, t, spec["rows"])
    return c.done()


DIAGRAMS = {
    "banner": banner,
    "pipeline": pipeline,
    "architecture": architecture,
    "input_contract": input_contract,
    "coupling_rule": coupling_rule,
    "benchmark": benchmark,
    "mark": mark,
    "vanillin_spin_systems": vanillin_spin_systems,
    "example_prediction": example_prediction,
    "example_prediction_vanillin": example_prediction_vanillin,
}


def main() -> int:
    global TRACE

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the committed SVGs match this source; write nothing",
    )
    ap.add_argument(
        "--trace",
        choices=("faithful", "ideal", "hybrid"),
        default=DEFAULT_TRACE,
        help="how the banner draws its two spectra (default: %(default)s, which plots the NPZ)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="write elsewhere, e.g. to compare --trace variants without touching docs/img",
    )
    ap.add_argument("--only", help="build just this diagram")
    args = ap.parse_args()
    TRACE = args.trace
    # Both sides resolved: an 8.3-shortened or symlinked TEMP would otherwise make an --out-dir
    # that IS this directory compare unequal, and the run would exit 2 blaming the caller.
    if args.check and (args.trace != DEFAULT_TRACE or args.out_dir.resolve() != OUT_DIR.resolve()):
        # Otherwise --check would compare the committed bytes against a variant nobody committed
        # and report every banner stale, which reads as a real staleness failure.
        print("--check verifies the committed defaults; drop --trace/--out-dir", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {k: v for k, v in DIAGRAMS.items() if args.only in (None, k)}
    if not wanted:
        print(f"no such diagram: {args.only}; have {', '.join(DIAGRAMS)}", file=sys.stderr)
        return 2

    stale: list[str] = []
    for name, fn in wanted.items():
        for suffix, palette in (("", LIGHT), ("-dark", DARK)):
            dest = args.out_dir / f"{name}{suffix}.svg"
            svg = fn(palette)
            if args.check:
                current = dest.read_text(encoding="utf-8") if dest.is_file() else ""
                if current != svg:
                    stale.append(dest.name)
            else:
                dest.write_text(svg, encoding="utf-8")
                print(f"{dest}  {len(svg.encode()) / 1024:.1f} KB")

    if args.check:
        if stale:
            print(f"stale, re-run without --check: {', '.join(stale)}", file=sys.stderr)
            return 1
        print(f"{2 * len(wanted)} committed SVG(s) match this source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
