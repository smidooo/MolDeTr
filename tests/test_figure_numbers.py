"""What the prediction figures print must be what ``docs/figure_predictions.json`` says.

The always-on half of a two-part contract. This ties the committed SVG to the data file and needs
no checkpoint, so it runs in every CI lane. ``tests/test_scripts_local.py`` ties that same file to
the 974 MB weights and **skips wherever they are absent, which is every CI lane** -- so CI can prove
the figure agrees with the file, and only a local run can prove the file is still true.

Neither half existed before. The numbers lived solely as pixels inside two PNGs, and the one
assertion that touched them checked that shifts were 0-12 ppm and couplings 0-30 Hz -- a bar the
superseded vanillin figure cleared while transposing two of its three couplings, printing 8.2 / 2.0 /
8.7 where the checkpoint says 8.74 / 1.97 / 8.25.

Cells are matched by ``id``, not by scraping numbers out of the document. A figure that prints
``8.74`` somewhere is not the same claim as a figure that prints it in row 1's ``max J`` column, and
the defect this exists to catch is precisely a value landing in the wrong row.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NUMBERS = REPO / "docs" / "figure_predictions.json"
IMG = REPO / "docs" / "img"
_SVG_NS = "{http://www.w3.org/2000/svg}"

SPEC = json.loads(NUMBERS.read_text(encoding="utf-8"))

#: How each column is rendered, so the assertion compares what a reader sees rather than a float.
#: The figure prints `max J` to one decimal and line width to two; asserting raw equality against
#: the JSON's full precision would fail on a figure that is completely correct.
COLUMNS = {"shift_ppm": "{:.3f}", "max_j_hz": "{:.1f}", "linewidth_hz": "{:.2f}"}


def _cells(svg_path: Path) -> dict[str, str]:
    """Every identified text node in the figure, by ``id``.

    Includes ``<tspan>`` children's text, since a composed run (a subscript, an italic symbol) is
    several nodes under one id -- concatenated in document order, which is reading order.
    """
    # stdlib ET rather than defusedxml: the input is a repo-local file this repository's own
    # generator wrote, committed, and re-verified byte-for-byte by `--check` on every run. It
    # carries no DOCTYPE, so there are no entities to expand. Same reasoning as
    # `tests/test_diagram_fonts.py`, which walks the same files.
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in root.iter():
        ident = node.get("id")
        if not ident:
            continue
        out[ident] = "".join(node.itertext()).strip()
    return out


@pytest.mark.unit
@pytest.mark.parametrize("figure", sorted(SPEC["figures"]))
def test_the_figure_prints_the_committed_predictions(figure: str) -> None:
    """Row by row, column by column, against the id the generator stamps on each cell."""
    svg = IMG / f"{figure}.svg"
    assert svg.is_file(), (
        f"{svg.name} is missing. It is generated: `python scripts/build_diagram_svgs.py`."
    )
    cells = _cells(svg)

    wrong: list[str] = []
    for row in SPEC["figures"][figure]["rows"]:
        for field, fmt in COLUMNS.items():
            ident = f"cell-{row['n']}-{field}"
            want = fmt.format(row[field])
            got = cells.get(ident)
            if got != want:
                wrong.append(f"{ident}: figure shows {got!r}, the data says {want!r}")
        ident = f"cell-{row['n']}-protons"
        if (got := cells.get(ident)) != (want := f"{row['protons']} H"):
            wrong.append(f"{ident}: figure shows {got!r}, the data says {want!r}")

    assert not wrong, (
        f"{svg.name} disagrees with docs/figure_predictions.json:\n  "
        + "\n  ".join(wrong)
        + '\n\nA missing cell reads as None: the generator must stamp `id="cell-<n>-<field>"` on '
        "every table value, so this compares a named cell rather than hunting a number anywhere in "
        "the document -- the defect this catches is a value landing in the wrong row."
    )


@pytest.mark.unit
@pytest.mark.parametrize("figure", sorted(SPEC["figures"]))
def test_the_figure_plots_the_spectrum_its_data_file_names(figure: str) -> None:
    """The curve must come from the NPZ the JSON names, checked through the figure's own ticks.

    Same falsifiable shape as ``test_the_banner_traces_plot_the_committed_spectrum``: the tick
    labels are text the generator lays out, the curve is a polyline built from the array, and the
    shifts come from the NPZ. Inventing a curve, flipping the axis, or swapping the file each breaks
    the agreement between a different pair of them.
    """
    np = pytest.importorskip("numpy")

    svg = IMG / f"{figure}.svg"
    assert svg.is_file(), f"{svg.name} is missing; run scripts/build_diagram_svgs.py"
    body = svg.read_text(encoding="utf-8")

    trace = re.search(r'<path\b[^>]*\bid="trace"[^>]*\bd="([^"]+)"', body) or re.search(
        r'<path\b[^>]*\bd="([^"]+)"[^>]*\bid="trace"', body
    )
    assert trace, 'no <path id="trace"> in the figure; the curve must be identifiable to be checked'

    nums = [float(n) for n in re.findall(r"[-+]?\d*\.?\d+", trace.group(1))]
    points = list(zip(nums[::2], nums[1::2]))
    assert len(points) > 200, f"the trace has {len(points)} vertices; that is not a real spectrum"

    # allow_pickle for `ground_truth`, an object array of dicts. The path comes from a committed
    # JSON naming a committed NPZ, both in this repository -- no untrusted input reaches here.
    npz = np.load(REPO / SPEC["figures"][figure]["npz"], allow_pickle=True)
    shifts = sorted(float(g["chemical_shift_ppm"]) for g in npz["ground_truth"])

    ticks = sorted(
        (float(x), body[m.end() : body.index("</text>", m.end())].strip())
        for m in re.finditer(r'<text\b[^>]*\bid="tick-\d"[^>]*>', body)
        for x in [float(re.search(r'\bx="([-\d.]+)"', m.group(0)).group(1))]
    )
    assert len(ticks) >= 2, f"expected labelled ppm ticks, found {ticks}"

    (x_left, left), (x_right, right) = ticks[0], ticks[-1]
    assert float(left) > float(right), (
        f"the tick at x={x_left:.0f} reads {left} and the one at x={x_right:.0f} reads {right}. "
        "A 1H axis descends left to right; this one ascends."
    )

    per_px = (float(left) - float(right)) / (x_right - x_left)
    base = max(y for _, y in points)
    full = base - min(y for _, y in points)
    signal = [
        (float(left) - (x - x_left) * per_px, base - y) for x, y in points if base - y > full * 0.10
    ]
    assert signal, "no signal above 10% of the panel's full scale; the trace is a flat line"

    for shift in shifts:
        mine = [(p, w) for p, w in signal if min(shifts, key=lambda s: abs(p - s)) == shift]
        assert mine, (
            f"{svg.name} puts nothing near {shift} ppm, though the spectrum it claims to plot "
            f"records a multiplet there."
        )
        centre = sum(p * w for p, w in mine) / sum(w for _, w in mine)
        assert abs(centre - shift) < 0.03, (
            f"the multiplet nearest {shift} ppm has its centre of mass at {centre:.3f} ppm, read "
            f"off the figure's own tick labels. The curve and the spectrum it claims to plot "
            f"disagree, so at least one of them is not the data."
        )
