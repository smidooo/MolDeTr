"""Brand invariants — the rules in `docs/BRAND.md` that a code change can silently violate.

Two families, deliberately separated:

* **code ↔ code**: the tricolor must be identical in both renderers, δ must never be uppercased
  into Δ, every marker must carry its number, and any blob mentioning `max J` must say what the live
  decode actually returns rather than promising a full coupling set. These need no external file.
* **`BRAND.md` ↔ code**: the palette and the declared version in `docs/BRAND.md` must match what
  the renderers actually use. Both families run unconditionally — `docs/BRAND.md` is committed, so
  there is no absent-file case to guard, and a brand test that can silently skip itself reports
  green while enforcing nothing.

Why a contract file at all: these invariants span modules, so no single unit test owns them. The
tricolor lives in two files that never import each other; the δ rule is a property of every header
string in the app; the `max J` caveat is the difference between a caption that is true and one that
overstates what the live decode returns.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from app_ui.plotting import MARKER_COLORS as GUI_TRICOLOR
from app_ui.plotting import assignment_rows, spectrum_figure
from moldetr.visualization import MARKER_COLORS as FIGURE_TRICOLOR

REPO = Path(__file__).resolve().parent.parent
BRAND_MD = REPO / "docs" / "BRAND.md"

BLUE, ORANGE, TEAL = "#2566b0", "#e08a1f", "#1f9e8c"


# --- code ↔ code -----------------------------------------------------------------------------------


@pytest.mark.unit
def test_tricolor_is_identical_in_both_renderers():
    """`app_ui/plotting.py` (GUI) and `moldetr/visualization.py` (PNG export) never import each
    other, so the palette is duplicated — the one arrangement where drift is invisible until a
    reader compares a screenshot with a figure in the paper.
    """
    assert GUI_TRICOLOR == FIGURE_TRICOLOR == [BLUE, ORANGE, TEAL]


@pytest.mark.unit
def test_categorical_colours_stay_capped_at_three():
    """BRAND.md caps categorical hues at ≤ 3 (well under the 6–8 CVD-safe maximum)."""
    assert len(GUI_TRICOLOR) <= 3


@pytest.mark.unit
@pytest.mark.parametrize("ppm", [True, False])
def test_shift_header_keeps_a_lowercase_delta(ppm):
    """`"δ".upper()` is `"Δ"`, which reads as *difference* in NMR — the shift column must never
    acquire it. This is the assertion that would fail if anyone reached for `.upper()` or a CSS
    `text-transform` on the assignment-table headers.
    """
    (row,) = assignment_rows(
        [{"proton_count": 1, "chemical_shift_ppm": 7.5, "chemical_shift_hz": 384.0}], ppm=ppm
    )
    # Collect every header carrying either character, so an uppercased δ cannot slip through as a
    # *different* key that a "δ in header" check would simply stop finding.
    assert [k for k in row if "δ" in k or "Δ" in k] == ["δ (PPM)" if ppm else "δ (HZ)"]


@pytest.mark.unit
def test_comparison_table_uses_capital_delta_only_for_differences(app_module):
    """The δ≠Δ rule forbids *uppercasing* δ — it does not forbid Δ where Δ genuinely means
    "difference". The comparison table needs both in one header (`Δδ (Hz)` = a difference of
    chemical shifts), so this pins the distinction rather than banning the character outright.
    """
    df = app_module._comparison_dataframe(
        [{"shift_ppm": 7.5, "proton_count": 1, "max_j_hz": 8.0}],
        [{"chemical_shift_ppm": 7.5, "proton_count": 1, "confidence": 0.9}],
    )
    assert "Δδ (Hz)" in df.columns  # Δ = difference, δ still lowercase
    assert "GT δ (ppm)" in df.columns and "GT Δ (ppm)" not in df.columns


@pytest.mark.unit
@pytest.mark.parametrize("n_detections", [1, 3, 5])
def test_every_marker_carries_its_number_not_just_a_colour(n_detections):
    """BRAND.md: "colour is never the only channel" — the figure must survive greyscale and every
    CVD type. Enforced as: the marker trace renders text, and that text is the 1-based row index,
    at any detection count (including past the 3-colour cycle, where hue alone starts repeating).
    """
    amp = np.abs(np.random.RandomState(0).rand(6144))
    preds = [
        {"proton_count": 1, "chemical_shift_in_points": float(800 * i)}
        for i in range(1, n_detections + 1)
    ]
    fig = spectrum_figure(amp, preds, ppm_left=10.0, ppm_right=0.0, points_per_hz=5.12)

    (markers,) = [t for t in fig.data if t.mode and "text" in t.mode]
    assert list(markers.text) == [str(i) for i in range(1, n_detections + 1)]


@pytest.mark.unit
@pytest.mark.parametrize("blob", ["SCOPE_NOTE", "FOOTNOTE", "OUTPUT_CAPTION"])
def test_every_max_j_mention_states_what_the_live_decode_returns(app_module, blob):
    """`max J` is the *largest* coupling, not the coupling set — a caption that says "coupling
    constants" without the caveat overstates what the live decode returns.

    The earlier form of this test required each blob to contain the literal token
    `structured_output`, on the reasoning that a reader should be pointed at the escape hatch. That
    inverted the truth: `structured_output/` holds the paper's committed 13-ROI benchmark, the app
    never writes to it, and the live decode cannot produce a full coupling set at all
    (`moldetr/postprocess.py` emits only the `max` component of the `[sum, min, max, std]`
    embedding). A user who followed the pointer asked where their couplings were, and the honest
    answer was nowhere. The enforced invariant is therefore the negative: a GUI caption must not
    send a user to a folder the app does not populate.
    """
    text = getattr(app_module, blob)
    assert "max J" in text
    assert re.search(r"largest|dominant", text), f"{blob} must qualify max J as largest/dominant"
    assert "structured_output" not in text, (
        f"{blob} names structured_output, which holds the paper's benchmark, not app output"
    )


DOC_SURFACES = (
    REPO / "README.md",
    REPO / "app.py",
    REPO / "docs" / "BRAND.md",
    REPO / "docs" / "SCOPE.md",
    REPO / "docs" / "USAGE_NOTES.md",
    REPO / "examples" / "README.md",
    REPO / "deploy" / "hf_space" / "README.md",
    REPO / "notebooks" / "MolDeTr_quickstart.ipynb",
)


FULL_SET = re.compile(r"full\s+(?:coupling\s+)?sets?\b", re.IGNORECASE)
BENCHMARK_QUALIFIER = re.compile(r"benchmark", re.IGNORECASE)
DENIAL = re.compile(r"\bnot\b|\bno\b|\bnever\b", re.IGNORECASE)


def _normalize(text: str) -> str:
    """Drop inline emphasis markers so `the *full* coupling set` matches like plain text.

    `README.md:366` writes it exactly that way, and a plain scan passed straight over it. That is
    the fourth form this guard has been too narrow to see.
    """
    return text.replace("*", "").replace("_", "").replace("`", "")


def _scannable_units(path: Path) -> list[tuple[str, str, bool]]:
    """`(label, text, lines_are_meaningful)` triples of normalized text.

    A notebook is scanned cell by cell: its raw JSON carries no paragraph structure, so one escaped
    blob would make the paragraph check whole-file and vacuous.
    """
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return [
            (
                f"{path.relative_to(REPO)} cell {index}",
                _normalize("".join(cell.get("source", []))),
                False,
            )
            for index, cell in enumerate(notebook.get("cells", []))
            if cell.get("cell_type") == "markdown"
        ]
    # Only the scanned text is normalized; the label keeps its real filename.
    return [(str(path.relative_to(REPO)), _normalize(path.read_text(encoding="utf-8")), True)]


def _paragraph_containing(text: str, start: int, end: int) -> str:
    left = text.rfind("\n\n", 0, start)
    right = text.find("\n\n", end)
    return text[(0 if left == -1 else left) : (len(text) if right == -1 else right)]


def _is_denied(text: str, start: int, end: int) -> bool:
    """A mention reads as a denial when a negation sits just before or after it, as in
    "not the full set" or "the full set is not available". Both are true statements; the
    unqualified promise is the thing that misled a reader.
    """
    before = text[max(0, start - 20) : start]
    after = text[end : end + 30]
    return bool(DENIAL.search(before) or DENIAL.search(after))


@pytest.mark.unit
def test_no_doc_mentions_the_full_set_without_the_benchmark_qualifier():
    """Three earlier versions of this guard were each narrower than the claim, which is the defect
    the file exists to catch.

    1. A line-by-line scan missed `examples/README.md`, whose claim straddles a line break.
    2. A blacklist holding "is in" missed the blobs saying the full set "comes from" the folder.
    3. That blacklist still missed the copula-free and plural forms: "For the full coupling set, use
       the exact `structured_output/` path", and "the full sets stay in `structured_output/`".

    A blacklist over natural language cannot be complete, so this asserts the invariant rather than
    banning phrasings: a paragraph mentioning the full coupling set must either carry the benchmark
    qualifier or deny availability. The full set exists only for the paper's committed 13-ROI
    benchmark, so both readings are true and the bare promise is not.
    """
    offenders = []
    for path in DOC_SURFACES:
        for label, text, has_lines in _scannable_units(path):
            for mention in FULL_SET.finditer(text):
                if _is_denied(text, mention.start(), mention.end()):
                    continue
                paragraph = _paragraph_containing(text, mention.start(), mention.end())
                if BENCHMARK_QUALIFIER.search(paragraph):
                    continue
                line = text.count("\n", 0, mention.start()) + 1
                location = f"{label}:{line}" if has_lines else label
                offenders.append(
                    f"  {location}: ...{re.sub(r'[ ]+', ' ', mention.group(0))[:70]}..."
                )
    assert not offenders, (
        "these mentions of the full coupling set carry no benchmark qualifier and are not denials,\n"
        "so they promise couplings the live decode never returns:\n" + "\n".join(offenders)
    )


# --- BRAND.md ↔ code -------------------------------------------------------------------------------


@pytest.mark.unit
def test_brand_md_declares_a_design_version():
    """`DESIGN_VERSION` is the handle that detects drift between the brand source of truth and the
    code; without it nothing can say whether the code is ahead of the brand or behind it.
    """
    assert re.search(r"DESIGN_VERSION:\s*v\d+", BRAND_MD.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_tricolor_hexes_are_the_ones_brand_md_publishes():
    """Every hex the renderers use must appear in the BRAND.md palette table, tagged as a marker.

    Matching on the table row (hex *and* the word "marker") rather than a bare substring means a
    hex that survives only as, say, a hover colour will not satisfy the marker contract.
    """
    brand = BRAND_MD.read_text(encoding="utf-8")
    for hex_code in GUI_TRICOLOR:
        (row,) = [ln for ln in brand.splitlines() if f"`{hex_code}`" in ln]
        assert "marker" in row.lower(), f"{hex_code} is in BRAND.md but not as a marker colour"
