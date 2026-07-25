"""Brand invariants — the rules in `design/BRAND.md` that a code change can silently violate.

Two families, deliberately separated:

* **code ↔ code** (always run): the tricolor must be identical in both renderers, δ must never be
  uppercased into Δ, every marker must carry its number, and any blob mentioning `max J` must also
  name the full-coupling escape hatch. These need no external file and are CI-enforced.
* **`BRAND.md` ↔ code** (skipped when the file is absent): `design/` is gitignored today, so the
  source-of-truth sync test degrades to a skip on a fresh clone rather than a spurious failure.
  Committing `design/BRAND.md` turns it on everywhere — see REQUIREMENTS/`SYNC.md`.

Why a contract file at all: these invariants span modules, so no single unit test owns them. The
tricolor lives in two files that never import each other; the δ rule is a property of every header
string in the app; the `max J` caveat is the difference between a caption that is true and one that
overstates what the live decode returns.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from app_ui.plotting import MARKER_COLORS as GUI_TRICOLOR
from app_ui.plotting import assignment_rows, spectrum_figure
from moldetr.visualization import MARKER_COLORS as FIGURE_TRICOLOR

REPO = Path(__file__).resolve().parent.parent
BRAND_MD = REPO / "design" / "BRAND.md"

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
def test_every_max_j_mention_names_the_full_coupling_path(app_module, blob):
    """`max J` is the *largest* coupling, not the coupling set — a caption that says "coupling
    constants" without the caveat overstates what the live decode returns.

    BRAND.md prints canonical Short/Long wordings "verbatim", but the three blobs are written for
    different widths (chip, footnote, caption) and are not byte-identical to it. Byte-equality
    would therefore fail on harmless rephrasing while catching nothing extra; the invariant worth
    enforcing is the semantic pair: say "largest", and point at `structured_output`.
    """
    text = getattr(app_module, blob)
    assert "max J" in text
    assert re.search(r"largest|dominant", text), f"{blob} must qualify max J as largest/dominant"
    assert "structured_output" in text, f"{blob} must point at the full-coupling path"


# --- BRAND.md ↔ code -------------------------------------------------------------------------------

_needs_brand_md = pytest.mark.skipif(
    not BRAND_MD.exists(), reason="design/BRAND.md is gitignored and absent (see SYNC.md contract)"
)


@pytest.mark.unit
@_needs_brand_md
def test_brand_md_declares_a_design_version():
    """`DESIGN_VERSION` is the handle SYNC.md uses to detect handoff drift; without it the SoT
    cannot say whether the code is ahead of the brand or behind it.
    """
    assert re.search(r"DESIGN_VERSION:\s*v\d+", BRAND_MD.read_text(encoding="utf-8"))


@pytest.mark.unit
@_needs_brand_md
def test_tricolor_hexes_are_the_ones_brand_md_publishes():
    """Every hex the renderers use must appear in the BRAND.md palette table, tagged as a marker.

    Matching on the table row (hex *and* the word "marker") rather than a bare substring means a
    hex that survives only as, say, a hover colour will not satisfy the marker contract.
    """
    brand = BRAND_MD.read_text(encoding="utf-8")
    for hex_code in GUI_TRICOLOR:
        (row,) = [ln for ln in brand.splitlines() if f"`{hex_code}`" in ln]
        assert "marker" in row.lower(), f"{hex_code} is in BRAND.md but not as a marker colour"
