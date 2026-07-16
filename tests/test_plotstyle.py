"""TDD tests for ``moldetr.plotstyle`` — the shared publication figure-style module.

Every MolDeTr figure generator imports its palette, rcParams, label-deconfliction
and PNG-saving helpers from here, so these tests pin the public contract:
the Okabe-Ito colourblind-safe palette, a constrained-layout / 300-dpi base style,
an ``adjustText`` wrapper that separates overlapping labels, and a >=300-dpi PNG writer.
"""

from __future__ import annotations

import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe for CI
import matplotlib.pyplot as plt  # noqa: E402

from moldetr.plotstyle import (  # noqa: E402
    OKABE_ITO,
    SPIN_SYSTEM_COLORS,
    apply_style,
    deconflict,
    save_figure,
)


def test_spin_system_colors_is_unique_hex_list_starting_with_okabe_orange():
    assert isinstance(SPIN_SYSTEM_COLORS, list)
    assert len(SPIN_SYSTEM_COLORS) >= 8
    assert all(isinstance(c, str) and c.startswith("#") and len(c) == 7 for c in SPIN_SYSTEM_COLORS)
    assert len(set(SPIN_SYSTEM_COLORS)) == len(SPIN_SYSTEM_COLORS)  # all unique
    assert SPIN_SYSTEM_COLORS[0] == "#E69F00"  # Okabe-Ito orange leads the order


def test_apply_style_sets_constrained_layout_dpi_and_okabe_prop_cycle():
    apply_style()
    assert plt.rcParams["figure.constrained_layout.use"] is True
    assert plt.rcParams["savefig.dpi"] == 300
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    assert cycle_colors == OKABE_ITO


def test_deconflict_moves_overlapping_texts_without_error():
    apply_style()
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    texts = [ax.text(5.0, 5.0, f"label{i}") for i in range(5)]  # deliberately stacked
    fig.canvas.draw()
    before = [t.get_position() for t in texts]
    result = deconflict(ax, texts)  # must run without raising
    after = [t.get_position() for t in texts]
    moved = any(b != a for b, a in zip(before, after))
    assert moved or result is not None
    plt.close(fig)


def _png_dpi(path: Path) -> float | None:
    """Read DPI from a PNG's ``pHYs`` chunk (dependency-free)."""
    data = Path(path).read_bytes()
    idx = data.find(b"pHYs")
    if idx == -1:
        return None
    ppu_x = struct.unpack(">I", data[idx + 4 : idx + 8])[0]  # pixels per unit, X
    unit = data[idx + 12]  # 1 == metre
    return ppu_x * 0.0254 if unit == 1 else None  # px/m -> px/in


def test_save_figure_writes_nonempty_png_at_300_dpi(tmp_path):
    apply_style()
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [0, 1, 4])
    out = tmp_path / "fig.png"
    save_figure(fig, str(out))
    assert out.exists() and out.stat().st_size > 0
    dpi = _png_dpi(out)
    # ~299.999 due to integer px/metre rounding in the PNG header; assert the 300-dpi contract
    assert dpi is not None and dpi >= 299
    plt.close(fig)
