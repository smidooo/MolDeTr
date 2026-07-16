#!/usr/bin/env python
"""Regenerate ``docs/banner.png`` — the MolDeTr hero image.

Composes one wide banner: the brand + tagline, the vanillin molecule and its 300 MHz ¹H spectrum
(the three aromatic protons colour-coded by spin system), and a benchmark footer with the paper's
experimental medians. The molecule, ``PROTON_COLORS`` and ``ASSIGNMENTS`` are imported from
``gen_molecule_figure`` so the banner and the molecule figure use one identical colour mapping.

Run:  python scripts/gen_banner.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; set before pyplot loads
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # import the sibling script when run directly
from gen_molecule_figure import ASSIGNMENTS, PROTON_COLORS, plot_spectrum_panel, render_molecule  # noqa: E402

from moldetr import plotstyle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "examples" / "roi_S8_example.npz"
OUT = ROOT / "docs" / "banner.png"

# ¹H via mathtext so the superscript renders on any font (never the literal "'H").
SUBTITLE = "chemistry-informed deep learning for $\\mathregular{^{1}}$H NMR multiplet detection"
OUTPUTS = "one forward pass  →  chemical shift δ · coupling J · proton count · line width (Hz)"
# Medians labelled exactly as the README / tables (aggregate_experimental.py output).
BENCHMARK = (
    "Experimental benchmark (13 ROIs):     "
    "|Δδ| 0.90 Hz     ·     |ΔJ| 0.20 Hz     ·     proton-count 93.5 %"
)


def _ensure_under(path: Path, limit: int = 300_000) -> None:
    """Keep the banner PNG under ``limit`` bytes (palette-quantise only if it overshoots)."""
    if path.stat().st_size <= limit:
        return
    Image.open(path).convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256).save(
        path, optimize=True
    )


def build_banner() -> "plt.Figure":
    """Assemble the hero banner figure (brand · molecule · spectrum · benchmark footer)."""
    data = np.load(NPZ, allow_pickle=True)  # trusted first-party example (object-array metadata)
    amp = np.real(data["spectrum_padded"]).astype(float)
    ppm_axis = np.asarray(data["ppm_axis_padded"], dtype=float)

    plotstyle.apply_style()
    plt.rcParams["figure.constrained_layout.use"] = False  # banner uses manual add_axes placement
    fig = plt.figure(figsize=(13, 5.0))
    fig.patch.set_facecolor("white")

    fig.text(0.018, 0.955, "MolDeTr", fontsize=34, fontweight="bold", color="#1a1a1a", va="top")
    fig.text(0.020, 0.845, SUBTITLE, fontsize=13.5, color="#555555", va="top")
    fig.text(0.020, 0.775, OUTPUTS, fontsize=10.5, color="#777777", va="top")

    ax_mol = fig.add_axes((0.012, 0.20, 0.25, 0.50))
    ax_mol.imshow(render_molecule())
    ax_mol.axis("off")

    ax_spec = fig.add_axes((0.335, 0.24, 0.645, 0.48))
    plot_spectrum_panel(ax_spec, amp, ppm_axis)
    ax_spec.set_ylabel("")  # cleaner hero: intensity is arbitrary, drop the y-axis
    ax_spec.set_yticks([])
    ax_spec.set_xlabel("δ (ppm)")
    ax_spec.spines["left"].set_visible(False)

    fig.text(0.5, 0.065, BENCHMARK, ha="center", va="center", fontsize=13, color="#333333")
    return fig


def main() -> None:
    fig = build_banner()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plotstyle.save_figure(fig, OUT)
    _ensure_under(OUT)
    print("wrote", OUT, f"({OUT.stat().st_size / 1024:.0f} KB)")
    # Colours are re-used from gen_molecule_figure, so banner and molecule figure always agree.
    print("proton colours (shared with the molecule figure):", dict(PROTON_COLORS))
    print("assignments:", dict(ASSIGNMENTS))


if __name__ == "__main__":
    main()
