#!/usr/bin/env python
"""Regenerate ``docs/img/vanillin_spin_systems.png`` — vanillin + its 300 MHz ¹H spectrum.

The molecule (drawn from SMILES with RDKit's ACS-1996 publication style) sits beside its own
spectrum; the three aromatic CH protons — H2 (meta doublet), H5 (ortho doublet), H6 (doublet of
doublets) — are colour-coded by spin system, each linked by colour to its multiplet in the
spectrum. ``PROTON_COLORS`` / ``ASSIGNMENTS`` / ``render_molecule`` are re-used by
``gen_banner.py`` so the banner and this figure share one identical colour mapping.

Run:  python scripts/gen_molecule_figure.py
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; set before pyplot loads
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402

from moldetr import plotstyle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "examples" / "roi_S8_example.npz"
OUT = ROOT / "docs" / "img" / "vanillin_spin_systems.png"

# Vanillin = 4-hydroxy-3-METHOXYbenzaldehyde: one O-CH3 (methoxy), NOT ethyl vanillin (O-CH2-CH3).
SMILES = "O=Cc1ccc(O)c(OC)c1"

# Each aromatic CH proton -> a fixed Okabe-Ito spin-system colour. gen_banner.py imports this dict
# unchanged so the molecule figure and the banner use identical colours for the same protons.
PROTON_COLORS = {
    "H2": plotstyle.SPIN_SYSTEM_COLORS[0],  # orange
    "H6": plotstyle.SPIN_SYSTEM_COLORS[1],  # sky blue
    "H5": plotstyle.SPIN_SYSTEM_COLORS[2],  # bluish green
}
# proton -> (aromatic carbon index in SMILES, chemical shift δ/ppm, multiplicity label).
# Carbon indices verified against the RDKit atom graph: C3 ortho to CHO (H6), C4 ortho to OH (H5),
# C10 between OCH3 and CHO (H2). δ values match the roi_S8 ground truth (6.959 / 7.420 / 7.385).
# Coupling labels use the paper's authoritative ground truth: ortho J = 8.1 Hz, meta J = 2.0 Hz.
ASSIGNMENTS: dict[str, tuple[int, float, str]] = {
    "H6": (3, 7.420, "dd, J = 8.1, 2.0 Hz"),
    "H2": (10, 7.385, "d, J = 2.0 Hz"),
    "H5": (4, 6.959, "d, J = 8.1 Hz"),
}


def _trim(im: Image.Image, margin: int = 12) -> Image.Image:
    """Crop the surrounding white border so the molecule fills its axes (kills dead space)."""
    rgb = im.convert("RGB")
    bbox = ImageChops.difference(rgb, Image.new("RGB", rgb.size, (255, 255, 255))).getbbox()
    if not bbox:
        return im
    left, top, right, bottom = bbox
    return im.crop(
        (
            max(0, left - margin),
            max(0, top - margin),
            min(im.width, right + margin),
            min(im.height, bottom + margin),
        )
    )


def render_molecule(width: int = 1400, height: int = 1200) -> np.ndarray:
    """Draw vanillin in ACS-1996 style with the three aromatic CH carbons colour-highlighted."""
    mol = Chem.MolFromSmiles(SMILES)
    AllChem.Compute2DCoords(mol)
    for proton, (cidx, _, _) in ASSIGNMENTS.items():
        mol.GetAtomWithIdx(cidx).SetProp("atomNote", proton)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    rdMolDraw2D.SetACS1996Mode(opts, rdMolDraw2D.MeanBondLength(mol))
    # ACS mode renders at a fixed ~9.6-px bond length — a tiny raster that blurs when upscaled.
    # Scale bond length + fonts + halo together so the drawing is crisp at the figure's display
    # size while keeping ACS proportions (bond widths, double-bond spacing, label padding).
    opts.fixedBondLength = 110.0
    opts.fixedFontSize = 52
    opts.bondLineWidth = 2.0
    opts.highlightRadius = 0.20
    opts.annotationFontScale = 0.7  # readable H2/H5/H6 atom notes
    atoms = [cidx for cidx, _, _ in ASSIGNMENTS.values()]
    colors = {cidx: mcolors.to_rgb(PROTON_COLORS[p]) for p, (cidx, _, _) in ASSIGNMENTS.items()}
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=atoms, highlightAtomColors=colors
    )
    drawer.FinishDrawing()
    return np.array(_trim(Image.open(io.BytesIO(drawer.GetDrawingText()))))


def _apex(amp: np.ndarray, xi: int, half: int = 18) -> float:
    lo, hi = max(0, xi - half), min(amp.shape[-1], xi + half + 1)
    return float(amp[lo:hi].max()) if hi > lo else float(amp.max())


def plot_spectrum_panel(ax, amp: np.ndarray, ppm_axis: np.ndarray) -> None:
    """Plot the normalised vanillin spectrum with one colour-coded marker per aromatic proton."""
    amp = amp / (float(np.max(np.abs(amp))) or 1.0)  # normalise -> clean 'a.u.' axis, no 1e6 offset
    span = float(amp.max() - amp.min()) or 1.0
    ax.plot(ppm_axis, amp, lw=1.0, color=plotstyle.SPECTRUM_COLOR, zorder=2)
    labels, xs = [], []
    for proton, (_, delta, mult) in ASSIGNMENTS.items():
        y_apex = _apex(amp, int(np.argmin(np.abs(ppm_axis - delta))))
        y_lab = y_apex + 0.13 * span
        color = PROTON_COLORS[proton]
        ax.plot(delta, y_apex, "o", ms=4.5, color=color, zorder=4)
        ax.plot([delta, delta], [y_apex, y_lab], color=color, lw=0.8, alpha=0.7, zorder=3)
        labels.append(
            ax.text(
                delta,
                y_lab,
                f"{proton}\n{mult}",
                color=color,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                zorder=6,
            )
        )
        xs.append(delta)
    ax.set_xlabel("Chemical shift δ (ppm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_ylim(amp.min() - 0.05 * span, amp.max() + 0.46 * span)
    ax.grid(axis="y", color=plotstyle.GRID_COLOR, lw=0.8)
    pad = max(0.24 * (max(xs) - min(xs)), 0.10)  # room for the (centred) multiplicity labels
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.invert_xaxis()  # NMR convention
    plotstyle.deconflict(ax, labels, arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6))


def main() -> None:
    data = np.load(NPZ, allow_pickle=True)  # trusted first-party example (object-array metadata)
    amp = np.real(data["spectrum_padded"]).astype(float)
    ppm_axis = np.asarray(data["ppm_axis_padded"], dtype=float)

    plotstyle.apply_style()
    fig = plt.figure(figsize=(12, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.85])
    ax_mol = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_mol.imshow(render_molecule())
    ax_mol.axis("off")
    ax_mol.set_anchor("C")
    plot_spectrum_panel(ax_spec, amp, ppm_axis)
    fig.suptitle(
        "Vanillin — three aromatic protons, three spin systems (300 MHz)",
        fontsize=13,
        fontweight="bold",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plotstyle.save_figure(fig, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
