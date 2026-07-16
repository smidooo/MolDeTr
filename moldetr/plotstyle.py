"""Shared publication figure style — the single source of truth for every MolDeTr figure.

Every figure generator imports from here so the paper's figures read as one system:
the Okabe-Ito colourblind-safe palette, a constrained-layout / 300-dpi Matplotlib base
style, an ``adjustText`` wrapper for separating overlapping labels, and a PNG writer.

Palette and base rcParams are ported from the origin repo's ``scientific-visualization``
skill (``color_palettes.OKABE_ITO_LIST`` and ``style_presets.get_base_style``).

Importing this module does NOT force a Matplotlib backend: the Gradio app selects ``Agg``
itself, and forcing a backend here would fight it. We only import ``pyplot`` (backend
selection stays lazy).

Typical use::

    from moldetr.plotstyle import apply_style, save_figure, deconflict, SPIN_SYSTEM_COLORS

    apply_style()
    fig, ax = plt.subplots()
    ...                     # plot, using SPIN_SYSTEM_COLORS[k] per spin system
    deconflict(ax, labels)  # LAST, after all plotting + axis-limit changes
    save_figure(fig, "figure.png")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt

if TYPE_CHECKING:  # import only for type-checkers; keeps runtime import light
    from os import PathLike

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text

# --- Palette ----------------------------------------------------------------

# Okabe-Ito (2008) colourblind-safe palette, canonical order (matches the skill's
# OKABE_ITO_LIST). This is the Matplotlib prop_cycle order.
OKABE_ITO: list[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# Fixed order for colouring spin systems consistently across every figure.
# Same Okabe-Ito colours, reordered so the two weakest-on-white (black, then pure
# yellow) fall at the end — spin systems 1..N pick the strong, high-contrast hues first.
SPIN_SYSTEM_COLORS: list[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
    "#F0E442",  # yellow (low contrast on white — placed last)
]

# --- Semantic colours (shared with moldetr.visualization) -------------------

SPECTRUM_COLOR = "#333333"  # the 1D spectrum trace — near-black, thin
GRID_COLOR = "#e6e6e6"  # faint gridlines
TRUTH_COLOR = "#e8963a"  # ground-truth overlay — warm orange


def get_base_style() -> dict[str, Any]:
    """Publication-quality Matplotlib rcParams (ported from the skill's ``get_base_style``).

    Constrained layout on, Okabe-Ito prop_cycle, sans-serif (Arial -> DejaVu Sans),
    top/right spines off, 300-dpi saves, and sensible tick / label sizes.
    """
    return {
        # Figure
        "figure.dpi": 100,  # on-screen; saves use savefig.dpi
        "figure.facecolor": "white",
        "figure.autolayout": False,
        "figure.constrained_layout.use": True,
        # Font
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        # Axes
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.labelweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.axisbelow": True,
        "axes.grid": False,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        # Ticks
        "xtick.major.size": 3,
        "xtick.major.width": 0.8,
        "xtick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.major.size": 3,
        "ytick.major.width": 0.8,
        "ytick.labelsize": 8,
        "ytick.direction": "out",
        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.5,
        # Legend
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.loc": "best",
        # Savefig
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        # Image
        "image.cmap": "viridis",
    }


def apply_style() -> None:
    """Apply the shared publication style to the global Matplotlib rcParams.

    Call once before building a figure. Silent by design (no stdout) so it is safe
    to call from libraries, the Gradio app, and batch scripts.
    """
    plt.rcParams.update(get_base_style())


def deconflict(
    ax: Axes,
    texts: list[Text],
    arrowprops: dict[str, Any] | None = None,
    **kw: Any,
) -> object | None:
    """Separate overlapping label ``texts`` on ``ax`` via ``adjustText.adjust_text``.

    Call this LAST — after all plotting *and* every axis-limit change — because
    ``adjustText`` positions labels using the final data-to-display transform; later
    ``set_xlim`` / ``set_ylim`` calls would invalidate the layout.

    ``arrowprops`` (e.g. ``dict(arrowstyle="->", color="gray")``) draws leader lines
    from each moved label back to its anchor. Extra keyword args pass straight through
    to ``adjust_text`` (``force_text``, ``expand``, ``only_move``, ...).

    Returns whatever ``adjust_text`` returns — in adjustText >= 1.0 a truthy
    ``(texts, arrows)`` tuple — or ``None`` when ``texts`` is empty. ``adjustText`` is
    imported lazily so the rest of this module works without the ``figures`` extra.
    """
    if not texts:
        return None
    try:
        from adjustText import adjust_text
    except ImportError:
        return None  # last-resort guard: labels stay put rather than crashing the caller

    if arrowprops is not None:
        kw["arrowprops"] = arrowprops
    return adjust_text(texts, ax=ax, **kw)


def save_figure(fig: Figure, path: str | PathLike[str], dpi: int = 300) -> None:
    """Write ``fig`` to ``path`` as a tight, white-background raster at ``dpi`` (>=300)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
