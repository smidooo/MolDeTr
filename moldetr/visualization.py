"""Annotated-spectrum plot, shared by predict.py, the evaluators, and the GUI.

Branded renderer with the earlier version's publication quality merged back in — same public
API (``plot_spectrum`` returns ``(figure, rows)``). Brand bits are marked ``# BRAND``:
- markers cycle the paper tricolor (blue / orange / teal) so plot numbers colour-link to the
  assignment-table rows
- ground truth is a dashed slate overlay (orange belongs to marker 2)
- grid / spine / title colours from the paper palette; IBM Plex Sans when installed
- table headers keep a lowercase ``δ`` — never uppercase δ programmatically
  ("δ".upper() == "Δ", which means *difference* in NMR)

Quality restored from the earlier renderer (all via :mod:`moldetr.plotstyle`, so no styling is
duplicated):
- 300-dpi PNGs (:func:`moldetr.plotstyle.save_figure`)
- the plotted intensity is peak-normalised — clean ``a.u.`` axis, no ``1eN`` offset
- the x-axis is cropped to the detected-signal region (no dead margins)
- overlapping numbers are de-conflicted with ``adjustText`` (:func:`moldetr.plotstyle.deconflict`,
  which imports ``adjustText`` lazily and is a no-op when it is not installed)

Only ``plotstyle.save_figure`` / ``plotstyle.deconflict`` are reused; ``apply_style`` is NOT
called here, so the brand palette + IBM Plex font below stay authoritative.

The static PNG renderer for ``predict.py --plot``; the interactive GUI spectrum lives in
``plotting.py`` (Plotly). ``plot_spectrum`` keeps the original ``(figure, rows)`` contract so
``predict.py``, the "Simulate" tab, and ``tests/test_visualization.py`` are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # headless-safe (CI, servers, Hugging Face Spaces); set before pyplot loads
from matplotlib import font_manager  # noqa: E402  # BRAND

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from moldetr import plotstyle  # noqa: E402  # save_figure + deconflict; no rcParams side effects

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text

SPECTRUM = "#20242b"  # BRAND: the 1D trace — ink, thin
MARKER_COLORS = ["#2566b0", "#e08a1f", "#1f9e8c"]  # BRAND: multiplet 1/2/3 cycle (paper tricolor)
ACCENT = MARKER_COLORS[0]  # BRAND: legend swatch for "detected"
TRUTH = "#5b6675"  # BRAND: ground truth — dashed slate
GRID = "#edf1f7"  # BRAND: faint gridlines / table borders
SPINE = "#9db0c6"  # BRAND: axis spines
NAVY = "#1f3a5f"  # BRAND: title + table header
MUTE = "#5b6675"  # BRAND: axis labels / ticks / leader lines

_available = {f.name for f in font_manager.fontManager.ttflist}  # BRAND
_BRAND_FONT = next(
    (f for f in ("IBM Plex Sans", "Source Sans 3") if f in _available), "DejaVu Sans"
)
plt.rcParams["font.family"] = _BRAND_FONT  # BRAND


def _shift_to_x(pts: float, n: int, ppm_left: float | None, ppm_right: float | None) -> float:
    """Map a point index to the x-axis (ppm if calibrated, else the point index)."""
    if ppm_left is not None and ppm_right is not None and n > 1:
        return ppm_left + (pts / (n - 1)) * (ppm_right - ppm_left)
    return pts


def _apex(amp: np.ndarray, xi: int, half: int = 18) -> float:
    """Peak height near point ``xi`` (a small window handles the exact-index jitter)."""
    lo, hi = max(0, xi - half), min(amp.shape[-1], xi + half + 1)
    return float(amp[lo:hi].max()) if hi > lo else float(amp.max())


def _rows(predictions: list[dict[str, Any]], ppm: bool) -> list[dict[str, Any]]:
    """Assignment table as a list of dicts (keys double as column headers)."""
    # BRAND: keep δ lowercase — do NOT .upper() these headers ("δ".upper() == "Δ" = difference).
    shift_col = "δ (ppm)" if ppm else "δ (Hz)"
    out = []
    for i, p in enumerate(predictions, 1):
        js = p.get("coupling_constants_hz") or []
        shift = p.get("chemical_shift_ppm") if ppm else p.get("chemical_shift_hz")
        lw = p.get("linewidth_hz")
        if shift is None:
            shift_str = "–"
        else:
            shift_str = f"{shift:.3f}" if ppm else f"{shift:.1f}"
        out.append(
            {
                "#": i,
                "protons": f"{p['proton_count']} H",
                shift_col: shift_str,
                "max J (Hz)": f"{js[0]:.1f}" if js else "–",
                "line width (Hz)": f"{lw:.2f}" if lw is not None else "–",
            }
        )
    return out


def _draw_table(ax: Axes, rows: list[dict[str, Any]]) -> None:
    """Render the assignment table into its own (borderless) axes."""
    ax.axis("off")
    headers = list(rows[0].keys())
    cells = [[str(r[h]) for h in headers] for r in rows]
    tbl = ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    for (row, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e6ebf2")  # BRAND
        cell.set_linewidth(0.8)
        if row == 0:  # header
            cell.set_facecolor(NAVY)  # BRAND
            cell.set_text_props(color="white", fontweight="bold")


def _peak_positions(
    predictions: list[dict[str, Any]], n: int, ppm_left: float | None, ppm_right: float | None
) -> list[float]:
    """x-axis positions of every detected peak (ppm or point index)."""
    return [
        _shift_to_x(p.get("chemical_shift_in_points", 0.0), n, ppm_left, ppm_right)
        for p in predictions
    ]


def _draw_markers(
    ax: Axes,
    predictions: list[dict[str, Any]],
    xs: list[float],
    amp: np.ndarray,
    span: float,
) -> list[Text]:
    """Apex dot + stem + numbered tricolor circle per prediction (colour 1/2/3 links to the table).

    Returns the number ``Text`` objects so the caller can de-conflict them. The number is drawn as
    a bbox circle *on the text* (BRAND tricolor) rather than a separate plotted marker, so it stays
    a single movable object — ``adjustText`` then moves circle + number together and adds a leader
    line, instead of stranding the number away from its circle.
    """
    texts: list[Text] = []
    for i, (p, cx) in enumerate(zip(predictions, xs), 1):
        color = MARKER_COLORS[(i - 1) % len(MARKER_COLORS)]  # BRAND
        y_apex = _apex(amp, int(round(p.get("chemical_shift_in_points", 0.0))))
        y_label = y_apex + 0.10 * span
        ax.plot(cx, y_apex, "o", markersize=3.5, color=color, zorder=4)  # BRAND: apex anchor
        ax.plot([cx, cx], [y_apex, y_label], color=color, linewidth=0.8, alpha=0.7, zorder=3)
        texts.append(
            ax.text(
                cx,
                y_label,
                str(i),
                color="white",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                zorder=5,
                bbox=dict(  # BRAND: tricolor circle, part of the text so deconflict keeps it attached
                    boxstyle="circle,pad=0.35", facecolor=color, edgecolor="white", lw=0.8
                ),
            )
        )
    return texts


def _draw_ground_truth(
    ax: Axes,
    ground_truth: list[dict[str, Any]] | None,
    n: int,
    ppm_left: float | None,
    ppm_right: float | None,
) -> None:
    """Overlay ground-truth positions as dashed slate vertical reference lines."""
    for j, g in enumerate(ground_truth or []):
        pts = g.get("chemical_shift_in_points", g.get("center_in_points", 0.0))
        gx = _shift_to_x(pts, n, ppm_left, ppm_right)
        ax.axvline(
            gx,
            color=TRUTH,
            linestyle="--",
            linewidth=1.1,
            alpha=0.7,
            zorder=1,  # BRAND: dashed slate
            label="Ground truth" if j == 0 else None,
        )


def _crop_x(ax: Axes, xs: list[float], x: np.ndarray) -> None:
    """Crop the x-axis to the detected-signal region (+~13 % pad) to kill dead margins."""
    if not xs:
        return
    lo, hi = min(xs), max(xs)
    x0, x1 = float(x[0]), float(x[-1])
    full = abs(x1 - x0) or 1.0
    pad = max(0.13 * (hi - lo), 0.04 * full)
    xlo = max(lo - pad, min(x0, x1))
    xhi = min(hi + pad, max(x0, x1))
    if xhi > xlo:
        ax.set_xlim(xlo, xhi)


def plot_spectrum(
    amplitudes: Any,
    predictions: list[dict[str, Any]],
    ppm_left: float | None = None,
    ppm_right: float | None = None,
    ground_truth: list[dict[str, Any]] | None = None,
    title: str = "MolDeTr — detected multiplets",
    save_path: str | Path | None = None,
    show_table: bool = True,
) -> tuple[Figure, list[dict[str, Any]]]:
    """Plot a 1D spectrum with numbered multiplet markers and an assignment table.

    Same contract as the original: returns ``(figure, rows)`` and writes a PNG when ``save_path`` is
    given. Markers cycle the paper tricolor so the numbers on the plot colour-link to the
    assignment-table rows. The plotted intensity is peak-normalised (clean ``a.u.`` axis, no ``1eN``
    offset), the x-axis is cropped to the signal region, overlapping numbers are de-conflicted with
    ``adjustText``, and the PNG is written at 300 dpi. ``show_table=False`` draws only the spectrum.
    """
    amp = np.real(np.asarray(amplitudes, dtype=float)).ravel()
    n = amp.shape[-1]
    peak = float(np.max(np.abs(amp))) if amp.size else 1.0
    amp = amp / (peak or 1.0)  # normalise -> clean 'a.u.' axis, no scientific-notation offset
    ppm = ppm_left is not None and ppm_right is not None
    if ppm_left is not None and ppm_right is not None:  # narrowed for the type-checker
        x = np.linspace(ppm_left, ppm_right, n)
    else:
        x = np.arange(n, dtype=float)  # float64 keeps both branches' dtype consistent
    base = float(amp.min()) if amp.size else 0.0
    top = float(amp.max()) if amp.size else 1.0
    span = (top - base) or 1.0
    rows = _rows(predictions, ppm)

    if show_table and rows:
        fig, (ax, ax_t) = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax_t = None

    ax.plot(x, amp, linewidth=1.0, color=SPECTRUM, zorder=2)  # BRAND: ink trace
    xs = _peak_positions(predictions, n, ppm_left, ppm_right)
    number_texts = _draw_markers(ax, predictions, xs, amp, span)
    _draw_ground_truth(ax, ground_truth, n, ppm_left, ppm_right)

    ax.set_xlabel(
        "Chemical shift δ (ppm)" if ppm else "Point index", fontsize=11, color=MUTE
    )  # BRAND
    ax.set_ylabel("Intensity (a.u.)", fontsize=11, color=MUTE)  # BRAND
    ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY)  # BRAND
    ax.set_ylim(base - 0.05 * span, top + 0.28 * span)  # headroom for de-conflicted numbers
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):  # BRAND
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=MUTE)  # BRAND
    _crop_x(ax, xs, x)  # kill dead x-margins before the invert below
    if ppm:
        ax.invert_xaxis()  # NMR convention: ppm decreases left to right
    if ground_truth:
        ax.plot([], [], "o", color=ACCENT, label="Detected (numbered)")
        ax.legend(loc="upper left", fontsize=9, frameon=False)

    if ax_t is not None:
        _draw_table(ax_t, rows)
    fig.tight_layout()
    # De-conflict LAST — after every plot call, tight_layout, and axis-limit change (crop + invert).
    # plotstyle.deconflict imports adjustText lazily and no-ops when it (or the text list) is absent.
    plotstyle.deconflict(ax, number_texts, arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.6))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plotstyle.save_figure(fig, save_path)  # 300-dpi tight PNG (shared writer)
    return fig, rows
