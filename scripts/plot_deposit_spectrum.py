#!/usr/bin/env python
"""Plot a deposited MolDeTr spectrum (.npz) with its annotations — standalone.

Self-contained (numpy + matplotlib only) so the Zenodo archive is plottable without
installing the model. Handles both deposited schemas:

  * experimental ROI — the annotated arrays are ``spectrum_padded`` (complex, 6144 pts)
    and ``ppm_axis_padded``; ``ground_truth`` / ``predictions`` hold the multiplets, and
    ``metadata`` carries ``points_per_hz`` / ``base_frequency_mhz`` / ``compound``. (The
    archive also stores ``*_raw`` and ``hz_axis_*`` variants, which this plotter ignores.)
  * synthetic sample — keys ``spec`` (complex) + ``labels``. ppm calibration comes from
    ``data_augmentation_meta.json`` (``ppm_left`` / ``ppm_right`` / ``ppm_scale`` +
    ``base_frequency``) beside the .npz; without it the x-axis falls back to point index.

Annotations are normalised to a common record — proton_count, chemical_shift_in_points,
coupling_constants_hz, linewidth_hz — via a small key adapter, then drawn as numbered
markers with an assignment table. Modelled on ``moldetr/visualization.py::plot_spectrum``.

Usage::

    python plot_deposit_spectrum.py --input roi_S8.npz
    python plot_deposit_spectrum.py --input 0.npz --meta data_augmentation_meta.json --out sample0.png

``allow_pickle`` is required for the object-array metadata / labels (first-party deposit data).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Palette + style — kept in sync with moldetr/plotstyle.py. The deposit ships standalone
# (numpy + matplotlib only), so the values are inlined here rather than imported from moldetr.
SPECTRUM_COLOR = "#333333"  # the 1D trace — near-black, thin
GRID_COLOR = "#e6e6e6"  # faint gridlines / table borders
TRUTH_COLOR = "#e8963a"  # ground-truth overlay — warm orange
# Okabe-Ito colourblind-safe spin-system colours (same order as plotstyle.SPIN_SYSTEM_COLORS).
SPIN_SYSTEM_COLORS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
    "#F0E442",
]
SPECTRUM, GRID, TRUTH = SPECTRUM_COLOR, GRID_COLOR, TRUTH_COLOR
ACCENT = SPIN_SYSTEM_COLORS[0]  # detections — the palette's lead colour (orange)

plt.rcParams.update(
    {
        "figure.constrained_layout.use": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def detect_schema(data) -> str:
    keys = set(data.keys())
    if {"spec", "labels"} <= keys:
        return "synthetic"
    if "metadata" in keys and any(k.startswith("spectrum") for k in keys):
        return "experimental"
    raise SystemExit(f"Unrecognised npz schema: keys={sorted(keys)}")


def amplitudes(data, schema) -> np.ndarray:
    key = (
        "spec"
        if schema == "synthetic"
        else ("spectrum_padded" if "spectrum_padded" in data else "spectrum_raw")
    )
    return np.real(np.asarray(data[key])).astype(float).ravel()


def ppm_endpoints(data, schema, meta_path):
    """(ppm at point 0, ppm at last point), or (None, None) if uncalibrated."""
    if schema == "experimental":
        if "ppm_axis_padded" in data:
            ax = np.asarray(data["ppm_axis_padded"], dtype=float)
            return float(ax[0]), float(ax[-1])
        meta = data["metadata"].item()
        return meta.get("left_ppm"), meta.get("right_ppm")
    if meta_path and Path(meta_path).exists():
        aug = json.loads(Path(meta_path).read_text())
        return float(aug["ppm_left"]), float(aug["ppm_right"])
    return None, None


def points_per_hz(data, schema, meta_path) -> float:
    if schema == "experimental":
        return float(data["metadata"].item().get("points_per_hz", 5.12))
    if meta_path and Path(meta_path).exists():
        aug = json.loads(Path(meta_path).read_text())
        span_hz = (aug["ppm_right"] - aug["ppm_left"]) * aug["base_frequency"]
        return abs(aug["ppm_scale"] / span_hz) if span_hz else 5.12
    return 5.12


def resolve_source(data, schema, which):
    """Pick the annotation source. Returns (source_key, which_label).

    For experimental files, fall back to whichever of predictions/ground_truth is present
    when the requested one is absent, and report the fallback so the plot isn't mislabelled.
    """
    if schema == "synthetic":
        return "labels", "labels"
    key = {"pred": "predictions", "gt": "ground_truth"}.get(which, "ground_truth")
    if key not in data or data[key] is None or len(list(data[key])) == 0:
        fallback = "ground_truth" if "ground_truth" in data else "predictions"
        if fallback != key:
            print(f"note: '{key}' not available in this file; drawing '{fallback}' instead.")
        key = fallback
    return key, ("pred" if key == "predictions" else "gt")


def to_records(data, schema, source_key, pph, ppm_l, ppm_r, n):
    """Normalise annotations to dicts: proton_count, chemical_shift_in_points,
    chemical_shift_ppm, coupling_constants_hz, linewidth_hz."""

    def pt_to_ppm(pts):
        if ppm_l is None or ppm_r is None or n <= 1:
            return None
        return ppm_l + (pts / (n - 1)) * (ppm_r - ppm_l)

    out = []
    if schema == "synthetic":
        for lab in data["labels"].tolist():
            pts = float(lab["center_position_in_points"])
            js = np.asarray(lab.get("coupling_constants_in_points", []), dtype=float) / pph
            out.append(
                {
                    "proton_count": int(lab["proton_number"]),
                    "chemical_shift_in_points": pts,
                    "chemical_shift_ppm": pt_to_ppm(pts),
                    "coupling_constants_hz": [float(j) for j in js if j > 0],
                    "linewidth_hz": float(lab.get("line_width_in_points", 0.0)) / pph,
                }
            )
        return out

    for p in list(data[source_key]):
        pts = float(p.get("chemical_shift_in_points", 0.0))
        js = [float(j) for j in (p.get("coupling_constants") or []) if float(j) > 0]
        shift = p.get("chemical_shift_ppm")
        out.append(
            {
                "proton_count": int(p["proton_count"]),
                "chemical_shift_in_points": pts,
                "chemical_shift_ppm": shift if shift is not None else pt_to_ppm(pts),
                "coupling_constants_hz": js,
                "linewidth_hz": p.get("linewidth_hz"),
            }
        )
    return out


def _apex(amp, xi, half=18):
    lo, hi = max(0, xi - half), min(amp.shape[-1], xi + half + 1)
    return float(amp[lo:hi].max()) if hi > lo else float(amp.max())


def _to_x(pts, n, ppm_l, ppm_r):
    """Map a point index to the x-axis (ppm if calibrated, else the point index)."""
    if ppm_l is not None and ppm_r is not None and n > 1:
        return ppm_l + (pts / (n - 1)) * (ppm_r - ppm_l)
    return pts


def _stagger(xs):
    """Two-level label heights: raise a label only when it crowds its left neighbour
    (a dependency-light stand-in for adjustText — the deposit stays numpy + matplotlib only)."""
    levels = [0] * len(xs)
    xr = (max(xs) - min(xs)) if len(xs) > 1 else 1.0
    last_x, last_lvl = None, 0
    for idx in sorted(range(len(xs)), key=lambda k: xs[k]):
        last_lvl = (
            (1 - last_lvl)
            if (last_x is not None and abs(xs[idx] - last_x) < 0.06 * (xr or 1.0))
            else 0
        )
        levels[idx] = last_lvl
        last_x = xs[idx]
    return levels


def _draw_markers(ax, recs, xs, amp, span):
    """Apex dot + stem + numbered circle per record, with a simple two-level height stagger."""
    levels = _stagger(xs)
    for i, (r, cx) in enumerate(zip(recs, xs), 1):
        y_apex = _apex(amp, int(round(r["chemical_shift_in_points"])))
        y_lab = y_apex + (0.10 + 0.11 * levels[i - 1]) * span
        ax.plot(cx, y_apex, "o", ms=3.5, color=ACCENT, zorder=4)
        ax.plot([cx, cx], [y_apex, y_lab], color=ACCENT, lw=0.7, alpha=0.6, zorder=3)
        ax.text(
            cx,
            y_lab,
            str(i),
            color="white",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            zorder=6,
            bbox=dict(boxstyle="circle,pad=0.28", fc=ACCENT, ec="white", lw=0.6),
        )


def _draw_truth(ax, gxs):
    """Overlay ground-truth positions as faint dotted TRUTH-coloured reference lines."""
    for j, gx in enumerate(gxs):
        ax.axvline(
            gx,
            color=TRUTH,
            ls=":",
            lw=1.1,
            alpha=0.8,
            zorder=1,
            label="Ground truth" if j == 0 else None,
        )


def _crop_x(ax, xs, x):
    """Crop the x-axis to the annotated-signal region (+~13 % pad) to kill dead margins."""
    if not xs:
        return
    lo, hi = min(xs), max(xs)
    x0, x1 = float(x[0]), float(x[-1])
    full = abs(x1 - x0) or 1.0
    pad = max(0.13 * (hi - lo), 0.04 * full)
    xlo, xhi = max(lo - pad, min(x0, x1)), min(hi + pad, max(x0, x1))
    if xhi > xlo:
        ax.set_xlim(xlo, xhi)


def _draw_table(axt, recs, ppm):
    """Render the assignment table into its own (borderless) axes."""
    axt.axis("off")
    unit = "ppm" if ppm else "pts"
    cells = []
    for i, r in enumerate(recs, 1):
        if ppm and r.get("chemical_shift_ppm") is not None:
            shift_str = f"{r['chemical_shift_ppm']:.3f}"
        else:  # uncalibrated: show the point position rather than a blank
            shift_str = f"{r['chemical_shift_in_points']:.0f}"
        cells.append(
            [
                str(i),
                f"{r['proton_count']} H",
                shift_str,
                (f"{max(r['coupling_constants_hz']):.1f}" if r["coupling_constants_hz"] else "–"),
                (f"{r['linewidth_hz']:.2f}" if r.get("linewidth_hz") is not None else "–"),
            ]
        )
    tbl = axt.table(
        cellText=cells,
        colLabels=["#", "protons", f"δ ({unit})", "max J (Hz)", "lw (Hz)"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    for (row, _), c in tbl.get_celld().items():
        c.set_edgecolor(GRID)
        c.set_linewidth(0.8)
        if row == 0:
            c.set_facecolor(ACCENT)
            c.set_text_props(color="white", fontweight="bold")


def plot(amp, recs, ppm_l, ppm_r, title, out, truth_recs=None):
    n = amp.shape[-1]
    peak = float(np.max(np.abs(amp))) if amp.size else 1.0
    amp = amp / (peak or 1.0)  # normalise -> clean 'a.u.' axis, no scientific-notation offset
    ppm = ppm_l is not None and ppm_r is not None
    x = np.linspace(ppm_l, ppm_r, n) if ppm else np.arange(n)
    base, top = float(amp.min()), float(amp.max())
    span = (top - base) or 1.0

    fig, (ax, axt) = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(x, amp, lw=1.0, color=SPECTRUM, zorder=2)
    xs = [_to_x(r["chemical_shift_in_points"], n, ppm_l, ppm_r) for r in recs]
    _draw_markers(ax, recs, xs, amp, span)
    if truth_recs:
        _draw_truth(ax, [_to_x(t["chemical_shift_in_points"], n, ppm_l, ppm_r) for t in truth_recs])

    ax.set_xlabel("Chemical shift δ (ppm)" if ppm else "Point index")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(base - 0.05 * span, top + 0.30 * span)
    ax.grid(axis="y", color=GRID, lw=0.8)
    _crop_x(ax, xs, x)
    if ppm:
        ax.invert_xaxis()  # NMR convention
    if truth_recs:
        ax.legend(loc="upper left")

    _draw_table(axt, recs, ppm)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot a deposited MolDeTr spectrum (.npz).")
    ap.add_argument(
        "--input", required=True, type=Path, help="deposited .npz (experimental or synthetic)"
    )
    ap.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="data_augmentation_meta.json for synthetic (default: beside the .npz)",
    )
    ap.add_argument(
        "--which",
        choices=["gt", "pred", "labels"],
        default=None,
        help="annotations to draw (default: gt for experimental, labels for synthetic)",
    )
    ap.add_argument("--out", type=Path, default=None, help="output PNG (default: <input>.png)")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    schema = detect_schema(data)
    amp = amplitudes(data, schema)
    n = amp.shape[-1]
    meta_path = args.meta or (args.input.parent / "data_augmentation_meta.json")
    ppm_l, ppm_r = ppm_endpoints(data, schema, meta_path)
    pph = points_per_hz(data, schema, meta_path)
    which = args.which or ("labels" if schema == "synthetic" else "gt")
    source_key, which_label = resolve_source(data, schema, which)
    recs = to_records(data, schema, source_key, pph, ppm_l, ppm_r, n)

    if schema == "experimental":
        m = data["metadata"].item()
        title = (
            f"{m.get('compound', '?')} — {m.get('base_frequency_mhz', 0):.0f} MHz ({which_label})"
        )
    else:
        title = f"{args.input.stem} — synthetic ({which_label})"

    # When predictions are drawn as markers, overlay ground truth (if present) as TRUTH reference lines.
    truth_recs = None
    if schema == "experimental" and which_label == "pred" and "ground_truth" in data:
        truth_recs = to_records(data, schema, "ground_truth", pph, ppm_l, ppm_r, n)

    out = args.out or args.input.with_suffix(".png")
    plot(amp, recs, ppm_l, ppm_r, title, out, truth_recs=truth_recs)


if __name__ == "__main__":
    main()
