"""Gradio GUI for MolDeTr — branded workbench with Detect + Simulate tabs.

Two tabs share the paper-branded theme (theme.py) and, on Detect, the interactive Plotly spectrum
(plotting.py):
- **Detect**: load a 1-D ¹H NMR window (.npz/.npy), get the multiplet assignment table + an
  interactive annotated spectrum (drag to box-zoom, double-click resets), with CSV / JSON export.
- **Simulate**: build a known spin system on the model's grid, optionally add training-range
  distortions, detect, and compare against ground truth. The Simulate plot stays on matplotlib
  (``moldetr.visualization.plot_spectrum``); the ground-truth overlay reuses that renderer.

Run locally:
    pip install -e ".[app]"
    python app.py

Deploys unchanged as a Hugging Face Space (set the checkpoint via ``MOLDETR_CHECKPOINT`` or place it
at ``moldetr/model/``). Weights are on Zenodo (DOI 10.5281/zenodo.21217102). ``theme.py`` and
``plotting.py`` must sit next to ``app.py``.

MolDeTr is research code accompanying the paper: it handles congested, strongly-coupled ¹H NMR
spectra and is largely field-agnostic — it works in Hz, so it was tested across 80–600 MHz (and
simulated down to ~5 MHz). Results can deviate for inputs outside its trained regime — unusual
distortions, non-standard pulse sequences or processing, mixtures, or windows wider than 1200 Hz.
``max J`` is the dominant coupling per multiplet; the full set comes from the committed
``structured_output`` path. See docs/SCOPE.md.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from moldetr.distort import distort
from moldetr.inference import build_model, load_checkpoint, run
from moldetr.postprocess import decode_predictions, load_extrema
from moldetr.simulate import simulate
from moldetr.validation import INPUT_LENGTH, POINTS_PER_HZ, validate_spectrum
from app_ui.plotting import (  # BRAND: interactive Plotly plots
    assignment_rows,
    comparison_figure,
    spectrum_figure,
)
from app_ui.theme import (
    CUSTOM_CSS,
    HEADER_HTML,
    MOLDETR_THEME,
)  # BRAND: palette / header / theme (at launch)

ROOT = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get(
    "MOLDETR_CHECKPOINT", str(ROOT / "moldetr" / "model" / "model_spin_system_ABCDEFG_exp2.pth")
)
EXTREMA = str(ROOT / "moldetr" / "assets" / "extrema.txt")

# The simulate->predict round-trip (phenotypes, coupling-matrix helper, GT matching) lives in the
# scripts/ deliverable; add it to the path so the "Simulate" tab reuses it rather than duplicating.
sys.path.insert(0, str(ROOT / "scripts"))
import simulate_and_predict as sp  # noqa: E402  (scripts/ was just placed on sys.path above)

AUTO, MANUAL, NONE = "Auto (from file)", "Manual (window ppm)", "None (report in Hz)"

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_checkpoint(build_model(), CHECKPOINT)
    return _MODEL


EXAMPLES_DIR = ROOT / "examples"


def _is_bundled_example(path: Path) -> bool:
    """True only for a file that ships inside this repo's ``examples/`` directory."""
    try:
        return path.resolve().is_relative_to(EXAMPLES_DIR)
    except (OSError, ValueError):  # unresolvable path (broken link, bad drive) → not ours
        return False


def _load(path: str, *, trusted: bool = False):
    """Load a spectrum (+ ppm calibration if present) from .npz/.npy. Array is returned as-is
    (possibly complex) so the caller can surface the dtype; validation takes the real part.

    ``trusted`` is what enables pickle, and only files we ship get it. Unpickling executes code
    carried in the archive, so an uploaded ``.npz`` must never take that path. The gate costs
    nothing: the one branch that touches an object array is the ``metadata`` fallback below, which
    is reached only when ``ppm_axis_padded`` is absent — and every bundled example has that axis.
    """
    p = Path(path)
    cal: dict = {}
    if p.suffix == ".npz":
        # allow_pickle is gated on provenance, never on the caller's convenience.
        data = np.load(p, allow_pickle=trusted)
        # Prefer the per-point ppm axis (correct for the ROI); metadata left/right_ppm span the full
        # spectrum and would mis-place peaks, so only fall back to them if the axis is absent.
        if "ppm_axis_padded" in data:
            axis = np.asarray(data["ppm_axis_padded"], dtype=float)
            cal = {"ppm_left": float(axis[0]), "ppm_right": float(axis[-1])}
        elif "metadata" in data:
            md = data["metadata"].item()
            cal = {"ppm_left": md.get("left_ppm"), "ppm_right": md.get("right_ppm")}
        for key in ("spectrum_padded", "spec"):
            if key in data:
                return np.asarray(data[key]), cal
        return np.asarray(data[list(data.keys())[0]]), cal
    return np.asarray(np.load(p)), cal


def _resolve_points_per_hz(points_per_hz) -> float:
    """Digital resolution in points/Hz, or ``ValueError`` naming what is wrong with it.

    A blank field means "unset" and falls back to the default. Zero and negatives are *stated*
    values that cannot be right, and the old ``float(x) if x else DEFAULT`` silently replaced 0
    with 5.12 — so clearing the box produced confident, wrongly-scaled results. A negative value
    is truthy and sailed through entirely, mirroring the axis and yielding negative line widths.
    """
    if points_per_hz is None or points_per_hz == "":
        return POINTS_PER_HZ
    pph = float(points_per_hz)
    if pph <= 0:
        raise ValueError("digital resolution must be positive (points/Hz)")
    return pph


def _spec_report(file, points_per_hz) -> str:
    """Post-upload input check — same logic, glyphs instead of emoji."""  # BRAND
    if file is None:
        return ""
    path = file if isinstance(file, str) else file.name
    try:
        raw, cal = _load(path, trusted=_is_bundled_example(Path(path)))
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Could not read the file: {exc}"
    try:
        pph = _resolve_points_per_hz(points_per_hz)
    except ValueError as exc:
        return f"⚠ Invalid input: {exc}"
    arr = np.asarray(raw).ravel()
    n = arr.shape[0]
    window = INPUT_LENGTH / pph
    ok_len = "✓" if n == INPUT_LENGTH else f"✗ needs exactly {INPUT_LENGTH}"
    ok_res = (
        "✓" if abs(pph - POINTS_PER_HZ) <= 0.01 else "⚠ not 1200 Hz — predictions may be unreliable"
    )
    dtype = "complex — the real (absorption) part is used" if np.iscomplexobj(arr) else "real ✓"
    finite = "✓" if np.all(np.isfinite(np.real(arr))) else "✗ contains NaN/Inf"
    axis = "yes ✓ (Auto works)" if cal.get("ppm_left") is not None else "no — use Manual or None"
    return (
        "**Input check**\n"
        f"- Length: **{n}** points {ok_len}\n"
        f"- Resolution: **{pph:g}** points/Hz → **{window:.0f} Hz** window {ok_res}\n"
        f"- Data type: {dtype}\n"
        f"- Finite values: {finite}\n"
        f"- ppm axis in file: {axis}"
    )


def predict(file, threshold, ppm_mode, manual_left, manual_right, points_per_hz):
    """Run detection and return (assignment table, annotated Plotly plot, status message)."""
    if file is None:
        return None, None, "Load a `.npz`/`.npy` spectrum, or pick an example below."
    if not Path(CHECKPOINT).exists():
        return (
            None,
            None,
            (
                f"Checkpoint not found at `{CHECKPOINT}`. "
                "Download it from Zenodo (10.5281/zenodo.21217102) into `moldetr/model/`."
            ),
        )
    path = file if isinstance(file, str) else file.name
    # `_spec_report` has always guarded this; `predict` did not, so the same bad file that produced
    # a tidy "⚠ Could not read" above the button rendered a Python traceback below it.
    try:
        raw, cal = _load(path, trusted=_is_bundled_example(Path(path)))
    except Exception as exc:  # noqa: BLE001 - any unreadable file must surface as a message
        return None, None, f"⚠ Could not read the file: {exc}"
    try:
        pph = _resolve_points_per_hz(points_per_hz)
    except ValueError as exc:
        return None, None, f"Invalid input: {exc}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            amplitudes = validate_spectrum(raw, points_per_hz=pph)
        except ValueError as exc:
            return None, None, f"Invalid spectrum: {exc}"
    warn_msg = " ".join(str(w.message) for w in caught)

    if ppm_mode == MANUAL and manual_left is not None and manual_right is not None:
        ppm_left, ppm_right = float(manual_left), float(manual_right)
    elif ppm_mode == AUTO:
        ppm_left, ppm_right = cal.get("ppm_left"), cal.get("ppm_right")
    else:  # NONE, or MANUAL without both bounds -> report shift in Hz
        ppm_left = ppm_right = None

    preds = decode_predictions(
        run(_get_model(), amplitudes),
        load_extrema(EXTREMA),
        pph,
        ppm_left=ppm_left,
        ppm_right=ppm_right,
        threshold=threshold,
    )
    fig = spectrum_figure(
        amplitudes, preds, ppm_left=ppm_left, ppm_right=ppm_right, points_per_hz=pph
    )  # BRAND: Plotly; Hz axis when no ppm calibration
    rows = assignment_rows(preds, ppm_left is not None and ppm_right is not None)
    table = pd.DataFrame(rows) if rows else pd.DataFrame()
    if preds:
        msg = f"Detected **{len(preds)}** multiplet(s). Numbers on the plot match the table rows."
    else:
        msg = "No multiplets passed the detection threshold — try lowering it."
    if warn_msg:
        msg += f"\n\n⚠ {warn_msg}"
    return table, fig, msg


_EXPORT_DIR: str | None = None


def _export_dir() -> str:
    """One temp directory per process, reused by every Detect click.

    `mkdtemp` *per click* leaked a directory on every detection — unbounded on a Space that stays
    up for weeks. The two export files are overwritten in place instead, which also means a stale
    download link can never serve a previous run's numbers.
    """
    global _EXPORT_DIR
    if _EXPORT_DIR is None or not os.path.isdir(_EXPORT_DIR):
        _EXPORT_DIR = tempfile.mkdtemp(prefix="moldetr_")
    return _EXPORT_DIR


def predict_ui(file, threshold, ppm_mode, manual_left, manual_right, points_per_hz):
    """predict() + CSV/JSON export files for the download buttons."""  # NEW
    table, fig, msg = predict(file, threshold, ppm_mode, manual_left, manual_right, points_per_hz)
    csv_path = json_path = None
    if table is not None and not table.empty:
        out = _export_dir()
        csv_path = os.path.join(out, "moldetr_prediction.csv")
        json_path = os.path.join(out, "moldetr_prediction.json")
        table.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(table.to_dict(orient="records"), fh, ensure_ascii=False, indent=2)
    return (
        table,
        fig,
        msg,
        gr.DownloadButton(value=csv_path, interactive=csv_path is not None),
        gr.DownloadButton(value=json_path, interactive=json_path is not None),
    )


# --- "Simulate" tab: reuse the scripts/ round-trip against the same model + decode + plot ---------

SIMULATE_INTRO = (
    "Simulate a known spin system on the model's grid (80 MHz, 15→0 ppm, 6144 pts), optionally add "
    "training-range distortions, then detect and compare against ground truth. Edit the per-spin "
    "shifts, the coupling, and the line width, or add noise / phase / broadening / baseline — every "
    "distortion slider is bounded to the range the model was trained on."
)

PHENOTYPE_CHOICES = sorted(sp.PHENOTYPES)


def _phenotype_defaults(name: str) -> tuple[str, float, float]:
    """Default editable fields (per-spin shift string, coupling J, line width) for a phenotype."""
    pheno = sp.PHENOTYPES[name]
    shifts_str = ", ".join(f"{s:g}" for s in pheno["shifts_ppm"])
    j_default = float(pheno["couplings"][0][2]) if pheno["couplings"] else 0.0
    return shifts_str, j_default, 1.0


def _parse_spin_shifts(text: str, n_expected: int, default: list[float]) -> list[float]:
    """Parse a comma/space/semicolon-separated per-spin shift list; fall back to the default."""
    if not text or not text.strip():
        return [float(s) for s in default]
    vals = [float(t) for t in text.replace(";", ",").replace(" ", ",").split(",") if t.strip()]
    if len(vals) != n_expected:
        raise ValueError(f"expected {n_expected} shift value(s), got {len(vals)}")
    return vals


def _build_gt_groups(
    shifts: list[float], pairs: list[tuple[int, int, float]], j_hz: float
) -> list[dict]:
    """Group equivalent spins into ground-truth multiplets (shift, proton count, max J)."""
    coupled = {s for pair in pairs for s in pair[:2]}
    groups: dict[float, list[int]] = {}
    for idx, shift in enumerate(shifts):
        groups.setdefault(round(float(shift), 4), []).append(idx)
    gt = []
    for shift_val, idxs in sorted(groups.items(), reverse=True):
        has_coupling = any(i in coupled for i in idxs)
        gt.append(
            {
                "shift_ppm": shift_val,
                "proton_count": len(idxs),
                "max_j_hz": float(j_hz) if has_coupling else None,
            }
        )
    return gt


def _comparison_dataframe(
    gt_groups: list[dict], preds: list[dict], tol_hz: float = 2.0
) -> pd.DataFrame:
    """GT-vs-detected table with a match status and explicit error columns.

    Each GT group is paired with its nearest-δ prediction (``match_to_gt``); predictions matched to
    no GT are appended as spurious rows. ``status``: ``✓ match`` (within ``tol_hz`` and right H) ·
    ``~ off`` · ``✗ missed`` (GT with no prediction) · ``+ extra`` (spurious prediction). These
    mirror the connector / marker colours in :func:`plotting.comparison_figure`.
    """
    matched = sp.match_to_gt(gt_groups, preds)
    matched_ids = {id(p) for _g, p in matched if p is not None}
    rows: list[dict] = []
    for i, (gt, pred) in enumerate(matched, 1):
        gt_j = "–" if gt["max_j_hz"] is None else f"{gt['max_j_hz']:.1f}"
        row = {
            "#": i,
            "status": "",
            "GT δ (ppm)": f"{gt['shift_ppm']:.2f}",
            "GT H": gt["proton_count"],
            "GT J (Hz)": gt_j,
        }
        if pred is None:
            row.update(
                {
                    "status": "✗ missed",
                    "pred δ (ppm)": "–",
                    "pred H": "–",
                    "Δδ (Hz)": "–",
                    "ΔH": "–",
                    "conf": "–",
                }
            )
        else:
            dd_hz = abs(float(pred["chemical_shift_ppm"]) - gt["shift_ppm"]) * sp.BASE_FREQ_MHZ
            dh = int(pred["proton_count"]) - gt["proton_count"]
            row.update(
                {
                    "status": "✓ match" if (dd_hz <= tol_hz and dh == 0) else "~ off",
                    "pred δ (ppm)": f"{float(pred['chemical_shift_ppm']):.3f}",
                    "pred H": int(pred["proton_count"]),
                    "Δδ (Hz)": f"{dd_hz:.2f}",
                    "ΔH": f"{dh:+d}",
                    "conf": f"{float(pred['confidence']):.2f}",
                }
            )
        rows.append(row)
    for k, pred in enumerate((p for p in preds if id(p) not in matched_ids), len(rows) + 1):
        rows.append(
            {
                "#": k,
                "status": "+ extra",
                "GT δ (ppm)": "–",
                "GT H": "–",
                "GT J (Hz)": "–",
                "pred δ (ppm)": f"{float(pred['chemical_shift_ppm']):.3f}",
                "pred H": int(pred["proton_count"]),
                "Δδ (Hz)": "–",
                "ΔH": "–",
                "conf": f"{float(pred['confidence']):.2f}",
            }
        )
    return pd.DataFrame(rows)


def _simulate_distort_kwargs(
    add_noise: bool, snr: float, phase0: float, broaden: float, baseline: float
) -> dict[str, float]:
    """Assemble ``distort`` kwargs from the sliders (neutral / identity values are skipped)."""
    dk: dict[str, float] = {}
    if add_noise:
        dk["noise_snr_log10"] = float(snr)
    if float(phase0) != 0.0:
        dk["phase0_deg"] = float(phase0)
    if float(broaden) > 0.0:
        dk["broaden_hz"] = float(broaden)
    if float(baseline) > 0.0:
        dk["baseline"] = float(baseline)
    return dk


def simulate_and_detect(
    phenotype: str,
    shifts_text: str,
    j_hz: float,
    width_hz: float,
    add_noise: bool,
    snr: float,
    phase0: float,
    broaden: float,
    baseline: float,
    threshold: float,
):
    """Simulate the (edited) phenotype, optionally distort, detect, and compare to ground truth."""
    if not Path(CHECKPOINT).exists():
        return (
            None,
            None,
            f"Checkpoint not found at `{CHECKPOINT}`. "
            "Download it from Zenodo (10.5281/zenodo.21217102) into `moldetr/model/`.",
        )
    pheno = sp.PHENOTYPES[phenotype]
    n_spins = len(pheno["shifts_ppm"])
    try:
        shifts = _parse_spin_shifts(shifts_text, n_spins, pheno["shifts_ppm"])
    except ValueError as exc:
        return None, None, f"Invalid shifts: {exc}"
    pairs = [(i, j, float(j_hz)) for (i, j, _j0) in pheno["couplings"]]
    couplings = sp.build_coupling_matrix(n_spins, pairs)
    try:
        spectrum, ppm_axis = simulate(
            shifts,
            couplings,
            [float(width_hz)] * n_spins,
            sp.BASE_FREQ_MHZ,
            sp.LEFT_PPM,
            sp.RIGHT_PPM,
            sp.N_POINTS,
        )
        dk = _simulate_distort_kwargs(add_noise, snr, phase0, broaden, baseline)
        if dk:
            spectrum = distort(spectrum, ppm_axis, **dk)
    except ValueError as exc:
        return None, None, f"Invalid parameters: {exc}"
    amplitudes = np.asarray(np.real(spectrum), dtype=float)
    preds = decode_predictions(
        run(_get_model(), amplitudes),
        load_extrema(EXTREMA),
        sp.POINTS_PER_HZ,
        ppm_left=sp.LEFT_PPM,
        ppm_right=sp.RIGHT_PPM,
        threshold=threshold,
    )
    gt_groups = _build_gt_groups(shifts, pheno["couplings"], j_hz)
    matched = sp.match_to_gt(gt_groups, preds)
    matched_ids = {id(p) for _g, p in matched if p is not None}
    spurious = [p for p in preds if id(p) not in matched_ids]
    fig = comparison_figure(
        amplitudes,
        matched,
        spurious,
        ppm_left=sp.LEFT_PPM,
        ppm_right=sp.RIGHT_PPM,
        base_freq_mhz=sp.BASE_FREQ_MHZ,
        tol_hz=2.0,
    )
    table = _comparison_dataframe(gt_groups, preds)
    n_match = sum(1 for _g, p in matched if p is not None)
    msg = (
        f"**Simulated `{phenotype}`** — {len(gt_groups)} ground-truth multiplet(s); the model "
        f"**detected** {len(preds)} ({n_match} matched, {len(spurious)} spurious). "
        "Teal ▽ = ground truth · clay ● = model detection; a connector turns **green** within "
        "tolerance and **amber** when off. Missed GT and spurious peaks are outlined in red."
    )
    return table, fig, msg


CONTRACT = (
    f"**Expected input.** A 1-D ¹H spectrum of **{INPUT_LENGTH} points** at **{POINTS_PER_HZ} "
    "points/Hz** (a 1200 Hz window), real-valued. Absolute intensity does not matter (each spectrum "
    "is min–max normalised), but relative intensities, SNR and line shape do. Every coupling partner "
    "of an in-window peak must also sit inside the window — see "
    "[`docs/INPUT_FORMAT.md`](docs/INPUT_FORMAT.md)."
)

# Field-agnostic scope disclaimer (matches docs/SCOPE.md and the README research-code callout).
PROTOTYPE = (
    "MolDeTr handles congested, strongly-coupled ¹H NMR spectra and is largely field-agnostic "
    "(it works in Hz; tested on 80–600 MHz). Predictions can deviate for inputs outside its trained "
    "regime — unusual distortions, non-standard pulse sequences or processing, mixtures, or windows "
    "wider than 1200 Hz. **max J** is the dominant coupling per multiplet (the full set is in the "
    "committed `structured_output` path). Sanity-check predictions against your own chemistry."
)

FOOTNOTE = (  # NEW
    "max J = largest coupling per multiplet — the full coupling set comes from the committed "
    "`structured_output` path. Sanity-check predictions against your own chemistry."
)

OUTPUT_CAPTION = (
    "Numbered markers on the plot correspond to the table rows. **max J** is the largest coupling "
    "per multiplet; the full set is in the committed `structured_output` path."
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="MolDeTr"
    ) as demo:  # BRAND: theme + css applied at launch() (gradio 6.x moved them off Blocks)
        gr.HTML(HEADER_HTML)  # BRAND: wordmark · eyebrow · prototype chip · links
        with gr.Tabs():
            with gr.Tab("Detect"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=0, min_width=400):  # BRAND: fixed input rail
                        gr.Markdown(CONTRACT)
                        spectrum = gr.File(
                            label="Spectrum (.npz / .npy)",
                            file_types=[".npz", ".npy"],
                            type="filepath",
                            elem_id="md-file",
                        )
                        spec_md = gr.Markdown(elem_id="md-check")
                        points_per_hz = gr.Number(
                            value=POINTS_PER_HZ, label="Digital resolution (points/Hz)", precision=4
                        )
                        ppm_mode = gr.Radio(
                            [AUTO, MANUAL, NONE],
                            value=AUTO,
                            label="Chemical-shift (ppm) axis",
                            elem_id="md-ppm",
                        )
                        with gr.Row():
                            manual_left = gr.Number(
                                value=None, label="Window left ppm (Manual only)"
                            )
                            manual_right = gr.Number(
                                value=None, label="Window right ppm (Manual only)"
                            )
                        threshold = gr.Slider(
                            0.0, 1.0, value=0.3, step=0.05, label="Detection threshold"
                        )
                        run_btn = gr.Button("Detect multiplets", variant="primary")
                        gr.Examples(
                            [
                                ["examples/roi_S10_example.npz"],
                                ["examples/roi_S8_example.npz"],
                                ["examples/synthetic_example.npz"],
                            ],
                            inputs=spectrum,
                            label="Examples",
                            example_labels=[  # BRAND: compound · field instead of filenames
                                "guajazulene · 500 MHz",
                                "vanillin · 300 MHz",
                                "synthetic",
                            ],
                            elem_id="md-examples",
                        )
                        with gr.Accordion("Research prototype — scope", open=False):  # BRAND
                            gr.Markdown(PROTOTYPE)
                    with gr.Column(scale=1):
                        status = gr.Markdown()
                        table = gr.Dataframe(
                            label="Assignment table",
                            interactive=False,
                            wrap=True,
                            elem_id="md-table",
                        )
                        with gr.Row():  # NEW: exports
                            csv_btn = gr.DownloadButton(
                                "Download CSV", interactive=False, size="sm"
                            )
                            json_btn = gr.DownloadButton(
                                "Download JSON", interactive=False, size="sm"
                            )
                        gr.Markdown(FOOTNOTE, elem_classes="md-footnote")  # NEW
                        plot = gr.Plot(
                            label="Annotated spectrum — drag to zoom · double-click resets",
                            elem_id="md-plot",
                        )
            with gr.Tab("Simulate"):
                gr.Markdown(SIMULATE_INTRO)
                with gr.Row():
                    with gr.Column(scale=2):
                        _defaults = _phenotype_defaults("ethyl")
                        sim_phenotype = gr.Dropdown(
                            PHENOTYPE_CHOICES, value="ethyl", label="Phenotype"
                        )
                        sim_shifts = gr.Textbox(
                            value=_defaults[0], label="Per-spin shifts δ (ppm, comma-separated)"
                        )
                        with gr.Row():
                            sim_j = gr.Number(value=_defaults[1], label="Coupling J (Hz)")
                            sim_width = gr.Number(value=_defaults[2], label="Line width FWHM (Hz)")
                        gr.Markdown("**Distortions** — each bounded to the model's trained range.")
                        with gr.Row():
                            sim_add_noise = gr.Checkbox(value=False, label="Add noise")
                            sim_snr = gr.Slider(
                                2.0, 5.0, value=3.0, step=0.1, label="Noise SNR (log10)"
                            )
                        sim_phase0 = gr.Slider(
                            -8.0,
                            8.0,
                            value=0.0,
                            step=0.5,
                            label="Zeroth-order phase (deg; 0 = off)",
                        )
                        sim_broaden = gr.Slider(
                            0.0,
                            3.0,
                            value=0.0,
                            step=0.1,
                            label="Broadening FWHM (Hz; 0 = off)",
                            info=(
                                "⚠ Outside the model's training distribution: the shipped model was "
                                "trained without line broadening, so enabling this feeds it an effect "
                                "it never saw and detection may degrade."
                            ),
                        )
                        sim_baseline = gr.Slider(
                            0.0, 0.1, value=0.0, step=0.01, label="Baseline tilt (0 = off)"
                        )
                        sim_threshold = gr.Slider(
                            0.0, 1.0, value=0.3, step=0.05, label="Detection threshold"
                        )
                        sim_btn = gr.Button("Simulate & Predict", variant="primary")
                    with gr.Column(scale=3):
                        sim_status = gr.Markdown(
                            "Pick a phenotype, adjust the parameters, then press "
                            "**Simulate & Predict**."
                        )
                        sim_plot = gr.Plot(label="Ground truth vs detected")
                        sim_table = gr.Dataframe(
                            label="GT vs detected (nearest-δ matched)", interactive=False, wrap=True
                        )
                        gr.Markdown(OUTPUT_CAPTION, elem_classes="md-footnote")

        # Explicit api_names: without them Gradio derives endpoint ids from the callback names, so
        # the two `_spec_report` wirings became `/_spec_report` and `/_spec_report_1` — a public API
        # surface named after private functions, positional in a way that shifts if either moves.
        # The e2e tier addresses these by name; keep them stable.
        spectrum.change(
            _spec_report,
            inputs=[spectrum, points_per_hz],
            outputs=spec_md,
            api_name="check_input_on_upload",
        )
        points_per_hz.change(
            _spec_report,
            inputs=[spectrum, points_per_hz],
            outputs=spec_md,
            api_name="check_input_on_resolution_change",
        )
        run_btn.click(
            predict_ui,  # NEW: wraps predict with the export files
            inputs=[spectrum, threshold, ppm_mode, manual_left, manual_right, points_per_hz],
            outputs=[table, plot, status, csv_btn, json_btn],
            api_name="detect",
        )
        sim_phenotype.change(
            _phenotype_defaults,
            inputs=sim_phenotype,
            outputs=[sim_shifts, sim_j, sim_width],
            api_name="phenotype_defaults",
        )
        sim_btn.click(
            simulate_and_detect,
            inputs=[
                sim_phenotype,
                sim_shifts,
                sim_j,
                sim_width,
                sim_add_noise,
                sim_snr,
                sim_phase0,
                sim_broaden,
                sim_baseline,
                sim_threshold,
            ],
            outputs=[sim_table, sim_plot, sim_status],
            api_name="simulate_and_detect",
        )
    return demo


# --- the single way this app is served -----------------------------------------------------------
# Gradio 6 moved `theme=`/`css=` from Blocks(...) onto .launch(), and omitting them raises nothing —
# the app just serves unstyled. Keeping the kwargs here, and routing every entry point (this module,
# `moldetr app`, and the browser/e2e fixtures) through launch_app(), makes "what ships" and "what is
# tested" the same object by construction. See tests/e2e/test_browser_branding.py.
LAUNCH_KWARGS = {"theme": MOLDETR_THEME, "css": CUSTOM_CSS}  # BRAND: gradio 6.x theming


def launch_app(demo: gr.Blocks | None = None, **overrides):
    """Launch the app exactly as production does. Returns ``(demo, launch_result)``.

    The Blocks handle comes back so callers can ``demo.close()`` — the test fixtures rely on it.
    """
    demo = build_ui() if demo is None else demo
    return demo, demo.launch(**{**LAUNCH_KWARGS, **overrides})


if __name__ == "__main__":
    launch_app()
