"""Gradio GUI for MolDeTr — branded workbench with Detect + Simulate tabs.

Two tabs share the paper-branded theme (theme.py) and, on Detect, the interactive Plotly spectrum
(plotting.py):
- **Detect**: load a 1-D ¹H NMR window (.npz/.npy), get the multiplet assignment table + an
  interactive annotated spectrum (drag to box-zoom, double-click resets), with CSV / JSON export.
- **Simulate**: build a known spin system on the model's grid, optionally add training-range
  distortions, detect, and compare against ground truth. Both tabs render with Plotly
  (``app_ui.plotting``); the ground-truth comparison uses ``comparison_figure``.

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
from typing import Any, Literal

import gradio as gr
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from moldetr.distort import distort
from moldetr.inference import build_model, load_checkpoint, run
from moldetr.postprocess import decode_predictions, load_extrema
from moldetr.simulate import COUPLING_EPS_HZ, coupling_blocks, simulate_systems
from moldetr.validation import INPUT_LENGTH, POINTS_PER_HZ, validate_spectrum
from app_ui import grading
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


def _checkpoint_error_message(exc: Exception) -> str:
    """Render a checkpoint-load refusal for the status box.

    `load_checkpoint` raises `RuntimeError` when the trust gate refuses a file, and that message is
    the only place `MOLDETR_ALLOW_UNTRUSTED_CHECKPOINT` — the documented way to run weights you
    trained yourself — is named. Unwrapped, it surfaced as a bare Gradio "Error" toast, so the one
    piece of information needed to resolve the situation was the piece that never arrived.

    Fenced rather than interpolated into the sentence: the gate's message is multi-line and carries
    both MD5s, and markdown collapses those lines into an unreadable run precisely where the user
    has to compare two hex digests.
    """
    return f"⚠ Could not load the checkpoint:\n\n```\n{exc}\n```"


EXAMPLES_DIR = ROOT / "examples"


def _is_bundled_example(path: Path) -> bool:
    """True only for a file that ships inside this repo's ``examples/`` directory.

    Relative paths resolve against ``ROOT``, **not** the process CWD. ``build_ui()`` wires the
    examples as relative paths (``"examples/roi_S10_example.npz"``), so a CWD-relative resolution
    silently disarms the gate whenever the app is launched from another directory — invisible
    today only because no bundled example needs pickle.

    This cannot widen the gate: a relative path can only resolve inside ``examples/`` if such a
    file actually ships there, and uploads always arrive as absolute temp paths.
    """
    try:
        candidate = path if path.is_absolute() else ROOT / path
        return candidate.resolve().is_relative_to(EXAMPLES_DIR)
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
        "✓"
        if abs(pph - POINTS_PER_HZ) <= 0.01
        else "⚠ not 1200 Hz, so predictions may be unreliable"
    )
    dtype = "complex; the real (absorption) part is used" if np.iscomplexobj(arr) else "real ✓"
    finite = "✓" if np.all(np.isfinite(np.real(arr))) else "✗ contains NaN/Inf"
    axis = "yes ✓ (Auto works)" if cal.get("ppm_left") is not None else "no; use Manual or None"
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

    try:
        model = _get_model()
    except RuntimeError as exc:  # the checkpoint trust gate; its text names the only remedy
        return None, None, _checkpoint_error_message(exc)

    preds = decode_predictions(
        run(model, amplitudes),
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
        msg = "No multiplets passed the detection threshold. Try lowering it."
    if warn_msg:
        msg += f"\n\n⚠ {warn_msg}"
    return table, fig, msg


_EXPORT_DIR: str | None = None


def _export_dir() -> str:
    """One temp directory per process, reused by every Detect click.

    `mkdtemp` *per click* leaked a directory on every detection — unbounded on a process that
    stays up for weeks. The two export files are overwritten in place instead.

    Reusing one path looks like it should let a later detection hand an earlier user someone
    else's numbers, but it cannot: ``gr.DownloadButton`` copies the file into Gradio's own
    **content-addressed** cache when the event returns, so the link a user holds points at a hash
    of the bytes they were shown, not at this file. Verified in
    ``test_download_links_are_content_addressed_so_the_shared_dir_is_safe`` — which exists
    precisely so a future Gradio that served this path directly would fail loudly here.
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
    "Build a spin system on the model's grid (80 MHz, 15→0 ppm, 6144 pts), distort it within the "
    "range the model was trained on, then run the detector on it and compare against ground truth "
    "you defined. Start from a known system or edit the matrix directly: shifts on the diagonal, "
    "couplings above it. A second, independent system can be switched on below the first and is "
    "summed with it at matching per-proton integrals. "
    "Once a spectrum is simulated, the distortion sliders re-distort it live — the spin dynamics "
    "are not solved again."
)

PHENOTYPE_CHOICES = sorted(sp.PHENOTYPES)

MATRIX_HINT = (
    "Each row is one spin. The **diagonal** holds its shift δ in ppm; cells **above** the diagonal "
    "hold the coupling J in Hz between that pair. Leave a pair at 0 and the two spins belong to "
    "separate spin systems, which are simulated independently and summed. Cells below the diagonal "
    "are ignored. For a second molecule, either leave its couplings to this one at 0 here, or use "
    "the **Second spin system** panel below — they describe the same computation."
)


#: Largest spin count the matrix editor offers, **per editor**. `simulate` pays 2**n per coupled
#: block, and `MAX_BLOCK_SPINS` caps a single block at 10, so offering more rows than that would only
#: ever produce systems the simulator refuses. With a second editor the combined spectrum may hold
#: more spins than this — legitimately, since the Hamiltonian is built per block and never on the
#: joined matrix.
MAX_MATRIX_SPINS = 8

#: What the optional second panel starts from. Any of `PHENOTYPE_CHOICES` works; this one is small,
#: strongly coupled, and pairs with the presets people reach for first.
SECOND_SYSTEM_PRESET = "AB"

#: Pople letters name the spins the way the spin-system literature and the paper's table do.
_POPLE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


#: The cell types `gr.Dataframe` accepts, spelled out so the grid's column list type-checks.
DataframeCell = Literal["str", "number", "bool", "date", "markdown", "html"]


def _matrix_datatype(n_spins: int) -> list[DataframeCell]:
    """Column types for the spin grid: a text label column, then one numeric column per spin."""
    columns: list[DataframeCell] = ["str"]
    columns.extend("number" for _ in range(n_spins))
    return columns


def _spin_label(index: int) -> str:
    """A, B, C ... then A', B', ... once the alphabet runs out (it will not, at 8 spins)."""
    letter = _POPLE[index % len(_POPLE)]
    return letter + "'" * (index // len(_POPLE))


def _cell_value(cell: object, row: int, col: int) -> float:
    """Read one grid cell as a float, treating blanks as zero and naming the cell on a typo.

    Gradio hands back ``""`` or ``None`` for a cleared cell and the raw string for anything it could
    not parse. Both must be resolved here: `float("")` raises, and a coupling that silently vanished
    looks exactly like a spectrum the user meant to produce.
    """
    if cell is None:
        return 0.0
    if isinstance(cell, str):
        if not cell.strip():
            return 0.0
    try:
        value = float(cell)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(
            f"row {row + 1}, column {col + 1} is not a number: {cell!r}. "
            "Use a plain decimal, or clear the cell for zero."
        ) from None
    # NaN and inf survive float(), then surface far away as "this spin system has no observable
    # transition" — a description of the symptom rather than of the cell the user typed into.
    if not np.isfinite(value):
        raise ValueError(
            f"row {row + 1}, column {col + 1} is not a finite number: {cell!r}. "
            "Use a plain decimal, or clear the cell for zero."
        )
    return value


def _matrix_to_system(rows: list[list[object]]) -> tuple[list[float], NDArray[np.float64]]:
    """Split the editor grid into per-spin shifts and a coupling matrix.

    The grid is ``[label, v0, v1, ...]`` per row, so every value is offset by one column. The
    **diagonal** carries chemical shifts in ppm and the **upper triangle** the couplings in Hz. Only
    the upper triangle is read, matching :func:`moldetr.simulate.simulate` and
    :func:`moldetr.simulate.coupling_blocks`, so a stale value below the diagonal cannot couple two
    spins the plotted spectrum treats as independent.

    Both grids let a user add or remove rows and columns directly, so the shape is checked rather
    than assumed. Indexing a ragged row unguarded raised ``IndexError``, which ``_simulate_stage``
    does not catch — it escaped the error-string channel and took the tab down with a Gradio toast
    instead of naming the row. A surplus column is a mismatch too, not something to drop silently.
    """
    n = len(rows)
    shifts: list[float] = []
    couplings = np.zeros((n, n), dtype=float)
    for i, row in enumerate(rows):
        if len(row) != n + 1:
            raise ValueError(
                f"row {i + 1} has {len(row) - 1} value(s) but the matrix has {n} spin(s). "
                "Use the spin-count slider to resize rather than editing rows directly."
            )
        shifts.append(_cell_value(row[i + 1], i, i + 1))
        for k in range(i + 1, n):
            couplings[i, k] = _cell_value(row[k + 1], i, k + 1)
    return shifts, couplings


def _resize_matrix(rows: list[list[object]], n_spins: int) -> list[list[object]]:
    """Rebuild the grid at ``n_spins`` rows, keeping every value that still has a home.

    Rebuilding from scratch on each slider step is simpler and throws away a half-entered system on
    a mis-click, so the overlap is copied across instead. New spins arrive at 0 ppm and uncoupled.
    """
    grid: list[list[object]] = []
    for i in range(n_spins):
        row: list[object] = [_spin_label(i)]
        for k in range(n_spins):
            keep = i < len(rows) and (k + 1) < len(rows[i])
            row.append(rows[i][k + 1] if keep else 0.0)
        grid.append(row)
    return grid


def _phenotype_grid(name: str) -> tuple[list[list[object]], list[list[object]]]:
    """Populate the matrix and the per-group width table from a named phenotype.

    The dropdown pre-fills the grid rather than feeding the simulator, so the matrix stays the one
    description of what is being simulated and a preset is just a starting point the user can edit.
    """
    pheno = sp.PHENOTYPES[name]
    shifts = [float(s) for s in pheno["shifts_ppm"]]
    matrix = sp.build_coupling_matrix(len(shifts), pheno["couplings"])

    rows: list[list[object]] = []
    for i, shift in enumerate(shifts):
        row: list[object] = [_spin_label(i)]
        for k in range(len(shifts)):
            row.append(shift if k == i else float(matrix[i, k]) if k > i else 0.0)
        rows.append(row)
    return rows, _width_rows(shifts, matrix, [float(w) for w in pheno["widths_hz"]])


def _width_rows(
    shifts: list[float], couplings: NDArray[np.float64], widths: list[float] | None = None
) -> list[list[object]]:
    """One editable line width per **spin system**: ``system | n H | FWHM (Hz)``.

    Per coupling block, not per ground-truth group, because that is the finest grain the simulator
    can honour. ``simulate`` collapses widths to a single mean *within* a block, so two groups of one
    coupled system are physically incapable of carrying different line shapes — for ethyl, widths of
    (1, 1, 1, 3, 3) produce a spectrum bit-identical to a uniform 1.8. Offering a row per group would
    put two controls on screen that silently average into one. Splitting blocks apart is what made
    per-system widths real in the first place.
    """
    rows: list[list[object]] = []
    for block in coupling_blocks(couplings):
        default = float(widths[block[0]]) if widths is not None and block[0] < len(widths) else 1.0
        rows.append([_block_label(block), _block_shifts(shifts, block), len(block), default])
    return rows


def _block_label(block: list[int]) -> str:
    """Name a spin system by its member spins — "A, B" — which is the key widths are matched on.

    Deliberately the spin letters and not the shifts. The label has to survive a shift edit, or
    retyping δ silently resets that system's line width; and it has to *change* when a coupling
    merges two systems, because the width then belongs to a system that no longer exists. Membership
    does exactly that; a shift list does neither.
    """
    return ", ".join(_spin_label(i) for i in block)


def _block_shifts(shifts: list[float], block: list[int]) -> str:
    """The shifts a system covers, down-field first — shown so the row is readable, never keyed on."""
    members = sorted({round(float(shifts[i]), 4) for i in block}, reverse=True)
    return ", ".join(f"{s:g}" for s in members)


def _widths_per_spin(
    shifts: list[float], couplings: NDArray[np.float64], width_rows: list[list[object]]
) -> list[float]:
    """Expand the per-system width table back to the one-entry-per-spin list `simulate` wants.

    Rows are matched to systems by **label**, not by position. Typing a coupling into the matrix
    merges two blocks without rebuilding the table, so a positional match hands row 2's width to
    whatever block now sits second — a different set of spins, with the label on screen contradicting
    the value applied. An unmatched system falls back to 1.0, which is visibly wrong rather than
    quietly wrong.
    """
    by_label: dict[str, float] = {}
    for n, row in enumerate(width_rows):
        if len(row) < 4:
            raise ValueError(
                f"line-width row {n + 1} is missing its FWHM value. "
                "Use the spin-count slider to rebuild the table."
            )
        by_label[str(row[0])] = _cell_value(row[3], n, 3)

    per_spin = [1.0] * len(shifts)
    for block in coupling_blocks(couplings):
        width = by_label.get(_block_label(block), 1.0)
        for spin in block:
            per_spin[spin] = width
    return per_spin


def _join_systems(
    first: tuple[list[float], NDArray[np.float64], list[float]],
    second: tuple[list[float], NDArray[np.float64], list[float]],
) -> tuple[list[float], NDArray[np.float64], list[float]]:
    """Lay two independent spin systems out on one block-diagonal coupling matrix.

    The cross terms stay zero, which is exactly what `coupling_blocks` reads as "separate systems",
    so the joined matrix goes through the unchanged `simulate_systems` path: each block simulated on
    a per-proton scale and summed under a single global peak rescale. That equivalence is what makes
    two editors and one matrix the same computation (`test_simulate_additivity`), and it only holds
    while the rescale happens **once, at the end** — adding two separately peak-normalised spectra
    would silently flatten the relative integrals.

    Widths are expanded to per-spin **before** joining, so the result never depends on the order
    `coupling_blocks` happens to return the combined blocks in. Matching a concatenated width table
    positionally against those blocks is the one way this could hand a system the wrong line shape.

    An empty second system needs no special case and does not have one: `joined[n:, n:]` is then a
    0×0 slice and the concatenations are no-ops, so the result *is* the first system. An earlier
    version guarded that explicitly; deleting the guard changed no test, which is how it was found
    to be dead. `test_two_spin_systems` pins the behaviour rather than the branch.
    """
    (shifts, couplings, widths), (shifts2, couplings2, widths2) = first, second

    n, n2 = len(shifts), len(shifts2)
    joined = np.zeros((n + n2, n + n2))
    joined[:n, :n] = couplings
    joined[n:, n:] = couplings2
    return shifts + shifts2, joined, widths + widths2


def _build_gt_groups(
    shifts: list[float], couplings_hz: NDArray[np.float64] | list[list[float]]
) -> list[dict]:
    """Group equivalent spins into ground-truth multiplets (shift, proton count, max J).

    ``max_j_hz`` is read **per group from the coupling matrix**: the largest coupling from any spin
    in the group to any spin outside it. Couplings *within* a group are excluded deliberately —
    equivalent protons may carry a mutual J, but it produces no observable splitting, so reporting it
    would claim a multiplet the spectrum does not show.

    Spins are grouped by **shift alone**, deliberately, and not by coupling block. Ground truth here
    describes what the spectrum shows, because it is compared against a detector that sees only the
    spectrum: protons sharing a shift produce one peak carrying their combined area whether or not a
    coupling connects them. Grouping by block would split a methoxy — three equivalent uncoupled
    protons — into three 1H multiplets where the spectrum has a single 3H line. See
    ``tests/test_gt_groups_from_matrix.py`` for the measured case and the one known limitation.

    Groups are ordered down-field first, matching how the spectrum is read and how the comparison
    table is numbered.
    """
    given = np.asarray(couplings_hz, dtype=float)
    # Mirror `coupling_blocks`: the upper triangle is the contract, so both read the same couplings.
    upper = np.triu(given, 1)
    j = upper + upper.T

    groups: dict[float, list[int]] = {}
    for idx, shift in enumerate(shifts):
        groups.setdefault(round(float(shift), 4), []).append(idx)

    gt = []
    for shift_val, idxs in sorted(groups.items(), reverse=True):
        members = set(idxs)
        outside = [
            abs(float(j[i, k]))
            for i in idxs
            for k in range(j.shape[0])
            # Same tolerance the block decomposition uses, so a stray 1e-15 cannot be "absent" to
            # one and a reported coupling to the other.
            if k not in members and abs(float(j[i, k])) > COUPLING_EPS_HZ
        ]
        gt.append(
            {
                "shift_ppm": shift_val,
                "proton_count": len(idxs),
                "max_j_hz": max(outside) if outside else None,
            }
        )
    return gt


def _predicted_max_j(pred: dict) -> float | None:
    """The model's single predicted coupling, or ``None`` when it emitted none.

    ``coupling_constants_hz`` holds **0 or 1** entries by construction -- ``PARAM_NAMES[3:7]`` are
    ``[sum, min, max, std]`` of the multiset, not four separate J values -- so there is one number to
    show, not a set. Mirrors ``sp._comparison_row``, which has had this column all along while the
    GUI table showed ``GT J`` with no predicted counterpart to compare it against.
    """
    js = pred.get("coupling_constants_hz") or []
    return float(js[0]) if len(js) else None


def _comparison_dataframe(gt_groups: list[dict], preds: list[dict]) -> pd.DataFrame:
    """GT-vs-detected table with a graded status and explicit error columns.

    Each GT group is paired with its nearest-δ prediction (``match_to_gt``); predictions matched to
    no GT are appended as spurious rows.

    ``status`` grades the **chemical shift alone** -- ``✓ excellent`` · ``✓ good`` · ``✓ ok`` ·
    ``~ fair`` · ``✗ off``, see :mod:`app_ui.grading`. It used to be conjunctive
    (``dd_hz <= 2.0 and dh == 0``), which forced ``~ off`` on a proton-count mismatch *at zero shift
    error*. Proton count now travels in ``ΔH`` and the coupling in ``ΔJ (Hz)``, each reported on its
    own terms; ``✗ missed`` and ``+ extra`` are unchanged, being the absence of a pairing rather than
    a grade. These mirror the connector / marker colours in :func:`plotting.comparison_figure`.
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
                    "status": grading.MISSED,
                    "pred δ (ppm)": "–",
                    "pred H": "–",
                    "pred J (Hz)": "–",
                    "Δδ (Hz)": "–",
                    "ΔH": "–",
                    "ΔJ (Hz)": "–",
                    "conf": "–",
                }
            )
        else:
            dd_hz = abs(float(pred["chemical_shift_ppm"]) - gt["shift_ppm"]) * sp.BASE_FREQ_MHZ
            dh = int(pred["proton_count"]) - gt["proton_count"]
            pred_j = _predicted_max_j(pred)
            dj = (
                None if (pred_j is None or gt["max_j_hz"] is None) else abs(pred_j - gt["max_j_hz"])
            )
            row.update(
                {
                    "status": grading.grade_shift(dd_hz),
                    "pred δ (ppm)": f"{float(pred['chemical_shift_ppm']):.3f}",
                    "pred H": int(pred["proton_count"]),
                    "pred J (Hz)": "–" if pred_j is None else f"{pred_j:.1f}",
                    "Δδ (Hz)": f"{dd_hz:.2f}",
                    "ΔH": f"{dh:+d}",
                    "ΔJ (Hz)": "–" if dj is None else f"{dj:.2f}",
                    "conf": f"{float(pred['confidence']):.2f}",
                }
            )
        rows.append(row)
    for k, pred in enumerate((p for p in preds if id(p) not in matched_ids), len(rows) + 1):
        pred_j = _predicted_max_j(pred)
        rows.append(
            {
                "#": k,
                "status": grading.EXTRA,
                "GT δ (ppm)": "–",
                "GT H": "–",
                "GT J (Hz)": "–",
                "pred δ (ppm)": f"{float(pred['chemical_shift_ppm']):.3f}",
                "pred H": int(pred["proton_count"]),
                "pred J (Hz)": "–" if pred_j is None else f"{pred_j:.1f}",
                "Δδ (Hz)": "–",
                "ΔH": "–",
                "ΔJ (Hz)": "–",
                "conf": f"{float(pred['confidence']):.2f}",
            }
        )
    return pd.DataFrame(rows)


def _simulate_distort_kwargs(
    add_noise: bool,
    snr: float,
    phase0: float,
    broaden: float,
    baseline: float,
    satellites: bool,
    sat_j: float,
) -> dict[str, float]:
    """Assemble ``distort`` kwargs from the sliders (neutral / identity values are skipped).

    ``satellites`` defaults **on** because training applied ¹³C satellites unconditionally --
    ``augment_distortions`` calls ``add_13C_satellites_with_variability`` on every sample, with no
    coin toss and no custom-values path. Leaving them off produced Simulate spectra systematically
    cleaner than anything the model was trained on, and the tab offered no control to fix it.
    ``distort`` treats them as opt-in (they apply only when a parameter is supplied), which is the
    inverse of training semantics, so parity has to be asserted here by the caller.
    """
    dk: dict[str, float] = {}
    if add_noise:
        dk["noise_snr_log10"] = float(snr)
    if satellites:
        dk["sat_j_hz"] = float(sat_j)
        dk["sat_intensity"] = 0.01  # midpoint of the trained 0.005-0.015 range
    if float(phase0) != 0.0:
        dk["phase0_deg"] = float(phase0)
    if float(broaden) > 0.0:
        dk["broaden_hz"] = float(broaden)
    if float(baseline) > 0.0:
        dk["baseline"] = float(baseline)
    return dk


#: What ``_simulate_stage`` hands to ``_detect_stage``: the clean **complex** spectrum plus the ppm
#: axis, phenotype label and ground-truth groups — everything needed to distort, detect and compare
#: without paying the 2**n eigendecomposition again.
SimCache = dict[str, Any]


def _simulate_stage(
    matrix_rows: list[list[object]],
    width_rows: list[list[object]],
    second_enabled: bool = False,
    matrix_rows2: list[list[object]] | None = None,
    width_rows2: list[list[object]] | None = None,
) -> SimCache | str:
    """Run the spin dynamics once and return everything the cheap stage needs.

    The spin system comes entirely from the matrix grid — shifts on the diagonal, couplings in the
    upper triangle — so there is one description of what is being simulated rather than a phenotype
    and an edited copy of it that can disagree. The preset dropdown fills the grid and then has no
    further say.

    A second, independent system can be supplied from its own grid. It is **off unless asked for**:
    the panel's controls always hold real values, so gating on `second_enabled` rather than on the
    grid being empty is what keeps every single-system caller — including the positional ones — bit
    identical to before this existed.

    The parameters here carry defaults because this helper is wired to no Gradio event; only
    `simulate_to_state` is, and its arity is checked against the wired input list.

    Returns a cache dict, or a **string** carrying the user-facing error. The string channel keeps
    the two stages composable into the original single-return-shape callback.

    The cached spectrum is the *clean* one, exactly as ``simulate_systems`` produced it, because
    re-distorting an already distorted spectrum would compound the effects as a slider is dragged.

    It is **real**: ``simulate_systems`` sums Lorentzian absorption lines and never forms an
    analytic signal, so there is no dispersion component to preserve or to strip. ``distort``
    documents a complex input and this path has always handed it a real one, on ``main`` as here, so
    the phase controls rotate a spectrum with no imaginary part rather than the analytic signal
    training used. That is a pre-existing limitation of the tab, noted for the distortion pass, not
    something this cache introduces.
    """
    if not Path(CHECKPOINT).exists():
        return (
            f"Checkpoint not found at `{CHECKPOINT}`. "
            "Download it from Zenodo (10.5281/zenodo.21217102) into `moldetr/model/`."
        )
    try:
        shifts, couplings = _matrix_to_system(matrix_rows)
        widths = _widths_per_spin(shifts, couplings, width_rows)
    except ValueError as exc:
        return f"Invalid spin matrix: {exc}"
    if second_enabled:
        try:
            shifts2, couplings2 = _matrix_to_system(matrix_rows2 or [])
            widths2 = _widths_per_spin(shifts2, couplings2, width_rows2 or [])
        except ValueError as exc:
            # Named, because two grids are on screen and "row 1" alone would be ambiguous.
            return f"Invalid second spin matrix: {exc}"
        shifts, couplings, widths = _join_systems(
            (shifts, couplings, widths), (shifts2, couplings2, widths2)
        )
    if not shifts:
        return "Add at least one spin to the matrix."
    try:
        # simulate_systems, not simulate: it splits the coupling matrix into independent blocks,
        # simulates each on a per-proton scale and sums them, so several spin systems in one window
        # keep the right relative integrals (one proton = one unit of area, everywhere). The default
        # "peak" rescale then restores max = 1 before distortion, which is what the distortion
        # magnitudes are calibrated against.
        spectrum, ppm_axis = simulate_systems(
            shifts,
            couplings,
            widths,
            sp.BASE_FREQ_MHZ,
            sp.LEFT_PPM,
            sp.RIGHT_PPM,
            sp.N_POINTS,
        )
    except ValueError as exc:
        return f"Invalid parameters: {exc}"
    blocks = coupling_blocks(couplings)
    return {
        "label": f"{len(shifts)} spin(s) in {len(blocks)} system(s)",
        "spectrum": spectrum,
        "ppm_axis": ppm_axis,
        "gt_groups": _build_gt_groups(shifts, couplings),
    }


def _distorted_amplitudes(
    cache: SimCache,
    add_noise: bool,
    snr: float,
    phase0: float,
    broaden: float,
    baseline: float,
    satellites: bool,
    sat_j: float,
) -> NDArray[np.float64]:
    """Apply the distortions to the cached clean spectrum and return real amplitudes.

    Always returns a fresh array, so ``cache`` survives any number of slider moves. ``distort``
    copies its input, but with every distortion at its neutral value it is skipped entirely and
    ``np.asarray(np.real(...))`` is a no-op on a real float64 array — it would hand back the cached
    array itself, and one in-place edit downstream would corrupt every later re-distortion.
    """
    spectrum = cache["spectrum"]
    dk = _simulate_distort_kwargs(add_noise, snr, phase0, broaden, baseline, satellites, sat_j)
    if dk:
        spectrum = distort(spectrum, cache["ppm_axis"], **dk)
    return np.array(np.real(spectrum), dtype=float, copy=True)


def _detect_stage(
    cache: SimCache | str,
    add_noise: bool,
    snr: float,
    phase0: float,
    broaden: float,
    baseline: float,
    threshold: float,
    satellites: bool,
    sat_j: float,
) -> tuple[pd.DataFrame | None, object | None, str]:
    """Distort, detect and compare — everything that must re-run when a slider moves, and no more.

    Accepts the error string ``_simulate_stage`` returns instead of a cache and passes it straight
    through, so a stage driven directly from a stored cache (rather than through
    ``simulate_and_detect``) still shows the message rather than indexing a ``str``.
    """
    if isinstance(cache, str):
        return None, None, cache
    try:
        amplitudes = _distorted_amplitudes(
            cache, add_noise, snr, phase0, broaden, baseline, satellites, sat_j
        )
    except ValueError as exc:
        return None, None, f"Invalid parameters: {exc}"
    label = cache["label"]
    gt_groups = cache["gt_groups"]
    try:
        model = _get_model()
    except RuntimeError as exc:  # the checkpoint trust gate; its text names the only remedy
        return None, None, _checkpoint_error_message(exc)

    preds = decode_predictions(
        # Skip the in-model noise floor when the user has already added calibrated noise here.
        # The floor (0.005 * max) exists to drag a perfectly clean FFT-resampled spectrum back
        # in-distribution; once "Add noise" is on, that job is done and the floor only masks the
        # slider. The two are directly comparable -- distort's std is max/(2*SNR) -- and they are
        # equal only at the slider's 2.0 minimum, so at the 3.0 default the requested noise sits
        # 10x under the floor and at 5.0 it is 1000x under. That is why the slider felt inert.
        # Detect (and predict.py) deliberately keep the floor: they feed the frozen decode.
        run(model, amplitudes, noise_frac=0.0 if add_noise else 0.005),
        load_extrema(EXTREMA),
        sp.POINTS_PER_HZ,
        ppm_left=sp.LEFT_PPM,
        ppm_right=sp.RIGHT_PPM,
        threshold=threshold,
    )
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
    )
    table = _comparison_dataframe(gt_groups, preds)
    n_match = sum(1 for _g, p in matched if p is not None)
    msg = (
        f"**Simulated {label}**: {len(gt_groups)} ground-truth multiplet(s); the model "
        f"**detected** {len(preds)} ({n_match} matched, {len(spurious)} spurious). "
        "Teal ▽ = ground truth · clay ● = model detection; a connector turns **green** within "
        "tolerance and **amber** when off. Missed GT and spurious peaks are outlined in red."
    )
    return table, fig, msg


def simulate_and_detect(
    matrix_rows: list[list[object]],
    width_rows: list[list[object]],
    add_noise: bool,
    snr: float,
    phase0: float,
    broaden: float,
    baseline: float,
    threshold: float,
    satellites: bool,
    sat_j: float,
    second_enabled: bool = False,
    matrix_rows2: list[list[object]] | None = None,
    width_rows2: list[list[object]] | None = None,
) -> tuple[pd.DataFrame | None, object | None, str]:
    """Simulate the matrix, optionally distort, detect, and compare to ground truth.

    Kept as a thin composition of the two stages, so the one-shot and cached paths cannot drift.

    It is the **direct-call** entry point — for tests and library users — and is wired to no event.
    The button runs :func:`simulate_to_state`, which merely carries `api_name="simulate_and_detect"`;
    that shared name is why an earlier version of this docstring claimed the Gradio event addressed
    this function. It does not, and the distinction is load-bearing: because `test_ui_graph`'s arity
    check only inspects *wired* callbacks, the second-system arguments can carry defaults here while
    `simulate_to_state`'s must not — which is what leaves every existing positional caller binding
    exactly as before.
    """
    cache = _simulate_stage(matrix_rows, width_rows, second_enabled, matrix_rows2, width_rows2)
    return _detect_stage(
        cache, add_noise, snr, phase0, broaden, baseline, threshold, satellites, sat_j
    )


def preset_grid(name: str) -> tuple[list[list[object]], list[list[object]], int]:
    """Fill the matrix and width table from a preset, and resize the spin slider to match."""
    rows, widths = _phenotype_grid(name)
    return rows, widths, len(rows)


def resize_spin_matrix(
    matrix_rows: list[list[object]], width_rows: list[list[object]], n_spins: float
) -> tuple[list[list[object]], list[list[object]]]:
    """Grow or shrink the matrix, keeping what was typed in **both** grids.

    The width table is re-derived rather than resized alongside the matrix, because its rows are
    spin *systems*: adding a spin coupled to an existing one enlarges a system instead of adding a
    row. Widths whose system survives the resize are carried across by label, so the count slider is
    not a way to lose them.
    """
    rows = _resize_matrix(matrix_rows, int(n_spins))
    try:
        shifts, couplings = _matrix_to_system(rows)
        rebuilt = _width_rows(shifts, couplings)
        carried = _widths_per_spin(shifts, couplings, width_rows)
    except (ValueError, IndexError):
        # A bad cell is reported when Simulate is pressed. Leave the width table untouched rather
        # than replacing it with an empty one, which silently reset every width to the default.
        return rows, width_rows
    for row, block in zip(rebuilt, coupling_blocks(couplings)):
        row[3] = carried[block[0]]
    return rows, rebuilt


def invalidate_cache() -> None:
    """Drop the cached spectrum without touching the grids.

    Used by the width table, whose edits change the spectrum but not which spin systems exist, so
    there is nothing to re-derive — unlike a matrix edit, which can merge two systems into one.
    """
    return None


def matrix_edited(
    matrix_rows: list[list[object]], width_rows: list[list[object]]
) -> tuple[None, list[list[object]]]:
    """React to a matrix edit: drop the cached spectrum and re-derive the width table.

    Two things go stale the moment a cell changes. The **cache** must go, or a slider re-distorts
    whatever was last simulated — edit five spins down to two, drag the phase slider, and the plot
    re-renders the old five, still labelled as five, beside a matrix saying otherwise. Clearing beats
    re-simulating on every keystroke, which is the ``2**n`` cost the cache exists to avoid.

    The **width table** goes stale differently: typing a coupling merges two systems into one, so the
    rows no longer describe the systems that exist. Rebuilding here keeps the screen honest, and
    widths whose system survived the edit are carried across by label.
    """
    try:
        shifts, couplings = _matrix_to_system(matrix_rows)
        rebuilt = _width_rows(shifts, couplings)
        carried = _widths_per_spin(shifts, couplings, width_rows)
    except (ValueError, IndexError):
        # A half-typed cell is reported when Simulate is pressed. Leave the table alone rather than
        # destroying the user's widths mid-edit.
        return None, width_rows
    for row, block in zip(rebuilt, coupling_blocks(couplings)):
        row[3] = carried[block[0]]
    return None, rebuilt


def simulate_to_state(
    matrix_rows: list[list[object]],
    width_rows: list[list[object]],
    add_noise: bool,
    snr: float,
    phase0: float,
    broaden: float,
    baseline: float,
    threshold: float,
    satellites: bool,
    sat_j: float,
    second_enabled: bool,
    matrix_rows2: list[list[object]],
    width_rows2: list[list[object]],
) -> tuple[SimCache | str, pd.DataFrame | None, object | None, str]:
    """Simulate once, hand the cache back for `gr.State`, and render the first result.

    The cache is returned alongside the outputs so a later slider move can re-distort it without
    paying the eigendecomposition again. It is deliberately the *same* cache
    :func:`simulate_and_detect` builds internally, so the live path and the one-shot API path cannot
    describe different spectra.

    Every parameter is **required**, including the three second-system ones appended last:
    `test_ui_graph` asserts that this function's no-default parameter count equals the wired input
    list, and a defaulted parameter would wire a control Gradio then never passes. The client can
    still omit the tail — gradio_client fills missing trailing arguments from the *components'*
    values, which is why the second panel shipping switched off keeps the 8-argument call honest.
    """
    cache = _simulate_stage(matrix_rows, width_rows, second_enabled, matrix_rows2, width_rows2)
    table, fig, msg = _detect_stage(
        cache, add_noise, snr, phase0, broaden, baseline, threshold, satellites, sat_j
    )
    return cache, table, fig, msg


def redistort(
    cache: SimCache | str | None,
    add_noise: bool,
    snr: float,
    phase0: float,
    broaden: float,
    baseline: float,
    threshold: float,
    satellites: bool,
    sat_j: float,
) -> tuple[pd.DataFrame | None, object | None, str]:
    """Re-apply the distortions to an already-simulated spectrum held in `gr.State`.

    This is the whole point of splitting the stages: dragging a distortion slider costs one
    `distort` plus one forward pass, not another ``2**n`` eigendecomposition. An empty state means
    the user moved a slider before pressing Simulate, which is a prompt rather than an error.
    """
    if cache is None:
        return None, None, "Press **Simulate & Predict** first, then move the distortion sliders."
    return _detect_stage(
        cache, add_noise, snr, phase0, broaden, baseline, threshold, satellites, sat_j
    )


CONTRACT = (
    f"**Expected input.** A 1-D ¹H spectrum of **{INPUT_LENGTH} points** at **{POINTS_PER_HZ} "
    "points/Hz** (a 1200 Hz window), real-valued. Absolute intensity does not matter (each spectrum "
    "is min–max normalised), but relative intensities, SNR and line shape do. Every coupling partner "
    "of an in-window peak must also sit inside the window; see "
    "[`docs/INPUT_FORMAT.md`](docs/INPUT_FORMAT.md)."
)

# Field-agnostic scope note (matches docs/SCOPE.md and the README callout). Named SCOPE_NOTE, not
# PROTOTYPE: the body stopped hedging about "research prototypes" and "well-resolved spectra" some
# time ago, but the name, the accordion title, and the header chip kept the old framing alive.
SCOPE_NOTE = (
    "MolDeTr handles congested, strongly-coupled ¹H NMR spectra and is largely field-agnostic "
    "(it works in Hz; tested on 80–600 MHz). Predictions can deviate for inputs outside its trained "
    "regime: unusual distortions, non-standard pulse sequences or processing, mixtures, or windows "
    "wider than 1200 Hz. **max J** is the dominant coupling per multiplet (the full set is in the "
    "committed `structured_output` path). Sanity-check predictions against your own chemistry."
)

FOOTNOTE = (  # NEW
    "max J = largest coupling per multiplet; the full coupling set comes from the committed "
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
        gr.HTML(HEADER_HTML)  # BRAND: wordmark · eyebrow · links
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
                        with gr.Accordion("Scope & limits", open=False):  # BRAND
                            gr.Markdown(SCOPE_NOTE)
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
                            label="Annotated spectrum (drag to zoom, double-click resets)",
                            elem_id="md-plot",
                        )
            with gr.Tab("Simulate"):
                gr.Markdown(SIMULATE_INTRO)
                with gr.Row():
                    with gr.Column(scale=2):
                        _grid, _widths = _phenotype_grid("ethyl")
                        _grid2, _widths2 = _phenotype_grid(SECOND_SYSTEM_PRESET)
                        sim_cache = gr.State(None)
                        sim_phenotype = gr.Dropdown(
                            PHENOTYPE_CHOICES,
                            value="ethyl",
                            label="Start from a known spin system",
                            elem_id="sim-preset",
                        )
                        sim_n_spins = gr.Slider(
                            1,
                            MAX_MATRIX_SPINS,
                            value=len(_grid),
                            step=1,
                            label="Number of spins",
                            elem_id="sim-nspins",
                        )
                        sim_matrix = gr.Dataframe(
                            value=_grid,
                            headers=["spin", *(_spin_label(i) for i in range(MAX_MATRIX_SPINS))],
                            datatype=_matrix_datatype(MAX_MATRIX_SPINS),
                            type="array",
                            label="Spin matrix — δ in ppm on the diagonal · J in Hz above it",
                            elem_id="sim-matrix",
                            static_columns=[0],
                        )
                        gr.Markdown(MATRIX_HINT, elem_classes="md-footnote")
                        sim_widths = gr.Dataframe(
                            value=_widths,
                            headers=["system", "δ (ppm)", "n H", "FWHM (Hz)"],
                            datatype=["str", "str", "number", "number"],
                            type="array",
                            label="Line width FWHM per spin system (coupled spins share one)",
                            elem_id="sim-widths",
                        )
                        # Collapsed and switched off, so the tab opens exactly as it did before.
                        # The controls hold real values throughout — the checkbox, not an emptied
                        # grid, is what says "one system", which keeps the grid always well formed.
                        with gr.Accordion("Second spin system (optional)", open=False):
                            sim_second_enabled = gr.Checkbox(
                                value=False,
                                label="Simulate a second, independent spin system",
                                info=(
                                    "Summed with the first at matching per-proton integrals — the "
                                    "same result as leaving their couplings at 0 in one matrix."
                                ),
                                elem_id="sim-second-enabled",
                            )
                            sim_phenotype2 = gr.Dropdown(
                                PHENOTYPE_CHOICES,
                                value=SECOND_SYSTEM_PRESET,
                                label="Start the second system from a known spin system",
                                elem_id="sim-preset-2",
                            )
                            sim_n_spins2 = gr.Slider(
                                1,
                                MAX_MATRIX_SPINS,
                                value=len(_grid2),
                                step=1,
                                label="Number of spins (second system)",
                                elem_id="sim-nspins-2",
                            )
                            sim_matrix2 = gr.Dataframe(
                                value=_grid2,
                                headers=[
                                    "spin",
                                    *(_spin_label(i) for i in range(MAX_MATRIX_SPINS)),
                                ],
                                datatype=_matrix_datatype(MAX_MATRIX_SPINS),
                                type="array",
                                label="Second spin matrix — δ in ppm on the diagonal · J in Hz above it",
                                elem_id="sim-matrix-2",
                                static_columns=[0],
                            )
                            sim_widths2 = gr.Dataframe(
                                value=_widths2,
                                headers=["system", "δ (ppm)", "n H", "FWHM (Hz)"],
                                datatype=["str", "str", "number", "number"],
                                type="array",
                                label="Line width FWHM (second system)",
                                elem_id="sim-widths-2",
                            )
                        gr.Markdown("**Distortions**: each bounded to the model's trained range.")
                        with gr.Row():
                            # Default ON: training applied satellites to every spectrum, so leaving
                            # them off makes this tab's output cleaner than anything the model saw.
                            sim_satellites = gr.Checkbox(
                                value=True,
                                label="¹³C satellites",
                                info="Applied to every training spectrum; on by default for parity.",
                                elem_id="sim-satellites",
                            )
                            sim_sat_j = gr.Slider(
                                40.0, 220.0, value=130.0, step=5.0, label="¹³C satellite ¹J (Hz)"
                            )
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
                                "Within the model's training distribution: the shipped checkpoint "
                                "saw added line broadening on roughly a third of its training "
                                "spectra."
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
                            "Edit a matrix, or start from a known system, then press "
                            "**Simulate & Predict**. The distortion sliders update live afterwards."
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
        # The preset only *fills* the matrix; from then on the matrix is the sole description of
        # what gets simulated, so a preset and an edited copy of it can never disagree.
        sim_phenotype.change(
            preset_grid,
            inputs=sim_phenotype,
            outputs=[sim_matrix, sim_widths, sim_n_spins],
            api_name="phenotype_defaults",
        )
        sim_n_spins.change(
            resize_spin_matrix,
            inputs=[sim_matrix, sim_widths, sim_n_spins],
            outputs=[sim_matrix, sim_widths],
            api_name="resize_spin_matrix",
        )
        # Order must match the middle of `simulate_to_state` (the wired callback) and the tail of
        # `redistort`: the distortion sliders, then threshold, then the two satellite controls
        # (appended last so the pre-existing positional call sites keep binding). No longer the
        # *tail* of the simulate callback — the second-system controls now follow these.
        _distortions = [
            sim_add_noise,
            sim_snr,
            sim_phase0,
            sim_broaden,
            sim_baseline,
            sim_threshold,
            sim_satellites,
            sim_sat_j,
        ]
        # The second system is appended after the distortions for the same reason the satellite
        # controls were: `gradio_client` binds positionally, so anything inserted earlier would shift
        # existing callers' arguments silently instead of erroring.
        sim_btn.click(
            simulate_to_state,
            inputs=[
                sim_matrix,
                sim_widths,
                *_distortions,
                sim_second_enabled,
                sim_matrix2,
                sim_widths2,
            ],
            outputs=[sim_cache, sim_table, sim_plot, sim_status],
            api_name="simulate_and_detect",
        )
        # Editing either grid invalidates the cache, so the sliders cannot re-distort a spectrum the
        # matrix no longer describes. A matrix edit also re-derives the width table, because a new
        # coupling can merge two spin systems into one and the rows must describe the systems that
        # actually exist. Also fires when a preset or a resize rewrites the grid.
        sim_matrix.change(
            matrix_edited,
            inputs=[sim_matrix, sim_widths],
            outputs=[sim_cache, sim_widths],
            api_name="matrix_edited",
        )
        sim_widths.change(invalidate_cache, outputs=sim_cache, api_name="invalidate_on_width_edit")
        # The second panel mirrors the first exactly — same handlers, its own components. Editing it
        # invalidates the cache but never simulates, because eager simulation would make every
        # keystroke in the grid cost a 2**n eigendecomposition. (`test_browser_simulate.py` asserts
        # zero `simulate_systems` calls while the *distortion sliders* move; nothing measures these
        # controls yet, so this is a design invariant rather than a covered one.)
        sim_phenotype2.change(
            preset_grid,
            inputs=sim_phenotype2,
            outputs=[sim_matrix2, sim_widths2, sim_n_spins2],
            api_name="phenotype_defaults_2",
        )
        sim_n_spins2.change(
            resize_spin_matrix,
            inputs=[sim_matrix2, sim_widths2, sim_n_spins2],
            outputs=[sim_matrix2, sim_widths2],
            api_name="resize_spin_matrix_2",
        )
        sim_matrix2.change(
            matrix_edited,
            inputs=[sim_matrix2, sim_widths2],
            outputs=[sim_cache, sim_widths2],
            api_name="matrix_edited_2",
        )
        sim_widths2.change(
            invalidate_cache, outputs=sim_cache, api_name="invalidate_on_width_edit_2"
        )
        # The checkbox changes *what is simulated*, so it must clear the cache like the grids do.
        # Without this it reached the button's input list and nothing else: a fresh press was always
        # right, but ticking the box after simulating left the sliders re-distorting a one-system
        # spectrum while the panel above said two. Invalidating is also the safe direction for the
        # inverse case — untick after simulating and the two-system result cannot persist.
        sim_second_enabled.change(
            invalidate_cache, outputs=sim_cache, api_name="invalidate_on_second_toggle"
        )
        # `.release` rather than `.change`: a slider fires continuously while dragged, and each
        # event costs a forward pass. `always_last` keeps the final position authoritative when
        # events are dropped mid-drag, so the view never settles on a stale value.
        # These do get endpoint ids, one per control — Gradio derives one for every wiring, and
        # `api_name=False` becomes the literal "false", "false_1", ..., which is the auto-derived
        # surface the graph tests exist to prevent. They are named but not *useful* to a
        # `gradio_client` caller: the cache lives in `gr.State`, which such a caller cannot hold, so
        # calling one only ever returns the "press Simulate first" prompt. `/simulate_and_detect`
        # remains the one programmatic entry point.
        # Sliders fire on `.release` so a drag costs one re-distort at the end rather than one per
        # pixel; the checkbox has no release event, so it re-distorts on change. Listed explicitly
        # rather than derived from the control type, which reads better and keeps the pairing
        # visible next to the names the endpoints take.
        _live_triggers = [
            (sim_add_noise.change, "noise"),
            (sim_snr.release, "snr"),
            (sim_phase0.release, "phase"),
            (sim_broaden.release, "broaden"),
            (sim_baseline.release, "baseline"),
            (sim_threshold.release, "threshold"),
            (sim_satellites.change, "satellites"),
            (sim_sat_j.release, "sat_j"),
        ]
        for _trigger, _name in _live_triggers:
            _trigger(
                redistort,
                inputs=[sim_cache, *_distortions],
                outputs=[sim_table, sim_plot, sim_status],
                # Named per control rather than shared: Gradio derives an endpoint id for every
                # wiring, and `api_name=False` becomes the literal "false", "false_1", ... which is
                # exactly the auto-derived surface the graph test exists to prevent.
                api_name=f"redistort_{_name}",
                trigger_mode="always_last",
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
