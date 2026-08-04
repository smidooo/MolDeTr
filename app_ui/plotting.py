"""Interactive Plotly spectrum for the MolDeTr GUI — brand-styled, zoomable.

Replaces matplotlib in the *GUI path only*: drag to box-zoom, double-click to reset,
Plotly re-ticks the axis for any zoom level. `moldetr/visualization.py` remains the
renderer for `predict.py --plot` (static PNG export).

Exports:
    spectrum_figure(amplitudes, predictions, ppm_left=None, ppm_right=None,
                    ground_truth=None) -> plotly.graph_objects.Figure
    assignment_rows(predictions, ppm) -> list[dict]   (table rows, safe δ headers)

Requires `plotly` (in the `app` extra of pyproject.toml and deploy/requirements-demo.txt).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]  # plotly ships no type stubs

MARKER_COLORS = ["#2566b0", "#e08a1f", "#1f9e8c"]  # multiplet 1/2/3 cycle (paper tricolor)
INK = "#20242b"
MUTE = "#5b6675"
GRID = "#edf1f7"
SPINE = "#9db0c6"
TRUTH = "#5b6675"
FONT = "IBM Plex Sans, ui-sans-serif, system-ui, sans-serif"


def _shift_to_x(pts, n, ppm_left, ppm_right):
    if ppm_left is not None and ppm_right is not None and n > 1:
        return ppm_left + (pts / (n - 1)) * (ppm_right - ppm_left)
    return pts


def _apex(amp, xi, half=18):
    lo, hi = max(0, xi - half), min(amp.shape[-1], xi + half + 1)
    return float(amp[lo:hi].max()) if hi > lo else float(amp.max())


def assignment_rows(predictions, ppm) -> list[dict]:
    """Assignment table rows. Headers are pre-uppercased LITERALS with lowercase δ —
    never uppercase these programmatically ("δ".upper() == "Δ" = difference)."""
    shift_col = "δ (PPM)" if ppm else "δ (HZ)"
    out = []
    for i, p in enumerate(predictions, 1):
        js = p.get("coupling_constants_hz") or []
        shift = p.get("chemical_shift_ppm") if ppm else p.get("chemical_shift_hz")
        lw = p.get("linewidth_hz")
        shift_str = "–" if shift is None else (f"{shift:.3f}" if ppm else f"{shift:.1f}")
        out.append(
            {
                "#": i,
                "PROTONS": f"{p['proton_count']} H",
                shift_col: shift_str,
                "MAX J (HZ)": f"{js[0]:.1f}" if js else "–",
                "LINE WIDTH (HZ)": f"{lw:.2f}" if lw is not None else "–",
            }
        )
    return out


# ── GT-vs-prediction comparison palette (Okabe-Ito-adjacent; verify colorblind contrast) ──
GT_INK = "#1f9e8c"  # teal  — ground truth
PRED_INK = "#e08a1f"  # clay  — prediction
MATCH_OK = "#2e9e5b"  # green — matched within tolerance
MATCH_OFF = "#d9a520"  # amber — matched but off
MISS_INK = "#d1495b"  # red   — missed GT / spurious prediction


def _ppm_to_index(ppm_val, n, ppm_left, ppm_right):
    """Nearest spectrum index for a ppm value (clamped to the array)."""
    if ppm_left == ppm_right or n <= 1:
        return 0
    frac = (ppm_val - ppm_left) / (ppm_right - ppm_left)
    return int(round(min(max(frac, 0.0), 1.0) * (n - 1)))


def comparison_figure(
    amplitudes, matched, spurious=None, *, ppm_left, ppm_right, base_freq_mhz, tol_hz=2.0
):
    """Ground-truth-vs-prediction spectrum with an intuitive match / miss / spurious overlay.

    ``matched`` is the ``match_to_gt`` output: a list of ``(gt, pred_or_None)`` where ``gt`` has
    ``shift_ppm``/``proton_count`` and ``pred`` has ``chemical_shift_ppm``/``proton_count``/
    ``confidence``. ``spurious`` is the predictions matched to no GT (false positives). Encoding:
    ground truth = teal ▽ on a lane above the spectrum; prediction = clay ● (opacity ∝ confidence);
    a connector links each matched pair, **green** when |Δδ| ≤ ``tol_hz`` else **amber**; a missed GT
    is a red-outlined ▽; a spurious prediction is a red ● with a dashed ring. x-axis is ppm.
    """
    spurious = spurious or []
    amp = np.real(np.asarray(amplitudes, dtype=float)).ravel()
    n = amp.shape[-1]
    base, top = (float(amp.min()), float(amp.max())) if amp.size else (0.0, 1.0)
    span = (top - base) or 1.0
    gt_lane = top + 0.16 * span
    pred_lane = top + 0.30 * span

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.linspace(ppm_left, ppm_right, n),
            y=amp,
            mode="lines",
            line=dict(color=INK, width=1.1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    for gt, pred in matched:
        gx = float(gt["shift_ppm"])
        missed = pred is None
        fig.add_trace(
            go.Scatter(
                x=[gx],
                y=[gt_lane],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=13,
                    color=GT_INK,
                    line=dict(color=MISS_INK if missed else GT_INK, width=2 if missed else 0),
                ),
                showlegend=False,
                hovertemplate=(
                    f"GT δ {gx:.3f} ppm · {gt['proton_count']}H"
                    + (" · MISSED" if missed else "")
                    + "<extra></extra>"
                ),
            )
        )
        if missed:
            continue
        px = float(pred["chemical_shift_ppm"])
        conf = float(pred.get("confidence", 1.0))
        dd_hz = abs(px - gx) * base_freq_mhz
        # Green only for a fully-correct match (position within tol AND right proton count), so the
        # connector colour agrees with the "✓ match" status in the comparison table.
        h_ok = int(pred.get("proton_count", -1)) == int(gt.get("proton_count", -2))
        conn = MATCH_OK if (dd_hz <= tol_hz and h_ok) else MATCH_OFF
        fig.add_trace(
            go.Scatter(
                x=[gx, px],
                y=[gt_lane, pred_lane],
                mode="lines",
                line=dict(color=conn, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[px],
                y=[pred_lane],
                mode="markers+text",
                text=[f"{pred['proton_count']}H"],
                textposition="top center",
                textfont=dict(size=10, color=MUTE),
                marker=dict(
                    symbol="circle", size=13, color=PRED_INK, opacity=max(0.35, min(1.0, conf))
                ),
                showlegend=False,
                hovertemplate=(
                    f"pred δ {px:.3f} ppm · {pred['proton_count']}H · conf {conf:.2f} · "
                    f"|Δδ| {dd_hz:.2f} Hz<extra></extra>"
                ),
            )
        )

    for pred in spurious:
        px = float(pred["chemical_shift_ppm"])
        fig.add_trace(
            go.Scatter(
                x=[px],
                y=[pred_lane],
                mode="markers",
                marker=dict(
                    symbol="circle-open-dot",
                    size=14,
                    color=MISS_INK,
                    line=dict(color=MISS_INK, width=2),
                ),
                showlegend=False,
                hovertemplate=(
                    f"spurious pred δ {px:.3f} ppm · {pred['proton_count']}H (no GT match)<extra></extra>"
                ),
            )
        )

    # Legend proxies — one entry per visual language, off-canvas.
    for label, color, sym in [
        ("Ground truth ▽", GT_INK, "triangle-down"),
        ("Prediction ●", PRED_INK, "circle"),
        ("matched (≤tol)", MATCH_OK, "line"),
        ("off", MATCH_OFF, "line"),
        ("missed / spurious", MISS_INK, "x"),
    ]:
        if sym == "line":
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="lines", line=dict(color=color, width=3), name=label
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(symbol=sym, size=11, color=color),
                    name=label,
                )
            )

    fig.update_layout(
        template="none",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, color=MUTE, size=12),
        margin=dict(l=54, r=16, t=16, b=48),
        showlegend=True,
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)),
        hovermode="closest",
    )
    fig.update_xaxes(
        title_text="Chemical shift δ (ppm)",
        autorange="reversed",
        showgrid=False,
        ticks="outside",
        tickcolor=SPINE,
        linecolor=SPINE,
        linewidth=1.2,
        zeroline=False,
        title_font=dict(color=MUTE),
    )
    fig.update_yaxes(
        title_text="Intensity (a.u.)",
        gridcolor=GRID,
        zeroline=False,
        showline=False,
        range=[base - 0.05 * span, pred_lane + 0.14 * span],
        title_font=dict(color=MUTE),
    )
    return fig


def spectrum_figure(
    amplitudes, predictions, ppm_left=None, ppm_right=None, ground_truth=None, points_per_hz=None
):
    """Branded interactive spectrum: ink trace, tricolor numbered markers, box zoom.

    x-axis: ppm when a calibration is given (``ppm_left``/``ppm_right``); otherwise a **Hz**
    axis when ``points_per_hz`` is known (window-relative, 0 at the left edge); else point index.
    """
    amp = np.real(np.asarray(amplitudes, dtype=float)).ravel()
    n = amp.shape[-1]
    ppm = ppm_left is not None and ppm_right is not None
    hz = (not ppm) and bool(points_per_hz)
    if ppm:
        x = np.linspace(ppm_left, ppm_right, n)
        x_of_pts = lambda pts: _shift_to_x(pts, n, ppm_left, ppm_right)  # noqa: E731
    elif hz:
        x = np.arange(n) / float(points_per_hz)  # window-relative Hz
        x_of_pts = lambda pts: pts / float(points_per_hz)  # noqa: E731
    else:
        x = np.arange(n)
        x_of_pts = lambda pts: pts  # noqa: E731
    base, top = (float(amp.min()), float(amp.max())) if amp.size else (0.0, 1.0)
    span = (top - base) or 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=amp,
            mode="lines",
            name="spectrum",
            line=dict(color=INK, width=1.1),
            hovertemplate=(
                "δ %{x:.4f} ppm" if ppm else ("ν %{x:.1f} Hz" if hz else "point %{x:.0f}")
            )
            + " · %{y:.3g}<extra></extra>",
        )
    )

    for j, g in enumerate(ground_truth or []):
        pts = g.get("chemical_shift_in_points", g.get("center_in_points", 0.0))
        gx = x_of_pts(pts)
        fig.add_shape(
            type="line",
            x0=gx,
            x1=gx,
            y0=base,
            y1=top + 0.06 * span,
            line=dict(color=TRUTH, width=1.2, dash="dash"),
            opacity=0.75,
            layer="below",
        )

    mx, my, mtext, mcolor = [], [], [], []
    for i, p in enumerate(predictions, 1):
        color = MARKER_COLORS[(i - 1) % len(MARKER_COLORS)]
        cx = x_of_pts(p.get("chemical_shift_in_points", 0.0))
        y_apex = _apex(amp, int(round(p.get("chemical_shift_in_points", 0.0))))
        y_mark = y_apex + 0.10 * span
        fig.add_shape(  # stem
            type="line",
            x0=cx,
            x1=cx,
            y0=y_apex + 0.01 * span,
            y1=y_mark - 0.035 * span,
            line=dict(color=color, width=1.2),
            opacity=0.65,
        )
        mx.append(cx)
        my.append(y_mark)
        mtext.append(str(i))
        mcolor.append(color)
    if mx:
        fig.add_trace(
            go.Scatter(
                x=mx,
                y=my,
                mode="markers+text",
                text=mtext,
                marker=dict(size=24, color=mcolor),
                textfont=dict(color="white", size=12, family=FONT),
                textposition="middle center",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        template="none",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, color=MUTE, size=12),
        margin=dict(l=54, r=16, t=16, b=48),
        showlegend=False,
        dragmode="zoom",  # drag = box zoom; double-click = reset (Plotly defaults)
        hovermode="closest",
        modebar=dict(remove=["lasso2d", "select2d", "autoScale2d"]),
    )
    fig.update_xaxes(
        title_text=(
            "Chemical shift δ (ppm)"
            if ppm
            else ("ν (Hz, window-relative)" if hz else "Point index")
        ),
        autorange="reversed" if ppm else True,  # NMR convention: ppm decreases left→right
        showgrid=False,
        ticks="outside",
        tickcolor=SPINE,
        linecolor=SPINE,
        linewidth=1.2,
        zeroline=False,
        title_font=dict(color=MUTE),
    )
    fig.update_yaxes(
        title_text="Intensity (a.u.)",
        gridcolor=GRID,
        zeroline=False,
        showline=False,
        range=[base - 0.05 * span, top + 0.20 * span],
        title_font=dict(color=MUTE),
    )
    return fig
