# MolDeTr GUI — brand handoff

Ports the **1a Branded workbench** design (see `MolDeTr Workbench.dc.html`) to the real
Gradio app. **Canonical tokens + wording live in [`BRAND.md`](BRAND.md)** — change values there
first, then propagate. Drop-in files:

| File | Goes to | What it is |
|---|---|---|
| `theme.py` | repo root (next to `app.py`) | `gr.themes.Base` subclass values + `CUSTOM_CSS` + `HEADER_HTML` |
| `plotting.py` | repo root (next to `app.py`) | interactive Plotly spectrum (box zoom, auto re-ticking) + table rows |
| `app.py` | replaces repo `app.py` | restructured `build_ui()` — header strip, input rail, exports, accordion disclaimer, Plotly plot |
| `visualization.py` | replaces `moldetr/visualization.py` | tricolor markers, brand grid/spines, safe table headers |

## Apply

```bash
cp handoff/theme.py         MolDeTr/theme.py
cp handoff/plotting.py      MolDeTr/plotting.py
cp handoff/app.py           MolDeTr/app.py
cp handoff/visualization.py MolDeTr/moldetr/visualization.py
pip install plotly          # GUI plot; also add `plotly>=5` to deploy/hf_space/requirements.txt
python app.py     # or push to the HF Space — deploys unchanged

# README redesign (optional, separate):
cp handoff/README_proposed.md MolDeTr/README.md
cp handoff/pipeline.png       MolDeTr/docs/img/pipeline.png   # referenced by the new "How it works"
cp handoff/img/*.png          MolDeTr/docs/img/               # brand-matched example figures (same filenames)
cp -r handoff/docs/*.md       MolDeTr/docs/                   # polished docs pages (GitHub alerts + nav strips)
cp handoff/banner-dark.png    MolDeTr/docs/banner-dark.png    # dark-mode hero (README <picture> variant)
# handoff/img already contains the -dark variants + benchmark / architecture /
# input_contract / coupling_rule / mark.png (favicon & HF-thumbnail source) — the
# wildcard above copies them all; README_proposed.md references them via <picture>.
```

**Demo GIF for the README:** open `MolDeTr Workbench.dc.html` with the Tweaks prop
`autoplay = true` (it loops load → detect → full-set → zoom on both examples) and screen-record
~12 s to `docs/img/demo.gif` (or .mp4 — GitHub plays both).

**Social preview:** upload `handoff/img/social_preview.png` (2560×1280, &lt;1 MB) in
GitHub → Settings → General → Social preview. Same design as `docs/banner.png`, so link
cards and the README hero match. Editable source: `social-banner-source.html`.

No new dependencies beyond **plotly** (GUI only). Model loading, validation, and prediction logic are untouched
(`# BRAND` / `# NEW` marks every changed line). Targeted at **Gradio 4.44**
(the version pinned in `deploy/hf_space/requirements.txt`); if a `.set()` key or CSS
selector errors on another version, drop that line — everything is cosmetic.

## Tokens

| Token | Value | Use |
|---|---|---|
| blue | `#2566b0` | primary button, links, slider, marker 1 |
| orange | `#e08a1f` | prototype chip, warnings, marker 2 |
| teal | `#1f9e8c` | success checks, marker 3 |
| navy | `#1f3a5f` | display text (Space Grotesk) |
| ink / mute / eyebrow | `#20242b` / `#5b6675` / `#74808f` | body / secondary / labels |
| panel / border / page | `#f1f5fa` / `#d5dfeb` / `#eef2f7` | fills |
| brick | `#9b3128` | errors (brand has no pure red) |

Type: **Space Grotesk** (identity — wordmark, eyebrows, buttons, table headers) ·
**IBM Plex Sans** (body, data) · **IBM Plex Mono** (filenames, code).

## What stock Gradio can't fully reproduce

- The mockup's segmented ppm control → approximated as pill-styled radio labels
- Colored dots inside table rows → colors live on the plot markers only
  (Dataframe can't render safe inline HTML); numbering still links them
- Card shadows on nested groups render slightly flatter than the mock

Zoom parity with the mock comes from Plotly (`plotting.py`): drag = box zoom,
double-click = reset, ticks regenerate for any range. The x-axis is ppm when the file
carries a calibration, a **window-relative Hz** axis when it doesn't (using `points_per_hz`),
and point index only as a last resort. `moldetr/visualization.py` (matplotlib) still renders
`predict.py --plot` PNGs and the GT overlay for the evaluators.

## One warning worth repeating

Never uppercase table headers via CSS `text-transform` or Python `.upper()` —
`δ` becomes `Δ`, which reads as "difference" in NMR. Headers ship as
pre-uppercased literals with lowercase δ (`δ (PPM)`).

## Matplotlib font

`visualization.py` uses IBM Plex Sans only if it's installed on the host
(`pip install --upgrade fonts-ibm-plex` or system package); otherwise it falls back
to DejaVu Sans silently. The HF Space works either way.
