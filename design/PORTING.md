# Porting the MolDeTr design into your Claude Code project

You feed Claude Code your GitHub repo (`smidooo/MolDeTr`). This folder (`handoff/`) is the
bridge: **runnable Python + Markdown + image assets** that drop into that repo, plus the
design references Claude Code should read to stay on-brand.

## 1. What actually ships into the repo (code + assets)

| From `handoff/` | Into the repo | Kind |
|---|---|---|
| `BRAND.md` | `BRAND.md` (root or `docs/`) | canonical tokens + wording — the source of truth |
| `theme.py` | `theme.py` | Gradio theme + CSS + header HTML |
| `plotting.py` | `plotting.py` | interactive Plotly spectrum (ppm / Hz axis, box zoom) |
| `app.py` | `app.py` | restructured Gradio UI (logic unchanged) |
| `visualization.py` | `moldetr/visualization.py` | branded matplotlib for `predict.py --plot` |
| `README_proposed.md` | `README.md` | redesigned README (light+dark `<picture>`) |
| `docs/*.md` | `docs/` | polished SCOPE / INPUT_FORMAT / USAGE_NOTES / DATA_SCHEMA |
| `img/*.png` | `docs/img/` | all figures + `-dark` variants + `mark.png` |
| `pipeline.png` | `docs/img/pipeline.png` | (also in `img/`) |
| `banner-dark.png` | `docs/banner-dark.png` | dark hero |
| `img/social_preview.png` | GitHub → Settings → Social preview | upload, not committed |
| `img/mark.png` / `mark-dark.png` | `docs/img/` → crop for `favicon.png` / HF thumbnail | brand mark |

One-shot apply is the copy block in [`README.md`](README.md).

## 2. What Claude Code should READ but NOT ship

The `*.dc.html` files at the project root are **design mockups**, not part of the Python app:

- `MolDeTr Workbench.dc.html` — the interactive GUI spec (the source of truth for `theme.py` +
  `app.py` layout). Set the Tweaks prop `autoplay = true` and screen-record it for the README demo GIF.
- `README Figures.dc.html` / `…Dark`, `MolDeTr Diagrams.dc.html` / `…Dark` — the generators
  behind every PNG in `img/`. Re-export from these if a value changes.
- `Current Gradio GUI.dc.html` — the faithful recreation of the app *before* the redesign.
- `README Redesign.dc.html`, `Gradio Handoff.dc.html` — rendered previews / index.

They are HTML/JS and will not run in the Python app; treat them as the visual contract.

## 3. Prompt to give Claude Code

> "Read `handoff/BRAND.md` for tokens and the δ≠Δ rule. Apply the copy block in
> `handoff/README.md`: replace `app.py`, `moldetr/visualization.py`, add `theme.py` and
> `plotting.py`, add `plotly>=5` to `deploy/hf_space/requirements.txt`, copy `handoff/img/*`
> into `docs/img/` and `banner-dark.png` into `docs/`, and replace `README.md` +
> `docs/*.md`. Then run `python app.py` and confirm the assignment table, Plotly zoom, and
> the input-check states render. Do not change the model, validation, or inference code —
> only the files marked `# BRAND` / `# NEW`."

## 4. After applying

```bash
pip install -e ".[app]" && pip install plotly
python app.py                      # smoke-test the GUI locally
python scripts/predict.py --input examples/roi_S8_example.npz --plot   # branded PNG
```

- The Space deploys unchanged (`theme.py` + `plotting.py` sit next to `app.py`).
- Everything is **Gradio 4.44**-targeted; a `.set()` key or CSS selector that errors on
  another version is cosmetic — delete that line.
- Keep `BRAND.md` as the single place tokens are defined; if you restyle, change it there first.

## 5. Consistency guardrails (carry into any future edits)

- **Never** `text-transform:uppercase` / `.upper()` a header containing δ (→ Δ = "difference").
- Colour is never the only channel — every marker keeps its **number** (CVD-safe; see
  `img/cvd_check.png`).
- Tricolor = blue `#2566b0` / orange `#e08a1f` / teal `#1f9e8c` for multiplets 1·2·3, everywhere.
- One canonical "max J vs full set" sentence (in `BRAND.md`) — reuse it verbatim.
