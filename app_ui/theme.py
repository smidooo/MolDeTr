"""MolDeTr Gradio theme — ports the paper brand (banner/TOC palette) to the GUI.

Usage in app.py:

    from theme import MOLDETR_THEME, CUSTOM_CSS, HEADER_HTML
    with gr.Blocks(title="MolDeTr") as demo:
        gr.HTML(HEADER_HTML)
        ...
    demo.launch(theme=MOLDETR_THEME, css=CUSTOM_CSS)  # gradio 6.x: theme + css at launch()

Targets the Gradio 6.x theme API (6.20, pinned in deploy/hf_space + pyproject).
If a `.set(...)` key errors on another version, delete that line — every key is cosmetic.
"""

import gradio as gr

# ---- paper palette (docs/banner.png · TOC graphic) --------------------------
BLUE = "#2566b0"  # primary / detections
BLUE_DARK = "#1f57a0"
ORANGE = "#e08a1f"  # accent 2 / warnings
TEAL = "#1f9e8c"  # accent 3 / success
NAVY = "#1f3a5f"  # display text
INK = "#20242b"  # body text
MUTE = "#5b6675"  # secondary text
EYEBROW = "#74808f"  # small labels
PANEL = "#f1f5fa"  # panel fill
PANEL_BD = "#d5dfeb"  # borders
PAGE_BG = "#eef2f7"  # page background
RED = "#9b3128"  # errors (harmonised brick — the brand has no pure red)

MOLDETR_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.Color(
        c50="#eaf1f9",
        c100="#d5e3f3",
        c200="#abc7e7",
        c300="#81abdb",
        c400="#4f88c9",
        c500=BLUE,
        c600=BLUE,
        c700=BLUE_DARK,
        c800="#194a86",
        c900="#143c6d",
        c950="#0f2e54",
    ),
    neutral_hue=gr.themes.colors.Color(
        c50="#f8fafd",
        c100=PANEL,
        c200="#e6ebf2",
        c300=PANEL_BD,
        c400="#cdd7e4",
        c500="#9db0c6",
        c600=EYEBROW,
        c700=MUTE,
        c800="#3a4553",
        c900=INK,
        c950="#14171c",
    ),
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill=PAGE_BG,
    body_text_color=INK,
    body_text_color_subdued=MUTE,
    background_fill_primary="#ffffff",
    background_fill_secondary=PANEL,
    border_color_primary=PANEL_BD,
    block_background_fill="#ffffff",
    block_border_color=PANEL_BD,
    block_border_width="1.5px",
    block_label_text_color=EYEBROW,
    block_label_background_fill="#ffffff",
    block_title_text_color=EYEBROW,
    block_shadow="0 12px 32px rgba(31,58,95,.07)",
    input_background_fill="#ffffff",
    input_border_color="#cdd7e4",
    input_border_width="1.5px",
    button_primary_background_fill=BLUE,
    button_primary_background_fill_hover=BLUE_DARK,
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_text_color=NAVY,
    slider_color=BLUE,
    checkbox_background_color_selected=BLUE,
    checkbox_border_color_selected=BLUE,
    checkbox_label_background_fill="#ffffff",
    checkbox_label_border_color=PANEL_BD,
    link_text_color=BLUE,
    link_text_color_hover=NAVY,
    table_border_color="#e6ebf2",
    table_even_background_fill="#ffffff",
    table_odd_background_fill="#f8fafd",
    loader_color=BLUE,
)

# ---- CSS for what the theme API can't express -------------------------------
# Selectors target Gradio 6.x; adjust if your version renders differently.
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap');

.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* header strip (markup comes from HEADER_HTML) */
#md-header { display: flex; align-items: center; justify-content: space-between;
  padding: 4px 2px 12px; border-bottom: 1.5px solid #d5dfeb; margin-bottom: 6px; }

/* block labels as eyebrows.
   NEVER add text-transform:uppercase to table headers — it would turn δ into Δ. */
label > span[data-testid="block-info"], .block-title {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 12px !important; font-weight: 600 !important;
  letter-spacing: .16em; text-transform: uppercase; color: #74808f !important; }

button.primary { font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important; box-shadow: 0 8px 20px rgba(37,102,176,.28) !important; }

/* file dropzone */
#md-file { border: 2px dashed #c6d2e1 !important; border-radius: 12px !important;
  background: #f8fafd !important; }

/* ppm radio → pill segments */
#md-ppm .wrap { gap: 8px; }
#md-ppm label { border: 1.5px solid #d5dfeb !important; border-radius: 10px !important;
  background: #fff !important; font-family: 'Space Grotesk', sans-serif; font-weight: 500; }
#md-ppm label.selected { border-color: #2566b0 !important; color: #1f3a5f !important;
  font-weight: 600; box-shadow: 0 2px 6px rgba(31,58,95,.12); }

/* examples → chips */
#md-examples button { border-radius: 999px !important; border: 1.5px solid #cdd7e4 !important;
  color: #1f3a5f; font-weight: 600; font-size: 12.5px; background: #fff; }
#md-examples button:hover { border-color: #2566b0 !important; color: #2566b0; }

/* assignment table */
#md-table table { font-variant-numeric: tabular-nums; }
#md-table thead th { font-family: 'Space Grotesk', sans-serif !important;
  font-size: 12px !important; letter-spacing: .06em; color: #74808f !important; }

.md-footnote { font-size: 12px; color: #74808f; }
"""

HEADER_HTML = """
<div id="md-header">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="display:flex;flex-direction:column;gap:4px;">
      <div style="display:flex;gap:4px;">
        <span style="width:18px;height:4px;border-radius:2px;background:#2566b0;"></span>
        <span style="width:18px;height:4px;border-radius:2px;background:#e08a1f;"></span>
        <span style="width:18px;height:4px;border-radius:2px;background:#1f9e8c;"></span>
      </div>
      <span style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:23px;color:#1f3a5f;letter-spacing:-.02em;line-height:1;">MolDeTr</span>
    </div>
    <span style="width:1px;height:30px;background:#d5dfeb;"></span>
    <span style="font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#74808f;">&#185;H NMR multiplet detection</span>
    <span style="display:inline-flex;align-items:center;gap:6px;background:#fbf1e2;border:1.5px solid #ecc78f;border-radius:999px;padding:4px 11px;font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:#9a6410;"><span style="width:6px;height:6px;border-radius:50%;background:#e08a1f;"></span>Research prototype</span>
  </div>
  <div style="display:flex;align-items:center;gap:22px;font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:13.5px;">
    <a href="https://doi.org/10.1021/acs.analchem.5c03465" target="_blank" style="color:#2566b0;text-decoration:none;">Paper</a>
    <a href="https://github.com/smidooo/MolDeTr" target="_blank" style="color:#2566b0;text-decoration:none;">GitHub</a>
    <a href="https://github.com/smidooo/MolDeTr/blob/main/docs/SCOPE.md" target="_blank" style="color:#2566b0;text-decoration:none;">Scope &amp; docs</a>
  </div>
</div>
"""
