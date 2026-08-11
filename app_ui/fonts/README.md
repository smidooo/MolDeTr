# Vendored webfont for the Gradio app

`SpaceGrotesk-latin-var.woff2` is embedded, as base64, into `CUSTOM_CSS` by `app_ui/theme.py`. It is
vendored rather than fetched so that no third-party host sits in the render path of the app.

## Why this exists separately from `docs/fonts/`

`docs/fonts/` holds *diagram* subsets — static single weights (`sg500`, `sg700`), pinned with
`fontTools` and cut down to the glyphs the SVGs use. Those cannot serve the app, whose UI text is
arbitrary. This file is the full latin subset with the weight axis intact, so one 22 KB payload
covers every weight `CUSTOM_CSS` asks for.

| | |
|---|---|
| **Family** | Space Grotesk |
| **Licence** | SIL OFL 1.1 — `OFL.txt`, byte-identical to `docs/fonts/OFL-SpaceGrotesk.txt` |
| **Axis** | `wght 300–700`, default 300 (`fvar`; the name table calls it *Space Grotesk Light*). `@font-face` narrows this to `font-weight: 500 700`, which is the range the stylesheet uses — the file itself is wider. |
| **Coverage** | 230 codepoints — ASCII, Latin-1 and typographic punctuation, i.e. Google's `latin` unicode-range. Includes `¹` (U+00B9, used in `HEADER_HTML`); **excludes** Greek, so `δ` falls back to `sans-serif` exactly as it did before this was vendored. |
| **Size** | 22 288 bytes |

## How it was produced

Downloaded, not built — this is Google's own `latin` subset, byte-for-byte what the browser used to
fetch at runtime, which keeps the rendering identical to the pre-vendoring behaviour:

```bash
# 1. the stylesheet, requested with a modern UA so Google serves woff2
curl -H 'User-Agent: Mozilla/5.0 … Chrome/120.0 …' \
  'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap'

# 2. the `src:` URL from its /* latin */ block  (weights 500/600/700 all name the same
#    file — Space Grotesk is variable, so one payload covers the range)
curl -o SpaceGrotesk-latin-var.woff2 \
  'https://fonts.gstatic.com/s/spacegrotesk/v22/V8mDoQDjQSkFtoMM3T6r8E7mPbF4Cw.woff2'
```

The `v22` in the path is upstream's version directory; a later Space Grotesk release will publish a
different URL, so re-derive step 2 from step 1 rather than reusing this link.

## What guards it

`tests/e2e/test_browser_branding.py` — the app must make no third-party request, the layout CSS must
survive a font CDN routed to hang, and the embedded face must actually load (`document.fonts.check`).
Recorded in `THIRD_PARTY.md`; background in issue #80.
