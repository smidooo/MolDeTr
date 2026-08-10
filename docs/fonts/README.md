# Vendored font subsets

These four `.woff2` files are embedded, as base64, into the SVG diagrams that
`scripts/build_diagram_svgs.py` generates. They are vendored rather than fetched at build time so
the diagrams can be regenerated offline and reproducibly.

| File | Family | Weight | Source |
|---|---|---|---|
| `sg500.woff2` | Space Grotesk | 500 | [`google/fonts` `ofl/spacegrotesk`](https://github.com/google/fonts/tree/main/ofl/spacegrotesk) |
| `sg700.woff2` | Space Grotesk | 700 | same |
| `plex400.woff2` | IBM Plex Sans | 400 | [`google/fonts` `ofl/ibmplexsans`](https://github.com/google/fonts/tree/main/ofl/ibmplexsans) |
| `plex600.woff2` | IBM Plex Sans | 600 | same |
| `mono400.woff2` | IBM Plex Mono | 400 | [`google/fonts` `ofl/ibmplexmono`](https://github.com/google/fonts/tree/main/ofl/ibmplexmono) |

`docs/BRAND.md` § Type names these families as the brand's identity and body faces.

## Licence

Both are **SIL Open Font License 1.1**. The full licences are `OFL-SpaceGrotesk.txt`, `OFL-IBMPlexSans.txt` and
`OFL-IBMPlexMono.txt`, retained here because the OFL requires the licence to travel with the font
data — and base64-embedding a subset into an SVG *is* distributing font data. Neither family is
sold, and neither subset is named in a way that claims to be the original font.

## How these were produced

From the upstream variable fonts, with `fontTools`:

1. `instancer.instantiateVariableFont` pins the weight axis (and IBM Plex's width axis at 100),
   dropping the variation tables. Embedding the variable font would carry every weight from 300–700
   into every SVG.
2. `subset.Subsetter` reduces the glyph set to printable ASCII plus the non-ASCII characters the
   diagrams use: `¹²³⁻·δ→≤≥±×÷—–''""…°Å`.
3. Saved with `flavor = "woff2"`.

One shared glyph superset is used for all diagrams rather than a per-diagram subset. A per-diagram
subset is smaller, but it means a later text edit can introduce a character the embedded font does
not carry — and inside a sandboxed `<img>` there is no system font to fall back to, so it renders
as tofu. A few KB is a fair price for making that impossible.

### Two coverage facts worth knowing before editing diagram text

- **Space Grotesk has no Greek at all.** `δ` is not in it. The generator's font stack names IBM
  Plex Sans second so that CSS's *per-glyph* fallback supplies it; do not "simplify" that stack to
  a single family.
- **IBM Plex Mono has no Greek either**, so `δ` cannot appear in a monospace run.
- **No family here has `⁻` (U+207B, superscript minus).** Write `E⁻¹` some other way, or it will be
  a missing glyph.
