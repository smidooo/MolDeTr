# Vendored font subsets

These five `.woff2` files are embedded, as base64, into the SVG diagrams that
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

`sg500.woff2` is committed and licensed but currently embedded by **nothing** — no `faces=` tuple in
`scripts/build_diagram_svgs.py` names it, so it reaches no SVG. Kept because the weight is part of
the brand and a future figure may want it; noted here so its absence from every diagram is not read
as a packaging bug.

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
   diagrams use.

   **Read the cmaps, not this line, for what is actually in the files.** This step used to publish
   the list `¹²³⁻·δ→≤≥±×÷—–''""…°Å`, and it was wrong in both directions: `⁻` is in none of the
   five faces (the last bullet below always said so, and the bullet was the correct half), and the
   list had aged behind the diagrams it describes. Note the quotes: the retired list named the
   *straight* ASCII `'` and `"`, which are covered trivially and state nothing, where the real
   non-ASCII coverage is the four curly ones. Measured coverage, 2026-08-11 — printable ASCII
   in full, plus `°±²³·¹Å×÷–—‘’“”…→≤≥` in every face and `δ` in the IBM Plex Sans faces only:

   ```bash
   python -c "from fontTools.ttLib import TTFont; f=TTFont('docs/fonts/plex400.woff2'); \
   print(''.join(chr(c) for c in sorted(f.getBestCmap()) if c > 0x7F))"
   ```

   `scripts/build_diagram_svgs.py`'s `GLYPHS` table pins that measurement, and
   `tests/test_diagram_fonts.py` holds every committed SVG to it — so a hand-written list can no
   longer be the thing a diagram is checked against. Update both after any re-subsetting.
3. Saved with `flavor = "woff2"`.

One shared glyph superset is used for all diagrams rather than a per-diagram subset. A per-diagram
subset is smaller, but it means a text edit can introduce a character that *this* diagram's font
lacks while another diagram carries it — a failure that depends on which figure you edited. A few
KB is a fair price for making that particular variant impossible.

It does **not** make the general case impossible, and it would be easy to read the paragraph above
as saying so. The superset is still a subset: five diagrams currently print characters no face
carries at all (issue #84 below). What closes that is `tests/test_diagram_fonts.py`, not the
sharing.

**What actually happens then is fallback, not tofu**, and this file used to claim the opposite —
"inside a sandboxed `<img>` there is no system font to fall back to, so it renders as tofu".
Observed 2026-08-11: a probe SVG embedding only `plex400` and printing `Δ ✓ ₂` inside an `<img>`
renders all three, in a stack ending `,sans-serif` **and** in one naming only `'IBM Plex Sans'` with
no generic at all. Both facts hold together — secure static mode blocks external *resource loads*,
which is why a referenced `@font-face` URL genuinely would fail and why these are embedded, but a
system font is not a resource load.

That makes the failure mode worse to detect rather than better: the character appears, in a
different face at a different weight, for readers whose OS happens to have it — and vanishes for
those whose OS does not. Reader-dependent rendering is precisely what vendoring exists to prevent.
Five diagrams are currently in that state; see
[issue #84](https://github.com/smidooo/MolDeTr/issues/84).

### Three coverage facts worth knowing before editing diagram text

- **Space Grotesk has no Greek at all.** `δ` is not in it. The generator's font stack names IBM
  Plex Sans second so that CSS's *per-glyph* fallback supplies it; do not "simplify" that stack to
  a single family.
- **IBM Plex Mono has no Greek either**, so `δ` cannot appear in a monospace run.
- **No family here has `⁻` (U+207B, superscript minus).** Write `E⁻¹` some other way, or it will be
  a missing glyph.
