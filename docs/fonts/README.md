# Vendored font subsets

These four `.woff2` files are embedded, as base64, into the SVG diagrams that
`scripts/build_diagram_svgs.py` generates. They are vendored rather than fetched at build time so
the diagrams can be regenerated offline and reproducibly.

| File | Family | Weight | Source |
|---|---|---|---|
| `sg700.woff2` | Space Grotesk | 700 | [`google/fonts` `ofl/spacegrotesk`](https://github.com/google/fonts/tree/main/ofl/spacegrotesk) |
| `plex400.woff2` | IBM Plex Sans | 400 | [`google/fonts` `ofl/ibmplexsans`](https://github.com/google/fonts/tree/main/ofl/ibmplexsans) |
| `plex600.woff2` | IBM Plex Sans | 600 | same |
| `mono400.woff2` | IBM Plex Mono | 400 | [`google/fonts` `ofl/ibmplexmono`](https://github.com/google/fonts/tree/main/ofl/ibmplexmono) |

`docs/BRAND.md` § Type names these families as the brand's identity and body faces.

`sg500.woff2` used to sit here too, committed and licensed but embedded by **nothing** — no `faces=`
tuple named it, so it reached no SVG while still being hash-pinned as though it mattered. Removed
when the subsets were rebuilt. If a future figure wants weight 500, add it to `FACES` in
`scripts/build_diagram_fonts.py` and rebuild; there is no reason to carry it unused.

## Licence

All are **SIL Open Font License 1.1**. The full licences are `OFL-SpaceGrotesk.txt`,
`OFL-IBMPlexSans.txt` and `OFL-IBMPlexMono.txt`, retained here because the OFL requires the licence
to travel with the font data — and base64-embedding a subset into an SVG *is* distributing font
data. Neither family is sold, and no subset is named in a way that claims to be the original font.

## How these were produced

```bash
python scripts/build_diagram_fonts.py --tables   # writes docs/fonts/*.woff2, prints two tables
python scripts/build_diagram_svgs.py             # rebuild: every SVG carries the bytes inline
```

**Mind the ordering when you have ADDED a character to a diagram.** The font build reads the
*committed* SVGs, so run `build_diagram_svgs.py` **first** to get the new character onto disk,
then the font build, then the SVG build again -- svgs, fonts, svgs. Run it the short way and
the fonts are subsetted from the stale SVGs, you paste two tables derived from them, and
`tests/test_diagram_fonts.py` goes red afterwards (`--check` stays green, since the SVGs really
do match the generator). Recovering is just running the cycle properly, but you will have
pasted the tables twice. `--dry-run` builds and reports without writing, which is the cheap way
to look before committing to a paste.

**This used to be prose, and that is what issue #84 was.** The recipe described a manual `fontTools`
session and listed the characters by hand, so the list aged behind the diagrams it described until
five figures printed eight characters no face carried. The script now **reads the character set out
of the committed SVGs**, so what the diagrams print and what the fonts carry cannot drift by more
than one rebuild. Do not go back to typing the list.

What the script does, in order: pin the variation axes with `instancer.instantiateVariableFont`
(weight, plus IBM Plex's width at 100) so the embedded font is not carrying every weight from
300–700; `subset.Subsetter` down to the derived character set; save with `flavor = "woff2"`.

Three details worth keeping:

- **The upstream commit and each source TTF's SHA-256 are pinned** in the script. `google/fonts`
  revises its fonts, and an unpinned fetch would let outlines drift into published figures under a
  diff that shows nothing but base64.
- **IBM Plex Mono has no variable font upstream** — it ships as static instances, so `mono400` is
  `IBMPlexMono-Regular.ttf` with no instancing step. The prose recipe this replaced described all
  three families as variable, which was wrong about that one.
- **The build is byte-reproducible for a given fontTools and brotli.** The toolchain is not
  pinned (`fonttools[woff]>=4.40`, brotli unconstrained) and woff2 is brotli-compressed, so a
  rebuild on a newer stack can legitimately produce different bytes from identical inputs -- if
  `FONT_SHA256` moves without an input change, check the versions before hunting a defect.

  Within one toolchain it is exact, and that took work: `head.modified` is a wall-clock stamp
  that `TTFont.recalcTimestamp` re-applies at save time, so two builds a second apart produced
  different files. Left alone it would have quietly disarmed `FONT_SHA256` — a hash that changes on
  every rebuild teaches you to paste the new value without reading it, which is exactly how the
  hand-written glyph list rotted.

### Coverage

Printable ASCII in full, plus `°±²³·¹Å×÷–—‘’“”•…→↕−≤≥` in every face. Beyond that the families
differ, and the differences are load-bearing:

| character | Space Grotesk | IBM Plex Sans | IBM Plex Mono |
|---|:--:|:--:|:--:|
| `Δ` U+0394 | yes | yes | **no** |
| `δ` U+03B4 | **no** | yes | **no** |
| `✓` U+2713 | **no** | yes | yes |
| `∅` U+2205 | yes | **no** | **no** |

`GLYPHS` in `scripts/build_diagram_svgs.py` pins that per family — never as a union — and
`tests/test_diagram_fonts.py` holds every committed SVG to it, character by character, honouring
each run's font stack. `FONT_SHA256` in the same test pins the binaries, so a re-subset that
silently *dropped* a glyph fails there instead of shipping. `--tables` prints both, ready to paste;
update both after any rebuild.

The set is a **union** of what the diagrams print and what the faces already carried, so coverage
can only grow. Deriving it purely from current usage would shrink it — `°±²³Å÷‘’“”…≥` are carried
and printed by nothing — and a set that tracks usage exactly puts you back in the world where the
answer to "does this character render?" depends on which figure you edited.

(That list is measured, not typed. A draft of this paragraph named `×` — which the same
change made `coupling_rule` print — and omitted `≥`. It is the one list on this page that no
test guards, which is exactly why it drifted. Re-measure it rather than editing it by eye.)

### What a missing character actually does

**Fallback, not tofu**, and this file used to claim the opposite — "inside a sandboxed `<img>` there
is no system font to fall back to, so it renders as tofu". Observed 2026-08-11: a probe SVG
embedding only `plex400` and printing `Δ ✓ ₂` inside an `<img>` renders all three, in a stack ending
`,sans-serif` **and** in one naming only `'IBM Plex Sans'` with no generic at all. Both facts hold
together — secure static mode blocks external *resource loads*, which is why a referenced
`@font-face` URL genuinely would fail and why these are embedded, but a system font is not a
resource load.

That makes the failure mode harder to detect, not milder: the character appears, in a different face
at a different weight, for readers whose OS happens to have it — and vanishes for those whose OS does
not. Reader-dependent rendering is precisely what vendoring exists to prevent.

### The two the subset could not reach

Re-subsetting closed six of the eight characters in issue #84. Two could not be closed that way and
were handled in the generator instead, so `KNOWN_UNCOVERED` in `tests/test_diagram_fonts.py` is now
empty — **every character every diagram prints comes from a font that diagram embeds.**

- **`✕` U+2715 in `coupling_rule` → `×` U+00D7.** Measured against all three upstream cmaps: no
  family here carries U+2715, so no subset could ever have fixed it. U+00D7 is covered by all three
  and is the conventional rejection marker beside a checkmark. Sized to the checkmark rather than to
  the glyph it replaced — the multiplication sign is optically smaller at equal point size.
- **`∅` U+2205 in `architecture` → split across two families.** Carried by Space Grotesk only, while
  the run printing it is `'IBM Plex Sans',sans-serif`. The prose stays in Plex Sans and the one
  symbol is emitted as a `<tspan>` in the Space Grotesk stack. That is a deliberate single-glyph
  face change in place of the browser making the same substitution invisibly, from whatever the
  reader's OS holds. DETR's no-object symbol was worth keeping over a reword.

`KNOWN_UNCOVERED` stays in the test as an exact-set comparison rather than being deleted: an empty
dict is the assertion that nothing is uncovered, and a new uncoverable character fails against it.

### Before editing diagram text

- **Space Grotesk has no Greek at all.** `δ` is not in it. The generator's font stack names IBM
  Plex Sans second so that CSS's *per-glyph* fallback supplies it; do not "simplify" that stack to
  a single family.
- **IBM Plex Mono has no Greek either**, so neither `δ` nor `Δ` can appear in a monospace run.
- **No family here has `⁻` (U+207B, superscript minus).** Write `E⁻¹` some other way, or it will be
  a missing glyph. Subscripts are the same story — compose them with `Canvas.sub`, as the banner's
  `T₂` does.
