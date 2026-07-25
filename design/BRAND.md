<!-- DESIGN_VERSION: v2 -->
# MolDeTr brand — single source of truth

Every artifact in this repo (README figures, the GUI theme, docs diagrams, the social
banner) pulls from the tokens below. **Change a value here first, then propagate.** This
file is the canonical reference the handoff `theme.py`, the figure generators, and the
docs all cite.

## Palette

| Token | Hex | Role |
|---|---|---|
| **blue** | `#2566b0` | primary · links · slider · multiplet marker **1** |
| blue-dark | `#1f57a0` | primary hover |
| **orange** | `#e08a1f` | warnings · marker **2** (v2: the header chip it also served is gone) |
| **teal** | `#1f9e8c` | success · confidence · marker **3** |
| **navy** | `#1f3a5f` | display text (Space Grotesk) |
| ink | `#20242b` | body text · spectrum trace |
| mute | `#5b6675` | secondary text |
| eyebrow | `#74808f` | small uppercase labels |
| latent | `#7d92b0` | NN hidden-layer nodes (diagrams) |
| panel | `#f1f5fa` | panel fill |
| border | `#d5dfeb` | hairline borders |
| page | `#eef2f7` | app background |
| **brick** | `#9b3128` | errors (the brand has no pure red — this is an oklch-harmonised brick) |

### Dark-figure palette (for `prefers-color-scheme: dark` image variants)

| Token | Hex |
|---|---|
| bg / card | `#141821` / `#1b2130` |
| text / dim | `#e8eef6` / `#93a2b5` |
| muted / eyebrow | `#9fb0c4` / `#8f9db0` |
| grid / border | `#28303f` / `#334054` |
| trace | `#e8eef6` |

The **tricolor stays identical** in dark mode — blue/orange/teal read well on both grounds.

## The multiplet tricolor & colour accessibility

Markers cycle **blue → orange → teal** for multiplets 1 · 2 · 3. These sit next to the
Okabe–Ito / Wong palette (the Nature-recommended colourblind-safe set: `#0072B2` blue,
`#E69F00` orange, `#009E73` bluish-green), so the brand tricolor is CVD-robust.

**Rule: colour is never the only channel.** Every marker also carries its **number**, and
tables repeat the value — so the figure survives greyscale printing and all CVD types
(verified: see `handoff/img/cvd_check.png`). Never encode a distinction by hue alone.
Cap categorical colours at ≤ 3 here (well under the 6–8 safe maximum).

## Type

- **Space Grotesk** (500/600/700) — identity: wordmark, eyebrows, titles, buttons, table headers, spin labels Hₓ.
- **IBM Plex Sans** (400–700) — body & data: axis, captions, atom labels.
- **IBM Plex Mono** (400/600) — filenames, code, DOIs.

## Brand mark

The **tricolor dash triple** (`▬▬▬` in blue/orange/teal) is the recurring mark — header,
figures, banner. Reuse it; don't invent new logos.

## Canonical wording — the two decode paths (use verbatim)

> **Short (captions, footnotes, tooltips):**
> "**max J** = the largest coupling per multiplet (the live decode path — `predict.py`, the GUI).
> The committed `structured_output` path recovers the full coupling set."

> **Long (README, SCOPE):**
> "The live tools reproduce the paper's predictions — δ, proton count, and the largest coupling
> **max J**. Deviations come from out-of-distribution acquisition or processing, not from spectral
> resolution. `max J` is only the largest coupling per multiplet; the committed `structured_output`
> / `aggregate_experimental` path recovers the full set (the paper's exact E⁻¹, 0.20 Hz median)."

## The δ ≠ Δ rule

Never uppercase table headers or labels via CSS `text-transform` or Python `.upper()` —
`"δ".upper()` → `"Δ"`, which means *difference* in NMR. Ship pre-uppercased literals with
lowercase δ (`δ [PPM]`).

## Changelog

- **v2** — scope framing corrected at the source. The "Research prototype" header chip is removed
  (orange no longer serves it), the GUI accordion is **Scope & limits**, and the canonical Long
  wording drops "On well-resolved spectra" — deviations come from out-of-distribution acquisition
  or processing, not from resolution. This file is now committed, so the token/wording sync is
  enforced by `tests/test_brand_contract.py` instead of being asserted here.
- **v1** — initial brand system: tokens, dark palette, tricolor + CVD rule, type, mark,
  canonical max-J wording, δ≠Δ rule. Manuscript-verified (0.89/0.20 Hz, 93.5 %).
