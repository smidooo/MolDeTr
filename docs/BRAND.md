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
| eyebrow | `#666f7d` | small uppercase labels (v2: was `#74808f`, which failed WCAG AA — see below) |
| latent | `#7d92b0` | NN hidden-layer nodes (diagrams) |
| panel | `#f1f5fa` | panel fill |
| border | `#d5dfeb` | hairline borders |
| page | `#eef2f7` | app background — **not** the figure ground, see `figure-page` |
| figure-page | `#f8fafd` | the ground inside a diagram frame (`docs/img/*.svg`) |
| **brick** | `#9b3128` | errors (the brand has no pure red — this is an oklch-harmonised brick) |

`page` and `figure-page` are separate roles that were briefly conflated. The app background is
`#eef2f7`; the diagram ground measured `#f8fafd` across every figure. Generators must not
substitute one for the other.

### Dark-figure palette (for `prefers-color-scheme: dark` image variants)

| Token | Hex |
|---|---|
| bg / card | `#141821` / `#1b2130` |
| panel / figure-page | `#232c3c` / `#141821` |
| text / dim | `#e8eef6` / `#93a2b5` |
| muted / eyebrow | `#9fb0c4` / `#8f9db0` |
| grid / border | `#28303f` / `#334054` |
| trace | `#e8eef6` |
| **navy fill / display text** | `#31517d` / `#e2e9f4` |
| connector / arrow | `#3a4557` / `#9db0c6` (unchanged) |
| track / rule | `#262f3d` / `#28303f` |
| brick / its wash / its edge | `#e08575` / `#241a18` / `#5c3a34` |
| teal wash / teal edge / teal text | `#16241f` / `#2f5850` / `#5cc2b0` |

The **tricolor stays identical** in dark mode — blue/orange/teal read well on both grounds.

**`navy` is two roles in dark, and only one of them is a colour swap.** In light, the box fill and
the display type are both `#1f3a5f`. In dark they diverge: a filled box becomes `#31517d` and the
type it once shared a value with becomes `#e2e9f4`. Carrying one token for both produces a
near-white box with dark text where the figure wants a blue box with light text — a defect only
dark-mode readers ever see. `scripts/build_diagram_svgs.py` keeps `navy` and `display` apart.

These dark values were **extracted, not chosen**: three figures shipped light and dark PNGs of
identical dimensions, so masking the light image by each token and taking the modal colour under
that mask in the dark image recovers the mapping exactly. Every row agreed at 100 % except the two
that legitimately split. Prefer re-running that extraction over inventing a value.

## The multiplet tricolor & colour accessibility

Markers cycle **blue → orange → teal** for multiplets 1 · 2 · 3. These sit next to the
Okabe–Ito / Wong palette (the Nature-recommended colourblind-safe set: `#0072B2` blue,
`#E69F00` orange, `#009E73` bluish-green), so the brand tricolor is CVD-robust.

**Rule: colour is never the only channel.** Every marker also carries its **number**, and
tables repeat the value — so the figure survives greyscale printing and all CVD types
(verified: see `handoff/img/cvd_check.png`). Never encode a distinction by hue alone.
Cap categorical colours at ≤ 3 here (well under the 6–8 safe maximum).

**Rule: text tokens must clear WCAG AA (4.5:1) on their own background.** The old
`eyebrow #74808f` did not — it measured **4.01:1 on white**, and it was defined for *small
uppercase labels*, which is precisely where the 4.5 threshold applies rather than the 3.0
large-text one. It therefore failed at its stated purpose. `#666f7d` measures **5.08:1**.
Caught by an axe-core scan in the browser tier, which now enforces this
(`tests/e2e/test_browser_a11y.py`) — CVD-safety and contrast are separate properties, and the
tricolor being CVD-robust never implied the greys were legible.

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
