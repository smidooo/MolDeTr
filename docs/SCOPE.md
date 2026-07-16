**Docs:** [README](../README.md) · **Scope &amp; limitations** · [Input format](INPUT_FORMAT.md) · [Usage notes](USAGE_NOTES.md) · [Data schema](DATA_SCHEMA.md)

---

# Scope and limitations

> **Research code accompanying the paper.** MolDeTr extracts δ, proton count, and couplings from real
> ¹H NMR spectra — including the congested, strongly-coupled cases it was built for. It is largely
> field-agnostic — it works in Hz, so it was tested across 80–600 MHz (and simulated down to ~5 MHz).
> Predictions can deviate for inputs outside its trained regime: unusual distortions, non-standard pulse
> sequences or processing, mixtures/impurities, or regions wider than the 1200 Hz window. `max J` is the
> dominant coupling per multiplet (the full set is in `structured_output/`). Sanity-check against your
> chemistry.

This page states, conservatively and cited to the paper, what MolDeTr can and cannot do. The README
carries a short summary; the detail lives here.

## What it can do

- **1-D ¹H NMR only**, single-component small molecules (one clean compound per spectrum).
- Up to **10 chemically-equivalent spin groups (multiplets)** per **1200 Hz** window. This is an
  engineering choice (the query budget), stated as scalable in the paper — not a physical limit.
- **Uncoupled, weakly-coupled, and strongly-coupled (higher-order)** systems, including overlapping
  and roof-topped multiplets (ABC, AA′BB′C) that defeat rule-based peak-picking — with no priors.
- **Largely field-agnostic**: because it works in Hz (resample to 5.12 points/Hz), it maintains consistent
  performance regardless of base frequency — tested on real **80–600 MHz** spectra (and simulated down to ~5 MHz).
- **Proton-count classes 1, 2, 3, 4, 6.**
- Experimental medians (reproduced from the committed matched pairs): chemical-shift **MedAE 0.90 Hz**,
  coupling **MedAE 0.20 Hz**, overall proton-count accuracy **93.5 %** (per-class 97 % / 88.75 % / 75 %
  for 1H / 2H / 3H). (0.90 Hz is the aggregate/reproduced figure used throughout; the README's *Reproducing
  the paper* section explains how it relates to the value printed in the article.)

## What it cannot do (out of scope)

- **Mixtures, impurities, or a solvent/water peak.** There is **no water suppression** — a residual
  water or solvent resonance in the window is explicitly out of scope.
- **Non-¹³C heteronuclear artifacts.** Only ¹³C satellites are modelled.
- **Chemical exchange, dynamics,** or any solvent/temperature/pH metadata.
- **2-D spectra, other nuclei, or other spectroscopies.** MolDeTr is 1-D ¹H only.
- **Extreme congestion** — far more overlapping spin systems in a window than can be resolved. This is
  fundamentally ill-posed for *any* method, not a MolDeTr-specific limit (distinct from the scalable
  ~10-multiplet query budget per 1200 Hz window).
- Anything **beyond the 1200 Hz window** or **beyond the trained distortion ranges** below.
- A multiplet whose **coupling partner sits outside the window** — draw the window so every spin
  system is complete.

## The training distribution (what "in range" means)

The model was trained on ~5 million simulated Lorentzian spectra with these ranges:

| Property | Range |
|---|---|
| SNR | 10² – 10⁵ |
| Baseline distortion | ≤ 5× noise |
| Phase | zero-order ±8°, plus first-order |
| ¹³C satellites | 0.5 – 1.5 % (¹J₍CH₎ 40–220 Hz) |
| Shim / line broadening | 0 – 3 Hz added |
| Chemical shift δ | 0 – 15 ppm |
| ¹H–¹H coupling J | 0.01 – 20 Hz |
| Line width | 0.3 – 2.2 Hz |
| Spin block size | N &lt; 6 spins |

Inputs far outside these ranges are out of distribution and degrade accuracy.

## About the coupling constants

> [!IMPORTANT]
> The model regresses a **permutation-invariant embedding** of each multiplet's couplings, not the
> individual constants. The committed `structured_output/` path inverts it **exactly** (the paper's
> 0.20 Hz median); the live demo reports a single **largest coupling, `max J`**.

The embedding is `[sum, min, max, std]`. (The paper's Supporting Information writes this embedding as
`[min, max, mean, sum]`; the released code and checkpoint use `[sum, min, max, std]` — the **code is
authoritative**, since the weights were trained with that definition. Treat the SI listing as an erratum
(it differs from the code by content too — `mean` where the code uses `std`, not only order).)
There are two ways this repo turns that into numbers:

- **Exact path (the paper's numbers).** The committed `structured_output/` + `aggregate_experimental.py`
  invert the embedding exactly — this is where the **0.20 Hz** median coupling error comes from.
- **Live demo (`predict.py`, the GUI).** These report a single **largest coupling, `max J`** (the max
  component of the embedding) — the dominant coupling per multiplet, not the full set. They reproduce the
  paper's proton counts, chemical shifts, and `max J` (e.g. vanillin `max J` 8.2/2.0/8.7 vs ground truth
  8.1/2.0/8.1 Hz); predictions can deviate for inputs outside the trained regime (the ranges above). For
  the full coupling set, use the exact path.

Coupling recovery is reliable for **≤ 3 distinct** couplings per multiplet; the paper's evaluation
centres on `max J` for this reason.

## A note on input noise (why the live tools inject it)

> [!NOTE]
> The live tools add a small **calibrated Gaussian noise (0.5 % of max amplitude, seeded)** before the
> model — deterministic, and it matches the paper's evaluation pipeline. Pass your processed spectrum
> as-is; do not add noise yourself.

The model was trained on spectra carrying realistic noise (SNR 10²–10⁵). A perfectly clean,
FFT-resampled region is **out of distribution** — the detector reads it less accurately (it can, for
example, miscount a triplet). This is why `predict.py`, the GUI, and the notebook inject the calibrated
noise, and why the live path is both deterministic *and* reproduces the paper's predictions. (Set
`noise_frac=0` in `moldetr.inference.normalize_spectrum` only if your input is already noisy.)

## Tested vs untested classes

> [!CAUTION]
> The **4-proton and 6-proton** classes exist in training but are **absent from the experimental test
> set** — only 1H / 2H / 3H are exercised on real data. Treat 4H/6H predictions on real spectra with
> extra caution.

---

**Next:** [Input format](INPUT_FORMAT.md) — prepare a window from your own data.
