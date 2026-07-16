**Docs:** [README](../README.md) · [Scope &amp; limitations](SCOPE.md) · [Input format](INPUT_FORMAT.md) · **Usage notes** · [Data schema](DATA_SCHEMA.md)

---

# Usage notes — how to read the output, and how it fails

MolDeTr is research code accompanying the paper (see [`SCOPE.md`](SCOPE.md)). This page is the practical
companion: what each number means, how much to trust it, and the failure modes to expect — with how
to avoid each one.

## How to read a prediction

| Output | What it is | How much to trust it |
|---|---|---|
| **δ (chemical shift)** | multiplet centre, in ppm (if calibrated) or Hz | **Most reliable.** Median error 0.90 Hz. |
| **proton count** | number of equivalent protons for the multiplet | Reliable for 1H/2H/3H (97 / 89 / 75 %). 4H/6H untested on real data. |
| **max J** | the **largest** coupling constant, in Hz | Reports the **dominant** coupling per multiplet; the full set is in `structured_output` (see [`SCOPE.md`](SCOPE.md#about-the-coupling-constants)). Can deviate for inputs outside the trained regime. |
| **line width** | peak width, in Hz (∝ 1/T₂) | Indicative; sensitive to shim and processing. |

> [!NOTE]
> Every number is an **estimate to sanity-check**, not a measurement. If a prediction contradicts
> what you know about the molecule, trust the chemistry.

## Failure modes and how to avoid them

| If this happens… | Why | What to do |
|---|---|---|
| Error: "needs exactly 6144 points" | wrong length | Resample your region to **5.12 points/Hz** over a **1200 Hz** window, crop/zero-pad to 6144 (see [`INPUT_FORMAT.md`](INPUT_FORMAT.md)). The validator catches this up front. |
| Warning about digital resolution | not 5.12 points/Hz (≠ 1200 Hz window) | Re-bin to 5.12 points/Hz. Off-resolution input shifts every peak position. |
| "complex — real part used" | you passed a complex spectrum | Pass the **real (absorption)** spectrum. The model uses the real part. |
| A peak at the window edge is wrong | its **coupling partner is outside** the window | Widen or re-centre the region so the whole spin system (≤ 1200 Hz) is inside it. |
| A big water/solvent peak throws it off | **out of scope** — no water suppression | Remove/suppress solvent and water first, or exclude that region. |
| Reported J looks too large / there's "only one" | the live demo reports only **max J** | This is by design (the model outputs a coupling *embedding*, not each J). For the full/accurate couplings use the `structured_output` path. |
| Predictions look generally off | baseline/phase distortion beyond the trained range, or an unusual sample | Phase and baseline-correct first (see the TopSpin recipe in `INPUT_FORMAT.md`); check the input is in the ranges in `SCOPE.md`. |
| A 4H or 6H prediction on real data | those classes were **not** in the experimental test set | Treat with extra caution; verify against the structure. |

## The one rule worth repeating

> [!IMPORTANT]
> A window need not contain the whole molecule, but **every proton that couples to a proton inside
> the window must also be inside it.** A multiplet whose coupling partner is out of the window is out
> of distribution, and its prediction will be wrong. Draw regions so each spin system is complete.

---

**Next:** [Data schema](DATA_SCHEMA.md) — the npz keys and label formats.
