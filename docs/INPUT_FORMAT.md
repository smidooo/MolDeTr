**Docs:** [README](../README.md) · [Scope &amp; limitations](SCOPE.md) · **Input format** · [Usage notes](USAGE_NOTES.md) · [Data schema](DATA_SCHEMA.md)

---

# Running MolDeTr on your own spectra

MolDeTr works on one spectral window at a time. It does not read raw vendor files, pick peaks, or
choose regions for you — you hand it a single preprocessed window and it returns the multiplets in it.
To use your own data, resample each region to the format below.

> [!TIP]
> **In a hurry?** The bundled [`examples/`](../examples/) files already follow this contract —
> `roi_S10_example.npz` is a good template for your own exporter. Smoke-test with:
> ```bash
> python scripts/predict.py --input examples/roi_S10_example.npz --plot
> ```

## The input contract

| Property | Requirement |
|---|---|
| Length | exactly **6144 points** |
| Digital resolution | **5.12 points/Hz** — i.e. a **1200 Hz** window |
| Values | **real** (absorption-mode). Complex input is reduced to its real part. |
| Intensity | any *global* scale — each spectrum is min–max normalised, so overall receiver gain does not matter and no integral reference is needed. **Relative** intensities, SNR and line shape still matter. |
| Field strength | not an input. MolDeTr works in Hz (largely field-agnostic), so 80–600 MHz spectra all map to the same 1200 Hz window. |

`moldetr/validation.py` enforces the hard parts (length, finiteness); `predict.py` and the GUI call it
and tell you exactly what to fix.

## Why 1200 Hz

6144 points ÷ 5.12 points/Hz = **1200 Hz**. In ppm that width depends on the spectrometer:

| Field | 1200 Hz in ppm |
|---|---|
| 80 MHz | ~15 ppm |
| 300 MHz | ~4 ppm |
| 500 MHz | ~2.4 ppm |
| 600 MHz | ~2 ppm |

Choose a region no wider than 1200 Hz.

## The rule that trips people up

> [!IMPORTANT]
> **Keep coupling partners together.** A window does **not** have to contain the whole molecule — it
> can hold several unrelated spin systems. But **every proton that couples to a proton inside the
> window must also be inside the window.** If a multiplet's coupling partner falls outside the region,
> the observed splitting points to a peak the model cannot see, and that multiplet will be predicted
> wrong. When you draw a region, make sure each spin system in it is complete.

## Preparing a window

1. Process the FID to a phased, baseline-corrected real 1D spectrum. Any reader works — the open-source
   [`nmrglue`](https://www.nmrglue.com/) library reads Bruker, Varian/Agilent, JCAMP, and more.
2. Pick a region ≤ 1200 Hz wide that fully contains the spin systems you want.
3. Resample that region to 5.12 points/Hz.
4. Zero-pad (or crop) to 6144 points.
5. Save it as a `.npz` (see keys below), or a plain `.npy` of the 6144-point array.

<details>
<summary><b>Bruker TopSpin recipe</b> — phase and baseline before you export</summary>

MolDeTr is sensitive to phase and baseline distortion beyond its trained range, so get a clean
absorption spectrum first:

1. **Fourier transform:** `ft` (or `efp` with a window function).
2. **Phase:** `apk` (automatic). Use `apk0`/`apk1` for zero-/first-order only, `apks` for a robust
   variant, or `apkf` to phase on a region; fine-tune manually if the baseline still rolls.
3. **Baseline:** `absn` (automatic, node-based — usually the best) or `abs` over the region; `bcm`
   and `sab` are alternatives. A flat baseline matters more than it looks.
4. Export the real spectrum (e.g. read `pdata/1` with `nmrglue`), then resample to 5.12 points/Hz
   and crop/pad to 6144 as above.

Heavy phase or baseline distortion is out of distribution and degrades the prediction — see
[`SCOPE.md`](SCOPE.md).
</details>

## What the loaders read from an `.npz`

- **`spectrum_padded`** (or **`spec`**) — the 6144-point real spectrum. **Required.**
- **`ppm_axis_padded`** — the per-point ppm axis; its first and last values calibrate the plot so shifts
  come out in ppm. *Optional* — without it, shifts are reported in Hz and points, or you can pass
  `--ppm-left`/`--ppm-right` to `predict.py`.
- Anything else in the file is ignored.

---

**Next:** [Usage notes](USAGE_NOTES.md) — how to read the output, and how it fails.
