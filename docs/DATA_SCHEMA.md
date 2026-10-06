**Docs:** [README](../README.md) · [Scope &amp; limitations](SCOPE.md) · [Input format](INPUT_FORMAT.md) · [Usage notes](USAGE_NOTES.md) · **Data schema**

---

# Data &amp; label schema

Two datasets feed MolDeTr. They historically used different field names/units; `moldetr/labels.py`
normalizes both into one canonical `Multiplet` (positions in points, **couplings always in Hz**).

> [!NOTE]
> The spectra below are stored **complex**, but the model consumes the **real (absorption) part**:
> `moldetr/validation.py` takes it and warns when it does. That is why
> [`INPUT_FORMAT.md`](INPUT_FORMAT.md) asks for real-valued input while these files are complex.
> The two pages describe the same contract from opposite ends.

## Synthetic: `data/custom_spin_systems/*.npz` (clean spectra; on Zenodo)

| npz key | meaning |
|---|---|
| `spec` | complex spectrum, 6144 points |
| `labels` | list of dicts (below) |

label dict: `proton_number`, `center_position_in_points`, `line_width_in_points`,
`bounding_box_range_in_points`, `coupling_constants_in_points` (couplings in **points**).

## Experimental: `roi_S*.npz` (preprocessed ROIs; on Zenodo, in `experimental_rois/`)

| npz key | meaning |
|---|---|
| `spectrum_padded` | complex spectrum, 6144 points. **The model input**: use its real part. `ground_truth` is indexed in this frame |
| `spectrum_raw` | the ROI before padding, 500-6144 points. **Not a model input**: on 7 ROIs it starts `padding_before` points into the padded frame |
| `ppm_axis_padded`, `hz_axis_padded` | axes for `spectrum_padded` (`*_raw`: for `spectrum_raw`) |
| `ground_truth` | list of label dicts (below) |
| `predictions` | one run of the article's own pipeline (noise unseeded, one draw), not a reference a fresh run must equal |
| `metadata` | `points_per_hz` (5.12), `padding_before`, `padding_after`, `base_frequency_mhz`, `compound`, `solvent`, `roi_id`, `num_spin_systems`, `target_roi_points` |

label dict: `proton_count`, `chemical_shift_in_points`, `coupling_constants` (**Hz**), `chemical_shift_ppm`.

> [!WARNING]
> **Load these files with `moldetr.roi.load_roi`, not by hand.** Measured 2026-10-06 on the real
> checkpoint (13 ROIs x 5 noise seeds, scored with the article's matcher `moldetr.roi.match_hungarian`):
>
> | model input built from | labels matched | false positives | proton accuracy |
> |---|---|---|---|
> | real part of `spectrum_padded` (what `load_roi`, the GUI and the scripts do) | 204 / 220 | 17 | 86.8 % |
> | `spectrum_raw`, zero-padded at the end | 110 / 220 | 116 | 49.5 % |
> | `spectrum_raw`, stretched to 6144 points | 113 / 220 | 169 | 45.5 % |
> | magnitude of `spectrum_padded` | 205 / 220 | 105 | 68.6 % |
>
> **ppm referencing on the 80 MHz ROIs is offset.** Labels and axes agree with each other, so every
> Hz-based metric is unaffected. Against literature shifts, though, the ethyl acetate ROIs (S1, S6)
> sit about 4.6 ppm low (CH3 at -3.3 ppm), the ethylbenzene ROIs (S2, S7) about 0.5 ppm high, and S4
> about 0.8 ppm high. Compare these ROIs in Hz or in points, not in absolute ppm.

## Canonical schema (`moldetr.labels.Multiplet`)

`proton_count`, `center_in_points`, `coupling_constants_hz`, `line_width_in_points`,
`bounding_box_range_in_points`, `chemical_shift_ppm`.

> [!NOTE]
> Adapters: `from_synthetic` (÷5.12 on couplings: points → Hz), `from_experimental` (couplings
> already Hz). Committed JSON (`roi_S*.json`, `experimental_matched_pairs.json`) uses the
> experimental field names.

---

**Back to:** [Scope &amp; limitations](SCOPE.md) · [Input format](INPUT_FORMAT.md) · [Usage notes](USAGE_NOTES.md)
