**Docs:** [README](../README.md) · [Scope &amp; limitations](SCOPE.md) · [Input format](INPUT_FORMAT.md) · [Usage notes](USAGE_NOTES.md) · **Data schema**

---

# Data &amp; label schema

Two datasets feed MolDeTr. They historically used different field names/units; `moldetr/labels.py`
normalizes both into one canonical `Multiplet` (positions in points, **couplings always in Hz**).

## Synthetic — `data/custom_spin_systems/*.npz` (clean spectra; on Zenodo)

| npz key | meaning |
|---|---|
| `spec` | complex spectrum, 6144 points |
| `labels` | list of dicts (below) |

label dict: `proton_number`, `center_position_in_points`, `line_width_in_points`,
`bounding_box_range_in_points`, `coupling_constants_in_points` (couplings in **points**).

## Experimental — `roi_S*.npz` (preprocessed ROIs; on Zenodo, in `experimental_rois/`)

| npz key | meaning |
|---|---|
| `spectrum_padded` | complex spectrum, 6144 points (model input) |
| `ppm_axis_*`, `hz_axis_*` | axes |
| `ground_truth` | list of label dicts (below) |
| `predictions` | the article's model predictions |
| `metadata` | `points_per_hz` (5.12), `base_frequency_mhz`, `left_ppm`, `right_ppm`, ... |

label dict: `proton_count`, `chemical_shift_in_points`, `coupling_constants` (**Hz**), `chemical_shift_ppm`.

## Canonical schema (`moldetr.labels.Multiplet`)

`proton_count`, `center_in_points`, `coupling_constants_hz`, `line_width_in_points`,
`bounding_box_range_in_points`, `chemical_shift_ppm`.

> [!NOTE]
> Adapters: `from_synthetic` (÷5.12 on couplings — points → Hz), `from_experimental` (couplings
> already Hz). Committed JSON (`roi_S*.json`, `experimental_matched_pairs.json`) uses the
> experimental field names.

---

**Back to:** [Scope &amp; limitations](SCOPE.md) · [Input format](INPUT_FORMAT.md) · [Usage notes](USAGE_NOTES.md)
