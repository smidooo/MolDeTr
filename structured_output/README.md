# `structured_output/` — ground-truth annotations & matched predictions

This folder holds the **JSON metadata** for the 13-ROI / 44-spin-system experimental test set
(the `.npz` spectral arrays themselves live on Zenodo, DOI
[10.5281/zenodo.21217102](https://doi.org/10.5281/zenodo.21217102), not in git). Three files are
canonical; the per-ROI files are convenience slices.

## Canonical files

| File | Role | Read by |
|---|---|---|
| `roi_definitions.json` | **Input ground truth.** The analyst-defined ROIs and their spin-system labels — the annotations the model is scored against. | the export pipeline; humans |
| `experimental_matched_pairs.json` | **The paper's numbers.** Hungarian-matched `[prediction, label]` pairs — the single source that reproduces the headline experimental medians. | `scripts/aggregate_experimental.py` |
| `all_rois_combined.json` | **Full per-ROI view.** Every ROI with its `export_info`, compound/solvent, and each matched prediction↔label. | humans; downstream tooling |

### `roi_definitions.json`
A JSON list, one object per ROI:
```jsonc
{
  "sid": "S8",
  "roi": { "start": 6000, "end": null, "padding": 0 },   // window in points
  "labels": [                                             // one per spin system
    { "proton_count": 1, "chemical_shift": 6.96, "coupling_constants": 8.0 }
  ],
  "override": { "compound": "Vanillin", "solvent": "DMSO-d6" }
}
```
`chemical_shift` is in ppm; `coupling_constants` in Hz (omitted for singlets).

### `experimental_matched_pairs.json`
```jsonc
{
  "matched_pairs_total": [
    [ { "proton_count": 3, "chemical_shift_in_points": 3982.4, "coupling_constants": [7.25, 0.0, 0.0] },   // prediction
      { "proton_count": 3, "chemical_shift": -3.3, "coupling_constants": 7.13, "chemical_shift_in_points": 3983 } ]  // label
  ]
}
```
`aggregate_experimental.py` reads this and reproduces |Δδ| 0.90 Hz · |ΔJ| 0.20 Hz · proton-count 93.5 %.

### `all_rois_combined.json`
```jsonc
{
  "export_info": { "version": "3.0", "date": "2025-10-29", "total_rois": 13,
                   "total_spin_systems": 44, "target_roi_points": 6144, "points_per_hz": 5.12 },
  "rois": [ { "roi_id": "S8", "compound": "Vanillin", "solvent": "DMSO-d6",
              "num_spin_systems": 3, "matches": [ { "match_id": 0, "prediction": {…}, "label": {…} } ] } ]
}
```

## Per-ROI files — `roi_S1.json … roi_S13.json`

The 13 `roi_S*.json` files (including the `roi_S5_R2` re-measurement) are single-ROI slices of
`all_rois_combined.json`. They are **kept intentionally**: a downstream refinement pipeline consumes
them as its canonical per-ROI metadata. Regenerate them, and the two combined files, by re-running the
export after editing `roi_definitions.json` — do not hand-edit them out of sync.

> `chemical_shift_in_points` is on the 6144-point / 5.12-points-per-Hz grid; a trailing run of `0.0`
> coupling constants is padding, not measured couplings.
