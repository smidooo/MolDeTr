# Examples

Three small illustrative spectra (the **full** datasets live on Zenodo, DOI
[10.5281/zenodo.21217102](https://doi.org/10.5281/zenodo.21217102)):

- `roi_S10_example.npz`: one preprocessed **experimental** ROI (Guajazulene, 500 MHz) with
  `spectrum_padded`, `ppm_axis_padded`, `ground_truth`, `metadata`. Three overlapping aromatic protons.
- `roi_S8_example.npz`: **vanillin** aromatic ABX (300 MHz), same keys. The live predictions recover
  the pattern — one proton per multiplet, both ortho couplings near 8 Hz against a ground truth of
  8.1, and the meta one near 2.0. Exact values are in `docs/figure_predictions.json`, measured from
  the published checkpoint and tied to it by `tests/test_scripts_local.py`; the full coupling set per
  multiplet is in the committed `structured_output` path.

  Numbers used to be quoted here and in the main README as a fixed triple. They are not any more, on
  purpose: the pair that was published did not reproduce from the shipped checkpoint, and nothing in
  the repo could notice, because the only copy of them was pixels inside a PNG.
- `synthetic_example.npz`: one **synthetic** spectrum with `spec` + `labels`.

## Try it
Download the checkpoint into `moldetr/model/` (see the main README), then:
```bash
python scripts/predict.py --input examples/roi_S10_example.npz --plot
```
Prints the detected multiplets (δ, J, proton count, line width) and writes `prediction.png`.
Or launch the GUI: `python app.py`.
