# Examples

Three small illustrative spectra (the **full** datasets live on Zenodo, DOI
[10.5281/zenodo.21217102](https://doi.org/10.5281/zenodo.21217102)):

- `roi_S10_example.npz` — one preprocessed **experimental** ROI (Guajazulene, 500 MHz):
  `spectrum_padded`, `ppm_axis_padded`, `ground_truth`, `metadata`. Three overlapping aromatic protons.
- `roi_S8_example.npz` — **vanillin** aromatic ABX (300 MHz), same keys. The live predictions reproduce
  the ground truth (proton counts, δ, and largest coupling `max J` 8.2/2.0/8.7 vs 8.1/2.0/8.1 Hz).
- `synthetic_example.npz` — one **synthetic** spectrum: `spec` + `labels`.

## Try it
Download the checkpoint into `moldetr/model/` (see the main README), then:
```bash
python scripts/predict.py --input examples/roi_S10_example.npz --plot
```
Prints the detected multiplets (δ, J, proton count, line width) and writes `prediction.png`.
Or launch the GUI: `python app.py`.
