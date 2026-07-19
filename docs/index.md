<p align="center">
  <img src="banner.png" alt="MolDeTr — chemistry-informed deep learning for ¹H NMR multiplet detection" width="820">
</p>

MolDeTr reads a 1-D ¹H NMR spectrum and returns the spin systems in it directly: for each group of
equivalent protons it gives the chemical shift (δ), the coupling (J), the proton count, and the line
width, in one forward pass, with no prior structure and no iterative fitting. It was trained on
quantum-mechanical spin-dynamics simulations and tested on real spectra from 80 to 600 MHz.

## Start here

- **[Read the paper](https://doi.org/10.1021/acs.analchem.5c03465)** — *Analytical Chemistry*, 2026
- **[Code on GitHub](https://github.com/smidooo/MolDeTr)** — model, training, and evaluation
- **[Weights and data on Zenodo](https://doi.org/10.5281/zenodo.21217101)** — the trained checkpoint and the spectra
- **[Try it in Colab](https://colab.research.google.com/github/smidooo/MolDeTr/blob/main/notebooks/MolDeTr_colab_demo.ipynb)** — the interactive Detect and Simulate app

## Documentation

- [Scope and limitations](SCOPE.md) — what MolDeTr is and is not built for
- [Input format](INPUT_FORMAT.md) — the 6144-point, 5.12 points/Hz window it expects
- [Usage notes](USAGE_NOTES.md) — failure modes and how to read the output
- [Data schema](DATA_SCHEMA.md) — the ground-truth ROI annotation format

<p align="center"><sub>Apache-2.0 · accompanies the paper (DOI <a href="https://doi.org/10.1021/acs.analchem.5c03465">10.1021/acs.analchem.5c03465</a>)</sub></p>
