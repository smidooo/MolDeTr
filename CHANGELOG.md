# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Scope framing aligned with the paper.** Removed the "research prototype" / "well-resolved spectra"
  language that understated the peer-reviewed method; clarified that deviations come from
  out-of-distribution acquisition/processing (unusual distortions, non-standard pulse sequences,
  mixtures, non-1200 Hz windows), not from spectral resolution.
- **Figures consolidated to four.** The README now embeds exactly four images — the banner, the
  guajazulene 500 MHz prediction, the vanillin molecule↔spectrum figure, and the GUI. The redundant
  standalone vanillin prediction was removed; its worked-example detail (proton counts, δ, `max J`
  8.2/2.0/8.7 vs 8.0/2.0/8.0 Hz) is folded into the molecule↔spectrum caption.
- **Shared plotting style.** Prediction figures render through one shared matplotlib style, so generated
  prediction plots stay visually consistent. The README banner and diagrams ship as design-tool assets.
- **Label de-confliction with adjustText.** Multiplet annotations are placed with `adjustText` so labels
  no longer overlap peaks or one another on congested windows.
- **Documentation single-source-of-truth.** De-duplicated caveats that had spread across ~13 files.
  Canonical homes: `docs/INPUT_FORMAT.md` (the input contract + the keep-coupling-partners-in-the-window
  rule), `docs/SCOPE.md` (scope/limits, the `max J` caveat, the research-prototype disclaimer, and the
  input-noise rationale), `docs/USAGE_NOTES.md` (how to read the output — output table + failure-mode
  table). Every other file now gives a one-line mention plus a relative link. Also reconciled the
  complex-vs-real input wording (`DATA_SCHEMA.md` ↔ `INPUT_FORMAT.md`: arrays may be stored complex, the
  model consumes the real part) and standardised the |Δδ| median on **0.90 Hz** (aggregate/reproduced),
  with the paper-vs-aggregate 0.89/0.90 note stated exactly once (README *Reproducing the paper*).
- **Link-rot soft-gating.** The not-yet-published HF Space, Colab notebook, and Zenodo **data** DOI
  (`10.5281/zenodo.21217102`) are marked *coming soon* instead of presented as live; the live software
  DOI (`10.5281/zenodo.21214876`) is unchanged.

### Added
- **Animated demo + docs site.** An animated Gradio demo GIF in the README, a GitHub Pages landing page
  (`docs/index.md`), and a `.github/` PR template + CODEOWNERS.

### Removed
- **Matplotlib banner + molecule-figure generators.** Dropped `scripts/gen_banner.py`,
  `scripts/gen_molecule_figure.py`, and the `[figures]` / `rdkit` extra. The README banner and diagrams now
  ship as design-tool assets, so the generators — which produced an off-brand matplotlib banner and would
  overwrite the shipped assets if run — are no longer needed.

## [1.0.0] - 2026-07-15

Initial public release accompanying the *Analytical Chemistry* article
(DOI: 10.1021/acs.analchem.5c03465).

### Added
- `moldetr/` — 1D Deformable-DETR model package (FPN backbone + deformable transformer,
  with a pure-PyTorch fallback so inference runs without a compiled CUDA op).
- `scripts/predict.py` — checkpoint-only inference on a single 1H NMR spectrum.
- `scripts/aggregate_experimental.py` — reproduces the article's headline experimental numbers exactly
  (|Δδ| 0.90 Hz, |ΔJ| 0.20 Hz, proton-count 93.5%) from committed matched pairs.
- `scripts/evaluate_experimental.py` — evaluation on the preprocessed ROI arrays (no vendor reader required).
- `scripts/{train,evaluate_synthetic,quick_validation}.py` — training, synthetic evaluation, smoke test.
- `tests/test_reproducibility.py` — asserts the bundled example spectra decode back to their stored
  predictions (proton counts exact; δ and max J within tolerance), guarding the deterministic inference path.
- `pyproject.toml`, `environment.yml`, `requirements-lock-linux64.txt` — installable package and environments.
- `structured_output/*.json` — ground-truth ROI annotations for the 13-ROI / 44-spin-system test set,
  documented in `structured_output/README.md`.
- Continuous integration (ruff + quick_validation + pytest) on CPU, across Linux/macOS/Windows.
- `app.py` — Gradio GUI (assignment table + annotated plot) with post-upload input validation and
  selectable ppm-axis handling (auto / manual / none).
- `docs/SCOPE.md` and `docs/USAGE_NOTES.md` — scope, limitations, and how to read the output.
- `examples/` — bundled example windows (guajazulene 500 MHz, vanillin 300 MHz, synthetic).
- `notebooks/MolDeTr_quickstart.ipynb`, `.github/ISSUE_TEMPLATE/`, and Hugging Face Space files.

### Fixed
- **Inference input noise.** `moldetr/inference.py` fed the model a noiseless, FFT-resampled spectrum,
  but the network was trained (and the paper evaluated) on spectra carrying ~0.5%-of-max Gaussian noise;
  the clean input was out of distribution and could misread congested regions. Inference now injects the
  same calibrated noise with a fixed seed, so predictions are deterministic and in-distribution, and the
  live demo reproduces the paper on well-resolved spectra.
- Live coupling decode (`predict.py`, the GUI): report the single largest coupling `max(J)` rather
  than the four permutation-invariant embedding statistics `[sum, min, max, std]`. The exact paper
  couplings are unchanged — they come from the committed `structured_output` path.

### Changed
- Documentation now states that on well-resolved spectra the live path reproduces the paper's
  predictions — chemical shift, proton count, and the largest coupling `max(J)` come out accurate
  (the earlier "approximation" framing predated the noise fix); the committed `structured_output`
  path still recovers the full per-coupling set. Also corrects the input scale-invariance wording
  (only the global scale is normalised away; relative intensity and SNR still matter).
- Second worked example switched from ethylbenzene (80 MHz) to vanillin (300 MHz) — a cleaner ABX
  case whose live prediction matches its ground truth — with a molecule↔spin-system figure.

### Notes
- Trained weights and spectra are archived on Zenodo (DOI: 10.5281/zenodo.21217102), not in git.
