---
title: MolDeTr — 1H NMR Multiplet Detection
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: apache-2.0
---

# MolDeTr — ¹H NMR multiplet detection (live demo)

Upload a preprocessed 1-D ¹H NMR window (6144 points at 5.12 points/Hz — a 1200 Hz window; full contract
in [`docs/INPUT_FORMAT.md`](https://github.com/smidooo/MolDeTr/blob/main/docs/INPUT_FORMAT.md)) and read
off each multiplet's chemical shift (δ), largest coupling (max J), and proton count in one pass.

> **Research code accompanying the paper.** MolDeTr extracts δ, proton count, and couplings from real
> ¹H NMR spectra — including the congested, strongly-coupled cases it was built for. It is largely
> field-agnostic — it works in Hz, so it was tested across 80–600 MHz (and simulated down to ~5 MHz).
> Predictions can deviate for inputs outside its trained regime: unusual distortions, non-standard pulse
> sequences or processing, mixtures/impurities, or regions wider than the 1200 Hz window. `max J` is the
> dominant coupling per multiplet (the full set is in the repo's `structured_output/` path). See the
> [main repository](https://github.com/smidooo/MolDeTr) and its
> [`docs/SCOPE.md`](https://github.com/smidooo/MolDeTr/blob/main/docs/SCOPE.md) for what the model can and
> cannot do.

## How this Space is built

This Space is a copy of the [MolDeTr repository](https://github.com/smidooo/MolDeTr); its entry point
is the repo's `app.py`. The trained checkpoint (~974 MB, not in git) is provided one of two ways:

- **HF Hub (recommended):** host `model_spin_system_ABCDEFG_exp2.pth` in a model repo and download it
  at startup — add near the top of `app.py`:
  ```python
  import os
  from huggingface_hub import hf_hub_download
  os.environ.setdefault(
      "MOLDETR_CHECKPOINT",
      hf_hub_download("smidooo/moldetr", "model_spin_system_ABCDEFG_exp2.pth"),
  )
  ```
- **Committed via LFS:** place the checkpoint at `moldetr/model/model_spin_system_ABCDEFG_exp2.pth`.

Weights and data are archived on Zenodo (DOI 10.5281/zenodo.21217102).
