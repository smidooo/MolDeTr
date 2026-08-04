# Security Policy

## Reporting a vulnerability
Please report security issues privately to **nicolas.schmid.research@gmail.com** rather than opening a
public issue. We aim to acknowledge reports within a few working days.

## Trust boundary — loading models and data
MolDeTr loads model weights and spectra with PyTorch and NumPy, which **deserialize arbitrary Python
objects**:

- `torch.load(...)` — the checkpoint. `moldetr/inference.py` loads with `weights_only=True` first and only
  falls back to `weights_only=False` for the fastai-format checkpoint.
- `numpy.load(..., allow_pickle=True)` — the `.npz` arrays carry object metadata, so pickling is required.

**Only load checkpoints and `.npz` files that you trust** — specifically the artifacts published on the
official Zenodo record (DOI `10.5281/zenodo.21217102`). Do not run these loaders on files from untrusted
sources; a malicious checkpoint/npz can execute arbitrary code.

## What is not in this repository
No credentials, API keys, private endpoints, or raw proprietary data are committed. The proprietary vendor
NMR reader used to preprocess the raw spectra is **not** included; only its preprocessed *outputs* are
deposited on Zenodo.

## Supported versions
Security fixes target the latest tagged release on `main`.
