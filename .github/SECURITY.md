# Security Policy

## Reporting a vulnerability
Please report security issues privately to **nicolas.schmid.research@gmail.com** rather than opening a
public issue. We aim to acknowledge reports within a few working days.

## Trust boundary — loading models and data
MolDeTr loads model weights and spectra with PyTorch and NumPy, which **deserialize arbitrary Python
objects**:

- `torch.load(...)` — the checkpoint. `moldetr/inference.py` loads with `weights_only=True` first. The
  published checkpoint is a fastai file carrying optimizer state, which the safe loader refuses, so a
  fallback to `weights_only=False` is unavoidable for it — but that fallback is **gated on the file's
  MD5 matching the published checkpoint**. Any other file that fails the safe load is refused rather
  than loaded.

  The gate is not cosmetic. `torch.load(..., weights_only=True)` raises `UnpicklingError` whenever it
  refuses a non-allowlisted global — which the published checkpoint and a malicious one both do — so
  the exception type cannot distinguish them, and an ungated fallback would turn the safe loader's
  *refusal* into the trigger for an unsafe load. Identity, not exception type, is what makes the
  fallback safe.

  If you are running weights you trained yourself, set `MOLDETR_ALLOW_UNTRUSTED_CHECKPOINT=1` to
  permit the fallback for an unrecognised file. It warns loudly and should only be used for
  checkpoints you produced.
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
