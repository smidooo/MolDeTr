# Contributing to MolDeTr

Thanks for your interest in improving MolDeTr. This is research software released alongside a
peer-reviewed article; contributions that improve clarity, portability, tests, and documentation
are especially welcome.

## Development setup
```bash
git clone https://github.com/smidooo/MolDeTr.git
cd MolDeTr
conda env create -f environment.yml
conda activate moldetr
pip install -e ".[dev]"
```

## Before opening a pull request
- **Format & lint:** `ruff check moldetr scripts` (and `ruff format` if you use it).
- **Smoke test:** `python scripts/quick_validation.py` must pass (3/3 gating checks).
- **Tests:** `pytest -q` must pass; add a test for any new behavior. The suite is tiered by marker:
  - **CI lane (fast, weight-free):** `pytest -m "not e2e and not browser and not network"` — exactly what CI runs; keep it green.
  - **App e2e:** `pytest -m e2e` (needs `.[app]`) · **Browser:** `pytest -m browser` (needs `playwright install`).
  - **Real checkpoint:** `MOLDETR_CHECKPOINT=/path/to/model_spin_system_ABCDEFG_exp2.pth pytest -m model`
    (fetch the weights first: `python scripts/download_weights.py`).
  - **Notebooks:** `pip install -e ".[notebooks]" && pytest --nbmake notebooks/*.ipynb` (downloads the checkpoint).
- Keep changes focused; describe the motivation and how you verified them.

## Reporting issues
Please include your OS, Python and PyTorch versions, whether you built the CUDA op or used the
pure-PyTorch fallback, and a minimal way to reproduce the problem.

## Weights & data
Model weights and spectra live on Zenodo (10.5281/zenodo.21217102), not in this repository. Do not
commit `.pth`, `.npz`, or raw Bruker binaries.

## Code of conduct
By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
