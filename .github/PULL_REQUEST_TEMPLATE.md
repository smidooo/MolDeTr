<!-- Thanks for contributing to MolDeTr. Keep this short; delete sections that don't apply. -->

## What & why
<!-- One or two sentences: what this changes and the reason. Link any issue (#123). -->

## Type
- [ ] Bug fix
- [ ] New feature / capability
- [ ] Docs / figures / packaging
- [ ] Refactor or cleanup (no behaviour change)

## Checks
- [ ] `pytest -m "not e2e and not browser"` passes (the CI default suite)
- [ ] `ruff check moldetr scripts tests` is clean
- [ ] No weights, `.npz` spectra, or other Zenodo-hosted data are committed
- [ ] Behaviour changes touching the model / loss / matcher come with a test

## Notes for the reviewer
<!-- Anything non-obvious: a design choice, a caveat, a number that moved. -->
