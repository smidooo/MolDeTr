# MolDeTr

[![DOI](https://zenodo.org/badge/1289888357.svg)](https://zenodo.org/badge/latestdoi/1289888357)
[![Paper](https://img.shields.io/badge/Paper-Anal.%20Chem.%202025-1e2d4d)](https://doi.org/10.1021/acs.analchem.5c03465)

**A Chemistry-Informed Deep Learning Model for Next-Generation Automated Analysis of ¹H NMR Spectra**

MolDeTr reads a raw, overlapping, strongly coupled 1D ¹H NMR spectrum and returns the resolved spin
system directly — for each group of equivalent spins it reports the chemical shift (δ), coupling
constants (J), proton count, and transverse relaxation time (T₂) — without prior structure, reference
standards, or iterative fitting. It is a one-pass detection transformer trained on quantum-mechanical
spin-dynamics simulations with realistic experimental distortions, and it generalises to experimental
spectra from 80 to 600 MHz.

This repository accompanies the peer-reviewed article in *Analytical Chemistry* and is archived on
Zenodo for long-term preservation.

> **Repository status.** Currently provides project and citation metadata. Code, usage instructions,
> and the accompanying materials will be added in a tagged release (which will be captured as a new
> archived Zenodo version).

## Paper
Schmid, N.; Wanner, M.; Fischetti, G.; Henrici, A.; Meshkian, M.; Bruderer, S.; Füchslin, R. M.;
Heitmann, B.; Wegner, J. D.; Sigel, R. K. O.; Wilhelm, D. **MolDeTr: A Chemistry-Informed Deep
Learning Model for Next-Generation Automated Analysis of ¹H NMR Spectra.** *Analytical Chemistry*
**2025**. DOI: [10.1021/acs.analchem.5c03465](https://doi.org/10.1021/acs.analchem.5c03465)

## Getting started
Installation and usage instructions will be provided with the code release.

## How to cite
Please cite the **article** as the primary reference for the method:

```bibtex
@article{Schmid2025MolDeTr,
  author  = {Schmid, Nicolas and Wanner, Marc and Fischetti, Giulia and Henrici, Andreas and
             Meshkian, Mohsen and Bruderer, Simon and Füchslin, Rudolf M. and Heitmann, Bjoern and
             Wegner, Jan Dirk and Sigel, Roland K. O. and Wilhelm, Dirk},
  title   = {{MolDeTr}: A Chemistry-Informed Deep Learning Model for Next-Generation
             Automated Analysis of $^{1}$H NMR Spectra},
  journal = {Analytical Chemistry},
  year    = {2025},
  doi     = {10.1021/acs.analchem.5c03465}
}
```

To cite the **software** (a specific archived version), use the Zenodo DOI in the badge above — the
concept DOI `<ZENODO_CONCEPT_DOI>` always resolves to the latest release. Machine-readable metadata is
in [`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button uses it.

## Availability
See the Data and Code Availability statements in the article.

## License
See [`LICENSE`](LICENSE). The terms for any released materials will be finalised with the content release.

## Contact
Corresponding authors: Nicolas Schmid (<nicolas.schmid.research@gmail.com>, ORCID
[0000-0003-1930-7654](https://orcid.org/0000-0003-1930-7654)); Dirk Wilhelm (<wilk@zhaw.ch>, ORCID
[0000-0001-5109-9803](https://orcid.org/0000-0001-5109-9803)).

## Acknowledgements
Supported by Innosuisse – Swiss Innovation Agency (Grant No. 2155007318).
