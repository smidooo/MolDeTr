# Third-party code in this distribution

This repository is distributed under the Apache License 2.0 (`LICENSE`). This file records
third-party code that has been part of it, and what was done about it.

## `moldetr/dataloader/shimming.py` — GPL — **removed**

| | |
|---|---|
| **Upstream** | [SHIMpanzee](https://github.com/smeerten/shimpanzee) |
| **Copyright** | © 2016–2017 Bas van Meerten and Wouter Franssen |
| **Licence** | **GPL-3.0-or-later** (upstream `LICENSE.md` is GPL v3, 29 June 2007; its headers read "either version 3 of the License, or (at your option) any later version"). GitHub reports `NOASSERTION` for that repository because its `LICENSE.md` is a Markdown-reformatted copy of the GPL — `====` underline headings, `&copy;`, HTML entities — which its licence detector does not match by content. A detection artefact, not an absence. |
| **Relationship** | adapted, not copied verbatim — the file's own header read *"Minimal SHIMpanzee code to simulate collate / modifed from https://github.com/smeerten/shimpanzee under GNU GPL licence"* |
| **What it provided** | `ShimSim`, a field-inhomogeneity (shim) simulator |
| **Status** | **removed** after v1.0.0. Present in the v1.0.0 release and its Zenodo archive; absent from every release after it. |

### Why it was removed

The file was GPL-derived and the repository as a whole is labelled Apache-2.0. Making the import
lazy — which this repository did — limits how far the GPL code reaches at *runtime*, but licensing
attaches to distribution, not to import. The file was present in every clone and every archive.

Rather than continue shipping GPL source under an Apache-2.0 label, the simulator was removed.

### What it costs, stated plainly

The shim branch was **roughly 50 %** of the 2024 training distribution. Removing the simulator
means this repository can no longer *re-apply* that branch, so full reproduction of the training-data
augmentation pipeline is **out of scope for the public release**.

This is a statement about reproduction, not about the model. The shipped weights **were** trained
with shim distortion (~50 % of samples) and line broadening (~35 %); `docs/SCOPE.md` documents the
ranges and they are unchanged. Anyone needing the shim simulator itself should obtain it from
SHIMpanzee upstream, under its own licence.

### What remains

- `moldetr.distort` is unaffected. It wraps only the five Apache-licensed `add_*` effects —
  noise, phase, baseline, ¹³C satellites and line broadening — and never reached the shim path.
- `moldetr.dataloader.data_augmentation.add_shim_distortions` still exists as a symbol and raises
  `NotImplementedError` pointing at this file, so a caller who invokes it directly gets an
  explanation rather than an `AttributeError`. Note the `toss_coin` branch inside
  `augment_distortions` that used to call it is unreachable in this tree — `toss_coin` is pinned to
  `0.99` — so the raise makes no promise about the augmentation distribution one way or the other.
- `tests/test_licensing.py` enforces all of the above, so an accidental re-introduction — a merge
  from an older branch, a restored file — fails the suite instead of going unnoticed.

## Deformable DETR and DETR — Apache-2.0 — **shipped**

| | |
|---|---|
| **Upstream** | [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), itself modified from [Deformable-Convolution-V2-PyTorch](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0); and [DETR](https://github.com/facebookresearch/detr) |
| **Copyright** | © 2020 SenseTime. All Rights Reserved. · © Facebook, Inc. and its affiliates. All Rights Reserved. · © 2018 Microsoft (via [DCN](https://github.com/msracver/Deformable-ConvNets), from which the CUDA kernels descend — see the header of `ms_deform_im2col_cuda.cuh`) |
| **Licence** | Apache License 2.0 — the same licence this repository ships under, so no boundary is crossed |
| **Where** | all of `moldetr/model/ops/**` — both the native sources (`src/ms_deform_attn.h`, `src/vision.cpp`, `src/cpu/*`, `src/cuda/*`) and the Python modules that wrap them (`functions/*.py`, `modules/*.py`, `setup.py`) — plus the Hungarian matcher `moldetr/matcher/matcher.py` |
| **Status** | **present and distributed**, including in the wheel |

Each of those files retains its own copyright and licence header, which is what Apache-2.0
§4(a)–(b) asks for; they are recorded here as well because this document is titled *third-party
code in this distribution* and would otherwise read as though the removed GPL file were the only
entry. It is the largest body of third-party code here, not the smallest.

## Fonts — SIL OFL 1.1 — **shipped**

Three families, in two places, all base64-embedded rather than fetched.

| | |
|---|---|
| **Upstream** | [Space Grotesk](https://github.com/floriankarsten/space-grotesk) · [IBM Plex Sans](https://github.com/google/fonts/tree/main/ofl/ibmplexsans) · [IBM Plex Mono](https://github.com/google/fonts/tree/main/ofl/ibmplexmono) |
| **Copyright** | © 2020 The Space Grotesk Project Authors · © 2017 IBM Corp. |
| **Licence** | **SIL Open Font License 1.1** for all three. OFL permits bundling and redistribution; it requires the licence to travel with the font data — and base64-embedding a subset *is* distributing font data — and forbids selling the font on its own. Neither family carries a Reserved Font Name, so subsetting under the original family name is compliant. This is not a copyleft obligation on the surrounding Apache-2.0 code. |
| **Where — app** | `app_ui/fonts/SpaceGrotesk-latin-var.woff2` (22 KB, latin subset, weight axis intact), embedded as a `data:` URI by `app_ui/theme.py`. Licence at `app_ui/fonts/OFL.txt`; provenance at `app_ui/fonts/README.md`. |
| **Where — diagrams** | `docs/fonts/{sg500,sg700,plex400,plex600,mono400}.woff2`, embedded into the generated SVGs by `scripts/build_diagram_svgs.py`. Licences at `docs/fonts/OFL-{SpaceGrotesk,IBMPlexSans,IBMPlexMono}.txt`; provenance at `docs/fonts/README.md`. |
| **Status** | **present and distributed** in the repository; none of it is in the wheel, since neither `app_ui` nor `docs` is a packaged module |

### Why the app font is vendored rather than fetched

The diagram subsets were always vendored. The *app* font was not: until issue #80 it was
`@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk…')` at the top of `CUSTOM_CSS`. A *pending* `@import` withholds the stylesheet it sits in, so a font CDN that
hangs — rather than failing — takes the entire layout with it. That is not hypothetical: CI run
`31390397645` failed `test_container_max_width_comes_from_custom_css` with `max-width: none` on a
branch whose diff could not touch styling, and its Playwright trace shows the Google Fonts request
issued and still unresolved 4.8 s later.

Vendoring also removes a third-party host from the render path of an artifact that carries a DOI and
is expected to keep working long after anyone is watching it.

`tests/e2e/test_browser_branding.py` holds all three halves: every request must be same-origin, the
layout CSS must still apply when a font CDN is routed to hang, and the embedded face must actually
load — the last one because the first two are both absences and neither can tell a working font from
a missing one.

## Attribution

The SHIMpanzee attribution above is retained deliberately. Version 1.0.0 of this software did
distribute the adapted file, and the credit is owed for that release regardless of its removal
from later ones.

## Status

This document records what was done. It is not a legal opinion, and no part of this repository's
licensing has been reviewed by anyone qualified to give one.
