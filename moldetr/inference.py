"""Checkpoint-only inference: build the shipped model, load weights, run one spectrum.

Checkpoint-only and CPU-capable (uses the pure-PyTorch deformable-attention fallback when
the CUDA op is not compiled). The construction mirrors ``init_learner`` for the production
(``config_big``) configuration.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch

from moldetr.checkpoint_meta import ALLOW_UNTRUSTED_ENV, TRUSTED_CHECKPOINT_MD5
from moldetr.checkpoint_meta import file_md5 as _md5
from moldetr.model.deformable_detr_nmr import Deformable_DETR_NMR
from moldetr.model.deformable_transformer import DeformableTransformer
from moldetr.model.fpn_backbone import FPN_BB
from moldetr.model.positional_embedding import LearnedPositionalEncoding
from moldetr.model.utils import ParamEmbedding


def build_model(
    d_model: int = 256,
    n_classes: int = 5,
    n_params: int = 7,
    n_groups: int = 8,
    num_queries: int = 10,
    input_length: int = 6144,
    num_decoder_layers: int = 6,
    n_levels: int = 4,
) -> Deformable_DETR_NMR:
    """Build the production model (defaults match conf/config_big.yaml)."""
    backbone = FPN_BB(
        input_length=input_length,
        number_of_classes=n_classes,
        num_multiplet_pred=num_queries // n_groups,
        kernel_size=11,
        num_params=n_params,
        pyramid_layers=9,
        channel_dim_up=d_model,
        pool_size=128,
        cnn_output_dimension=d_model,
    )
    positional = LearnedPositionalEncoding(d_model=d_model, max_len=input_length)
    param_embed = ParamEmbedding(
        num_params=n_params, hidden_dim=d_model, num_decoder_layers=num_decoder_layers
    )
    transformer = DeformableTransformer(
        d_model=d_model,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=1024,
        dropout_ratio=0.1,
        n_levels=n_levels,
        n_points=4,
        param_embed=param_embed.parameter_embed,
    )
    model = Deformable_DETR_NMR(
        backbone=backbone,
        positional_encoding=positional,
        transformer=transformer,
        num_classes=n_classes,
        num_params=n_params,
        num_queries=num_queries,
        hidden_dim=d_model,
        backbone_output_dim=d_model,
        n_groups=n_groups,
        d_model=d_model,
        n_levels=n_levels,
        channel_size=d_model,
        parameter_embed=param_embed.parameter_embed,
    )
    return model.eval()


def _require_trusted_checkpoint(ckpt_path, cause: BaseException) -> None:
    """Decide whether dropping `weights_only` protection is permissible for *this* file.

    ``torch.load(..., weights_only=True)`` raises ``pickle.UnpicklingError`` whenever it refuses a
    global that is not on its allowlist (``torch/serialization.py``). The published fastai checkpoint
    trips that because it carries optimizer state — and a hostile checkpoint trips it for the reason
    the guard exists. **The exception cannot tell the two apart**, so catching something narrower
    would not help; the decision has to be made on the file's identity.

    Before this gate, the refusal itself was what unlocked the unsafe load:

        weights_only=True → refuses payload → raises → except Exception → weights_only=False → runs it
    """
    if os.environ.get(ALLOW_UNTRUSTED_ENV, "").strip().lower() in {"1", "true", "yes"}:
        warnings.warn(
            f"{ALLOW_UNTRUSTED_ENV} is set, so {ckpt_path} is being loaded with weights_only=False, "
            "which executes arbitrary code from the file. Only do this for checkpoints you produced.",
            RuntimeWarning,
            stacklevel=3,
        )
        return

    digest = _md5(ckpt_path)
    if digest == TRUSTED_CHECKPOINT_MD5:
        return

    raise RuntimeError(
        f"Refusing to load {ckpt_path} without weights_only protection.\n"
        f"  expected MD5 {TRUSTED_CHECKPOINT_MD5}  (the published checkpoint)\n"
        f"  actual   MD5 {digest}\n"
        "torch's safe loader rejected this file and it is not the published checkpoint, so loading "
        "it would execute arbitrary code from it. Fetch the published weights with "
        "`python scripts/download_weights.py`, or, if this is a checkpoint you trained and trust, "
        f"set {ALLOW_UNTRUSTED_ENV}=1."
    ) from cause


def load_checkpoint(model, ckpt_path, map_location: str = "cpu"):
    """Load a fastai-saved checkpoint (dict with a 'model' state_dict) with strict matching.

    Prefers the safe ``weights_only`` load. The fallback the fastai format needs is gated on the file
    being the published checkpoint — see :func:`_require_trusted_checkpoint`.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except Exception as safe_load_refused:
        # Deliberately still a broad catch. Narrowing to UnpicklingError would add a regression risk
        # (a future torch could raise something else for the same condition and the published
        # checkpoint would stop loading) while buying no security: the trust gate below, not the
        # exception type, is what makes the fallback safe.
        _require_trusted_checkpoint(ckpt_path, safe_load_refused)
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return model


def normalize_spectrum(
    amplitudes, input_length: int = 6144, noise_seed: int = 0, noise_frac: float = 0.005
) -> torch.Tensor:
    """Min-max normalize a real 1D spectrum to [0, 1] and shape it as (1, 1, L).

    Enforces the input contract (length, finiteness) via ``validate_spectrum`` so a wrong-sized
    array fails with a clear message instead of an opaque error deep in the backbone.

    The model was trained on spectra carrying realistic noise (SNR 10²–10⁵). A perfectly clean,
    FFT-resampled ROI is *out of distribution* — the detector reads such inputs less accurately (e.g.
    miscounting a triplet). Matching the paper's evaluation, we inject calibrated Gaussian noise
    (``noise_frac`` of the maximum amplitude, default 0.5 %) before the backbone. The noise is seeded
    (``noise_seed``) so inference stays deterministic and reproducible across ``predict.py``, the GUI,
    and the notebook. Set ``noise_frac=0`` to disable (not recommended for real, processed spectra).
    """
    from moldetr.validation import validate_spectrum

    a = validate_spectrum(amplitudes, warn=False).astype(np.float32)
    if noise_frac:
        rng = np.random.RandomState(noise_seed)
        a = a + rng.normal(0.0, noise_frac * float(np.max(a)), a.shape).astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-12)
    return torch.from_numpy(a).float()[None, None, :]


def run(model, amplitudes, noise_seed: int = 0, noise_frac: float = 0.005) -> torch.Tensor:
    """Forward one spectrum; return a flat (n_groups*num_queries, num_classes+num_params) block.

    ``noise_frac`` is forwarded so the in-model noise floor is reachable from the public API. It
    previously was not: ``normalize_spectrum`` accepted it, but ``run`` neither took nor passed it,
    so every production caller was pinned to 0.005 and ``docs/SCOPE.md``'s advice to "set
    ``noise_frac=0``" described an affordance no supported call path offered.
    """
    with torch.no_grad():
        out = model(normalize_spectrum(amplitudes, noise_seed=noise_seed, noise_frac=noise_frac))
    return out[0].reshape(-1, out.shape[-1])
