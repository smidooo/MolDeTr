"""Checkpoint-only inference: build the shipped model, load weights, run one spectrum.

Checkpoint-only and CPU-capable (uses the pure-PyTorch deformable-attention fallback when
the CUDA op is not compiled). The construction mirrors ``init_learner`` for the production
(``config_big``) configuration.
"""

from __future__ import annotations

import numpy as np
import torch

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


def load_checkpoint(model, ckpt_path, map_location: str = "cpu"):
    """Load a fastai-saved checkpoint (dict with a 'model' state_dict) with strict matching."""
    # Prefer the safe weights_only load; fall back for the fastai checkpoint that stores
    # optimizer state (source is the DOI-pinned Zenodo deposit, i.e. trusted).
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except Exception:
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


def run(model, amplitudes, noise_seed: int = 0) -> torch.Tensor:
    """Forward one spectrum; return a flat (n_groups*num_queries, num_classes+num_params) block."""
    with torch.no_grad():
        out = model(normalize_spectrum(amplitudes, noise_seed=noise_seed))
    return out[0].reshape(-1, out.shape[-1])
