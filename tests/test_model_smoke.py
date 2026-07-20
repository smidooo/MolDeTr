"""Weight-free model-forward smoke test — exercises the whole DETR build + forward path on CPU, in CI.

With random weights (no checkpoint) we cannot assert *what* it detects, but we lock the *contract*: the
model builds, forwards a 6144-point spectrum without error, produces a finite
(n_groups*num_queries, n_classes+n_params) block, and that block decodes into structurally valid
detections. This pulls fpn_backbone / deformable_transformer / the deform-attn CPU op / positional_embedding
/ resnet_block / dnn into the fast CI lane (previously reachable only under the -m model gate).
"""

import numpy as np
import torch

from moldetr.inference import build_model, run
from moldetr.postprocess import PROTON_COUNTS, decode_predictions

N_CLASSES, N_PARAMS, N_GROUPS, N_QUERIES = 5, 7, 8, 10


def test_build_model_forwards_on_cpu_with_the_expected_contract():
    torch.manual_seed(0)
    model = build_model()  # defaults: n_classes=5, n_params=7, n_groups=8, num_queries=10
    model.eval()
    out = run(model, np.zeros(6144, dtype=np.float32))
    assert out.shape == (N_GROUPS * N_QUERIES, N_CLASSES + N_PARAMS)
    assert torch.isfinite(out).all()


def test_random_model_output_decodes_to_structurally_valid_detections(extrema):
    torch.manual_seed(0)
    model = build_model()
    model.eval()
    out = run(model, np.zeros(6144, dtype=np.float32))
    preds = decode_predictions(out, extrema, points_per_hz=5.12, threshold=0.3)
    assert isinstance(preds, list)
    for p in preds:  # random weights may yield none; whatever survives must be well-formed
        assert p["proton_count"] in PROTON_COUNTS
        assert all(j >= 0 for j in p.get("coupling_constants_hz", []))
