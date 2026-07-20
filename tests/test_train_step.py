"""Weight-free training-step smoke: one forward+backward through the model completes cleanly.

Asserts the step runs end-to-end with **finite** gradients (no NaN/Inf, no backward crash) and actually
moves the model. We assert grad *flow* and *finiteness*, not "loss decreases" (flaky on random init).

Note: a plain L1 on the final output reaches only the parameters that feed it (~38 %); the intermediate
decoder layers / backbone levels get gradient from the auxiliary per-layer terms of the real training
objective (``combined_loss_func``). A full dead-subgraph check would wire that objective up — deferred; this
smoke still catches a broken forward/backward, NaN gradients, or a fully-detached graph.
"""

import torch
import torch.nn.functional as F

from moldetr.inference import build_model


def test_one_backward_produces_finite_gradients_and_updates_the_model():
    torch.manual_seed(0)
    model = build_model()
    model.train()
    out = model(torch.randn(1, 1, 6144))  # (1, n_groups, n_queries, n_classes+n_params)
    loss = F.l1_loss(out, torch.randn_like(out))
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for _, p in model.named_parameters() if p.grad is not None]
    assert grads, "backward produced no gradients at all (fully detached graph)"
    assert all(torch.isfinite(g).all() for g in grads)  # no NaN/Inf in any gradient
    assert any(g.abs().sum() > 0 for g in grads)  # the step actually moves something
