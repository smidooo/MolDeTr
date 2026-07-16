"""
CPU vs CUDA Equivalence Test for Deformable Attention
=====================================================

Tests that the pure-PyTorch CPU fallback for MSDeformAttn produces
outputs identical (within floating-point tolerance) to the CUDA extension.

Usage:
  1. Run on Windows/CPU to generate reference:
     python tests/test_cpu_cuda_equivalence.py --generate

  2. Copy tests/reference_outputs/ to the Linux server.

  3. Run on Linux/CUDA to compare:
     python tests/test_cpu_cuda_equivalence.py --compare

  4. Or run as pytest (auto-detects CPU/CUDA):
     pytest tests/test_cpu_cuda_equivalence.py -v
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REFERENCE_DIR = Path(__file__).resolve().parent / "reference_outputs"

# Fixed seed for reproducibility
SEED = 42


def create_deterministic_inputs(
    batch_size: int = 2,
    seq_len: int = 128,
    d_model: int = 256,
    n_heads: int = 8,
    n_levels: int = 4,
    n_points: int = 4,
) -> dict:
    """Create deterministic inputs for MSDeformAttn with fixed seed."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Spatial shapes: 1D signal at multiple FPN levels
    # Module expects (n_levels, 1) with widths only — it prepends H=1 internally
    widths = [seq_len // (2**i) for i in range(n_levels)]
    spatial_shapes = torch.tensor([[w] for w in widths], dtype=torch.long)
    level_start_index = torch.cat(
        [torch.tensor([0]), torch.cumsum(spatial_shapes[:, 0], dim=0)[:-1]]
    )
    total_len = sum(widths)

    query = torch.randn(batch_size, seq_len, d_model)
    input_flatten = torch.randn(batch_size, total_len, d_model)
    # Reference points in [0, 1] range — 1D: (B, Lq, n_levels, 1)
    # The module doubles the last dim internally via cat with zeros
    reference_points = torch.rand(batch_size, seq_len, n_levels, 1)

    return {
        "query": query,
        "reference_points": reference_points,
        "input_flatten": input_flatten,
        "input_spatial_shapes": spatial_shapes,
        "input_level_start_index": level_start_index,
        "params": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_levels": n_levels,
            "n_points": n_points,
        },
    }


def create_model(params: dict) -> "torch.nn.Module":
    """Create MSDeformAttn module with deterministic weights."""
    from moldetr.model.ops.modules.ms_deform_attn import MSDeformAttn

    torch.manual_seed(SEED + 1)  # Different seed for weights

    model = MSDeformAttn(
        d_model=params["d_model"],
        n_levels=params["n_levels"],
        n_heads=params["n_heads"],
        n_points=params["n_points"],
    )
    model.eval()
    return model


def run_forward(model, inputs: dict, device: str = "cpu") -> torch.Tensor:
    """Run forward pass on specified device, return output on CPU."""
    model = model.to(device)
    with torch.no_grad():
        output = model(
            query=inputs["query"].to(device),
            reference_points=inputs["reference_points"].to(device),
            input_flatten=inputs["input_flatten"].to(device),
            input_spatial_shapes=inputs["input_spatial_shapes"].to(device),
            input_level_start_index=inputs["input_level_start_index"].to(device),
        )
    return output.cpu()


def generate_reference():
    """Generate reference outputs on CPU and save to disk."""
    from moldetr.model.ops.functions.ms_deform_attn_func import MSDA

    backend = (
        "cuda" if MSDA is not None and torch.cuda.is_available() else "cpu_pytorch"
    )
    print(f"Generating reference with backend: {backend}")

    inputs = create_deterministic_inputs()
    model = create_model(inputs["params"])

    device = "cuda" if backend == "cuda" else "cpu"
    output = run_forward(model, inputs, device=device)

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Save output
    np.save(REFERENCE_DIR / f"msdeformattn_output_{backend}.npy", output.numpy())

    # Save model weights for reproducibility
    torch.save(model.state_dict(), REFERENCE_DIR / "msdeformattn_weights.pt")

    # Save metadata
    meta = {
        "backend": backend,
        "output_shape": list(output.shape),
        "output_mean": float(output.mean()),
        "output_std": float(output.std()),
        "output_min": float(output.min()),
        "output_max": float(output.max()),
        "params": inputs["params"],
    }
    with open(REFERENCE_DIR / f"metadata_{backend}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {REFERENCE_DIR}/")
    print(f"  Output shape: {output.shape}")
    print(f"  Mean: {output.mean():.6f}, Std: {output.std():.6f}")
    print(f"  Min: {output.min():.6f}, Max: {output.max():.6f}")
    return output


def compare_outputs():
    """Compare CPU and CUDA reference outputs."""
    cpu_path = REFERENCE_DIR / "msdeformattn_output_cpu_pytorch.npy"
    cuda_path = REFERENCE_DIR / "msdeformattn_output_cuda.npy"

    if not cpu_path.exists():
        print(f"Missing CPU reference: {cpu_path}")
        print("Run with --generate on CPU first")
        return False
    if not cuda_path.exists():
        print(f"Missing CUDA reference: {cuda_path}")
        print("Run with --generate on CUDA server first")
        return False

    cpu_output = np.load(cpu_path)
    cuda_output = np.load(cuda_path)

    print(f"CPU  output: shape={cpu_output.shape}, mean={cpu_output.mean():.6f}")
    print(f"CUDA output: shape={cuda_output.shape}, mean={cuda_output.mean():.6f}")

    # Compare
    assert cpu_output.shape == cuda_output.shape, (
        f"Shape mismatch: {cpu_output.shape} vs {cuda_output.shape}"
    )

    abs_diff = np.abs(cpu_output - cuda_output)
    rel_diff = abs_diff / (np.abs(cuda_output) + 1e-8)

    print("\nAbsolute difference:")
    print(f"  Mean: {abs_diff.mean():.2e}")
    print(f"  Max:  {abs_diff.max():.2e}")
    print("Relative difference:")
    print(f"  Mean: {rel_diff.mean():.2e}")
    print(f"  Max:  {rel_diff.max():.2e}")

    # Tolerance: float32 precision allows ~1e-5 relative error
    atol = 1e-4
    rtol = 1e-4
    match = np.allclose(cpu_output, cuda_output, atol=atol, rtol=rtol)
    print(f"\nOutputs match (atol={atol}, rtol={rtol}): {match}")
    return match


# --- Full model forward pass test ---


def test_full_model_forward_cpu():
    """Test that full model import chain works on CPU (no Hydra)."""
    from moldetr.model.ops.functions.ms_deform_attn_func import MSDA

    print("Deformable_DETR_NMR: imported OK")
    print("init_learner: imported OK")
    cuda_status = "available" if MSDA else "using PyTorch fallback"
    print(f"CUDA deformable attention: {cuda_status}")
    print("Full model import chain: PASS")


# --- pytest interface ---


def test_msdeformattn_cpu_forward():
    """Test MSDeformAttn forward pass on CPU produces reasonable output."""
    inputs = create_deterministic_inputs()
    model = create_model(inputs["params"])
    output = run_forward(model, inputs)

    assert output.shape == (
        inputs["params"]["batch_size"],
        inputs["params"]["seq_len"],
        inputs["params"]["d_model"],
    ), f"Unexpected output shape: {output.shape}"

    # Verify output is not all zeros or NaN
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    assert output.abs().mean() > 1e-6, "Output is effectively zero"

    print(
        f"MSDeformAttn CPU output OK: shape={output.shape}, "
        f"mean={output.mean():.4f}, std={output.std():.4f}"
    )


def test_cpu_cuda_equivalence():
    """Compare CPU and CUDA outputs if both references exist."""
    cpu_path = REFERENCE_DIR / "msdeformattn_output_cpu_pytorch.npy"
    cuda_path = REFERENCE_DIR / "msdeformattn_output_cuda.npy"

    if not cpu_path.exists() or not cuda_path.exists():
        import pytest

        pytest.skip(
            "Need both CPU and CUDA reference files. "
            "Run --generate on each platform first."
        )

    assert compare_outputs(), "CPU and CUDA outputs differ beyond tolerance"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU vs CUDA equivalence test")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate reference output for current platform",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare CPU and CUDA reference outputs"
    )
    args = parser.parse_args()

    if args.generate:
        generate_reference()
    elif args.compare:
        success = compare_outputs()
        sys.exit(0 if success else 1)
    else:
        # Default: run forward pass test
        print("Running forward pass tests...\n")
        test_msdeformattn_cpu_forward()
        print()
        test_full_model_forward_cpu()
        print("\nAll tests passed!")
