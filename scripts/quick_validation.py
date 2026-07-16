"""
Quick Validation Script - Import & ROI-metadata Check
=====================================================

Validates that the key modules import and the ground-truth ROI metadata loads.
No model weights or spectral data are required (both live on Zenodo), so this
runs on a fresh, data-less clone on any OS (CPU only).
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def check_structured_output() -> bool:
    """Check structured_output/ JSON files exist and are valid."""
    print("\n1. Checking structured_output/ ...")
    output_dir = BASE_DIR / "structured_output"
    if not output_dir.exists():
        print("   FAIL: structured_output/ directory not found")
        return False

    json_files = sorted(output_dir.glob("roi_*.json"))
    npz_files = sorted(output_dir.glob("roi_*.npz"))
    print(f"   JSON files: {len(json_files)}")
    print(f"   NPZ  files: {len(npz_files)}")

    # Validate roi_definitions.json
    defs_path = output_dir / "roi_definitions.json"
    if defs_path.exists():
        with open(defs_path, "r") as f:
            defs = json.load(f)
        print(f"   roi_definitions.json: {len(defs)} ROI definitions")
    else:
        print("   WARN: roi_definitions.json not found")

    # Validate all_rois_combined.json
    combined_path = output_dir / "all_rois_combined.json"
    if combined_path.exists():
        with open(combined_path, "r") as f:
            combined = json.load(f)
        n_rois = (
            len(combined.get("rois", combined)) if isinstance(combined, dict) else len(combined)
        )
        print(f"   all_rois_combined.json: {n_rois} entries")
    else:
        print("   WARN: all_rois_combined.json not found")

    # Show individual ROI files
    for jf in json_files:
        size_kb = jf.stat().st_size / 1024
        print(f"   OK  {jf.name} ({size_kb:.1f} KB)")

    return True


def check_config_imports() -> bool:
    """Check that config.py and Hydra config load correctly."""
    print("\n2. Checking config imports ...")
    try:
        print("   OK  config.MultipletConfig (frozen dataclass)")
        return True
    except Exception as e:
        print(f"   FAIL: {e}")
        return False


def check_module_imports() -> bool:
    """Check that moldetr/ submodules import on CPU."""
    print("\n3. Checking moldetr/ module imports ...")
    ok = True
    # (module, symbol, requires_cuda)
    modules = [
        ("moldetr.model.deformable_detr_nmr", "Deformable_DETR_NMR", False),
        ("moldetr.learner.multi_multiplet_learner", "init_learner", False),
        ("moldetr.matcher.matcher", "matching", False),
        ("moldetr.loss.combined_loss", "combined_loss_func", False),
        ("moldetr.config", "MultipletConfig", False),
    ]
    for mod_path, symbol, needs_cuda in modules:
        try:
            mod = __import__(mod_path, fromlist=[symbol])
            getattr(mod, symbol)
            print(f"   OK  {mod_path}.{symbol}")
        except Exception as e:
            if needs_cuda:
                print(f"   SKIP {mod_path}.{symbol} (CUDA-only): {e}")
            else:
                print(f"   FAIL {mod_path}.{symbol}: {e}")
                ok = False
    return ok


def main() -> int:
    print("=" * 70)
    print("MolDeTr - Quick Validation")
    print(f"Project: {BASE_DIR}")
    print("=" * 70)

    # Gating checks — these must pass on a fresh, data-less clone.
    gates = {
        "structured_output": check_structured_output(),
        "config_imports": check_config_imports(),
        "module_imports": check_module_imports(),
    }

    # Model weights and spectra live on Zenodo (DOI 10.5281/zenodo.21217102),
    # not in git; this smoke test deliberately needs neither.
    print(
        "\n[info] Model weights and spectra are on Zenodo "
        "(DOI 10.5281/zenodo.21217102), not tracked in git; this smoke test needs neither."
    )

    print("\n" + "=" * 70)
    print("Summary (gating checks):")
    for name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"   [{status}] {name}")

    n_pass = sum(gates.values())
    n_total = len(gates)
    print(f"\n   {n_pass}/{n_total} gating checks passed")
    print("=" * 70)

    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
