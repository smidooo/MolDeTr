"""The public contract downstream packages are allowed to depend on.

Three properties, each of which a downstream consumer (the private ``nmrsynth`` generator) was
previously forced to work around:

1. **A public transitions API.** ``nmrsynth.compat._engine`` reached into ``moldetr.simulate`` for
   ``_IM``, ``_build_hamiltonian`` and ``_embed`` because no public equivalent existed. Private
   names carry no compatibility promise, so a rename here would break it silently at a distance.
2. **``py.typed``.** Without it, type checkers treat this package as untyped and downstreams need a
   per-module ``ignore_missing_imports`` override, which suppresses real errors alongside the noise.
3. **Simulation does not require torch.** ``moldetr.simulate`` and ``moldetr.distort`` are pure
   NumPy/SciPy. Installing them should not drag in a multi-hundred-megabyte deep-learning stack.

The torch check runs in a subprocess with an import blocker rather than in-process, because
``tests/conftest.py`` imports torch itself — an in-process assertion could never fail.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # `tomllib` landed in 3.11; this package supports 3.10, so use the backport there.
    import tomli as tomllib

REPO = Path(__file__).resolve().parent.parent


# --- 1. the public transitions API ----------------------------------------------------------------


@pytest.mark.unit
def test_build_hamiltonian_is_public_and_infers_spin_count() -> None:
    """The public wrapper takes no redundant ``n_spins``; it reads it off the shifts."""
    from moldetr.simulate import build_hamiltonian

    shifts = np.array([100.0, 140.0], dtype=float)
    couplings = np.array([[0.0, 7.0], [7.0, 0.0]], dtype=float)

    hamiltonian, fx = build_hamiltonian(shifts, couplings)

    assert hamiltonian.shape == (4, 4), "2 spins -> a 2**2 product space"
    assert fx.shape == (4, 4)
    # H is Hermitian for a real shift/coupling system; a transposed-coupling bug breaks this.
    assert np.allclose(hamiltonian, hamiltonian.conj().T)


@pytest.mark.unit
def test_build_hamiltonian_matches_the_private_implementation() -> None:
    """The wrapper must not quietly diverge from the function the model was built on."""
    from moldetr.simulate import _build_hamiltonian, build_hamiltonian

    shifts = np.array([10.0, 55.0, 120.0], dtype=float)
    couplings = np.zeros((3, 3), dtype=float)
    couplings[0, 1] = couplings[1, 0] = 6.0
    couplings[1, 2] = couplings[2, 1] = 3.5

    public_h, public_fx = build_hamiltonian(shifts, couplings)
    private_h, private_fx = _build_hamiltonian(shifts, couplings, 3)

    assert np.array_equal(public_h, private_h)
    assert np.array_equal(public_fx, private_fx)


@pytest.mark.unit
def test_lowering_operators_are_public_and_correctly_embedded() -> None:
    """``I-`` on each spin of the product space — the ``_IM`` + ``_embed`` pair, made public."""
    from moldetr.simulate import lowering_operators

    operators = lowering_operators(2)

    assert len(operators) == 2
    assert all(op.shape == (4, 4) for op in operators)
    # I- lowers |alpha> to |beta>: exactly one nonzero per spin's single-quantum block.
    assert all(np.count_nonzero(op) == 2 for op in operators)
    # The two spins must differ; embedding on the wrong axis would make them identical.
    assert not np.array_equal(operators[0], operators[1])


@pytest.mark.unit
def test_transitions_is_public() -> None:
    """Single-quantum frequencies and intensities, without reaching for ``_transitions``."""
    from moldetr.simulate import build_hamiltonian, transitions

    shifts = np.array([100.0, 140.0], dtype=float)
    couplings = np.array([[0.0, 7.0], [7.0, 0.0]], dtype=float)

    freqs, intensities = transitions(*build_hamiltonian(shifts, couplings))

    assert freqs.shape == intensities.shape
    assert freqs.size > 0
    assert np.all(freqs > 0.0), "single-quantum transitions are reported as positive frequencies"
    # An AX system with J = 7 Hz gives four lines around the two shifts.
    assert freqs.size == 4


@pytest.mark.unit
def test_the_borrowed_private_names_still_exist() -> None:
    """Removing them would break the very downstream this change exists to unblock.

    The public API is additive: ``nmrsynth`` migrates on its own schedule, and until it does the
    private names must keep working.
    """
    import moldetr.simulate as simulate

    for name in ("_IM", "_build_hamiltonian", "_embed"):
        assert hasattr(simulate, name), f"downstream still borrows {name}"


# --- 2. py.typed ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_py_typed_marker_ships_with_the_package() -> None:
    """PEP 561: without this file, type checkers treat the whole package as untyped."""
    assert (REPO / "moldetr" / "py.typed").is_file()


@pytest.mark.unit
def test_py_typed_is_declared_as_package_data() -> None:
    """Present in the tree but missing from package-data means it is absent from the wheel."""
    with (REPO / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    package_data = config["tool"]["setuptools"]["package-data"]
    assert "py.typed" in package_data.get("moldetr", []), (
        "py.typed must be declared as package-data or it will not be installed"
    )


# --- 3. simulation without torch ------------------------------------------------------------------


_TORCH_FREE_PROBE = textwrap.dedent(
    """
    import importlib.abc, importlib.machinery, sys

    BLOCKED = ("torch", "fastai")

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            root = fullname.split(".")[0]
            if root in BLOCKED:
                raise ImportError(f"{fullname} is blocked: this path must not need it")
            return None

    sys.meta_path.insert(0, Blocker())

    import moldetr.simulate
    import moldetr.distort

    # Prove the blocker actually bites, so a silently-ineffective probe cannot pass.
    try:
        import torch
    except ImportError:
        print("OK")
    else:
        raise AssertionError("the import blocker did nothing; this test proves nothing")
    """
)


@pytest.mark.unit
def test_simulation_and_distortion_import_without_torch() -> None:
    """Pure NumPy/SciPy code must not require the deep-learning stack to import."""
    result = subprocess.run(
        [sys.executable, "-c", _TORCH_FREE_PROBE],
        capture_output=True,
        text=True,
        cwd=REPO,
    )

    assert result.returncode == 0, (
        f"importing moldetr.simulate/distort needed torch:\n{result.stderr}"
    )
    assert "OK" in result.stdout


@pytest.mark.unit
def test_torch_and_fastai_are_optional_dependencies() -> None:
    """They belong to the ``model`` extra, not to the base install."""
    with (REPO / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    base = " ".join(config["project"]["dependencies"])
    assert "torch" not in base, "torch must not be a base dependency"
    assert "fastai" not in base, "fastai must not be a base dependency"

    model_extra = " ".join(config["project"]["optional-dependencies"]["model"])
    assert "torch" in model_extra
    assert "fastai" in model_extra
