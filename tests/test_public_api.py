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

    # Three shifts must give a 2**3 space off the same call shape -- that is the inference.
    # (There was an `allclose(H, H.conj().T)` assertion here, commented as catching a
    # transposed-coupling bug. It cannot: only the i<j triangle is read, so H is Hermitian by
    # construction for *every* input, including couplings[1,0]=999. It was removed rather than
    # left to reassure a future reader that a case is covered when it is not.)
    bigger, _ = build_hamiltonian(np.array([10.0, 55.0, 120.0]), np.zeros((3, 3)))
    assert bigger.shape == (8, 8), "3 spins -> a 2**3 product space, read off the shifts"


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
def test_build_hamiltonian_rejects_couplings_it_would_silently_ignore() -> None:
    """Only ``i < j`` is read, so an unmirrored lower triangle must not pass as valid input.

    Filling the lower triangle instead of the upper one is a plausible convention, and before this
    guard it produced a *decoupled* Hamiltonian with no error: an AX pair came back as two singlets
    at 100/140 Hz rather than the four lines at 96.196/103.196/136.804/143.804.
    """
    from moldetr.simulate import build_hamiltonian

    shifts = np.array([100.0, 140.0], dtype=float)
    lower_only = np.array([[0.0, 0.0], [7.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="lower triangle"):
        build_hamiltonian(shifts, lower_only)


@pytest.mark.unit
def test_build_hamiltonian_rejects_asymmetric_couplings() -> None:
    """A lower entry that contradicts its mirror is a bug, not a convention: the upper one wins."""
    from moldetr.simulate import build_hamiltonian

    shifts = np.array([100.0, 140.0], dtype=float)
    contradictory = np.array([[0.0, 7.0], [-99.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="lower triangle"):
        build_hamiltonian(shifts, contradictory)


@pytest.mark.unit
def test_build_hamiltonian_accepts_both_documented_coupling_conventions() -> None:
    """Regression guard on the fix above: symmetric and upper-triangular must both stay valid.

    ``simulate`` and ``coupling_blocks`` both accept an upper-triangular matrix -- the natural
    output of an editor whose contract is "fill the upper triangle" -- so a symmetry check that
    rejected it would break the module's own documented convention.
    """
    from moldetr.simulate import build_hamiltonian

    shifts = np.array([100.0, 140.0], dtype=float)
    symmetric = np.array([[0.0, 7.0], [7.0, 0.0]], dtype=float)
    upper_only = np.array([[0.0, 7.0], [0.0, 0.0]], dtype=float)

    assert np.array_equal(
        build_hamiltonian(shifts, symmetric)[0], build_hamiltonian(shifts, upper_only)[0]
    )


@pytest.mark.unit
def test_build_hamiltonian_rejects_the_empty_system() -> None:
    """Zero spins passed the shape check and returned a 0-d scalar where an array is declared."""
    from moldetr.simulate import build_hamiltonian

    with pytest.raises(ValueError, match="at least one spin"):
        build_hamiltonian(np.zeros(0), np.zeros((0, 0)))


@pytest.mark.unit
def test_build_hamiltonian_honours_the_declared_block_ceiling() -> None:
    """``MAX_BLOCK_SPINS`` is the module's own limit; the public entry point ignored it.

    The body eagerly builds ``3n`` matrices of ``4**n`` complex128, so one spin past the ceiling
    is hundreds of megabytes -- the check has to happen before allocation, not after.
    """
    from moldetr.simulate import MAX_BLOCK_SPINS, build_hamiltonian

    n = MAX_BLOCK_SPINS + 1
    with pytest.raises(ValueError, match="MAX_BLOCK_SPINS|too large|at most"):
        build_hamiltonian(np.arange(n) * 10.0, np.zeros((n, n)))


@pytest.mark.unit
def test_the_ceiling_error_stays_readable_for_an_absurd_spin_count() -> None:
    """The guard exists to make this failure diagnosable, so its own message must not blow up.

    ``2**n_spins`` interpolated into the message is an ``n_spins``-bit integer rendered in decimal.
    For 6144 -- this project's own spectrum length, and exactly what a caller passing a spectrum
    where shifts belong would produce -- that is an 1850-digit number, and past ~14285 it trips
    CPython's integer-to-string limit and raises a *different* ValueError from inside the
    formatting.
    """
    from moldetr.simulate import lowering_operators

    with pytest.raises(ValueError) as excinfo:
        lowering_operators(6144)

    message = str(excinfo.value)
    assert len(message) < 200, f"the diagnostic is {len(message)} characters long: {message[:120]}…"
    assert "6144" in message, "it still has to say what was actually passed"


@pytest.mark.unit
def test_lowering_operators_rejects_a_non_positive_spin_count() -> None:
    """``lowering_operators(-3)`` returned ``[]``, which propagates as an all-zero spectrum."""
    from moldetr.simulate import lowering_operators

    with pytest.raises(ValueError, match="at least one spin"):
        lowering_operators(-3)


@pytest.mark.unit
def test_transitions_rejects_an_fx_that_does_not_match_the_hamiltonian() -> None:
    """Frequencies come from ``H``, so a mismatched ``F_x`` yields right lines, wrong intensities."""
    from moldetr.simulate import build_hamiltonian, transitions

    hamiltonian, _ = build_hamiltonian(np.array([100.0, 140.0]), np.array([[0.0, 7.0], [7.0, 0.0]]))
    _, wrong_fx = build_hamiltonian(np.array([10.0, 20.0, 30.0]), np.zeros((3, 3)))

    with pytest.raises(ValueError, match="same shape|shape"):
        transitions(hamiltonian, wrong_fx)


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

    # Everything above also holds for the *raising* operator I+, so none of it checks the one
    # thing the name promises. Direction lives in the position of the nonzeros: I- is strictly
    # lower-triangular (row > col), I+ is its transpose. Verified against a sabotaged I+ build.
    for op in operators:
        assert all(row > col for row, col in np.argwhere(op != 0)), (
            "lowering operators must be strictly lower-triangular; this is I+ (raising)"
        )


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

    # Everything above passes on a *decoupled* pair too -- it also yields four lines, as two
    # degenerate pairs at 100/100/140/140 -- and on freqs*2. So assert the physics, not the count:
    order = np.argsort(freqs)
    lines, amps = freqs[order], intensities[order]
    assert lines[1] - lines[0] == pytest.approx(7.0), (
        f"the doublet splitting must recover J = 7 Hz; got {lines} (a decoupled system gives "
        "two degenerate pairs, which the line count alone cannot distinguish)"
    )
    assert lines[3] - lines[2] == pytest.approx(7.0)
    # Second-order intensity asymmetry: the inner lines of an AB pair lean toward each other.
    assert amps[1] > amps[0], f"expected the roof effect on the inner lines, got {amps}"
    assert amps[2] > amps[3]


@pytest.mark.unit
def test_the_borrowed_private_names_still_exist() -> None:
    """Removing them would break the very downstream this change exists to unblock.

    The public API is additive: ``nmrsynth`` migrates on its own schedule, and until it does the
    private names must keep working.
    """
    import moldetr.simulate as simulate

    # `_lorentzian_sum` is the fourth: nmrsynth's tests/test_lineshape.py imports it to check its
    # own lineshape against this one. It was missing from this guard, so renaming it would have
    # broken the downstream suite with no tripwire on either side.
    for name in ("_IM", "_build_hamiltonian", "_embed", "_lorentzian_sum"):
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


@pytest.mark.unit
def test_every_extra_backing_a_network_command_provides_the_model_stack() -> None:
    """The other half of the contract: an extra that needs the network must *declare* it.

    Asserting only "torch is in the model extra" leaves the sufficiency half untested, and that is
    the half that broke: ``app`` and ``dev`` self-reference ``moldetr[model]`` while ``eval`` did
    not, even though ``scripts/evaluate_synthetic.py`` -- the script the extra exists for --
    imports torch at module scope and the README states all three already include it.
    """
    with (REPO / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    extras = config["project"]["optional-dependencies"]
    for name in ("app", "dev", "eval"):
        assert "moldetr[model]" in " ".join(extras[name]), (
            f"the {name!r} extra backs a documented command that loads the checkpoint, so it "
            "must pull in the model extra or that command fails on a clean install"
        )


#: Mimics a real missing dependency: ``ModuleNotFoundError`` with ``.name`` set, exactly as the
#: import machinery raises it, so the CLI's handler is exercised the way a user would trigger it.
_CLI_WITHOUT_TORCH = textwrap.dedent(
    """
    import importlib.abc, sys

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            root = fullname.split(".")[0]
            if root in ("torch", "fastai"):
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    sys.meta_path.insert(0, Blocker())

    # Prove the blocker actually bites, so a silently-ineffective probe cannot pass.
    try:
        import torch
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("the import blocker did nothing; this test proves nothing")

    from moldetr.cli import main
    main(["predict", "--demo"])
    """
)


_SET_SEED_WITHOUT_TORCH = textwrap.dedent(
    """
    import importlib.abc, sys
    import numpy as np

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname.split(".")[0] in ("torch", "fastai"):
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    sys.meta_path.insert(0, Blocker())
    try:
        import torch
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("the import blocker did nothing; this test proves nothing")

    np.random.seed(1234)
    before = np.random.get_state()[1][:8].tolist()

    from moldetr.reproducibility import set_seed
    try:
        set_seed(42)
    except ImportError:
        pass

    after = np.random.get_state()[1][:8].tolist()
    print("SAME" if before == after else f"MUTATED {before} -> {after}")
    """
)


@pytest.mark.unit
def test_set_seed_leaves_the_global_rng_alone_when_it_cannot_finish() -> None:
    """It fails either way on a base install; it must not half-seed the process on the way out.

    ``set_seed`` seeded Python and NumPy and only *then* imported torch, so a torch-free install
    got a partially-reseeded global RNG plus an exception -- the caller cannot tell how far it got.
    """
    result = subprocess.run(
        [sys.executable, "-c", _SET_SEED_WITHOUT_TORCH], capture_output=True, text=True, cwd=REPO
    )

    assert "SAME" in result.stdout, (
        f"set_seed mutated the global RNG before failing:\n{result.stdout}{result.stderr}"
    )


@pytest.mark.unit
def test_cli_names_the_remedy_when_the_model_stack_is_missing() -> None:
    """PyTorch became an extra in v1.1.0, so this failure is newly reachable by a documented path.

    The remedy lives only in the README; a user who pip-installs and runs the entry point sees a
    bare traceback. The error has to carry the fix.
    """
    result = subprocess.run(
        [sys.executable, "-c", _CLI_WITHOUT_TORCH], capture_output=True, text=True, cwd=REPO
    )

    assert result.returncode != 0, "predict cannot succeed without the deep-learning stack"
    combined = result.stdout + result.stderr
    assert "moldetr[model]" in combined, (
        f"the failure must name the fix, not only the missing module:\n{combined}"
    )


@pytest.mark.unit
def test_a_broken_but_installed_torch_is_not_reported_as_missing() -> None:
    """``ImportError.name`` reports the *innermost* failure, so it cannot tell the two apart.

    A ``DLL load failed while importing _C`` inside an installed torch -- the most common way this
    stack breaks on Windows, which CI runs -- arrives as ``name='torch._C'`` and roots at
    ``torch``. Inferring absence from that sends a user to ``pip install``, which answers
    "Requirement already satisfied", after the hint has replaced the traceback that would have
    shown them the real line. Absence has to be proven, not inferred.
    """
    from moldetr.cli import _missing_extra

    broken = ModuleNotFoundError("DLL load failed while importing _C", name="torch._C")

    assert _missing_extra("predict", broken) is None, (
        "torch is installed in this environment, so this must fall through to the real traceback"
    )


#: `moldetr app` needs gradio *before* it needs torch (``app.py`` imports gradio at module scope),
#: and gradio comes from the ``app`` extra -- ``moldetr[model]`` would not install it.
_CLI_APP_WITHOUT_GRADIO = textwrap.dedent(
    """
    import importlib.abc, sys

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname.split(".")[0] in ("gradio", "plotly", "torch", "fastai"):
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    sys.meta_path.insert(0, Blocker())

    # Prove the blocker actually bites, so a silently-ineffective probe cannot pass.
    try:
        import gradio
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("the import blocker did nothing; this test proves nothing")

    from moldetr.cli import main
    main(["app"])
    """
)


@pytest.mark.unit
def test_cli_app_names_the_app_extra_not_the_model_extra() -> None:
    """The GUI's first missing import is gradio, and ``moldetr[model]`` does not provide it."""
    result = subprocess.run(
        [sys.executable, "-c", _CLI_APP_WITHOUT_GRADIO], capture_output=True, text=True, cwd=REPO
    )

    assert result.returncode != 0, "the app cannot launch without gradio"
    combined = result.stdout + result.stderr
    assert "moldetr[app]" in combined, (
        f"`moldetr app` must point at the extra that actually supplies gradio:\n{combined}"
    )
