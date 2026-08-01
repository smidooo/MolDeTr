"""The distribution's licence boundary, asserted rather than documented.

MolDeTr ships under Apache-2.0 (``LICENSE``). ``moldetr/dataloader/shimming.py`` was adapted
from `SHIMpanzee <https://github.com/smeerten/shimpanzee>`_ under the GNU GPL, and was removed
from the public repository so that no GPL source is distributed under an Apache-2.0 label.

The removal is easy to undo by accident -- a merge from an older branch, a restored file, a
vendored copy under a new name -- and nothing else in the suite would notice. These tests fail
loudly if it comes back, and pin the two properties that made removal safe: ``moldetr.distort``
never reached the shim path anyway, and the public symbol still exists so callers get an
explanation instead of an ``AttributeError``.

``THIRD_PARTY.md`` is checked for the SHIMpanzee attribution because v1.0.0 *did* ship the file;
the attribution is owed for that release whether or not the file is present today.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.unit
def test_gpl_shimming_module_is_not_distributed() -> None:
    """No file named ``shimming.py`` anywhere under the shipped package."""
    offenders = sorted(
        p.relative_to(REPO).as_posix() for p in (REPO / "moldetr").rglob("shimming.py")
    )
    assert offenders == [], (
        f"GPL-derived shim source is back in an Apache-2.0 distribution: {offenders}. "
        "See THIRD_PARTY.md -- removing it was decision D-1, not a cleanup."
    )


#: Identifiers from the SHIMpanzee implementation itself -- the spherical-harmonic shim terms.
#: The *names* ``ShimSim`` and ``shimpanzee`` deliberately survive in the removal docstring and in
#: the ``NotImplementedError`` message, so matching on those would flag the documentation that is
#: supposed to be there. These appear only in the code.
_GPL_IMPLEMENTATION_MARKERS = ("class ShimSim", "ZX2_ZY2LIM", "zx2_zy2", "X2_Y2")


@pytest.mark.unit
def test_no_gpl_shim_implementation_survives_under_any_filename() -> None:
    """The filename check above is blind to the threat this module's docstring actually names.

    ``shim_sim.py``, ``_shimming.py``, ``moldetr/vendor/shimpanzee.py``, or ``ShimSim`` pasted back
    into ``data_augmentation.py`` all pass a glob for ``shimming.py``. Match the implementation
    instead of the filename. Scoped to ``moldetr/`` -- the shipped package -- so this file's own
    marker list does not match itself.
    """
    offenders: list[str] = []
    for source in (REPO / "moldetr").rglob("*.py"):
        text = source.read_text(encoding="utf-8", errors="replace")
        found = [marker for marker in _GPL_IMPLEMENTATION_MARKERS if marker in text]
        if found:
            offenders.append(f"{source.relative_to(REPO).as_posix()} contains {found}")

    assert offenders == [], (
        "GPL-derived shim implementation is back in an Apache-2.0 distribution:\n"
        + "\n".join(offenders)
        + "\nSee THIRD_PARTY.md -- removing it was decision D-1, not a cleanup."
    )


@pytest.mark.unit
def test_gpl_shimming_module_is_not_importable() -> None:
    """Catches a copy shipped from somewhere the path check above would miss."""
    # Unpinned, this also passes if `moldetr.dataloader` itself becomes unimportable -- green for
    # entirely the wrong reason.
    with pytest.raises(ModuleNotFoundError, match=r"moldetr\.dataloader\.shimming"):
        importlib.import_module("moldetr.dataloader.shimming")


@pytest.mark.unit
def test_add_shim_distortions_explains_itself_instead_of_vanishing() -> None:
    """The symbol survives removal, so callers get a reason rather than an AttributeError."""
    from moldetr.dataloader.data_augmentation import add_shim_distortions

    import numpy as np

    with pytest.raises(NotImplementedError) as excinfo:
        add_shim_distortions(np.zeros(64, dtype=complex))

    message = str(excinfo.value)
    assert "THIRD_PARTY.md" in message, (
        "the error must point at the document explaining the removal"
    )
    assert "GPL" in message


@pytest.mark.unit
def test_distort_still_imports_without_the_shim() -> None:
    """``moldetr.distort`` wraps only the five Apache-licensed ``add_*`` effects.

    It never imported ``shimming`` nor called ``add_shim_distortions``; that is exactly why the
    removal costs the public distortion API nothing. Asserted, not assumed.
    """
    import numpy as np

    distort = importlib.import_module("moldetr.distort")

    assert callable(distort.distort)
    assert "add_shim_distortions" not in dir(distort)

    # The namespace check above catches the likeliest reintroduction (someone adding the name to
    # distort.py's explicit import list) but not the claimed property, which is *reachability*: a
    # function-local import or an indirect call would leave dir() untouched. Since
    # add_shim_distortions now raises unconditionally, running every effect proves the shim path
    # is not reached -- if it were, this call would raise NotImplementedError.
    spectrum = np.zeros(512, dtype=complex)
    spectrum[256] = 1.0
    out = distort.distort(
        spectrum,
        np.linspace(10.0, 0.0, 512),
        noise_snr_log10=2.0,
        phase0_deg=5.0,
        phase1=0.1,
        baseline=True,
        sat_j_hz=120.0,
        broaden_hz=1.0,
        seed=0,
    )
    assert out.shape == spectrum.shape


@pytest.mark.unit
def test_third_party_notice_attributes_shimpanzee() -> None:
    """v1.0.0 shipped the file; the attribution is owed for that release regardless."""
    notice = REPO / "THIRD_PARTY.md"
    assert notice.is_file(), "THIRD_PARTY.md must ship with the distribution"

    text = notice.read_text(encoding="utf-8")
    assert "SHIMpanzee" in text
    # Was `"shimpanzee" in text.lower()`, which the line above already implies and so could never
    # fail on its own. Point it at the upstream instead, which is what attribution actually needs.
    assert "github.com/smeerten/shimpanzee" in text, (
        "attribution needs to say where the code came from, not just name it"
    )
    assert "GPL" in text
