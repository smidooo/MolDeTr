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


@pytest.mark.unit
def test_gpl_shimming_module_is_not_importable() -> None:
    """Catches a copy shipped from somewhere the path check above would miss."""
    with pytest.raises(ModuleNotFoundError):
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
    distort = importlib.import_module("moldetr.distort")

    assert callable(distort.distort)
    assert "add_shim_distortions" not in dir(distort)


@pytest.mark.unit
def test_third_party_notice_attributes_shimpanzee() -> None:
    """v1.0.0 shipped the file; the attribution is owed for that release regardless."""
    notice = REPO / "THIRD_PARTY.md"
    assert notice.is_file(), "THIRD_PARTY.md must ship with the distribution"

    text = notice.read_text(encoding="utf-8")
    assert "SHIMpanzee" in text
    assert "shimpanzee" in text.lower()
    assert "GPL" in text
