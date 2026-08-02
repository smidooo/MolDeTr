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
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # `tomllib` landed in 3.11; this package supports 3.10, so use the backport there.
    import tomli as tomllib

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
_GPL_IMPLEMENTATION_MARKERS = (
    # The off-axis shim terms.
    "class ShimSim",
    "ZX2_ZY2LIM",
    "zx2_zy2",
    "X2_Y2",
    # ...and the axis-independent core. The four above all live on one axis of the original, so a
    # reduced copy taking only the `zonly=True` path -- grid, on-axis Z harmonics, field
    # accumulation, FID synthesis -- matched none of them. These do not depend on which harmonics
    # were kept, which is what makes the set a scan rather than a keyword filter.
    "setupGrid",
    "fidSurface",
    "Mfield",
    "startGame",
    # The upstream header's own typo: a fingerprint of a verbatim copy under any filename.
    "modifed from",
)


def _gpl_markers_in(root: Path) -> list[str]:
    """Every ``path -> markers`` hit under ``root``, as human-readable lines.

    Split out of the test below so the marker set can be exercised against a fixture rather than
    only against a tree that is expected to be clean -- a scan that is only ever run where it must
    find nothing is a scan nobody has watched succeed.
    """
    hits: list[str] = []
    for source in sorted(root.rglob("*.py")):
        text = source.read_text(encoding="utf-8", errors="replace")
        found = [marker for marker in _GPL_IMPLEMENTATION_MARKERS if marker in text]
        if found:
            hits.append(f"{source.relative_to(root).as_posix()} contains {found}")
    return hits


@pytest.mark.unit
def test_no_gpl_shim_implementation_survives_under_any_filename() -> None:
    """The filename check above is blind to the threat this module's docstring actually names.

    ``shim_sim.py``, ``_shimming.py``, ``moldetr/vendor/shimpanzee.py``, or ``ShimSim`` pasted back
    into ``data_augmentation.py`` all pass a glob for ``shimming.py``. Match the implementation
    instead of the filename. Scoped to ``moldetr/`` -- the shipped package -- so this file's own
    marker list does not match itself.
    """
    offenders = _gpl_markers_in(REPO / "moldetr")

    assert offenders == [], (
        "GPL-derived shim implementation is back in an Apache-2.0 distribution:\n"
        + "\n".join(offenders)
        + "\nSee THIRD_PARTY.md -- removing it was decision D-1, not a cleanup."
    )


#: A *reduced* re-introduction: the upstream's own ``startGame(order=4, zonly=False)`` path keeps
#: the grid, the on-axis Z harmonics, the field accumulation and the FID synthesis, and drops every
#: off-axis term. Restoring "just enough shim" is the realistic accident, and it is the one a
#: marker set drawn entirely from the off-axis terms cannot see.
_REDUCED_Z_ONLY_COPY = """
import numpy as np

class FieldSim:
    def setupGrid(self, n=64):
        self.z = np.linspace(-1, 1, n)
        self.Z1 = 0.5 * np.sqrt(3 / np.pi) * self.z
        self.Z2 = 0.25 * np.sqrt(5 / np.pi) * (3 * self.z ** 2 - 1)

    def startGame(self, order=2, zonly=True):
        self.Mfield = np.zeros_like(self.z)

    def apply(self, z1=0, z2=0):
        self.Mfield += z1 * self.Z1 + z2 * self.Z2
        return self.fidSurface()
"""


@pytest.mark.unit
def test_the_marker_set_catches_a_reduced_z_only_copy(tmp_path: Path) -> None:
    """Verbatim copies are the easy case; the marker set has to survive a partial one."""
    (tmp_path / "field_helpers.py").write_text(_REDUCED_Z_ONLY_COPY, encoding="utf-8")

    assert _gpl_markers_in(tmp_path), (
        "a z-only reduction of the GPL simulator went undetected -- the markers are all drawn "
        "from the off-axis terms such a copy drops"
    )


@pytest.mark.unit
def test_the_marker_set_does_not_flag_the_mentions_that_deliberately_survive(
    tmp_path: Path,
) -> None:
    """``ShimSim``/``shimpanzee`` live on in the removal docstring and the raise; that is the point."""
    (tmp_path / "innocent.py").write_text(
        '"""add_shim_distortions: the ShimSim simulator (SHIMpanzee, '
        'https://github.com/smeerten/shimpanzee) was removed -- see THIRD_PARTY.md."""\n',
        encoding="utf-8",
    )

    assert _gpl_markers_in(tmp_path) == [], "prose naming the upstream is not an implementation"


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
    # Only the versioned form is asserted: a bare `"GPL" in text` is implied by the line below and
    # could never fail on its own -- the same subsumption removed above. "GNU General Public
    # License" without a version is under-specified anyway, since GPLv2 and GPLv3 differ in their
    # patent and termination terms and Apache-2.0 is one-way compatible into v3 only.
    assert "GPL-3.0" in text, "name the GPL version; upstream is GPL-3.0-or-later"
    for holder in ("Bas van Meerten", "Wouter Franssen"):
        assert holder in text, f"attribution needs the copyright holder: {holder}"


@pytest.mark.unit
def test_the_retracted_fails_loudly_claim_does_not_survive_where_it_ships() -> None:
    """It was corrected in two files and quietly survived in a third that this release now ships.

    ``add_shim_distortions`` cannot "fail loudly": ``augment_distortions`` pins ``toss_coin`` to
    ``0.99``, so the branch that would call it is unreachable and the raise never fires from there.
    Correcting two of three copies is exactly how the original claim propagated across this
    project's artefacts once before, so the third copy gets a tripwire rather than another sweep.

    Scoped to what is distributed: ``THIRD_PARTY.md`` (now in the wheel via ``license-files``) and
    the package itself. ``CHANGELOG.md`` quotes the phrase to *retract* it, which is its job.
    """
    shipped = [REPO / "THIRD_PARTY.md", *sorted((REPO / "moldetr").rglob("*.py"))]
    offenders = [
        path.relative_to(REPO).as_posix()
        for path in shipped
        if "fails loudly" in path.read_text(encoding="utf-8", errors="replace")
    ]

    assert offenders == [], (
        "the retracted 'fails loudly' claim is still in a shipped artefact: "
        + ", ".join(offenders)
        + " -- the toss_coin branch is unreachable, so nothing fails loudly from it"
    )


@pytest.mark.unit
def test_third_party_notice_covers_the_code_that_actually_ships() -> None:
    """The file is titled "third-party code in this distribution" and listed only the removed one.

    The deformable-attention sources under ``moldetr/model/ops/src/`` and ``moldetr/matcher`` carry
    SenseTime and Facebook copyright headers and *do* ship in the wheel. Their in-file notices are
    intact, so attribution is preserved where it legally matters -- but a document that presents
    itself as an enumeration reads as exhaustive, and this one was not.
    """
    text = (REPO / "THIRD_PARTY.md").read_text(encoding="utf-8")

    # "DETR" is dropped from this list on purpose: it is a substring of "Deformable DETR" above, so
    # it could never fail on its own -- the same subsumption this file removed elsewhere. Microsoft
    # is here because the CUDA kernels descend from DCN and carry its copyright (see the header of
    # ms_deform_im2col_cuda.cuh), which the first version of this enumeration missed.
    for owed in ("Deformable DETR", "SenseTime", "Facebook", "Microsoft"):
        assert owed in text, f"{owed} code ships in this distribution and is not recorded"


@pytest.mark.unit
def test_third_party_notice_is_declared_so_it_ships() -> None:
    """Present in the tree but undeclared means absent from the wheel -- LICENSE shipped alone.

    A notice that reaches only people who browse the repository is not much of a notice; the wheel
    is what most consumers actually install.
    """
    with (REPO / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    declared = config["tool"]["setuptools"].get("license-files", [])
    assert "THIRD_PARTY.md" in declared, (
        "THIRD_PARTY.md must be declared in [tool.setuptools] license-files or it is not "
        "installed; the v1.1.0 wheel carried only dist-info/licenses/LICENSE"
    )
