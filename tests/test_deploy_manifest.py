"""Deploy-manifest contract: the runtime manifests must not drift below the packaged floor.

`pyproject.toml` is the single source of truth for the gradio floor. `>=6.21` is not cosmetic:
gradio's icon-only tab-overflow button shipped with no accessible name until
gradio-app/gradio#13639 (released in 6.21.0), which axe reports as a *critical* `button-name`
violation on the Simulate tab.

Nothing enforced that outside `pyproject.toml` before this file existed, and the gap was not
hypothetical — the floor was raised across three commits and `deploy/` was never revisited, so the
demo manifest sat 21 minor versions below it while being the file the Colab demo actually installs
(`notebooks/MolDeTr_colab_demo.ipynb`). It escaped notice because it was named for a Hugging Face
Space that does not exist, so it read as dead deployment scaffolding.

The `gradio-floor` CI job cannot catch this: it installs from the `pyproject.toml` extras, so it
re-proves the fix is present *at the pinned floor* and never opens these manifests. This test is
the only thing that reads them.

Kept deliberately free of `yaml` and of a bare `import tomllib`: PyYAML is not a dependency, and
`tomllib` is 3.11+ while this package declares `requires-python = ">=3.10"` and CI runs a 3.10 leg.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # `tomllib` landed in 3.11; this package supports 3.10, so use the backport there.
    import tomli as tomllib

REPO = Path(__file__).resolve().parent.parent
DEMO_REQUIREMENTS = REPO / "deploy" / "requirements-demo.txt"
SPACE_README = REPO / "deploy" / "hf_space" / "README.md"
ENVIRONMENT_YML = REPO / "environment.yml"


def _app_extra() -> list[str]:
    with (REPO / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    return config["project"]["optional-dependencies"]["app"]


def _model_extra() -> list[str]:
    """`fastai` and `torch` live in the `model` extra; `app` only pulls it in via `moldetr[model]`."""
    with (REPO / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    return config["project"]["optional-dependencies"]["model"]


def _environment_dependencies() -> list[str]:
    """Requirement-like lines from `environment.yml`, read without a YAML parser.

    PyYAML is not a dependency (module docstring); `tests/test_public_api.py` reads `ci.yml` with a
    regex for the same reason. Conda entries are `  - name<spec>`, optionally with a trailing
    comment (`pytorch-cuda=11.7  # remove this line ...`), so the comment is stripped or it would
    become part of the specifier.
    """
    lines = ENVIRONMENT_YML.read_text(encoding="utf-8").splitlines()
    entries = [re.sub(r"^\s*-\s*", "", line) for line in lines if re.match(r"^\s*-\s", line)]
    return [re.sub(r"\s*#.*$", "", entry).strip() for entry in entries]


def _requirement(specs: list[str], distribution: str) -> str:
    """The one requirement string for `distribution`, e.g. `gradio>=6.21,<7`."""
    matches = [s for s in specs if re.match(rf"^{distribution}\b", s.strip())]
    assert len(matches) == 1, f"expected exactly one {distribution} requirement, got {matches}"
    return matches[0].strip()


def _floor(specifier: str) -> tuple[int, ...]:
    """The `>=` floor of a requirement, as a comparable tuple."""
    match = re.search(r">=\s*([\d.]+)", specifier)
    assert match, f"{specifier!r} declares no >= floor"
    return tuple(int(part) for part in match.group(1).split("."))


@pytest.mark.unit
def test_demo_manifest_declares_the_same_gradio_requirement_as_the_app_extra():
    """Compared as the *whole* specifier, not just the floor, so a dropped `<7` ceiling fails too.

    A ceiling-less manifest is how the untracked staging copy of this file ended up admitting a
    hypothetical gradio 7.x that `pyproject.toml` explicitly excludes.
    """
    packaged = _requirement(_app_extra(), "gradio")
    declared = _requirement(DEMO_REQUIREMENTS.read_text(encoding="utf-8").splitlines(), "gradio")
    assert declared == packaged, (
        f"{DEMO_REQUIREMENTS.name} declares {declared!r} but pyproject's app extra declares "
        f"{packaged!r}; the Colab demo installs the former"
    )


@pytest.mark.unit
def test_space_sdk_version_is_at_or_above_the_packaged_floor():
    """`sdk_version` is the strongest version declaration in the repo.

    The HF Spaces builder installs exactly that gradio version, and a `requirements.txt` range that
    *contains* it leaves it untouched — so a stale pin here silently wins over the floor.
    """
    match = re.search(r"^sdk_version:\s*(\S+)", SPACE_README.read_text(encoding="utf-8"), re.M)
    assert match, f"{SPACE_README.name} front-matter declares no sdk_version"
    pinned = tuple(int(part) for part in match.group(1).split("."))
    floor = _floor(_requirement(_app_extra(), "gradio"))
    assert pinned >= floor, (
        f"sdk_version pins {match.group(1)} but the packaged floor is >={'.'.join(map(str, floor))}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("manifest", "read_specs"),
    [
        (DEMO_REQUIREMENTS, lambda: DEMO_REQUIREMENTS.read_text(encoding="utf-8").splitlines()),
        (ENVIRONMENT_YML, _environment_dependencies),
    ],
    ids=["requirements-demo", "environment-yml"],
)
def test_runtime_manifests_declare_the_same_fastai_requirement_as_the_model_extra(
    manifest, read_specs
):
    """The whole specifier, ceiling included — the gradio lesson applied to the other pinned dep.

    `pyproject.toml` caps fastai deliberately (see the measured `L.starmap` note there): the shipped
    checkpoint was trained against a specific stack and no CI lane exercises `learner.load()`. Both
    manifests declared a bare `fastai>=2.7`, so each would admit a fastai the package excludes —
    `deploy/requirements-demo.txt` is what the Colab demo installs, and `environment.yml` is the
    documented way to create the environment.

    This is the guard that would have caught the 2026-08-10 drift: when the ceiling moved `<2.8` ->
    `<2.9`, four surfaces quoted the old value and three were fixed. A test does that job; vigilance
    demonstrably does not.
    """
    packaged = _requirement(_model_extra(), "fastai")
    declared = _requirement(read_specs(), "fastai")
    assert declared == packaged, (
        f"{manifest.name} declares {declared!r} but pyproject's model extra declares {packaged!r}"
    )


@pytest.mark.unit
def test_demo_manifest_covers_every_distribution_the_app_extra_names():
    """A manifest can drift by omission as well as by version.

    `moldetr[...]` self-references are skipped — the demo manifest installs the package separately
    (`pip install -e . --no-deps`), so it must name the third-party deps itself.
    """
    named = {
        re.match(r"^([A-Za-z0-9_.-]+)", spec.strip()).group(1)
        for spec in _app_extra()
        if not spec.strip().startswith("moldetr")
    }
    declared = DEMO_REQUIREMENTS.read_text(encoding="utf-8")
    missing = [d for d in sorted(named) if not re.search(rf"^{d}\b", declared, re.M)]
    assert not missing, f"{DEMO_REQUIREMENTS.name} is missing {missing}"
