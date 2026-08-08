"""Declared markers must be real — a selector that filters nothing reads exactly like one that does.

Every CI lane selects `-m "not e2e and not browser and not network"`. That expression *looks* like
three deliberate exclusions. `network` was applied to no test at all, so the clause excluded the empty
set: a no-op wearing the costume of a policy. `slow` and `data` were declared and equally unused.

Nothing could catch that, because both halves fail silently by construction. pytest does not warn
about a declared-but-unused marker (there is nothing anomalous about it at collection time), and
`-m "not <unknown>"` is not an error either — an unknown name simply matches nothing, so the
expression evaluates and the lane goes green. The failure has no symptom.

So these tests guard the *class*, not the three instances: any future marker that is declared and
never used, or named in a CI selector and never applied, fails here.

Deliberately a static scan rather than a pytest sub-invocation: collecting this suite imports torch
and the Gradio app, so seven `--collect-only` runs would cost minutes to answer a question the source
text already answers. The scan is complete because no conftest applies markers dynamically — verified:
neither `tests/conftest.py` nor `tests/e2e/conftest.py` defines `pytest_collection_modifyitems`,
`pytest_configure`, or calls `add_marker`. If one ever does, this file's premise breaks and
`test_no_conftest_applies_markers_dynamically` below turns red to say so.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # `tomllib` is 3.11+; this package supports 3.10, matching tests/test_deploy_manifest.py.
    import tomli as tomllib

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"

#: EVERY workflow, not just `ci.yml`. The first version of this module scanned one file, which was
#: correct when one file existed — by the time `nightly.yml` (`-m model`), `integrations.yml`
#: (`-m network`) and `security.yml` had been added, three of the four selector sites were outside
#: the guard's view. A guard scoped to one file silently stops covering the thing it was written for
#: the moment a second file appears, which is this module's own subject matter.
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))

#: Markers pytest and its plugins provide. Not declared in `pyproject.toml`, so they are not this
#: file's business, and they appear in the source scan like any other mark.
BUILTIN_MARKERS = frozenset(
    {
        "parametrize",
        "skip",
        "skipif",
        "xfail",
        "usefixtures",
        "filterwarnings",
        "tryfirst",
        "trylast",
        "no_cover",
        "timeout",
        "asyncio",
    }
)

#: Words that are operators in a `-m` expression, not marker names.
SELECTOR_KEYWORDS = frozenset({"not", "and", "or"})


def _declared_markers() -> dict[str, str]:
    """`{name: description}` from `[tool.pytest.ini_options] markers`."""
    with (REPO / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    entries = config["tool"]["pytest"]["ini_options"]["markers"]
    return {entry.split(":", 1)[0].strip(): entry for entry in entries}


def _markers_used_in_tests() -> set[str]:
    """Every marker genuinely applied in the suite, read from the AST rather than the text.

    Catches the decorator form, the module-level `pytestmark = ...` form, and the callable form
    (as used in tests/test_simulate_predict.py) alike, because all three are the same attribute
    access once parsed.

    **A regex over the source was wrong, and this module's own docstring is what proved it.** The
    previous version matched `pytest.mark.(\\w+)` in raw text, so the illustrative example written
    one paragraph above registered a marker named `x` as "used". Harmless in that instance — nothing
    declares `x` — but the general case is not: a marker mentioned only in a comment or a docstring
    would count as applied, so someone could delete every real use of a marker and this guard would
    still pass. That is exactly the class of failure this module exists to catch, reproduced inside
    the guard itself.

    Parsing sidesteps it by construction: string and comment contents are not attribute accesses.
    """
    used: set[str] = set()
    for path in TESTS.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # a deliberately broken fixture should not take the guard down
            continue
        for node in ast.walk(tree):
            # `pytest.mark.NAME` parses as Attribute(attr=NAME, value=Attribute(attr="mark", ...)).
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "mark"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "pytest"
            ):
                used.add(node.attr)
    return used - BUILTIN_MARKERS


def _markers_named_in_ci_selectors() -> set[str]:
    """Marker names appearing in any `-m <expr>` across every workflow."""
    text = "\n".join(w.read_text(encoding="utf-8") for w in WORKFLOWS)
    names: set[str] = set()
    for line in text.splitlines():
        # Only `-m` that belongs to a pytest invocation is a marker expression. `python -m pip` and
        # `python -m playwright` also appear in this workflow, and a naive scan reads their module
        # names as markers — which is how the first draft of this test reported `pip` and
        # `playwright` as undeclared markers. Anchoring past the word `pytest` also keeps
        # `python -m pytest -m unit` correct, where the first `-m` names the module and the second
        # names the marker.
        _, sep, args = line.partition("pytest")
        if not sep:
            continue
        for expression in re.findall(r'-m\s+"([^"]+)"', args) + re.findall(
            r"-m\s+([A-Za-z_]\w*)", args
        ):
            names.update(set(re.findall(r"[A-Za-z_]\w*", expression)) - SELECTOR_KEYWORDS)
    return names


@pytest.mark.unit
def test_every_declared_marker_is_applied_to_at_least_one_test():
    """A declared marker nobody uses is a promise the suite does not keep."""
    declared = _declared_markers()
    used = _markers_used_in_tests()
    unused = sorted(set(declared) - used)
    assert not unused, (
        f"{len(unused)} marker(s) declared in pyproject.toml but applied to no test: {unused}. "
        'Either apply them or remove the declaration — an unused marker makes `-m "not <name>"` '
        "read as a deliberate exclusion while excluding nothing."
    )


@pytest.mark.unit
def test_every_marker_named_in_a_ci_selector_is_declared_and_used():
    """The sharper guard: `-m \"not network\"` in a lane that has no network tests is a no-op.

    Separate from the test above because a selector can name a marker that was never declared at all
    — pytest treats an unknown name as matching nothing and exits green, so the mistake is invisible
    from the lane's output.
    """
    declared = set(_declared_markers())
    used = _markers_used_in_tests()
    named = _markers_named_in_ci_selectors()

    undeclared = sorted(named - declared)
    assert not undeclared, (
        f"CI selects on marker(s) that pyproject.toml does not declare: {undeclared}. "
        "An unknown marker in a `-m` expression matches nothing and the lane still passes."
    )

    inert = sorted(named - used)
    assert not inert, (
        f"CI selects on marker(s) that no test carries: {inert}. Those clauses filter an empty set "
        "while reading as deliberate policy."
    )


@pytest.mark.unit
def test_no_conftest_applies_markers_dynamically():
    """Guards this module's own premise.

    The static scan above is only complete while markers are applied in test source. A conftest hook
    that attaches them at collection time would make the scan under-count and the guards above
    silently wrong — the exact shape of failure this file exists to prevent, so it gets its own test
    rather than a comment.
    """
    offenders = []
    for conftest in TESTS.rglob("conftest.py"):
        text = conftest.read_text(encoding="utf-8", errors="replace")
        for hook in ("add_marker", "pytest_collection_modifyitems", "pytest_configure"):
            if hook in text:
                offenders.append(f"{conftest.relative_to(REPO)}:{hook}")
    assert not offenders, (
        f"markers may now be applied dynamically ({offenders}), so the static scan in this module "
        "can under-count. Rework the scan to collect via pytest before trusting it again."
    )
