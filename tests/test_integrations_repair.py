"""The automatic paper-relation repair must be gated, scoped and proven — not merely present.

`integrations.yml` detects that the newest Zenodo software deposit has lost its `isSupplementTo`
link to the article, and now repairs it by running `scripts/zenodo_add_paper_doi.py --confirm` with a
`deposit:write` + `deposit:actions` credential. That credential can edit **every** deposit this
account owns, including the data deposit, so what guards the write matters more than that it happens.

**A presence test would pass on a step gated wrongly.** Grepping for the step's name says nothing
about whether it fires on the right event, reads the right outcome, or writes at all. Every
assertion below is on a *condition*, and each names the mutation that must turn it red — because a
guard that has never been seen to fail is a guard nobody has tested.

Two of these encode failure modes that were live in the first draft of the wiring:

**The detector's `continue-on-error` silently disarms the issue filer.** The filer fires on
`always() && (failure() || ...)`, and `failure()` is False when every failing step is
`continue-on-error`. A relation that could not be repaired would therefore file nothing — the
delivery mechanism this workflow is built around vanishing for exactly the fault it exists to catch.
`test_the_issue_filer_still_fires_when_the_repair_could_not_close_the_gap` holds that shut.

**A misspelled node id would fire the repair.** `pytest` exits 4 on an unmatched node id, which
`continue-on-error` renders as `outcome == 'failure'` — indistinguishable from "the relation is
missing". A typo would write to a public archival record. `test_the_detector_names_a_test_that_exists`
checks the vacuity from the other side, against the test file itself.

Parsed by hand rather than with PyYAML, which is not a dependency of this project;
`tests/test_deploy_manifest.py` and `tests/test_integrations_isolation.py` set that precedent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "integrations.yml"
INTEGRATIONS_TESTS = REPO / "tests" / "test_integrations.py"

#: The one test the repair is allowed to react to. Any *other* network failure — a rate-limited
#: badge, an unreachable weight mirror — must never reach the credential.
PAPER_TEST = "test_latest_software_record_supplements_the_article"

#: Events on which a write to Zenodo is legitimate. An allowlist rather than `!= 'pull_request'`:
#: if a trigger is ever added to `on:`, the repair stays refused until someone names it here.
WRITE_EVENTS = {"release", "schedule", "workflow_dispatch"}

#: The credential this file exists to keep scoped. Named explicitly rather than matching any
#: `secrets.` reference: this job also carries a `HEALTHCHECK_URL_INTEGRATIONS` freshness-ping
#: secret in its own step (see `.github/actions/heartbeat-ping`) -- a low-privilege URL with none of
#: the blast radius `deposit:write` has, and correctly none of this file's business. A bare
#: `"secrets." in block` check would flag that unrelated secret as if it were the write token.
WRITE_CREDENTIAL = "secrets.ZENODO_DEPOSIT_TOKEN"


def _steps(text: str) -> list[str]:
    """The job's `steps:` list, one raw text block per item.

    Blocks rather than parsed values: the assertions below care about `if:` expressions and shell
    bodies, which a naive scalar parse would flatten in ways that hide the thing being checked.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "steps:"), None)
    assert start is not None, (
        f"no `steps:` list in {WORKFLOW.name}; the parser is looking at nothing"
    )

    body_indent = len(lines[start]) - len(lines[start].lstrip()) + 2
    item = " " * body_indent + "- "
    blocks: list[list[str]] = []
    for line in lines[start + 1 :]:
        if line.startswith(item):
            blocks.append([line])
        elif line.strip() and not line.startswith(" " * body_indent):
            break  # dedented out of the steps list
        elif blocks:
            blocks[-1].append(line)
    assert blocks, f"`steps:` in {WORKFLOW.name} parsed as empty"
    return ["\n".join(block) for block in blocks]


def _field(block: str, key: str) -> str:
    """The value of `key:` in a step block, block scalars folded onto one line.

    Comment lines are dropped: `integrations.yml` carries more prose than YAML, and a `#` paragraph
    explaining why a step is gated the way it is would otherwise read as part of the gate.

    The leading `- ` of a list item is stripped before matching. Without that, the *first* key of
    every step — usually `name:` — is invisible to this function, which returned `""` for it
    silently. The only consumer was a failure message, so the guard that fires exactly when a
    second step gains the credential would have named none of them.
    """
    lines = block.splitlines()
    head = next(
        (
            i
            for i, line in enumerate(lines)
            if line.strip().removeprefix("- ").startswith(f"{key}:")
        ),
        None,
    )
    if head is None:
        return ""

    first = lines[head].strip().removeprefix(f"{key}:").strip().lstrip(">|-").strip()
    collected = [first] if first else []
    key_indent = len(lines[head]) - len(lines[head].lstrip())
    for line in lines[head + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= key_indent:
            break
        if not line.strip().startswith("#"):
            collected.append(line.strip())
    return " ".join(part for part in collected if part)


def _by_id(blocks: list[str], step_id: str) -> str:
    match = [block for block in blocks if _field(block, "id") == step_id]
    assert len(match) == 1, (
        f"expected exactly one step with `id: {step_id}` in {WORKFLOW.name}, found {len(match)}. "
        "The repair wiring addresses steps by id; a rename breaks the gate silently."
    )
    return match[0]


def _relation_steps(blocks: list[str]) -> list[str]:
    """Every step that runs the paper-relation test on its own, in document order.

    There are two by design — the detector and the re-check — so this deliberately returns a list
    rather than pretending to identify one. An earlier version described itself as finding "the"
    such step while matching both and silently taking the first.
    """
    named = [
        block
        for block in blocks
        if PAPER_TEST in _field(block, "run") and "--deselect" not in _field(block, "run")
    ]
    assert named, (
        f"no step runs `{PAPER_TEST}` on its own. The repair below must be gated on that one test "
        "and not on the broad network sweep, or a rate-limited badge triggers a Zenodo write."
    )
    ids = [_field(block, "id") for block in named]
    assert all(ids), f"the step(s) running {PAPER_TEST} need an `id:` so the repair can read them"
    return named


def _detector_id(blocks: list[str]) -> str:
    """The id of the *first* step running the paper-relation test — the one the repair is gated on.

    Resolved rather than hardcoded: hardcoding it would let a rename move the detector out from
    under the repair's `if:` while this file kept asserting things about a step that no longer gates
    anything. Document order is what makes "first" meaningful, and
    `test_the_re_check_runs_after_the_repair` is what keeps that order from drifting.
    """
    return _field(_relation_steps(blocks)[0], "id")


def _selected_test(block: str) -> str:
    """The test function a step selects by node id, or "" if it selects none."""
    node = re.search(r"tests/test_integrations\.py::(\w+)", _field(block, "run"))
    return node.group(1) if node else ""


@pytest.fixture(scope="module")
def workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def steps(workflow: str) -> list[str]:
    return _steps(workflow)


@pytest.mark.unit
def test_the_workflow_never_runs_on_a_pull_request(workflow: str) -> None:
    """Parsed from the `on:` block, not grepped — the file's prose discusses pull requests at length.

    This is the load-bearing mitigation for holding a credential in a PUBLIC repository: GitHub
    withholds secrets from fork PRs, but a same-repo PR trigger would hand it to any branch.

    Mutation: add `pull_request:` under `on:`.
    """
    lines = workflow.splitlines()
    start = next(i for i, line in enumerate(lines) if line.rstrip() == "on:")
    triggers = set()
    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith(" "):
            break
        if re.match(r"^  [a-z_]+:", line):
            triggers.add(line.strip().rstrip(":"))

    assert triggers, "the `on:` block parsed as empty, so this test proves nothing"
    assert triggers <= WRITE_EVENTS, (
        f"{WORKFLOW.name} gained trigger(s) {sorted(triggers - WRITE_EVENTS)}. This job holds a "
        "Zenodo deposit:write credential; every trigger it answers is a way to reach that token."
    )


@pytest.mark.unit
def test_the_credential_is_scoped_to_the_repair_step_alone(workflow: str, steps: list[str]) -> None:
    """A job-level `env:` would hand the token to lychee-action and the issue filer for free.

    Scoped to `WRITE_CREDENTIAL` specifically, not "any `secrets.` reference" -- see that
    constant's docstring for why a bare substring match would misfire on this job's unrelated
    freshness-ping secret.

    Mutations: hoist the secret to job-level `env:`; add it to a second step.
    """
    before_steps = workflow.split("steps:")[0]
    assert WRITE_CREDENTIAL not in before_steps, (
        f"a `{WRITE_CREDENTIAL}` reference appears above the step list — at job or workflow level "
        "it is exported into every step, including the third-party action this job runs."
    )

    carrying = [block for block in steps if WRITE_CREDENTIAL in block]
    assert len(carrying) == 1, (
        f"expected exactly one step to reference `{WRITE_CREDENTIAL}`, found {len(carrying)}: "
        f"{[_field(block, 'name') for block in carrying]}"
    )
    assert WRITE_CREDENTIAL in _field(carrying[0], "env"), (
        "the secret must arrive through the repair step's own `env:` block"
    )


@pytest.mark.unit
def test_the_repair_is_gated_on_the_detector_and_on_a_named_event(steps: list[str]) -> None:
    """Both halves, and the events must be a subset of what the workflow actually answers.

    Mutations: gate on `always()`; gate on `steps.lychee.outcome`; drop the event clause; name an
    event that is not in `on:`.
    """
    repair = _by_id(steps, "repair")
    condition = _field(repair, "if")
    detector = _detector_id(steps)

    assert f"steps.{detector}.outcome == 'failure'" in condition, (
        f"the repair must fire only when `{detector}` failed. Its condition is: {condition!r}"
    )

    # The allowlist is read out of the condition and parsed as JSON, not scanned for known-bad
    # names. Intersecting the quoted words with a known set first was a tautology: it filtered out
    # exactly the strings it was meant to catch, so smuggling in `pull_request_target` passed, and
    # so did inverting the gate to `github.event_name != 'release'` — which refuses the release
    # path this exists for and fires on everything else.
    array = re.search(r"contains\(fromJSON\('(\[.*?\])'\)\s*,\s*github\.event_name\s*\)", condition)
    assert array, (
        "the repair must gate its events with `contains(fromJSON('[...]'), github.event_name)`. "
        f"Anything else — an inequality, a bare comparison — is not an allowlist. Got: {condition!r}"
    )
    allowed = set(json.loads(array.group(1)))
    assert allowed, f"the repair's event allowlist is empty. Condition: {condition!r}"
    assert allowed <= WRITE_EVENTS, (
        f"the repair would write on {sorted(allowed - WRITE_EVENTS)}, which is not an event this "
        "workflow is allowed to hold a deposit:write credential for."
    )


@pytest.mark.unit
def test_the_detector_names_a_test_that_exists(steps: list[str]) -> None:
    """A misspelled node id exits 4, reads as `failure`, and fires the repair on a typo.

    Checked against the test file rather than against the workflow's own spelling, which is the only
    direction that can catch it. Mutation: misspell the node id in the workflow.
    """
    source = INTEGRATIONS_TESTS.read_text(encoding="utf-8")
    for block in _relation_steps(steps):
        step_id = _field(block, "id")
        selected = _selected_test(block)
        assert selected, (
            f"step `{step_id}` must select the paper-relation test by full node id, not by "
            f"file or `-k`. Its run line is: {_field(block, 'run')!r}"
        )
        assert f"def {selected}(" in source, (
            f"step `{step_id}` selects `{selected}`, which does not exist in "
            f"{INTEGRATIONS_TESTS.name}. pytest exits 4 on an unmatched node id, "
            "`continue-on-error` renders that as a failure, and the repair would write to a "
            "published archival record because of a typo."
        )
        assert selected == PAPER_TEST, (
            f"step `{step_id}` selects `{selected}`, a real test but not the one the repair "
            "reacts to. Pointing the re-check at a test that always passes is how a broken "
            "deposit ends the job green."
        )


@pytest.mark.unit
def test_the_broad_sweep_does_not_re_run_the_test_the_repair_reacts_to(steps: list[str]) -> None:
    """Otherwise the broad step reports the pre-repair state and contradicts the re-verification.

    Mutation: drop the `--deselect`.
    """
    broad = [block for block in steps if "-m network" in _field(block, "run")]
    assert len(broad) == 1, f"expected one broad `-m network` step, found {len(broad)}"
    assert "--deselect" in _field(broad[0], "run") and PAPER_TEST in _field(broad[0], "run"), (
        "the broad network sweep must `--deselect` the paper-relation test; the dedicated "
        "re-verification step owns that verdict."
    )


@pytest.mark.unit
def test_the_re_check_exists_and_runs_after_the_repair(steps: list[str]) -> None:
    """The step that proves the repair took is the one thing nothing else here checks.

    `--confirm` exiting 0 is distrusted throughout this wiring because it only says the fixer's own
    read-back was happy. That argument is worth nothing if the re-check can be deleted, reordered
    ahead of the repair, or pointed at a different test — each of which turns a broken deposit into
    a green job. `test_the_detector_names_a_test_that_exists` covers what it selects; this covers
    that it exists at all and runs at the right moment.

    Mutations: delete the `reverify` step; move it above the repair.
    """
    order = [_field(block, "id") for block in steps]
    assert "reverify" in order, (
        "no step with `id: reverify`. Without it the repair is never proven, and the terminal "
        "red-restoring step below reads an outcome that does not exist."
    )
    assert order.index("reverify") > order.index("repair"), (
        "the re-check runs before the repair, so it reports the state the repair was meant to "
        f"change. Step order is: {[step for step in order if step]}"
    )

    # Addressed by **id**, deliberately, and this is not redundant with
    # `test_the_detector_names_a_test_that_exists`. That test walks the steps that *contain* the
    # paper test's node id, so a re-check pointed at some other test drops out of the set entirely
    # and is checked by nothing — the guard stops seeing the step precisely when the step stops
    # doing its job. Identity has to come from somewhere the mutation cannot move.
    selected = _selected_test(_by_id(steps, "reverify"))
    assert selected == PAPER_TEST, (
        f"the re-check runs `{selected or '(no node id)'}` rather than `{PAPER_TEST}`. A re-check "
        "pointed at a test that passes anyway turns every unrepaired deposit into a green job, "
        "which is the outcome the whole detect/repair/re-check sequence exists to prevent."
    )


@pytest.mark.unit
def test_a_repair_that_did_not_take_still_ends_the_job_red(steps: list[str]) -> None:
    """`continue-on-error` on the detector and the re-check means something must restore the red.

    Mutations: delete the terminal step; give it `continue-on-error: true` — which lets it run,
    exit 1, and leave the job green anyway. Either is a green tick over a deposit that does not
    point at its own paper.
    """
    detector = _detector_id(steps)
    assert _field(_by_id(steps, detector), "continue-on-error") == "true", (
        "the detector must not abort the job, or the repair below never runs"
    )

    terminal = [
        block
        for block in steps
        if _field(block, "run").strip() == "exit 1"
        and f"steps.{detector}.outcome == 'failure'" in _field(block, "if")
        and "steps.reverify.outcome != 'success'" in _field(block, "if")
        and _field(block, "continue-on-error") != "true"
    ]
    assert terminal, (
        "no step turns the job red when the relation was missing and the re-check did not pass. "
        "Both are `continue-on-error`, so without this the job ends green over a broken record."
    )


@pytest.mark.unit
def test_the_issue_filer_still_fires_when_the_repair_could_not_close_the_gap(
    steps: list[str],
) -> None:
    """`failure()` is False when every failing step is `continue-on-error`.

    This is the regression that the repair wiring introduces if written naively: the one fault the
    workflow exists to report becomes the one fault it reports nothing about.

    Mutations: revert the condition to `always() && (failure() || cancelled() || lychee)`; swap the
    `||`s joining the clauses for `&&`, which leaves every substring in place while making the
    filer essentially unable to fire.
    """
    filer = [block for block in steps if "gh issue create" in block]
    assert len(filer) == 1, f"expected one issue-filing step, found {len(filer)}"
    condition = " ".join(_field(filer[0], "if").split())

    detector = _detector_id(steps)
    unrepairable = f"(steps.{detector}.outcome == 'failure' && steps.reverify.outcome != 'success')"
    assert unrepairable in condition, (
        "the issue filer does not read the relation steps' outcomes. Both are `continue-on-error`, "
        f"so `failure()` is False and an unrepairable relation files nothing. Condition: {condition!r}"
    )

    # Presence of the clause is not enough — it has to be *reachable*. Checking the substrings
    # alone accepted a condition whose every `||` had become `&&`, which can essentially never be
    # true and silently restores the very regression this test exists to prevent.
    #
    # Both directions are checked, because `&&` binds tighter than `||` in GitHub expressions. A
    # single `&&` on either side absorbs this clause into a conjunction with `failure()` and
    # `cancelled()` — mutually exclusive in practice — while leaving `<clause> ||` intact further
    # along, which an OR-check alone reads as healthy.
    assert f"&& {unrepairable}" not in condition and f"{unrepairable} &&" not in condition, (
        "the unrepairable-relation clause is AND-joined to a neighbour. `&&` binds tighter than "
        "`||`, so the clause can only fire together with whatever it is joined to, and the filer "
        f"stays silent for the fault it exists to report. Condition: {condition!r}"
    )
    assert f"|| {unrepairable}" in condition or f"{unrepairable} ||" in condition, (
        "the unrepairable-relation clause is not OR-joined to the rest of the filer's condition, "
        f"so it cannot independently fire the issue. Condition: {condition!r}"
    )


@pytest.mark.unit
def test_the_off_release_report_distinguishes_repaired_from_no_credential(steps: list[str]) -> None:
    """`repair` exits 0 when no secret is set, so its *outcome* cannot mean "it repaired something".

    Keyed on the outcome, the off-release clause fires on every cron run of a repository that
    simply has no credential, and the issue body then states that a relation "was removed from an
    already-published record and put back" — a paragraph of confident fiction, weekly, describing
    an event that did not happen. The step publishes `ran` instead, and only after the fixer
    actually returned.

    Mutation: key the clause on `steps.repair.outcome == 'success'`.
    """
    repair_body = _field(_by_id(steps, "repair"), "run")
    assert "ran=true" in repair_body and "ran=false" in repair_body, (
        "the repair step must publish a `ran` output covering both paths, so the filer can tell a "
        "real repair from an early exit"
    )

    condition = " ".join(_field(next(b for b in steps if "gh issue create" in b), "if").split())
    assert "steps.repair.outputs.ran == 'true'" in condition, (
        "the off-release report must key on the repair's `ran` output, not on its outcome. "
        f"Condition: {condition!r}"
    )
    assert "steps.repair.outcome == 'success'" not in condition, (
        "`outcome == 'success'` is true when the step exited 0 having done nothing at all"
    )


@pytest.mark.unit
def test_the_token_reaches_the_script_by_file_and_never_by_argument(steps: list[str]) -> None:
    """`--token-file` keeps the value out of `argv`, which is world-readable via `ps`.

    Mutation: switch to a `--token "$ZENODO_TOKEN"` style flag.
    """
    body = _field(_by_id(steps, "repair"), "run")
    invocation = next(
        (part for part in body.split(" python ")[1:] if "zenodo_add_paper_doi" in part), ""
    )
    assert invocation, f"the repair step runs no fixer. Body: {body!r}"

    assert "--confirm" in invocation, "a dry run repairs nothing"
    assert "--token " not in invocation, "the token value must never appear in argv"

    target = re.search(r'--token-file\s+"?([^\s"]+)"?', invocation)
    assert target, "the token must arrive as a file, not on the command line"

    # Resolve one level of shell indirection: the step names the path once and passes the variable,
    # so asserting on the invocation text alone would look for `$RUNNER_TEMP` where it cannot be.
    path = target.group(1)
    if path.startswith("$"):
        variable = path.strip("${}")
        assigned = re.search(rf'\b{re.escape(variable)}=[\'"]?([^\s\'"]+)', body)
        assert assigned, f"--token-file passes ${variable}, which the step body never assigns"
        path = assigned.group(1)

    assert path.startswith("$RUNNER_TEMP"), (
        f"the token file resolves to {path!r}. It must live under $RUNNER_TEMP: inside the checkout "
        "it survives the step and is sweepable into an uploaded artifact."
    )

    # The remaining three properties were intended rather than asserted, which is the exact charge
    # this module levels at everything else. `set -x` is the sharpest: GitHub masks the secret's
    # value, but trace output of the surrounding shell is noise nobody reads and a habit worth not
    # forming in a step that holds a credential.
    assert "umask 077" in body, (
        "create the token file inside `(umask 077; ...)`. A `chmod` after the redirection leaves a "
        "window in which the file is world-readable."
    )
    assert "trap 'rm -f" in body, (
        "remove the token file with a `trap ... EXIT`, not a trailing `rm`: a trailing one does not "
        "run when the fixer exits non-zero, which is precisely when it matters."
    )
    assert "set -x" not in body and "-euxo" not in body, (
        "shell tracing is on in the step that handles the credential"
    )


@pytest.mark.unit
def test_lychee_fails_when_it_checks_zero_links(steps: list[str]) -> None:
    """`--accept 200,206,403,429` and `continue-on-error: true` on the lychee step both look, in
    isolation, like a step that could pass having checked nothing -- if the `README.md 'docs/**/*.md'`
    glob ever stopped matching, or lychee's arg parsing changed, the step could exit 0 with zero
    links checked and the compensating `exit 1` at "Fail the job if the link sweep failed" (gated on
    `steps.lychee.outcome == 'failure'`) would never fire.

    That gap does not need a new counter: `lycheeverse/lychee-action`'s own `failIfEmpty` input
    defaults to `true` and is NOT set here, so it is active. Verified 2026-08-29 against the `@v2`
    tag's `action.yml` (`failIfEmpty`, default `true`) and `entrypoint.sh`: it greps its own markdown
    summary for `Total | 0` and exits 1 regardless of the `fail:` input when it matches -- which is
    exactly "checked zero links, fail". `continue-on-error: true` changes the step's `conclusion`,
    not its `outcome`, so `steps.lychee.outcome == 'failure'` still fires on that exit and the
    downstream `exit 1` step still runs. This test pins that the workflow does not disable it.

    That upstream behavior is unpinned by version and this test cannot see it: `@v2` is a floating
    tag, so what it resolves to (and whether `failIfEmpty` keeps defaulting to `true`) can drift
    without this repository changing anything. This test is a guard against THIS repository
    disabling the default, not a guarantee the default's meaning stays fixed -- if a future upstream
    release changes it, nothing here would notice until the zero-links case actually occurs.

    Mutation: add `failIfEmpty: false` (or `failIfEmpty: "false"`) to the lychee step's `with:`
    block -- the assertion below must go red.
    """
    lychee = next((block for block in steps if "lycheeverse/lychee-action" in block), None)
    assert lychee is not None, "no step uses lycheeverse/lychee-action -- has it been renamed?"

    with_block = _field(lychee, "with")
    assert (
        "failIfEmpty" not in with_block or "false" not in with_block.split("failIfEmpty")[1][:20]
    ), (
        "the lychee step sets `failIfEmpty: false` (or similar), which disables the action's own "
        "guard against silently checking zero links. Remove the override -- the default `true` is "
        "what makes a broken glob or a lychee arg-parsing change fail loudly instead of passing."
    )
