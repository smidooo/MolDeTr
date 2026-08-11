# Releasing MolDeTr: maintainer checklist

Cutting a release here is **irreversible**. The repository carries Zenodo's GitHub integration — a
release-scoped webhook on repository `1289888357`, configured in Zenodo and independent of anything
in the README — so publishing a GitHub release automatically archives the tarball and mints a
permanent version DOI under the software concept DOI `10.5281/zenodo.21214876`. Zenodo records are
immutable: whatever is in the tree at tag time is public forever. That is why the GPL `shimming.py`
had to be removed *before* v1.1.0 rather than after.

The front-page DOI badge is **not** part of that mechanism, and changing it mints nothing. It is a
static `img.shields.io` badge pinned to the concept DOI. Do not restore the `zenodo.org/badge/…`
form: Zenodo serves badges `cache-control: no-cache` under an `x-ratelimit-limit: 120` per-IP cap,
so GitHub's shared camo image proxy — which fetches README images server-side for the whole site —
exhausts the quota and the badge renders as `502 Invalid upstream response (429)`. Measured on this
page: 4 of 5 fetches failed, while all nine `img.shields.io` badges returned `200`.
`tests/test_readme_badges.py` enforces both halves.

The practical consequence: **a tag is not a checkpoint you can move.** Do not tag to "see if it
works", and do not tag a commit you have not already decided to publish.

This file covers the **software** record only. The separate **data** deposit
([10.5281/zenodo.21217101](https://zenodo.org/doi/10.5281/zenodo.21217101)) does not move with a
code release.

That link used to point at `ZENODO_DEPOSIT.md`, which is **gitignored** — so it resolved on the
maintainer's disk and 404'd for every actual reader, including on the docs site. Nothing noticed,
because the link checker that owns this file had never had a run: `integrations.yml` was added on
2026-08-05 with a Monday cron whose first firing was due 2026-08-10, and its first real execution
was a manual dispatch on 2026-08-09 — which found this immediately. Before adding a docs link here,
check that the target is actually tracked (`git ls-files`), not merely present.

## Before tagging

- [ ] **`CHANGELOG.md` `[Unreleased]` is current.** It is the section most likely to be behind, and
      it is invisible in a diff review because nothing *fails* when it is stale. Precedent: v1.0.0
      shipped while `main` was 61 commits ahead and `[Unreleased]` had fallen behind by 7 PRs.
      Compare against `git log <last-tag>..main` rather than memory.
- [ ] **Rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`** and open a fresh empty `[Unreleased]`.
- [ ] **Bump `version` in `pyproject.toml` — and nothing else.** `CITATION.cff` deliberately carries
      no `version:` or `date-released:` field, because citation surfaces should pin the *concept*
      DOI, which always resolves to the latest release.
- [ ] **CI is green on `main`.** The merge-gating family is **14 legs, 7 of them required** (the two
      ubuntu legs plus the whole e2e tier). macOS and Windows stay advisory — read them anyway before
      a release, since advisory means unblocking, not unimportant. A commit carries 25 check-runs
      once Security and Pages are counted; those never gate a merge.

      This read "11 checks" until 2026-08-11, and **that was correct when written on 2026-08-03** —
      6 matrix legs plus the 5-strong e2e tier. It went stale four days later when `7d9c924` (PR #37)
      added py3.12 to all three OS. **Re-derive the count** from
      `gh api repos/smidooo/MolDeTr/branches/main/protection` and
      `gh api repos/smidooo/MolDeTr/commits/<sha>/check-runs` rather than carrying this number
      forward — it has now gone stale twice. The maintainer's requirements doc restates these counts
      and has to be updated in the same pass; it lives under the gitignored `docs/requirements/`, so
      it is not a link a reader of this repository can follow.
- [ ] **Check the skip and deselect counts, not just the green tick.** A tier that silently skipped
      reports success. `-rs` is on the browser steps for exactly this reason.
- [ ] **Anything Python 3.11+-only will pass locally and fail CI.** The local venv is 3.12; CI's
      oldest leg is 3.10. The same asymmetry applies to dependency floors: `numpy` and `gradio` are
      pinned with a floor that only the dedicated floor jobs actually install.

Changes reach `main` through numbered PRs, not direct pushes. Admin override would succeed —
`enforce_admins` is false so a flaky leg can be worked around — but the convention is the audit
trail, so use it.

## After publishing — the paper relation, which is repaired for you

**The link to the paper does not survive versioning, and this was never a checklist problem.** The
`isSupplementTo` relation pointing at the article was present on v0.1.0 and absent from every
release since — **six for six**: v1.0.0, v1.1.0, v1.1.1, v1.2.0, v1.3.0 and v1.4.0. Seven Zenodo
records exist under the concept DOI and only the first was ever born correct; the other six were
restored after the fact — the last of them without anyone noticing, which is the point of the
section below.

The sixth was **measured**, not predicted, and that is a different claim from the five before it.
On 2026-08-11 the release-triggered run recorded Zenodo minting v1.4.0 at 11:40:32 and the guard
failing at 11:40:35 with `Present relations: [('isSupplementTo', '.../tree/v1.4.0')]`. Re-derive
this count at each release rather than incrementing it.

Zenodo is not carrying it forward and does not prefill it from the previous record: **v1.2.0 was
published without the relation even though v1.1.1 had already been hand-edited to carry it.** That
single fact rules out both "the last version seeds the next one" and "somebody forgot" — a step
performed correctly five times and lost five times is a missing automation, not a lapse of
attention.

Writing it down did not help either, though the sample is smaller than it looks: this file was
created on 2026-08-03, so only v1.2.0 and v1.3.0 were ever published while the instruction existed.
It was dropped on both.

So it is no longer a step you perform. `tests/test_integrations.py` asserts it, and
`.github/workflows/integrations.yml` runs on `release: published` — waiting out Zenodo's
asynchronous webhook first, because a check that fires before minting reads the *previous* record,
finds the relation there, and passes while the new release carries nothing. The weekly cron still
runs, and not only as a backstop: it is the only thing that catches a relation undone later, or a
release whose minting failed after the release-time run gave up waiting.

**Since v1.4.0 the repair is automatic too, and there is nothing to do after publishing.** Detecting
a fault five times and then asking a human to type the fix is still a manual step; it had simply
moved. The job now runs the relation check as its own step, and if that step fails it runs
`scripts/zenodo_add_paper_doi.py --confirm` with the `ZENODO_DEPOSIT_TOKEN` repository secret, then
**re-checks** — a repair exiting 0 proves only that the fixer's own read-back was happy. If the
re-check does not pass, the job ends red and files the issue as before.

Three properties are load-bearing, and `tests/test_integrations_repair.py` holds each of them shut
against a named mutation:

- **The repair is gated on that one check**, not on the job. A rate-limited badge or an unreachable
  weight mirror must never reach a `deposit:write` credential.
- **The credential is scoped to the single step** that needs it. At job level it would be exported
  into every step, including the third-party `lychee-action`.
- **The workflow answers no `pull_request` trigger.** That is what keeps a token in a *public*
  repository tolerable, and it is now asserted rather than merely intended.

The token is a repository secret named `ZENODO_DEPOSIT_TOKEN` (scopes `deposit:write` +
`deposit:actions`), deliberately **not** the same one as `~/.secrets/.zenodo_token` — a separate
token can be revoked without breaking local tooling, and its use is attributable. Zenodo cannot
scope a token to one deposit, so it can edit every deposit this account owns; that is the accepted
cost, taken with the mitigations above.

**When the repair itself fails** — an `External reference check failed` issue naming
`repair=` / `reverify=` outcomes, or red in the Actions tab — the manual path is unchanged:

```bash
python scripts/zenodo_add_paper_doi.py             # DRY RUN — resolves the newest record, prints
python scripts/zenodo_add_paper_doi.py --confirm   # apply
```

It resolves the newest release from the concept DOI, so there is no record id to keep current. It
appends without replacing, discards the draft if anything fails mid-flight so a record is never
stranded in edit state, and is idempotent. It needs a Zenodo token at `~/.secrets/.zenodo_token`
with scopes `deposit:write` + `deposit:actions`. **A metadata edit mints no new DOI** — the record
id and its DOI are unchanged by this, so neither expect nor fear a new version record.

To check by hand, read the public API rather than the edit form, so you are reading what a citation
tool would read:

```bash
curl -s https://zenodo.org/api/records/<new-version-id> \
  | python -c "import json,sys; m=json.load(sys.stdin)['metadata']; \
print([(r['relation'], r['identifier']) for r in m.get('related_identifiers', [])])"
```

A correct v1.3.0 (record `21856870`) reports both relations:

```
[('isSupplementTo', 'https://github.com/smidooo/MolDeTr/tree/v1.3.0'),
 ('isSupplementTo', '10.1021/acs.analchem.5c03465')]
```

While you are there, confirm the concept DOI still resolves to the new version, and that the
creator list and licence carried over intact — neither is guarded automatically.

## Why there is no `.zenodo.json` — considered and rejected, 2026-08-09

The obvious-looking improvement is to declare the relation in a `.zenodo.json`, so Zenodo mints the
record correctly instead of the guard above catching it afterwards. It was investigated and
**rejected**. The reasoning is recorded here because the idea is attractive enough to be proposed
again by anyone who has not checked.

**It would silence `CITATION.cff` entirely.** Per Zenodo's documentation, when a repository has
both, *"Zenodo will only use the `.zenodo.json` metadata. The `CITATION.cff` will be completely
ignored."* Precedence runs `.zenodo.json` > `CITATION.cff` > GitHub contributor statistics.

That is a bad trade **here specifically**, because `CITATION.cff` is already doing the job. All six
software records under concept `21214876` carry 11 creators with 2 ORCIDs, matching the file
exactly. Adding a `.zenodo.json` to fix one field would mean re-declaring the whole author list,
title, description and licence in a second file that no test covers and that can silently drift from
the first. `CITATION.cff` is also the single source of truth for `_declared_dois()` in
`tests/test_integrations.py`, so it cannot simply be deleted in favour of the new file.

**And supplying `related_identifiers` means owning the whole list.** Measured, rather than assumed,
against records that already exist:

| | `.zenodo.json` `related_identifiers` | GitHub link on the minted records |
|---|---|---|
| `citation-file-format/citation-file-format` (11 records) | 3 declared | only their own hand-written `…/releases/tag/<v>` — no auto link |
| `nipy/nipype` (records incl. 2026-08-07) | none declared | Zenodo's auto `isSupplementTo → …/tree/<tag>` |

So the auto GitHub link that every MolDeTr record currently carries would become a per-release
string to maintain by hand. That this is a real hazard rather than a theoretical one shows in the
first repository: record `1242911` has `version: 1.0.3-2` and a link pointing at
`…/releases/tag/1.0.3-1`, permanently, in an immutable record.

*Caveat, stated so the evidence is not read as stronger than it is:* the "with" arm's records span
2017–2021 while the control is current, and no recent example could be found to close that gap.

**What is done instead.** The guard above: `integrations.yml` runs on `release: published`, the test
asserts the relation on the newest record, a failure files an issue, and
`scripts/zenodo_add_paper_doi.py` repairs it.

**That last step was taken on 2026-08-11, before cutting v1.4.0.** This paragraph used to end "if
that ever becomes too slow a loop, the better next step is to run the fixer *from* the release job
… at the cost of putting a `deposit:write` + `deposit:actions` token into Actions secrets." Five
hand-repairs was that loop, and v1.4.0 would have been the sixth. The cost was paid deliberately:
the token is a repository secret, and the mitigations that make it tolerable in a public repository
are listed under *After publishing* above and asserted in `tests/test_integrations_repair.py`.

It remains the right shape for the reason given here — it keeps `CITATION.cff` authoritative and
duplicates no metadata, which is exactly what a `.zenodo.json` would not do.
