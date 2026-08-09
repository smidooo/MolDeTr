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

This file covers the **software** record only. The separate **data** deposit (concept DOI
`10.5281/zenodo.21217101`) does not move with a code release — see [Zenodo data
deposit](ZENODO_DEPOSIT.md).

## Before tagging

- [ ] **`CHANGELOG.md` `[Unreleased]` is current.** It is the section most likely to be behind, and
      it is invisible in a diff review because nothing *fails* when it is stale. Precedent: v1.0.0
      shipped while `main` was 61 commits ahead and `[Unreleased]` had fallen behind by 7 PRs.
      Compare against `git log <last-tag>..main` rather than memory.
- [ ] **Rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`** and open a fresh empty `[Unreleased]`.
- [ ] **Bump `version` in `pyproject.toml` — and nothing else.** `CITATION.cff` deliberately carries
      no `version:` or `date-released:` field, because citation surfaces should pin the *concept*
      DOI, which always resolves to the latest release.
- [ ] **CI is green on `main`.** `main` is branch-protected with 11 checks, 7 of them required
      (the two ubuntu legs plus the whole e2e tier). macOS and Windows stay advisory — read them
      anyway before a release, since advisory means unblocking, not unimportant.
- [ ] **Check the skip and deselect counts, not just the green tick.** A tier that silently skipped
      reports success. `-rs` is on the browser steps for exactly this reason.
- [ ] **Anything Python 3.11+-only will pass locally and fail CI.** The local venv is 3.12; CI's
      oldest leg is 3.10. The same asymmetry applies to dependency floors: `numpy` and `gradio` are
      pinned with a floor that only the dedicated floor jobs actually install.

Changes reach `main` through numbered PRs, not direct pushes. Admin override would succeed —
`enforce_admins` is false so a flaky leg can be worked around — but the convention is the audit
trail, so use it.

## After publishing — the paper relation, which is checked for you

**The link to the paper does not survive versioning, and this was never a checklist problem.** The
`isSupplementTo` relation pointing at the article was present on v0.1.0 and absent from every
release since — **five for five**: v1.0.0, v1.1.0, v1.1.1, v1.2.0 and v1.3.0. The last was minted
on 2026-08-09, four days after a tool existed to fix it. Six Zenodo records exist under the concept
DOI and only the first was ever born correct; the other five were restored after the fact.

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

**When the guard fires** — as an `External reference check failed` issue, or red in the Actions tab:

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
