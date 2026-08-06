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

## After publishing — re-check the paper relation

**The link to the paper does not survive versioning.** Zenodo prefills a new version's metadata from
the previous version, and the `isSupplementTo` relation pointing at the article is **not** carried
over. It was present on v0.1.0, then silently absent for three consecutive releases, and had to be
restored by hand on 2026-08-03.

Nothing warns you about this, and the release still looks completely normal without it. So after
every publish:

1. Open the new version record on Zenodo.
2. Confirm it carries `isSupplementTo` → `10.1021/acs.analchem.5c03465`.
3. If missing, add it and save.

Verify from the public API rather than the edit form, so you are reading what a citation tool would
read:

```bash
curl -s https://zenodo.org/api/records/<new-version-id> \
  | python -c "import json,sys; m=json.load(sys.stdin)['metadata']; \
print([(r['relation'], r['identifier']) for r in m.get('related_identifiers', [])])"
```

A correct v1.1.1 (record `21757166`) reports both relations:

```
[('isSupplementTo', 'https://github.com/smidooo/MolDeTr/tree/v1.1.1'),
 ('isSupplementTo', '10.1021/acs.analchem.5c03465')]
```

While you are there, confirm the concept DOI still resolves to the new version, and that the
creator list and licence carried over intact.
