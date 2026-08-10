# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **The `fastai` ceiling moved to `<2.9`, and the comment that guarded it named the wrong pair.**
  `pyproject.toml` carried `fastai>=2.7,<2.8` annotated *"fastai 2.8.x + fastcore 2.x drops
  `L.starmap` → `learner.load()` crashes"*. Measured against the wheels on 2026-08-10, the hazard
  runs the other way round. fastai **2.7.19** — the version that ceiling selected — is the one that
  calls it: `Optimizer.set_hypers` is `L(kwargs.items()).starmap(self.set_hyper)`
  (`optimizer.py:46`), and `Learner.load()` reaches it through `create_opt()` (`learner.py:425`).
  fastcore 2.2.10 contains no `starmap` at all. 2.7.19 survives only because it pins
  `fastcore<1.8` itself. fastai **2.8.8** replaced the call with `.map(star(...))`
  (`optimizer.py:47`) and requires `fastcore>=1.14.6` — so 2.8.x with fastcore 2.x is the pairing
  upstream intends, and the ceiling had been holding this project on the fragile side of the
  change. The bump is a move away from the defect, not toward it. The corrected reasoning now sits
  in all **four** places that quoted the old one: `pyproject.toml`, `.github/dependabot.yml`, the
  py3.13 note in `.github/workflows/ci.yml` (whose "zero slack" argument rested on 2.7.19's
  `torch<2.7`; 2.8.8 allows `torch<3`), and `README.md`'s *Supported versions* paragraph. That last
  one was missed on the first pass and this entry said "three" — the same
  fix-it-everywhere failure the entry itself is about, one file short.
- **`actions/checkout` v4 → v7** in the three failure-reporter jobs — the last `@v4` call sites,
  the other twelve having already moved. v7's breaking change affects only `pull_request_target`
  and `workflow_run`, and no workflow here uses either trigger.
- **`huggingface_hub` floor raised to `>=1.26.1`** in `deploy/requirements-demo.txt`. Inert in
  practice: CI installs it unconstrained and has been resolving 1.27.0.

### Notes
- **No CI lane exercises `learner.load()`, in any configuration.** Its only callers are
  `scripts/evaluate_synthetic.py:917` and `scripts/train.py:39`. The unit suite is weight-free; the
  nightly `-m model` lane runs `scripts/predict.py` and `scripts/evaluate_experimental.py` through
  `tests/test_scripts_local.py` and never invokes `evaluate_synthetic.py`; `train.py` is frozen.
  So the fastai question above could not have been decided by waiting for a lane to go red or
  green — a green run installing fastai 2.8.8 proves imports work and nothing more. It was decided
  by reading the two wheels. Recorded because the same blind spot will apply to the next `fastai`
  or `torch` decision.

### Fixed
- **Both runtime manifests admitted a fastai the package excludes.** `deploy/requirements-demo.txt`
  and `environment.yml` each declared a bare `fastai>=2.7` — no ceiling — while `pyproject.toml`
  caps it at `<2.9` deliberately. The demo manifest is what the Colab demo installs and
  `environment.yml` is the documented way to create the environment, so both would have resolved a
  fastai 2.9+ on a machine the maintainer never sees. Structurally the identical defect
  `test_demo_manifest_declares_the_same_gradio_requirement_as_the_app_extra` was written to catch on
  that same file for gradio; it simply was not asserted for the other pinned dependency.
- **Dependabot's fastai ignore rule could not see the change it was written to block.**
  `update-types` classifies a *version* change, so it does not fire on a range **widening** — which
  is exactly how `<2.8` -> `<2.9` arrived unreviewed on 2026-08-10 despite an explicit
  `version-update:semver-minor` ignore. Both entries now carry `versions:` as well, which is matched
  against the candidate version itself. `torch` additionally gains the semver-minor ignore it never
  had: the asymmetry with fastai had no stated reason, and no CI lane can judge either bump. The
  cost is recorded in the config rather than hidden — this also suppresses a torch security bump
  that arrives as a minor.

### Added
- **A drift guard for the fastai specifier**, parametrized over both runtime manifests
  (`tests/test_deploy_manifest.py`). Confirmed RED on both before the fix and GREEN after, and
  mutation-tested: changing the packaged ceiling to `<2.99` turns both cases red, so the assertion
  compares rather than merely runs. This is the durable answer to the 2026-08-10 drift, where four
  surfaces quoted the old ceiling and three were corrected — a test does that job and vigilance
  demonstrably does not.

 `deploy/requirements-lock-linux64.txt` is `conda list --explicit`
  output — line 4 is the literal `@EXPLICIT`, and the body is conda package URLs, not requirement
  specifiers. Dependabot's pip ecosystem discovers anything matching `requirements*.txt` under the
  configured directory, parsed it, and died on
  `InstallationError: Invalid requirement: '@EXPLICIT'`. That aborts the whole `update_graph` job,
  so **no pip dependency graph has been submitted for three weeks** — a lane that is not a required
  check and therefore went red without blocking anything or telling anyone. Renamed to
  `deploy/conda-lock-linux64.txt`, which is both the fix and the honest name: the file has never
  been installable with pip, and the old name said it was. The three live references
  (`README.md`'s bit-exact install command, `environment.yml`'s header, a `pyproject.toml` comment)
  move with it; the two `CHANGELOG.md` mentions are history and stay as written. No CI job
  referenced the file, and `deploy/requirements-demo.txt` is genuine pip and untouched.
- **`docs/RELEASING.md` linked to a gitignored file**, so the link resolved on the maintainer's disk
  and 404'd for every actual reader, the docs site included. `docs/ZENODO_DEPOSIT.md` is ignored at
  `.gitignore:49` and has never been in the repository. It now points at the data deposit's Zenodo
  record instead. Found the first time `integrations.yml` ever executed: it was added on 2026-08-05
  with a Monday cron whose first firing was due 2026-08-10, so the lane that owns documentation
  links had never actually run, and a manual dispatch on 2026-08-09 caught this within two seconds.
  Worth stating plainly — the check was correct from the day it was written and still found nothing
  for four days, because *scheduled* is not the same as *run*.
- **The release cross-check read the newest release off list position, and that is the wrong clock.**
  `tests/test_integrations.py::_newest_published_release_tag` returned `published[0]["tag_name"]`,
  trusting GitHub to hand back `/releases` in the order Zenodo mints. GitHub sorts that endpoint by
  `created_at`; Zenodo archives in `published_at` order. The two already disagree on this
  repository — v1.0.0 carries `created_at 2026-07-17T08:12:10Z` against `published_at
  2026-07-14T20:01:05Z`, three days apart and in the opposite direction — and they agree at position
  0 today only by luck, which is what kept this latent. The case that separates them is a hotfix cut
  from an older commit and published after a newer minor: Zenodo's newest record would be the
  hotfix, GitHub would list the minor first, and the guard would compare the deposit against a tag
  Zenodo never archived, then fail blaming the relation. Now filtered, then sorted by `published_at`
  descending — in that order, because drafts carry `published_at: None`, which does not compare
  against a string. `test_newest_release_is_the_one_published_last` is a crafted-payload unit test
  in which list order and publish order disagree; it was confirmed red before the sort and green
  after, so it is a test that bites rather than one that merely passes.

### Changed
- `_api` in `scripts/zenodo_add_paper_doi.py` now declares `-> Any`. Deliberately `Any` and not
  `dict | list | None`: the function is a generic JSON round trip that never validates the parsed
  shape, so a narrower annotation would claim something it does not enforce. This file is outside
  the mypy gate (`pyproject.toml [tool.mypy]` covers the pure-NumPy half only), so this is
  readability, not a type-check fix.

### Added
- **The deposit's link to the paper is now guarded instead of remembered.** The `isSupplementTo`
  relation pointing a software record at the article was present on v0.1.0 and absent from every
  release since — **five for five** (v1.0.0, v1.1.0, v1.1.1, v1.2.0, v1.3.0), the last minted four
  days after a tool existed to fix it. The decisive evidence that this was never a diligence
  failure: v1.2.0 was published without the relation *even though v1.1.1 had already been
  hand-edited to carry it*, which rules out Zenodo seeding a new version from the previous one. A
  step performed correctly five times and lost five times is a missing automation. Writing it down
  did not help either, on the smaller sample available: `docs/RELEASING.md` dates from 2026-08-03,
  so only v1.2.0 and v1.3.0 were published while the instruction existed, and it was dropped on
  both. `integrations.yml` now also runs on
  `release: published`, and `tests/test_integrations.py` asserts that the newest software record
  relates the article. Two details carry the check. It **waits out Zenodo's asynchronous webhook**
  first, because a run that fires before minting resolves the concept DOI to the *previous* record,
  finds the relation there, and passes while the release that just went out carries nothing. And it
  **cross-checks the record's version against the newest published GitHub release**, so "Zenodo has
  not caught up" reports itself rather than hiding as a green tick — prereleases included, since
  Zenodo archives one like any other release. The weekly cron stays: it is the only thing that
  catches a relation undone later, or a release whose minting failed after the release-time run
  gave up waiting.
- **`scripts/zenodo_add_paper_doi.py`** — what to run when that guard fires, and the first version
  of this tool to live in the repository rather than on one machine, which is why the checklist
  could not previously name a command. It resolves the newest release from the concept DOI (no
  record id to keep current — the hardcoded one it replaces was obsolete within two releases),
  appends without replacing, is idempotent, and discards the draft if anything fails mid-flight so
  a record is never stranded in edit state. The credential comes from `--token-file` only: a
  `ZENODO_TOKEN` variable exists on the maintainer's machine, it is stale, and it 403s even on a
  read, so an environment fallback would swap a working token for a broken one and blame Zenodo —
  `tests/test_scripts.py` pins that omission, since it is a plausible future "fix". Editing
  metadata mints no new DOI, and restricted records stay restricted because the PUT re-sends the
  whole metadata object.

  Both halves share one `paper_relation_present`. The PowerShell original decided "already present"
  on the identifier alone, so a record relating the article under `references` would have read as
  correct to the fixer while a stricter guard kept failing it — a detect/fix pair that cannot
  converge. Every live record was already backfilled by the time this was written, so there is no
  longer a Zenodo record that *lacks* the relation to test against: the guard is proven by crafted
  payloads and mutation instead, which is the only reason it is more than decoration.

## [1.3.0] - 2026-08-09

### Added
- **A second, independent spin system in the Simulate tab.** The simulator could always do this —
  `simulate_systems` splits the coupling matrix into disconnected blocks, simulates each on a
  per-proton scale and sums them under one global peak rescale — but the only way to reach it was to
  know that convention and hand-assemble a block-diagonal grid cell by cell, because picking a preset
  *replaces* the whole matrix. An optional **Second spin system** panel now gives that second system
  its own preset dropdown, spin-count slider, matrix and line-width table, so the maintainer's own
  example (`AB` + `AA'BB'`) is two dropdowns rather than a 6×6 grid filled by hand. It is collapsed
  and switched **off** by default, so a simulation that ignores it is bit-identical to before —
  including over the wire, where `gradio_client` fills the three new trailing arguments from the
  components' own values. One behaviour does change even with the box off: touching *any*
  second-system control clears the cached spectrum, so the distortion sliders ask for a fresh
  **Simulate & Predict** rather than re-distorting a spectrum the panels no longer describe. That is
  the same invalidation the first panel's grids have always done, and it is the safe direction —
  the alternative is a plot and table labelled with a spin system that is no longer on screen.
  The physics is untouched: the two grids are laid out block-diagonally and handed to the same single
  `simulate_systems` call, which is why `test_simulate_additivity` and `test_simulate_blocks` already
  cover the result and needed no changes. One consequence worth naming — `MAX_MATRIX_SPINS = 8`
  bounds *an editor*, not the spectrum, so `ethyl` + `AA'BB'C` (10 spins) is now reachable and
  legal, the real limit being `MAX_BLOCK_SPINS = 10` per coupled block.

### Changed
- **The Simulate tab grades a match instead of calling it right or wrong, and now shows the
  predicted coupling.** `status` was `dd_hz <= 2.0 and dh == 0`, which collapsed "0.3 Hz out" and
  "40 Hz out" into one `~ off`, and forced `~ off` on a proton-count mismatch *at zero shift error* —
  hiding a perfect δ behind an unrelated defect. Shift now grades `✓ excellent` · `✓ good` · `✓ ok` ·
  `~ fair` · `✗ off` at ≤1 / ≤2 / ≤4 / ≤10 Hz, with the proton count reported beside it in `ΔH`
  rather than folded into the verdict. Couplings get a hybrid tolerance, `max(0.5 Hz, 0.10 × J)`,
  because one fixed number cannot serve both a 2 Hz meta coupling and a 130 Hz ¹³C satellite: the
  same 6.5 Hz error is `✓ excellent` on the satellite and `✗ off` on the meta coupling. Two new
  columns, `pred J (Hz)` and `ΔJ (Hz)`, close a gap where the table showed `GT J (Hz)` with no
  predicted counterpart to compare it against — the CLI twin (`scripts/simulate_and_predict.py`) has
  had them all along. The three coincidentally-equal `tol_hz = 2.0` literals across `app.py` and
  `app_ui/plotting.py` are replaced by one `app_ui/grading.py`. The *figure* deliberately keeps the
  conjunctive rule: a green connector reads as "this detection is good", which a right-δ/wrong-H
  detection is not, so only the table decouples.

### Fixed
- **The spin-matrix editor mislabelled every column past its seed preset's width.** `headers` and
  `datatype` are fixed when the component is built and no handler returns `gr.update`, so a grid
  seeded from a 5-spin preset kept five labels forever: resizing to 8 spins rendered the surplus
  columns as positional indices (`… | E | 6 | 7 | 8`) rather than `F G H`. Latent since the matrix
  editor landed, and immediately visible in the new second panel, which seeds from a 2-spin preset —
  choosing `AA'BB'` there showed `spin | A | B | 4 | 5`. Both grids now size their headers to
  `MAX_MATRIX_SPINS`, which costs nothing at render time because Gradio draws as many columns as the
  *data* has and consults `headers` only for their labels. Found by looking at a screenshot: the
  unit, browser and axe-a11y tiers were all green, because none of them asserted on column labels.
- **A refused checkpoint now explains itself instead of showing a bare "Error" toast.** The trust
  gate under *Security* below made `load_checkpoint` raise `RuntimeError`, but `app.py::_get_model()`
  was called inline inside `run(...)` at both call sites with nothing catching it — while every other
  failure in `predict` and `_detect_stage` returns a status string. Gradio therefore showed a bare
  "Error", and the refusal text is the only place `MOLDETR_ALLOW_UNTRUSTED_CHECKPOINT` is named: the
  one piece of information a user needs in order to run weights they trained themselves was exactly
  the piece that never reached the screen, so the security fix had broken the documented own-weights
  workflow. `_get_model()` is hoisted out of the nested call at both sites and returns through the
  existing `(None, None, message)` channel. The message is fenced rather than interpolated into a
  sentence, because the gate's text is multi-line and carries both MD5s, and markdown collapses those
  lines into an unreadable run precisely where the user has to compare two hex digests.
- **The simulator's ppm window is derived from the field instead of declared beside it.** The
  checkpoint expects 6144 points across a 1200 Hz window — 5.12 points/Hz — and the simulator is
  parameterised in ppm, so that resolution held only while
  `(LEFT_PPM - RIGHT_PPM) * BASE_FREQ_MHZ == 1200`, with all three declared independently. At the
  shipped 80 MHz over a 15 → 0 ppm window it holds exactly, which is why nothing had gone wrong. Set
  `BASE_FREQ_MHZ` to 600 and leave the ppm bounds alone, though, and the window becomes 9000 Hz at
  0.68 points/Hz — an eighth of the trained resolution, with nothing raised anywhere. The detector
  still returns numbers and they are confidently wrong: not an error, an answer that looks like every
  other answer. `LEFT_PPM`/`RIGHT_PPM` now come from `ppm_window(base_freq_mhz)`, so the invariant is
  unrepresentable rather than merely documented. At 80 MHz the derivation yields exactly (15.0, 0.0)
  — byte-for-byte the previous constants, so no behaviour changes on the frozen-checkpoint path.
- **Two README figures rendered below the 2× device-pixel floor.** Nine of the eleven local figures
  are pinned at ≥2.00× their displayed width via `<img … width="N">`; `docs/img/gui.png` (1425×1182,
  ~1.4×) and `docs/img/demo.gif` (960×867, ~1.0×) were the only two written as Markdown
  `![…](…)`, which cannot carry a width at all. Both are regenerated by
  `scripts/capture_gui_media.py`, which drives the real app through the same `app.launch_app` entry
  point the browser tests serve — so the scale is a parameter rather than a property of whoever took
  the screenshot, and the figures show real checkpoint output rather than the stubbed model's.
  `gui.png` is now 2850×2364 (3.31×) and `demo.gif` 1720×1280 at 2.00× — and *smaller* than before
  (377 KB against 465 KB), because it holds five key states rather than approximating motion. It
  stays a GIF because GitHub strips `<video>` from rendered Markdown for both repo-relative and
  `raw` URLs, taking any nested `<img>` fallback with it. `tests/test_readme_figures.py` holds the
  floor.
- **The DOI badge on the README front page renders again.** It was served from
  `zenodo.org/badge/1289888357.svg`, and GitHub does not load README images in the browser — it
  fetches them server-side through its shared `camo` proxy, so Zenodo saw the whole of GitHub as a
  single client against an `x-ratelimit-limit: 120` per-IP-per-minute cap, on a response Zenodo
  marks `cache-control: no-cache` and camo therefore cannot cache. Measured: 4 of 5 fetches through
  camo returned `502 Invalid upstream response (429)`, while all nine `img.shields.io` badges on the
  same page returned `200`. The badge is now a CDN-cached shields.io badge (`max-age=432000`) pinned
  to the **concept** DOI `10.5281/zenodo.21214876`. That settles a second inconsistency in the same
  line: the old `latestdoi` form named the newest *version* DOI, while `docs/RELEASING.md` requires
  citation surfaces to pin the concept DOI and the README's own Availability section already did.
  Pinning the static `zenodo.org/badge/DOI/….svg` form was rejected — measured, it carries the same
  `no-cache` header and the same origin, so it stays flaky. Zenodo's release webhook is untouched
  and releases still mint DOIs; `tests/test_readme_badges.py` now guards both rules.

### Security
- **The `weights_only=False` fallback is gated on the checkpoint's identity.** `.github/SECURITY.md`
  stated that `inference.py` loads with `weights_only=True` first and only falls back to
  `weights_only=False` for the fastai-format checkpoint. The first half was true; the second was not.
  `load_checkpoint` wrapped the safe load in a bare `except Exception` and retried unsafely, and torch
  raises `pickle.UnpicklingError` whenever `weights_only=True` refuses a non-allowlisted global —
  exactly what a hostile checkpoint triggers. **The safe loader's refusal was therefore what unlocked
  the unsafe load.** Catching a narrower exception does not fix this: the published fastai checkpoint
  carries optimizer state and fails the safe load with the *same* `UnpicklingError` a malicious file
  does, so exception type cannot separate them and the decision has to be made on the file's identity.
  The fallback is now gated on the checkpoint's MD5. Measured on a file that fails the safe load,
  `weights_only` calls go from `[True, False]` to `[True]` plus a refusal.
- **The checkpoint's identity is checked before the blanket opt-in, not after.**
  `_require_trusted_checkpoint` consulted `MOLDETR_ALLOW_UNTRUSTED_CHECKPOINT` before computing the
  digest, so setting it once — the documented way to run weights you trained yourself — stopped the
  *published* checkpoint being identified for the rest of the process. Every later load in that
  process took the escape hatch instead of the trust anchor and warned that it was "executing
  arbitrary code from the file" about a file whose digest the gate could have recognised; a warning
  that fires on the known-good checkpoint is a warning users learn to skip. Hashing first restores the
  intended precedence: identity wins, and the opt-in is what it was documented to be — a fallback for
  files the anchor does not recognise. The untrusted path still warns, and still loads. The cost is
  one streaming MD5 on the opt-in path that used to skip it: seconds on a 974 MB file, once per load,
  and only on the branch where the safe loader already refused.

### Tests
The recurring defect this cycle was a guard that reports green while performing none of its work.
Three were found, every one of them by reading a log rather than by anything failing:

- **The dependency audit had never audited anything.** `security.yml` ran `pip-audit --strict`, and
  `--strict` promotes "could not be audited" to a fatal error — which this project triggers on every
  run by construction, because it installs itself with `pip install -e` and its own distribution can
  never resolve on PyPI. The one run the lane had ever had died 1.7 s in and printed no advisory
  table, while `continue-on-error: true` reported the job green. `--skip-editable` does *not* rescue
  it (that yields a `SkippedDependency`, which `--strict` also fatals on); dropping `--strict` does.
  The first real audit reports 45 advisories across `torch`, `pillow` and `setuptools`, all with
  fixes available. Still deliberately report-only — the noise floor now has one reading, not none.
- **The rot watcher would never have run a single check.** `integrations.yml` installed pytest alone,
  reasoning that `tests/test_integrations.py` imports only the standard library — true of that file,
  wrong about the run. pytest loads `tests/conftest.py` before collecting anything under `tests/`, and
  that conftest imported numpy and torch at module scope, so collection died before a single check
  ran, and the job's own failure handler then filed a `ModuleNotFoundError` issue that reads as a
  broken dependency rather than as a job that never worked. A watcher built to prevent hollow-green
  that is itself hollow-green is worse than absent, because its noise trains the maintainer to ignore
  it. Caught before it ever fired: the lane had been added the day before and its cron had not run.
- **The nightly went green while the two tests it exists to run skipped.** Its first dispatch reported
  `11 passed, 8 skipped`, and the workflow's own anti-hollow-green guard passed anyway — the failure
  this whole effort is about, occurring inside the machinery built to prevent it. Three defects, none
  visible from the tick. The clearest: `experimental_rois.zip` nests everything under
  `experimental_rois/`, so a plain extract left the arrays one directory below `MOLDETR_ROI_NPZ_DIR`,
  while the reassuring `roi count: 13` came from a recursive `find` that counted files no consumer
  could see. Now `unzip -j` to flatten and `-maxdepth 1` to count, failing below 13.
- **The declared markers and classifiers now describe reality.** Every lane selected
  `-m "not e2e and not browser and not network"`, which reads as three deliberate exclusions.
  `network` was applied to no test, so the clause excluded the empty set; `slow` and `data` were
  likewise declared and never used. pytest does not warn about a declared-but-unused marker, and
  `-m "not <unknown>"` is not an error — an unknown name matches nothing and the lane goes green, so
  the defect has no symptom. Measured proof it was inert: the default lane deselected 35 tests both
  before and after removing the clause, identically. The `network` marker later returned, this time
  with tests behind it (below).
- **Coverage and formatting are gated, both scoped by measurement rather than aspiration.**
  pytest-cov was a dev dependency with the command sitting in a comment, and only `ruff check` ran,
  never `ruff format --check`. Measured before adding rather than after: 24 files would have been
  reformatted, so the naive step would have been red on arrival — and 18 of them are the frozen
  training stack, the same set the coverage run finds at 0–45 %. Two independent measurements landing
  on one boundary. The exclude list widens to cover that stack and the remaining 6, all test files,
  are reformatted: 24 → 6 → 0.
- **The artifact a user installs is now tested, from outside the repository.** Every job installed
  with `pip install -e .`, so nothing had ever touched the wheel a user actually gets — and an
  editable install cannot detect a missing `py.typed`, a dropped package-data glob, an unregistered
  console script, or a licence file that never reaches `dist-info`, because the source tree supplies
  all of them regardless. The job builds, runs `twine check --strict`, installs the wheel into a clean
  venv and asserts from outside the tree. That `cd` is load-bearing, and the first assertion is that
  `moldetr.__file__` resolves under `site-packages`, because the alternative passes against an empty
  wheel.
- **The real-checkpoint tests run nightly.** The thirteen `model`-marked tests had never run anywhere
  but a maintainer's laptop, and they are not thin: `tests/test_model_contract.py` carries a vanillin
  oracle captured from a live decode against checkpoint md5 `faf842d1a1d8beae67e0544e28f226b5`,
  asserting δ and J to tolerance. So this is not a new reproduction check — one already existed and is
  good. It is the lane that executes it, which is why the change is one workflow file and no new
  tests.
- **The external references nothing was watching are now watched.** This closes the defect that
  started the overhaul: the README's DOI badge began rendering as `502 Invalid upstream response
  (429)` and a *user* reported it, because every check in `ci.yml` runs against the repository's own
  contents. Re-adding the `network` marker required three things to move together, and all three are
  here — the declaration, the tests in `tests/test_integrations.py`, and `and not network` restored to
  all three `ci.yml` selectors, without which the fast lane would start making internet calls on every
  push. Confirmed by the deselect count: 35 → 43.
- **A failing lane now tells someone.** `main` went red on 2026-08-08, stayed red all day, and three
  PRs were opened on top of it; it was found by looking. Nothing here was going to say so —
  `nightly.yml` and `security.yml` had no failure-reporting step at all, and `ci.yml`'s two
  `if: failure()` steps only upload artifacts. `.github/actions/report-failure` files a deduplicated
  issue or comments on the open one, and is wired into all three silent lanes: one implementation
  rather than three copies, because three copies drift and only one of them ever gets tested.
- **The webkit lane: one real race removed, and the remaining failure now reports what it measured.**
  `browser e2e (webkit)` is the only CI job that has ever needed a re-run, and
  `test_moving_a_distortion_slider_does_not_re_simulate` accounted for two of the three. The gesture
  had a genuine race: `page.mouse` is viewport-absolute and performs no actionability check, so the
  press landed at coordinates captured by an earlier `bounding_box()`, and Gradio re-measures its
  layout on every ResizeObserver fire — a press landing off the input emits no `.release` at all.
  `hover()` re-resolves the locator and waits for it to stop moving before the pointer lands. The
  other failure, `test_spectrum_plot_zooms_and_resets`, is deliberately **not** fixed: its only
  evidence was "Locator expected to be visible", which cannot distinguish Plotly drawing late from
  Plotly never running. The retained traces now answer that, identically across two runs —
  `js-plotly-plot`, `plotly` and `main-svg` appear in none of the 15 DOM snapshots across the full
  30 s while `#md-plot` is present throughout, and a `ResizeObserver loop completed with undelivered
  notifications` pageError fires ~1.9 s in. Reproducible rather than coincidence, and tracked as an
  open issue rather than patched speculatively.
- **Dependabot is enabled, configured against alert fatigue rather than missed updates.** This tree is
  torch-sized and the code is frozen at a paper release, so ungrouped weekly PRs would produce a
  stream a maintainer learns to close unread — strictly worse than not enabling it. Two pip groups
  split by what a failure would mean: dev-tooling breaks loudly and locally in CI so it can move
  together, while runtime can change model behaviour so it gets its own PR and its own nightly result.
  Verified the patterns partition the tree with nothing left over: 9 dev-tooling, 15 runtime, 0
  unclassified. `torch` and `fastai` major/minor bumps are ignored. The first grouped run bumped six
  actions.

## [1.2.0] - 2026-08-05

### Changed
- **The `gradio` floor is now `>=6.21,<7`, up from `>=6.0`** (#26). Gradio's icon-only tab-overflow
  button shipped with no accessible name until gradio-app/gradio#13639, which landed in 6.21.0 and
  which axe reports as a critical `button-name` violation. Installs that resolve gradio
  unconstrained have picked the fix up incidentally since 2026-07-29; this makes it a requirement.
  **If you pinned gradio below 6.21, `pip install -e ".[app]"` will now ask to upgrade it.**
- **The repository root is five entries lighter.** `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` and
  `SECURITY.md` moved into `.github/`, which GitHub resolves for the Security tab, the contributing
  link on new issues and the Community Standards checklist exactly as it does the root;
  `design/BRAND.md` moved to `docs/BRAND.md`; and the 40 KB `requirements-lock-linux64.txt` moved
  to `deploy/`. Nothing that is resolved *by path* was touched: `LICENSE` and `CITATION.cff` stay
  at the root because the licence detector and the citation widget read the root only, and
  `THIRD_PARTY.md` stays because `pyproject.toml` declares it in `license-files` and it ships
  inside the wheel.
- **The demo dependency manifest is named for what actually installs it.**
  `deploy/hf_space/requirements.txt` → `deploy/requirements-demo.txt`. There is no Hugging Face
  Space — a Gradio Space requires an HF PRO account — so the file's only live consumer is the Colab
  demo notebook, which `pip install -r`s it directly. The old name read as dead deployment
  scaffolding, which is plausibly why three commits raised the gradio floor without anyone
  revisiting it. `deploy/hf_space/README.md` stays where it is: it carries the Space front-matter
  and must be copied as `README.md` if a Space is ever created.

### Removed
- **The `design/` handoff scaffolding.** `HANDOFF_README.md`, `INTEGRATE.md`, `PORTING.md` and
  `SYNC.md` described a GUI migration that shipped in `e816d64`. Nothing in the code, CI or README
  referenced them, and their instructions had gone stale: they still routed `theme.py` and
  `plotting.py` to the repository root, where neither has lived since that migration, and copied
  them from a `handoff/` directory that is not part of this repository. `design/BRAND.md` — the one
  file with a consumer — survives as `docs/BRAND.md`.

### Fixed
- **The brand contract can no longer skip itself.** `tests/test_brand_contract.py` guarded its two
  `BRAND.md`↔code tests with `skipif(not BRAND_MD.exists())`, whose reason string claimed the file
  was gitignored. It is committed, so the guard never fired — but had the file ever gone missing,
  those two tests would have skipped and the suite would have reported green while enforcing
  nothing. The guard is gone: a missing `docs/BRAND.md` now fails them (verified by removing it —
  2 failed, not 2 skipped).
- **`validate_spectrum` is annotated, so `py.typed` no longer over-promises** (#25). The function is
  public and the package ships a `py.typed` marker, but neither its argument nor its return carried
  an annotation — so every typed downstream consumer silently received `Any` while the marker
  advertised otherwise. Verified against the declared dependency floor (Python 3.10 / numpy 2.2.6)
  as well as 3.12 / numpy 2.5.1, because checking only the newest numpy is what previously shipped
  a `py.typed` contract that was wrong for this package's own `numpy>=1.23`.
- **The demo manifest no longer sits below the packaged gradio floor.** It declared
  `gradio>=6.0,<7` — 21 minor versions under `pyproject.toml`'s `>=6.21` — and
  `deploy/hf_space/README.md` pinned `sdk_version: 6.20.0`, the exact version carrying the critical
  `button-name` violation. Colab installs resolved the newest 6.x and so avoided the defect by
  luck, which is precisely the accident that raising the floor was meant to stop relying on.
- **Two `TODO` comments that misdescribed working code.** `moldetr/matcher/matcher.py` carried
  `# TODO: Implement the GIoU cost` directly above a `giou_cost()` that is implemented, called by
  the matcher and weighted through `giou_cost_weighting` — a reader auditing the paper's loss
  function would have concluded the term was never written. `moldetr/learner/multi_multiplet_learner.py`
  labelled `single_parameter_loss_partial` "just temporary for debugging" although it feeds
  `single_parameter_loss_metric` on the live metric path. Ten commented-out `print(` lines were
  removed alongside them. No behaviour change.

### Tests
- **The declared gradio floor is now asserted, not just installed** (#27 and follow-up). A
  `gradio-floor` CI job pins `gradio==6.21.0` in the same resolution pass as the extras and runs the
  unit, e2e and chromium-a11y tiers against it. Alongside it,
  `test_tab_overflow_control_has_an_accessible_name` forces the tab strip to overflow and asserts
  the control has an accessible name — the defect the floor exists to prevent, which no previous
  scan could see because the control stays `display:none` until the tabs genuinely do not fit.
- **The three gradio declarations are now held in step by a test** (`tests/test_deploy_manifest.py`).
  It parses the floor out of `pyproject.toml`'s `app` extra, then asserts that
  `deploy/requirements-demo.txt` declares the identical specifier — ceiling included, so a dropped
  `<7` fails too — that `deploy/hf_space/README.md`'s `sdk_version` is at or above it, and that the
  manifest names every distribution the extra does. The `gradio-floor` job cannot catch this drift:
  it installs from the pyproject extras and never opens these files.

## [1.1.1] - 2026-08-02

Findings from the post-release code review of #21 — the review this project requires on >100-LOC
diffs touching the distort hub, which was skipped before v1.1.0 shipped — and from a second,
adversarial review of the branch that fixed them. No change to the shipped checkpoint, to
`moldetr.distort`'s behaviour, or to any number the paper reports.

### Fixed
- **The `eval` extra now provides the network.** Moving torch behind a `model` extra added
  `moldetr[model]` to `app` and `dev` but not to `eval`, so the documented
  `pip install -e ".[eval]"` → `python scripts/evaluate_synthetic.py` failed on a clean install
  with `ModuleNotFoundError: torch`. Conda users were unaffected (`environment.yml` carries
  pytorch); the break was confined to the pip path.
- **`moldetr predict` / `moldetr app` now name the remedy** when a needed extra is absent, instead
  of raising a bare `ModuleNotFoundError` whose fix appears only in the README — and they name the
  *right* extra: `moldetr app` reaches `gradio` before it reaches torch, so it points at
  `moldetr[app]`, which `moldetr[model]` would not have installed.
- **Absence is proven rather than inferred.** `ImportError.name` reports the innermost module the
  import machinery failed on, so a `DLL load failed while importing _C` inside a perfectly good
  torch arrives as `name='torch._C'` and roots at `torch`. Reading that as "not installed" sent
  the user to a `Requirement already satisfied` *and*, because the hint is raised as `SystemExit`,
  discarded the traceback that would have located the real fault. `find_spec` is now used to
  confirm the package is genuinely missing; an installed-but-broken one keeps its traceback.
- **The ceiling error stays readable.** `MAX_BLOCK_SPINS` rejection interpolated `2**n_spins` into
  its message: passing a 6144-point spectrum where shifts belong produced an 1850-digit number,
  and past ~14285 it tripped CPython's integer-to-string limit and raised from inside the
  diagnostic. It now writes `2**n` rather than evaluating it.
- **Input validation on the public spin-physics API.** `build_hamiltonian()` rejects a coupling
  matrix whose lower triangle is not mirrored above the diagonal — only `i<j` is read, so such a
  matrix silently produced a *decoupled* Hamiltonian (an AX pair came back as two singlets). Also
  rejects the empty system (which returned a 0-d scalar where `NDArray[complex128]` is declared),
  a spin count past `MAX_BLOCK_SPINS`, `lowering_operators(n<1)`, and an `fx` whose shape does not
  match the Hamiltonian. Symmetric and upper-triangular matrices both remain valid; `simulate`,
  `simulate_systems` and `coupling_blocks` are deliberately unchanged, since their shared
  upper-triangle convention is pinned by `tests/test_simulate_blocks.py`.
- **`set_seed()` imports torch before seeding.** It previously seeded Python and NumPy first, so a
  torch-free install got a half-reseeded process plus an exception.
- **`scripts/quick_validation.py`'s config gate can fail.** `check_config_imports()` wrapped a bare
  `print` in `try`/`except` — nothing was imported, so it reported PASS unconditionally while its
  docstring claimed it checked that the config loads. CI runs this script as a gate.

### Changed
- **`lowering_operators` now shares the `MAX_BLOCK_SPINS` ceiling.** `n < 1` and
  `n > MAX_BLOCK_SPINS` both raise where they previously returned `[]` and an oversized list
  respectively. This narrows a function released in v1.1.0 within the same cycle. `simulate` and
  `transitions` do **not** share the ceiling — recorded as a gap in `_validated_spin_count`'s
  docstring rather than silently implied.
- **`THIRD_PARTY.md` records the code that actually ships.** It listed only the removed GPL file
  while omitting Deformable DETR (© 2020 SenseTime) and DETR (© Facebook), whose sources ship in
  the wheel and carry their own headers. It now also names the GPL version (GPL-3.0-or-later) and
  the SHIMpanzee copyright holders, and is declared in `license-files` so it reaches the wheel —
  v1.1.0 shipped `LICENSE` alone.
- **Corrected two docstrings that described the wrong mechanism.** `add_shim_distortions` claimed
  the `toss_coin` branch "fails loudly rather than silently altering the augmentation
  distribution"; that branch is statically unreachable (`toss_coin` is pinned to `0.99`), so it is
  the pin, not the removal, that alters the distribution. `reproducibility.py` gave the raise as
  the *reason* for unreachability. Neither the pin nor any behaviour was changed. `[four-skills]`

### Tests
- Public-API tests now assert behaviour rather than shape. Executed against deliberately broken
  builds, the previous assertions passed for: `lowering_operators` returning the *raising*
  operator, `transitions` returning `freqs*2`, and a fully decoupled system. They now pin the
  J-splitting, the roof effect, and the strictly-lower-triangular structure of `I⁻`.
- Removed a Hermitian assertion that could not fail — only `i<j` is read, so `H` is Hermitian by
  construction for every input — along with its comment claiming it caught transposed couplings.
- The licence re-introduction guard matches the shim *implementation* rather than the filename
  `shimming.py`, closing the "vendored copy under a new name" case its own docstring named.
- The packaging contract test asserts sufficiency (every extra backing a network command declares
  `moldetr[model]`), not just that torch lives in the `model` extra. `[four-skills]`

## [1.1.0] - 2026-08-01

### Added
- **Public spin-physics API.** `moldetr.simulate` now exports `build_hamiltonian()`,
  `lowering_operators()` and `transitions()`. Downstream simulation code previously had to reach for the
  private `_IM` / `_build_hamiltonian` / `_embed`, which carry no compatibility promise — a rename here
  would have broken it silently at a distance. `build_hamiltonian()` infers the spin count from the shifts
  and raises on a coupling matrix that disagrees. The private originals are unchanged and still exported.
- **`py.typed` (PEP 561).** Type checkers now see the package as typed, so downstreams no longer need an
  `ignore_missing_imports` override that suppresses real errors alongside the noise. Declared as
  package-data, so it actually ships in the wheel.
- **The paper's 25 Table S2 spin-system presets** in the Simulate dropdown — 11 strongly coupled, 9 weakly
  coupled, 5 uncoupled — replacing a choice of three hand-written phenotypes with the regimes the model was
  actually trained on. Built as a group-level table plus an expander, because the per-spin form has a trap:
  N equivalent protons need N rows at the same shift, not one row labelled `proton_count = N`. Example
  molecules follow the ACS proof corrections, not the pre-proof SI source.
- **¹³C satellites on the Simulate tab, on by default.** Training applied satellites to *every* spectrum,
  unconditionally; the tab applied none and offered no control, so its output was systematically cleaner
  than anything the model was trained on. Adds a checkbox and a J slider bounded to the trained 40–220 Hz.
- **Simulate tab.** Pick a built-in spin-system phenotype, edit its per-spin shifts, coupling and line
  width, optionally apply training-range distortions (noise / phase / broadening / baseline), then detect
  and compare against ground truth in one round trip.
- **Ground-truth-vs-prediction comparison view.** An intuitive overlay — ground truth as teal markers on a
  lane above the spectrum, predictions as clay markers with opacity ∝ confidence, connectors coloured green
  within tolerance and amber outside it, missed ground truth and spurious detections called out in red —
  plus a matching table with explicit `Δδ` / `ΔH` error columns and a `✓ match` / `~ off` / `✗ missed` /
  `+ extra` status per row.
- **Frontend verification layers.** The GUI had almost no automated coverage. Added: unit tests for the
  coordinate helpers that position every marker; a brand-contract suite (tricolor parity across the two
  renderers, the δ≠Δ rule, the "colour is never the only channel" numbering rule, `max J` wording,
  `BRAND.md`↔code token sync); a Blocks-graph contract (component/event counts, event-vs-callback arity,
  elem_id presence and uniqueness, no orphaned components); browser-level branding assertions; and a
  selector-liveness meta-test that parses `CUSTOM_CSS` and fails when a rule stops matching the DOM.
- **`design/` brand docs are now committed.** `BRAND.md` was declared the source of truth while being
  gitignored, so nothing could check it. It ships with the repo (images excepted) and the sync test reads it.

- **Animated demo + docs site.** An animated Gradio demo GIF in the README, a GitHub Pages landing page
  (`docs/index.md`), and a `.github/` PR template + CODEOWNERS.
- **Comprehensive test & validation suite (313 tests, ~11 perspectives).** A weight-free CI lane now
  exercises the full DETR build + forward pass on CPU, a one-step training update (finite gradients), the
  metrics, transforms/normalization (order-invariant coupling embedding + `Normalize` round-trip), config
  parsing, and seeded reproducibility — plus property-based (Hypothesis) and robustness fuzzing,
  schema/data-contract guards for the 13-ROI test set, and matcher/loss integration checks. CPU/GPU-parity
  goldens (`tests/reference_outputs/*.npy`) put the deformable-attention op under CI (no GPU needed to
  compare). Heavy/opt-in tiers are gated by pytest markers
  (`unit`/`e2e`/`browser`/`model`/`data`/`network`), a `[notebooks]` extra (nbmake) executes the Colab
  notebooks end-to-end, and `CONTRIBUTING.md` documents how to run each tier.

### Changed
- **PyTorch and fastai are now an optional `model` extra, not base dependencies.** The spin-physics half
  (`moldetr.simulate`, `moldetr.distort`) is pure NumPy/SciPy, so simulation-only consumers no longer
  install several hundred megabytes they never call. **This is user-visible:** a bare `pip install -e .`
  no longer provides inference — anything that loads the checkpoint or runs the network needs
  `pip install -e ".[model]"`. The `app`, `dev` and `eval` extras self-reference `moldetr[model]`, so every
  documented command is unaffected, and CI already installs CPU PyTorch explicitly. The import-graph
  property this relies on already held; only the dependency declaration was wrong.
- **Public copy reads plainer.** A language pass over the README, both notebook narratives, and the
  GUI microcopy: em-dashes replaced with ordinary punctuation, rhetorical lead-ins dropped, captions
  tightened. Numbers, DOIs, links, and CSS selectors are unchanged, and UI strings pinned by tests
  were updated together with their assertions. One sentence of new content: the README now documents
  the `--plot` ground-truth overlay that shipped undocumented above.
- **Scope framing aligned with the paper.** Removed the "research prototype" / "well-resolved spectra"
  language that understated the peer-reviewed method; clarified that deviations come from
  out-of-distribution acquisition/processing (unusual distortions, non-standard pulse sequences,
  mixtures, non-1200 Hz windows), not from spectral resolution.
  **Completed 2026-07-25:** this entry previously overstated itself. The framing had been removed from the
  disclaimer *body* only — the header chip still read "Research prototype", the GUI accordion was still
  titled "Research prototype — scope", the constant was still named `PROTOTYPE`, and the README screenshot
  alt-text still described the chip. The chip is now gone, the accordion is **Scope & limits**, and
  `BRAND.md` is bumped v1 → v2 so the brand source of truth moved first, per its own contract.
- **One launch path for the app.** `launch_app()` is the single way MolDeTr is served — `python app.py`,
  `moldetr app`, and every test fixture route through it. Gradio 6 moved `theme=`/`css=` from `Blocks(...)`
  onto `.launch()` and raises nothing when they are omitted, so the app had been served *unstyled* in every
  automated test: a dead selector or a lost font could not have been detected.
- **Distortion phase parity.** The analytic signal is conjugated (`np.conj(hilbert(...))`) to match the
  convention the shipped model was trained under.
- **GUI modules moved into `app_ui/`** (`plotting.py`, `theme.py`); `app.py` stays at the repo root as the
  Gradio entry point. README documents the layout.
- **Every event has an explicit `api_name`.** Gradio otherwise derives endpoint ids from callback names,
  which published `/_spec_report` and `/_spec_report_1` — a public API surface named after private helpers,
  with the suffix decided by registration order.

### Fixed
- **The "trained without line broadening" warning was false, and is gone.** The Simulate tab told users
  broadening was outside the training distribution, and `REQUIREMENTS.md` carried the same error as a hard
  constraint. Both came from reading `augment_distortions` in its *current* state, where `toss_coin = 0.99`
  is hardcoded and the shim and broadening branches are dead — but that literal postdates the shipped
  weights by seven weeks. Dated from primary sources: the checkpoint was last written 2024-10-14, the pin
  landed 2024-12-01, and at the commit current on the weights' date the line still drew at random. Training
  therefore applied shim (~50 %) and line broadening (~35 %); the slider is *in* distribution and
  `docs/SCOPE.md`'s range table was right all along. A test now guards the copy, since nothing else in the
  suite asserts on component info text.
- **`--seed` and the noise floor are reachable.** `run()` accepted `noise_seed` but neither took nor
  forwarded `noise_frac`, so every production caller was pinned to 0.005 and `docs/SCOPE.md`'s advice to
  "set `noise_frac=0`" described an affordance no supported call path offered. The Simulate tab also now
  skips the floor once the user has ticked "Add noise", where it only masked the slider.
- **First-order phase is range-checked against its window-derived trained bound**, rather than against a
  constant that did not depend on the window.
- **The training-distribution table stated two ranges incorrectly.** Baseline distortion is bounded by 1×
  the noise, not 5×: `add_baseline_distortion` draws from ±`min_peak/sino*base_scale` = 0.5/SNR while
  `add_noise` uses a std of 0.5/SNR on the peak-normalised spectrum — the same number. The first-order
  phase range was restated in the units it is actually bounded in.
- **WebKit a11y flake on the Simulate tab.** The scan had no settle beyond `to_be_visible()`, which resolves
  the instant the panel display flips, so a docs-only PR went red on a critical button-name violation. The
  offender was named from the failing run's Playwright trace rather than inferred: Gradio's icon-only tab
  overflow toggle, which upstream ships with no `aria-label`, transiently rendered while a late webfont
  changed tab widths. The new wait asserts exactly axe's precondition, so a permanently nameless button
  still fails instead of being masked — proven by a test that injects one.
- **`moldetr evaluate-synthetic` works again.** The dispatcher imported the script as
  `scripts.evaluate_synthetic`, and `@hydra.main(config_path="../conf")` only resolves that relative
  path when the task function's module is `__main__`, so every invocation through the console script
  failed with "Primary config module 'conf' not found" while `python scripts/evaluate_synthetic.py`
  worked. Hydra-decorated scripts now run as `__main__`.
- **`moldetr app --help` prints usage instead of starting a server.** The `app` branch ignored its
  remaining arguments, so `--help` fell through to `launch_app()` and the terminal hung on a running
  app, contradicting the dispatcher's documented "`moldetr <cmd> --help` shows that command's own
  options".
- **Colab demo launch cell.** `notebooks/MolDeTr_colab_demo.ipynb` still ran `from theme import ...`
  after that module moved to `app_ui/theme.py`, so Runtime → Run all crashed at the last cell for
  every Colab user. The cell now calls `launch_app(share=True)`, the same entry point every other
  frontend uses. A weight-free import guard (`tests/test_notebooks_static.py`) now runs in the fast
  CI lane and fails when a shipped notebook references a module or symbol that no longer exists.
- **`predict.py --plot` now shows stored ground truth.** ROI `.npz` files carry a `ground_truth`
  array and the shared renderer can draw it as dashed reference lines, but the CLI never passed it
  along. Plots from annotated files now include the overlay, with a stdout note of the count.
  Checkpoint selection via `$MOLDETR_CHECKPOINT` was already in place and tested.
- **Unreadable files no longer produce a traceback.** `predict` called `_load` unguarded while the input
  check had always wrapped it, so the same corrupt file gave a tidy message above the button and a Python
  traceback below it.
- **Uploaded `.npz` files are no longer unpickled.** `np.load(..., allow_pickle=True)` ran on user uploads,
  which executes code carried in the archive. Pickle is now gated on provenance — only files shipped under
  `examples/` get it, and no bundled example needed it on the paths actually walked.
- **A stated `points/Hz` of 0 is refused instead of silently replaced** with 5.12, and negatives — which are
  truthy and previously sailed through entirely — no longer yield a mirrored axis and negative line widths.
  A blank field still means "unset" and uses the default.
- **The export directory no longer leaks.** `tempfile.mkdtemp()` ran per Detect click, creating a directory
  on every detection; one per process is now reused.
- **Dead CSS removed.** `.block-title` matched nothing in Gradio 6 while appearing to style block labels.
- **Figures consolidated to four.** The README now embeds exactly four images — the banner, the
  guajazulene 500 MHz prediction, the vanillin molecule↔spectrum figure, and the GUI. The redundant
  standalone vanillin prediction was removed; its worked-example detail (proton counts, δ, `max J`
  8.2/2.0/8.7 vs 8.0/2.0/8.0 Hz) is folded into the molecule↔spectrum caption.
- **Shared plotting style.** Prediction figures render through one shared matplotlib style, so generated
  prediction plots stay visually consistent. The README banner and diagrams ship as design-tool assets.
- **Label de-confliction with adjustText.** Multiplet annotations are placed with `adjustText` so labels
  no longer overlap peaks or one another on congested windows.
- **Documentation single-source-of-truth.** De-duplicated caveats that had spread across ~13 files.
  Canonical homes: `docs/INPUT_FORMAT.md` (the input contract + the keep-coupling-partners-in-the-window
  rule), `docs/SCOPE.md` (scope/limits, the `max J` caveat, the research-prototype disclaimer, and the
  input-noise rationale), `docs/USAGE_NOTES.md` (how to read the output — output table + failure-mode
  table). Every other file now gives a one-line mention plus a relative link. Also reconciled the
  complex-vs-real input wording (`DATA_SCHEMA.md` ↔ `INPUT_FORMAT.md`: arrays may be stored complex, the
  model consumes the real part) and standardised the |Δδ| median on **0.90 Hz** (aggregate/reproduced),
  with the paper-vs-aggregate 0.89/0.90 note stated exactly once (README *Reproducing the paper*).
- **External resources now live (soft-gating removed).** The earlier *coming soon* placeholders are gone:
  the Zenodo **data** deposit is published (concept DOI `10.5281/zenodo.21217101`, always resolving to the
  latest version — currently v1.1.1; the initial version DOI is `10.5281/zenodo.21217102`), the Hugging Face
  **model** repo (`huggingface.co/smidooo/moldetr`) is live, and the interactive demo runs on Colab. The
  software DOI (`10.5281/zenodo.21214876`) is unchanged.

- **Four latent bugs the new suite surfaced.** (1) `metrics/multiplet_metrics.py` referenced
  `fastai.metrics.accuracy_multi`/`.accuracy` with only `import fastai`, which raises `AttributeError` at
  call time on current fastai — a training-time crash; it now imports `fastai.metrics` explicitly. (2) The
  Hungarian matcher wrote a stray `cost.txt` into the working directory on every call — removed. (3) The
  `moldetr` CLI leaked its rewritten `sys.argv` back to the caller — now restored in a `finally`. (4)
  `scripts/download_weights.py` could overwrite a good checkpoint with a truncated download on `--force`;
  it now verifies the temp file's SHA-256 before replacing.
- **CI lane honours the `network` marker.** The fast selector is
  `-m "not e2e and not browser and not network"`, matching the marker's documented "skipped in the default
  CI lane" contract (previously a future `network`-marked test would have run in CI against its own contract).

### Removed
- **`moldetr/dataloader/shimming.py` — the GPL-derived shim simulator.** It was adapted from
  [SHIMpanzee](https://github.com/smeerten/shimpanzee) under the GNU GPL and sat inside a repository
  labelled Apache-2.0 that ships only `LICENSE`. Making the import lazy limited how far the GPL code
  reached at runtime, but licensing attaches to distribution, not to import — the file was in every clone
  and every archive. **What it costs, stated plainly:** the shim branch was ~50 % of the 2024 training
  distribution, so re-applying it is now out of scope for the public release. That is a statement about
  reproduction, not about the model — the shipped weights *were* trained with it, and `docs/SCOPE.md`'s
  ranges are deliberately unchanged. `add_shim_distortions` keeps its signature and raises
  `NotImplementedError` pointing at the new `THIRD_PARTY.md`, which retains the SHIMpanzee attribution
  because v1.0.0 did ship the file. `moldetr.distort` is unaffected and always was — it wraps only the five
  Apache-licensed `add_*` effects, which `tests/test_licensing.py` now asserts rather than merely
  documenting.
- **Matplotlib banner + molecule-figure generators.** Dropped `scripts/gen_banner.py`,
  `scripts/gen_molecule_figure.py`, and the `[figures]` / `rdkit` extra. The README banner and diagrams now
  ship as design-tool assets, so the generators — which produced an off-brand matplotlib banner and would
  overwrite the shipped assets if run — are no longer needed.

## [1.0.0] - 2026-07-15

Initial public release accompanying the *Analytical Chemistry* article
(DOI: 10.1021/acs.analchem.5c03465).

### Added
- `moldetr/` — 1D Deformable-DETR model package (FPN backbone + deformable transformer,
  with a pure-PyTorch fallback so inference runs without a compiled CUDA op).
- `scripts/predict.py` — checkpoint-only inference on a single 1H NMR spectrum.
- `scripts/aggregate_experimental.py` — reproduces the article's headline experimental numbers exactly
  (|Δδ| 0.90 Hz, |ΔJ| 0.20 Hz, proton-count 93.5%) from committed matched pairs.
- `scripts/evaluate_experimental.py` — evaluation on the preprocessed ROI arrays (no vendor reader required).
- `scripts/{train,evaluate_synthetic,quick_validation}.py` — training, synthetic evaluation, smoke test.
- `tests/test_reproducibility.py` — asserts the bundled example spectra decode back to their stored
  predictions (proton counts exact; δ and max J within tolerance), guarding the deterministic inference path.
- `pyproject.toml`, `environment.yml`, `requirements-lock-linux64.txt` — installable package and environments.
- `structured_output/*.json` — ground-truth ROI annotations for the 13-ROI / 44-spin-system test set,
  documented in `structured_output/README.md`.
- Continuous integration (ruff + quick_validation + pytest) on CPU, across Linux/macOS/Windows.
- `app.py` — Gradio GUI (assignment table + annotated plot) with post-upload input validation and
  selectable ppm-axis handling (auto / manual / none).
- `docs/SCOPE.md` and `docs/USAGE_NOTES.md` — scope, limitations, and how to read the output.
- `examples/` — bundled example windows (guajazulene 500 MHz, vanillin 300 MHz, synthetic).
- `notebooks/MolDeTr_quickstart.ipynb`, `.github/ISSUE_TEMPLATE/`, and Hugging Face Space files.

### Fixed
- **Inference input noise.** `moldetr/inference.py` fed the model a noiseless, FFT-resampled spectrum,
  but the network was trained (and the paper evaluated) on spectra carrying ~0.5%-of-max Gaussian noise;
  the clean input was out of distribution and could misread congested regions. Inference now injects the
  same calibrated noise with a fixed seed, so predictions are deterministic and in-distribution, and the
  live demo reproduces the paper on well-resolved spectra.
- Live coupling decode (`predict.py`, the GUI): report the single largest coupling `max(J)` rather
  than the four permutation-invariant embedding statistics `[sum, min, max, std]`. The exact paper
  couplings are unchanged — they come from the committed `structured_output` path.

### Changed
- Documentation now states that on well-resolved spectra the live path reproduces the paper's
  predictions — chemical shift, proton count, and the largest coupling `max(J)` come out accurate
  (the earlier "approximation" framing predated the noise fix); the committed `structured_output`
  path still recovers the full per-coupling set. Also corrects the input scale-invariance wording
  (only the global scale is normalised away; relative intensity and SNR still matter).
- Second worked example switched from ethylbenzene (80 MHz) to vanillin (300 MHz) — a cleaner ABX
  case whose live prediction matches its ground truth — with a molecule↔spin-system figure.

### Notes
- Trained weights and spectra are archived on Zenodo (DOI: 10.5281/zenodo.21217102), not in git.
