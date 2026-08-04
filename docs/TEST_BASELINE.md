# Test baseline — 2026-07-25

Captured before the frontend-verification workstream (W1) begins, so that every later change has a
known-good reference. Branch `feat/gt-vs-pred-viz` @ `816becb`. Windows 11, Python 3.12, `.venv`.

> **Superseded by the outcome below** — see *After the frontend workstream* at the foot of this
> file for the post-W1 numbers. This section is kept as the "before" half of the comparison.

```
MOLDETR_CHECKPOINT=C:\Users\nicol\Documents\NewCode\MolDeTr_zenodo_staging_v3\model_spin_system_ABCDEFG_exp2.pth
md5 faf842d1a1d8beae67e0544e28f226b5  (973,617,196 bytes) — matches scripts/download_weights.py:24
```

## Lanes — all green

| Lane | Command | Result |
|---|---|---|
| CI lane | `pytest -q -rs -m "not e2e and not browser and not network"` | **249 passed, 1 skipped, 5 deselected** (3:18) |
| in-process e2e | `pytest -q -rs -m e2e` | **2 passed**, 253 deselected |
| browser | `pytest tests/e2e -m browser -q` | **3 passed** |
| real-weights | `pytest -q -rs -m model` | **3 passed, 1 skipped** |
| lint (CI scope) | `ruff check moldetr scripts tests` | clean |
| lint (frontend) | `ruff check app.py app_ui` | clean |
| smoke | `python scripts/quick_validation.py` | **3/3 gating checks passed** |

The single skip is intentional: `tests/test_scripts_local.py:57` — *"Zenodo ROI npz absent"*.

## Coverage — starting point, not yet gated

```
app.py               249 stmts   62 miss   75%
app_ui/plotting.py   109 stmts    8 miss   93%
app_ui/theme.py       16 stmts    0 miss  100%
frontend TOTAL       374 stmts   70 miss   81%
repo TOTAL          2627 stmts  846 miss   68%
```

`app_ui/theme.py` at **100 %** is the clearest argument for W1: every statement in it is executed
(they are module-level constant assignments) while the theme it defines is **never applied in any
test**. Line coverage cannot see this class of defect; layers L2 and L7 exist to.

## Two environment defects found and fixed while capturing this

### 1. The documented "4 pre-existing `test_scripts` failures" were 5, with one cause — now fixed

Previously accepted as a known-red baseline and never re-checked. Actual failures:

```
tests/test_scripts.py::test_quick_validation_passes
tests/test_scripts.py::test_predict_demo_without_checkpoint_fails_cleanly
tests/test_scripts.py::test_predict_reads_moldetr_checkpoint_env
tests/test_scripts.py::test_evaluate_experimental_clean_clone_fails_cleanly
tests/test_scripts_local.py::test_predict_on_example_emits_physical_detections
```

**Root cause: `moldetr` was not installed in `.venv`.** Its `dist-info` was a corrupt shell (no
`RECORD`, no `__editable__*.pth`). The suite still passed in-process only because
`tests/conftest.py:25` injects the repo root into `sys.path` — and that does *not* propagate to the
subprocesses `tests/test_scripts.py:21-30 _run()` spawns. For a script invocation Python sets
`sys.path[0]` to the script's own directory (`REPO/scripts`), never the cwd, so `import moldetr`
died in every child.

The discriminator that proves it: `test_app_imports_and_builds_ui` passed throughout, because it is
invoked as `python -c "import app"`, where `sys.path[0]` is `''` = cwd = repo root.

**Fix:** `uv pip install --python .venv/Scripts/python.exe -e . --no-deps`. All five went green;
the tests were correct all along, so **no `xfail` was added**. CI never saw this because
`ci.yml:27` does an editable install on every run.

> Repair note: the first attempt (`-e ".[dev,app]"`, resolving all extras) tried to replace the
> working CPU torch and aborted mid-uninstall with a Windows access-denied, deleting
> `torch/__init__.py` and degrading torch to an empty namespace package. Restored with
> `uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0+cpu" --no-deps
> --reinstall-package torch`. **Use `--no-deps` when registering the package in this venv.**

### 2. `nbmake` silently disables the entire browser tier on Windows

`.venv/Lib/site-packages/nbmake/pytest_plugin.py:21` calls
`asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())` from **`pytest_addoption`** —
so it fires on *every* pytest invocation on Windows, whether or not a notebook is involved. Selector
loops cannot spawn subprocesses, so Playwright's driver died with a bare `NotImplementedError`
during fixture setup and all 3 browser tests errored.

Verified chain: bare Playwright works; importing gradio does not change the policy; `demo.launch()`
does not change it; but inside pytest the policy is `WindowsSelectorEventLoopPolicy`, and
`-p no:nbmake` makes the lane pass.

**Fix:** `tests/e2e/conftest.py` restores the Proactor policy, guarded to `win32`. Linux CI is
unaffected (its selector loops support subprocesses). The browser lane now passes with no flags.

## Notes for the work that follows

- `ruff` is already clean over `app.py`/`app_ui` — **L0 is about wiring them into CI**
  (`ci.yml:29` lints only `moldetr scripts tests`), not about fixing violations.
- `axe-playwright-python` 0.1.8 installed for L8; Playwright browsers chromium + firefox + webkit all
  present locally for L10. CI still installs chromium only (`ci.yml:70`).
- `pytest-cov` is installed but absent from the `dev` extra in `pyproject.toml` — declare it.
- Only **2 of 8** CI legs are required on `main` (`ubuntu-latest / py3.10`, `py3.11`); `e2e` and
  `browser-e2e` are advisory, contradicting `docs/requirements/REQUIREMENTS.md:64`.

---

# After the frontend workstream — 2026-07-26

| Lane | Before | After |
|---|---|---|
| CI lane | 249 passed / 1 skipped | **299 passed / 1 skipped** |
| in-process e2e | 2 | **5** |
| browser | 3 (chromium only) | **24 × chromium/firefox/webkit = 72** |
| real-weights (`-m model`) | 3 (the selector reached 4 of 10 gated tests) | **11 / 1 skipped** |
| lint | `moldetr scripts tests` | **+ `app.py app_ui`**, clean |
| smoke | 3/3 | 3/3 |
| headline medians | 0.90 / 0.20 / 93.5 | **unchanged** |

The single remaining skip is still `tests/test_scripts_local.py:57` (Zenodo ROI npz absent).

## Layers built

L0 lint wiring · L1 coordinate helpers · L2 `test_brand_contract.py` · L3 `test_ui_graph.py` ·
L4 callback error states · L5 named `gradio_client` endpoints · L6 `test_browser_journeys.py` ·
L7 `test_browser_selectors.py` · L8 `test_browser_a11y.py` · L10 cross-browser matrix ·
L11 the vanillin oracle.

## L9 (visual baselines) — deliberately not built

Screenshot goldens were scoped but dropped. The ground they would cover is already held, and held
better, by layers that fail with a readable reason instead of a pixel diff:

| What L9 would catch | Already caught by |
|---|---|
| stylesheet not applied | L7 computed-style assertions (and the selector-liveness meta-test) |
| a rule stops matching after a Gradio bump | L7 — it found `.block-title` dead on its first run |
| brand colours drift | L2 tricolor parity + `BRAND.md` token sync |
| contrast / a11y regressions | L8 axe scan — it found `eyebrow` at 4.01:1 |
| layout or control breakage | L6 journeys, across three engines |

Against that, screenshot goldens cost binary churn in a public repo on every intentional UI edit,
and flake on font rendering and engine-specific antialiasing — the cross-browser matrix would
multiply that by three. Revisit only if a visual regression ever ships that the layers above
missed; that would be the evidence this trade-off was wrong.

## `docs/requirements/` stays gitignored

`REQUIREMENTS.md` is local project-management guidance and is not read by any test. Publishing it
is a call about what belongs in a public paper companion, not a correctness fix — so it stays
local, and this note records that the gitignore is deliberate rather than an oversight.

The brand source of truth is the opposite case. `docs/BRAND.md` is committed precisely because
`tests/test_brand_contract.py` reads it in CI: a source of truth the suite cannot open is one
nothing enforces.
