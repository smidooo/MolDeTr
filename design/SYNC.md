# Keeping Claude Code in sync with these designs

Claude Code reads your **GitHub repo**. So the reliable link is: **the design source of
truth lives *in* the repo**, and Claude Code re-reads it every run. Don't rely on Claude
Code "remembering" a past handoff — give it files it can diff.

## Recommended setup (one-time)

1. Commit this whole `handoff/` folder into the repo as **`design/`**:
   ```bash
   cp -r handoff MolDeTr/design && cd MolDeTr && git add design && git commit -m "Add design source of truth"
   ```
2. Add one line to the repo's `CLAUDE.md` (create it if absent) so every Claude Code session picks it up:
   > Design source of truth is `design/`. Before changing any UI, README figure, or the
   > Gradio theme, read `design/BRAND.md` and `design/PORTING.md`. Apply changes from
   > `design/` — never restyle by hand. The `*.dc.html` mockups are the visual contract.

That's the whole link: repo `design/` + a `CLAUDE.md` pointer. Now every design change
you ship here flows to Claude Code the moment you update `design/` and it reads the repo.

## When you change something in the design workspace (here)

1. I re-export the affected asset(s) — a figure PNG, a token in `BRAND.md`, a `.py` file.
2. You (or I, if the repo is connected) drop the changed files into `design/` and commit.
3. Bump the version stamp so drift is detectable:
   - `DESIGN_VERSION` at the top of `BRAND.md` (e.g. `v1 → v2`) + a one-line entry in the
     `## Changelog` there naming what changed.
4. Tell Claude Code: *"design/ updated to vN — re-apply per design/PORTING.md."* It diffs
   `design/` against the live `app.py` / `README.md` / `docs/img/` and re-applies only what moved.

## What is authoritative vs generated

| Authoritative (edit here → propagate) | Generated (re-export, don't hand-edit) |
|---|---|
| `BRAND.md` tokens & wording | `img/*.png` (from the `.dc.html` generators) |
| `theme.py` / `plotting.py` / `app.py` / `visualization.py` | `README_proposed.md` figures (paths only) |
| the `.dc.html` mockups (visual contract) | `banner-dark.png`, `pipeline*.png` |

If a token changes in `BRAND.md`, the figures must be re-exported from their `.dc.html`
generators (they don't read `BRAND.md` at runtime — inline styles). Ask me and I'll re-run them.

## Fastest path each time

Keep this design workspace as the **editor**, the repo `design/` as the **published copy**,
and `CLAUDE.md` as the **pointer**. Change here → update `design/` + bump version → Claude
Code re-applies. One folder, one pointer, one version stamp.
