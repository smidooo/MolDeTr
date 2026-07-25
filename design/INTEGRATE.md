# Integrate into `multiplet_detection_detr` (local Claude Code project)

This design workspace can't write to your local disk. **Claude Code, running inside
`C:\Users\nicol\Documents\PhD2\Code\multiplet_detection_detr`, does the placement.**
Give it the unzipped `handoff/` folder (drop it in the repo, e.g. as `design/`) and the
prompt at the bottom.

## Where each file goes

| From `handoff/` | Destination in the repo | Action |
|---|---|---|
| `theme.py` | `theme.py` (repo root, next to `app.py`) | add |
| `plotting.py` | `plotting.py` (repo root) | add |
| `app.py` | `app.py` | **replace** |
| `visualization.py` | `moldetr/visualization.py` | **replace** |
| `README_proposed.md` | `README.md` | **replace** |
| `docs/SCOPE.md` `INPUT_FORMAT.md` `USAGE_NOTES.md` `DATA_SCHEMA.md` | `docs/` | **replace** |
| `img/*.png` (all, incl. `-dark`, `benchmark`, `architecture`, `input_contract`, `coupling_rule`, `mark`, `pipeline`, `gui`, the vanillin/guajazulene figures) | `docs/img/` | add/replace |
| `banner-dark.png` | `docs/banner-dark.png` | add |
| `BRAND.md` `PORTING.md` `SYNC.md` | keep as `design/` (source of truth) | add |
| `img/social_preview.png` | **upload** in GitHub → Settings → Social preview (not committed) | manual |
| `img/cvd_check.png` | reference only (in `design/`); not shipped | — |

Then add `plotly>=5` to `deploy/hf_space/requirements.txt` and `pip install plotly`.

**Don't ship** the `*.dc.html` mockups into the app — they're the visual contract; keep
them under `design/` for reference only.

## Windows copy commands (if you prefer to do it by hand)

Run from the repo root after unzipping `handoff/` there:
```powershell
Copy-Item handoff\theme.py, handoff\plotting.py, handoff\app.py .
Copy-Item handoff\visualization.py moldetr\visualization.py
Copy-Item handoff\README_proposed.md README.md
Copy-Item handoff\docs\*.md docs\
Copy-Item handoff\img\*.png docs\img\
Copy-Item handoff\banner-dark.png docs\banner-dark.png
Copy-Item handoff -Recurse design   # keep BRAND/PORTING/SYNC + mockups as source of truth
```

## Copy-paste prompt for Claude Code

> Read `design/PORTING.md` and `design/BRAND.md`. Integrate the branded design into this
> repo: add `design/theme.py` and `design/plotting.py` at the root; replace `app.py` with
> `design/app.py` and `moldetr/visualization.py` with `design/visualization.py`; replace
> `README.md` with `design/README_proposed.md`; replace the four `docs/*.md` with the ones
> in `design/docs/`; copy every PNG from `design/img/` into `docs/img/` and
> `design/banner-dark.png` into `docs/banner-dark.png`; add `plotly>=5` to
> `deploy/hf_space/requirements.txt`. Do NOT touch model, dataloader, training, or inference
> code — only the files marked `# BRAND` / `# NEW`. Then run `python app.py` and confirm the
> assignment table, Plotly zoom, and input-check states render. Add to `CLAUDE.md`: "Design
> source of truth is `design/`; read `design/BRAND.md` before any UI/README/figure change."

After this, the loop in `SYNC.md` keeps us in sync: I re-export here → you refresh `design/`
+ bump `DESIGN_VERSION` → Claude Code re-applies.
