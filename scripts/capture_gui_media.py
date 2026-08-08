"""Regenerate the README's app media -- ``docs/img/gui.png`` and ``docs/img/demo.gif``.

``tests/test_readme_figures.py`` holds every README figure to a 2x device-pixel floor: an asset shown
at *N* CSS pixels needs *2N* native pixels, or the browser upscales it and it reads as soft. Both app
captures predated that rule and were the only two assets below it -- a hand-taken 1425x1182 screenshot
(~1.4x) and a 960x867 GIF (~1.0x).

Taking them by hand again would fix today's numbers and leave the next ones to chance, so this drives
the real app through the same ``app.launch_app`` entry point the browser tests serve. Scale becomes a
parameter instead of a property of whoever's laptop took the shot.

**Both modes use the real checkpoint on purpose.** ``tests/e2e/conftest.py`` serves a stubbed model,
which is right for tests and wrong here: a front-page figure implying real output must not show
invented numbers. Fetch the weights first if they are absent::

    python scripts/download_weights.py           # ~974 MB, Zenodo 10.5281/zenodo.21217102
    python scripts/capture_gui_media.py          # -> docs/img/gui.png   (2850x2364)
    python scripts/capture_gui_media.py --gif    # -> docs/img/demo.gif  (1720x1424)

**Why a GIF and not a video.** Measured against GitHub's own markdown API on 2026-08-08: a ``<video>``
tag is stripped from rendered Markdown -- to an empty ``<p>`` -- for both a repo-relative ``src`` and
an absolute ``.../raw/main/...`` URL, taking any nested ``<img>`` fallback with it. Only
``https://github.com/user-attachments/assets/...`` survives, and that host is fed by dragging a file
into a GitHub comment: not scriptable, not versioned in the repo, and invisible on PyPI, where this
README is also the package long description. So the animation stays a GIF; it just stops being a 1x
one.

The frames are grabbed as stills rather than transcoded from a recording, which keeps the whole path
in Pillow and avoids an ffmpeg dependency the project does not otherwise have.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: CSS pixels for the still. The viewport is the *layout* being captured; ``--scale`` decides how many
#: device pixels back each one. 1425x1182 reproduces the framing of the figure this replaces.
VIEWPORT = {"width": 1425, "height": 1182}

#: A GIF frame is stored at the viewport's own pixel size -- ``device_scale_factor`` does not apply --
#: so the 2x has to come from a physically larger viewport: 1720 is 2x the 860 the README declares.
GIF_VIEWPORT = {"width": 1720, "height": 1280}

#: Which bundled example to detect. `guajazulene` carries a ppm axis, so the table shows δ in ppm
#: rather than the Hz fallback -- matching the README alt text, which describes a ppm column.
EXAMPLE = "guajazulene"

DETECT = "Detect multiplets"
DEFAULT_PNG = ROOT / "docs" / "img" / "gui.png"
DEFAULT_GIF = ROOT / "docs" / "img" / "demo.gif"

#: Key states, not a steady frame rate. A full-viewport screenshot costs a few hundred ms at this
#: size, so smooth motion is unaffordable in a GIF anyway; a held beat per state is both far smaller
#: and easier to read than a stuttering approximation of one.
FRAME_HOLD_MS = {"empty": 1400, "loaded": 2000, "detecting": 700, "table": 1900, "plot": 3000}


def _run_demo(page, url: str, on_frame: Callable[[str], None] | None = None) -> None:
    """The demo journey, shared by both modes so the selectors cannot drift apart."""
    from playwright.sync_api import expect

    def shot(state: str) -> None:
        if on_frame is not None:
            on_frame(state)

    page.goto(url)
    page.locator("#md-file").wait_for()
    shot("empty")

    page.locator("#md-examples").get_by_role("button", name=EXAMPLE).click()
    expect(page.locator("#md-check")).to_contain_text("Input check")
    shot("loaded")

    page.get_by_role("button", name=DETECT).click()
    shot("detecting")

    expect(page.get_by_text("Detected", exact=False)).to_be_visible(timeout=120_000)
    shot("table")

    # Wait on the plot itself, not a fixed sleep: the table fills before Plotly finishes drawing, and
    # a capture taken in that window shows a populated table beside an empty chart panel.
    expect(page.locator("#md-plot .js-plotly-plot")).to_be_visible(timeout=120_000)
    page.wait_for_function("() => !!document.querySelector('#md-plot .js-plotly-plot .main-svg')")
    shot("plot")


def _write_gif(frames: list[tuple[str, bytes]], out: Path) -> None:
    """Assemble the captured stills into a looping GIF with a per-state duration."""
    from PIL import Image

    images = [Image.open(io.BytesIO(png)).convert("RGB") for _state, png in frames]
    durations = [FRAME_HOLD_MS[state] for state, _png in frames]
    # `optimize` re-uses one palette and emits only changed regions between frames; on a UI capture,
    # where most of the layout is identical shot to shot, that is most of the file size.
    images[0].save(
        out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True
    )


def capture(out: Path, scale: int, gif: bool) -> None:
    """Serve the real app, run one detection, and write a still or an animation."""
    from playwright.sync_api import sync_playwright

    import app as app_module

    demo, (_fastapi, url, _share) = app_module.launch_app(
        prevent_thread_lock=True, server_name="127.0.0.1", show_error=True, quiet=True
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[str, bytes]] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            context = (
                browser.new_context(viewport=GIF_VIEWPORT)
                if gif
                # device_scale_factor multiplies the backing store without changing layout, so the
                # framing matches VIEWPORT while the file gains real detail.
                else browser.new_context(viewport=VIEWPORT, device_scale_factor=scale)
            )
            page = context.new_page()
            _run_demo(
                page,
                url,
                on_frame=(lambda s: frames.append((s, page.screenshot()))) if gif else None,
            )
            if not gif:
                page.screenshot(path=str(out))
            context.close()
            browser.close()
    finally:
        demo.close()

    if gif:
        _write_gif(frames, out)
        box = GIF_VIEWPORT
    else:
        box = {"width": VIEWPORT["width"] * scale, "height": VIEWPORT["height"] * scale}
    kb = out.stat().st_size // 1024
    detail = f"{len(frames)} frames, " if gif else ""
    print(f"wrote {out.relative_to(ROOT)} at {box['width']}x{box['height']} ({detail}{kb} KB)")
    print(f'README should declare width="{box["width"] // 2}" or less to stay >=2x')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--scale", type=int, default=2, help="device pixel ratio (still only)")
    parser.add_argument("--gif", action="store_true", help="record the demo instead of a still")
    args = parser.parse_args()
    capture(args.output or (DEFAULT_GIF if args.gif else DEFAULT_PNG), args.scale, args.gif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
