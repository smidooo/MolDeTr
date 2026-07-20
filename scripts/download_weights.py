#!/usr/bin/env python
"""Download the trained MolDeTr checkpoint from Zenodo into the default model path.

The weights (~974 MB) are archived on Zenodo (DOI 10.5281/zenodo.21217102), not in git. This fetches
that exact file, verifies its MD5 against the value Zenodo publishes, and writes it where the model
loader expects it (``moldetr/model/model_spin_system_ABCDEFG_exp2.pth``), so a fresh clone can run
``scripts/predict.py`` / ``app.py`` without a manual download.

    python scripts/download_weights.py            # fetch + verify into moldetr/model/
    python scripts/download_weights.py --force     # re-download even if a file is already present
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_NAME = "model_spin_system_ABCDEFG_exp2.pth"
# The immutable v1.0.0 record (10.5281/zenodo.21217102); the checkpoint is byte-identical across versions.
ZENODO_URL = f"https://zenodo.org/api/records/21217102/files/{CHECKPOINT_NAME}/content"
EXPECTED_MD5 = "faf842d1a1d8beae67e0544e28f226b5"
DEFAULT_OUT = ROOT / "moldetr" / "model" / CHECKPOINT_NAME


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> Path:
    """Stream to a .part file with a progress line; return it (caller verifies before replacing)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(
                        f"\r  {done / 1e6:7.0f} / {total / 1e6:.0f} MB ({100 * done / total:5.1f} %)",
                        end="",
                        flush=True,
                    )
    print()
    return tmp


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download + verify the MolDeTr checkpoint from Zenodo."
    )
    ap.add_argument(
        "--output", type=Path, default=DEFAULT_OUT, help="where to write the checkpoint"
    )
    ap.add_argument(
        "--force", action="store_true", help="re-download even if a valid file is present"
    )
    args = ap.parse_args()

    if args.output.exists() and not args.force:
        if _md5(args.output) == EXPECTED_MD5:
            print(f"Checkpoint already present and verified: {args.output}")
            return
        print(f"Existing file checksum mismatch — re-downloading ({args.output}).")

    print("Downloading the MolDeTr checkpoint (~974 MB) from Zenodo 10.5281/zenodo.21217102 ...")
    try:
        tmp = _download(ZENODO_URL, args.output)
    except Exception as e:
        raise SystemExit(f"Download failed: {e}\nURL: {ZENODO_URL}")

    got = _md5(tmp)
    if got != EXPECTED_MD5:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"MD5 mismatch: got {got}, expected {EXPECTED_MD5}. Left the existing file untouched."
        )
    tmp.replace(args.output)
    print(f"Verified (md5 {got[:12]}...) -> {args.output}")


if __name__ == "__main__":
    main()
