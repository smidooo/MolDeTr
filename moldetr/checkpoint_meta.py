"""Identity of the published checkpoint — deliberately dependency-free.

Two consumers need these constants and they cannot share them any other way:

* ``moldetr.inference.load_checkpoint`` gates its ``weights_only=False`` fallback on the digest, so
  a file that fails the safe load is only trusted when it *is* the published checkpoint.
* ``scripts/download_weights.py`` verifies what it fetched.

``scripts/`` is not shipped in the wheel (``[tool.setuptools.packages.find] include = ["moldetr*"]``),
so ``inference.py`` cannot import the downloader's constants; hence this module, which imports
nothing heavier than ``hashlib`` and is therefore safe to pull in from anywhere in the package.

**The downloader deliberately does NOT import this module, and keeps its own copy of the digest.**
That looks like duplication and is a considered trade. ``python scripts/download_weights.py`` puts
``scripts/`` on ``sys.path[0]``, not the repo root, and it is the *bootstrap* step — it exists so a
fresh clone can fetch the 974 MB weights *before* the package is installed. Importing ``moldetr``
there would make the one script that must work on an uninstalled tree depend on the tree being
installed. Two literals plus a contract test that pins them together
(``tests/test_checkpoint_trust.py``) is the cheaper failure mode than a bootstrap that breaks, and it
matches how this repo already guards ``pyproject.toml`` against the deploy manifests.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: File name of the published checkpoint, as archived on Zenodo and as the loaders expect it on disk.
CHECKPOINT_NAME = "model_spin_system_ABCDEFG_exp2.pth"

#: The immutable v1.0.0 record (DOI 10.5281/zenodo.21217102). The checkpoint is byte-identical
#: across versions, so the version record is pinned rather than the concept record: this constant
#: identifies *one exact file*, which is the whole point of a trust anchor.
ZENODO_RECORD = "21217102"
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{CHECKPOINT_NAME}/content"

#: MD5 of that exact file. This is a trust anchor, not an integrity-against-corruption check.
#: MD5 is retained because it is the digest Zenodo itself publishes for the record, so this value can
#: be checked by hand against the deposit. It is not relied on for collision resistance: the fallback
#: it guards is a defence-in-depth measure over an already-documented trust boundary
#: (see .github/SECURITY.md), not the only thing standing between a user and arbitrary code.
TRUSTED_CHECKPOINT_MD5 = "faf842d1a1d8beae67e0544e28f226b5"

#: Escape hatch for people running their own weights. README.md documents ``--checkpoint`` for
#: user-supplied files, so refusing every unrecognised checkpoint would break a supported workflow;
#: requiring an explicit opt-in keeps that workflow while making the unsafe load a deliberate act.
ALLOW_UNTRUSTED_ENV = "MOLDETR_ALLOW_UNTRUSTED_CHECKPOINT"


def file_md5(path: str | Path, _chunk: int = 1 << 20) -> str:
    """Streaming MD5 so a 974 MB checkpoint is not read into memory to be hashed."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_chunk), b""):
            digest.update(block)
    return digest.hexdigest()
