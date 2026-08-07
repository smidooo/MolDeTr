"""The checkpoint trust boundary — `SECURITY.md` promises it, and nothing enforced it.

`SECURITY.md` states that `moldetr/inference.py` "loads with ``weights_only=True`` first and only
falls back to ``weights_only=False`` for the fastai-format checkpoint". The first half was true; the
second was not, and the gap is the whole point of this module.

``torch.load(..., weights_only=True)`` raises ``pickle.UnpicklingError`` whenever it refuses a global
that is not on its allowlist (``torch/serialization.py`` — the two ``_get_wo_message`` raise sites).
That is *exactly* what a hostile checkpoint triggers. The old code caught it with a bare
``except Exception`` and retried at ``weights_only=False``, so the safe loader's refusal was the very
thing that unlocked the unsafe load:

    weights_only=True → refuses payload → raises → except Exception → weights_only=False → executes it

Critically, **exception type cannot separate the two cases**: the published fastai checkpoint stores
optimizer state, so it fails the safe load with the same ``UnpicklingError`` a hostile file does. The
fix therefore cannot be "catch something narrower" — it has to positively identify the file. The
fallback is now gated on the checkpoint hashing to the published value, with an explicit opt-in for
people running their own weights (``README.md`` documents ``--checkpoint``, so refusing unknown files
outright would break a supported workflow).

The hostile fixture here is deliberately inert: a plain class instance is not on torch's allowlist,
so it trips the identical guard as a real payload while executing nothing. Writing a genuinely
malicious pickle to disk to test this would be irresponsible and unnecessary.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from moldetr import inference


class _NotOnTorchsAllowlist:
    """Inert stand-in for whatever a checkpoint might smuggle in. Executes nothing."""

    def __init__(self, tag: str = "payload") -> None:
        self.tag = tag


def _write_checkpoint_that_fails_the_safe_load(path: Path) -> Path:
    """A checkpoint that `weights_only=True` refuses — the shape of both the fastai file and an attack."""
    torch.save({"model": {"w": torch.zeros(1)}, "opt": _NotOnTorchsAllowlist()}, path)
    return path


@pytest.mark.unit
def test_an_untrusted_checkpoint_is_refused_not_loaded_unsafely(tmp_path, monkeypatch):
    """The core guarantee. A file that fails the safe load and is *not* the published checkpoint must
    raise — not silently escalate to ``weights_only=False``.

    Asserts on the call log rather than only on the exception: the failure mode being closed is
    "the unsafe load happened", and a test that only checked for a raise would still pass if the
    unsafe load ran first and then something else threw.
    """
    hostile = _write_checkpoint_that_fails_the_safe_load(tmp_path / "hostile.pth")

    calls: list[bool] = []
    real_load = torch.load

    def recording_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(inference.torch, "load", recording_load)

    with pytest.raises(Exception) as excinfo:
        inference.load_checkpoint(inference.build_model(), str(hostile))

    assert False not in calls, (
        f"the unsafe load ran on an untrusted checkpoint (weights_only calls: {calls}) — "
        "the fallback still escalates a refusal into an execution"
    )
    assert calls == [True], f"expected exactly one safe load attempt, got {calls}"

    message = str(excinfo.value).lower()
    assert "trust" in message or "checksum" in message or "md5" in message, (
        f"the refusal must explain itself and name the opt-in; got: {excinfo.value!r}"
    )


@pytest.mark.unit
def test_the_published_checkpoint_is_still_allowed_to_fall_back(tmp_path, monkeypatch):
    """The other half of the gate: the real checkpoint must keep loading.

    Verified by pointing the trusted digest at this fixture rather than by downloading 974 MB, so the
    *gate logic* is proven in the fast lane. That the real file's digest matches the published
    constant is a separate claim, asserted by ``test_trusted_digest_matches_the_published_checkpoint``
    under ``-m model``.
    """
    trusted = _write_checkpoint_that_fails_the_safe_load(tmp_path / "trusted.pth")
    monkeypatch.setattr(inference, "TRUSTED_CHECKPOINT_MD5", inference._md5(trusted))

    calls: list[bool] = []
    real_load = torch.load

    def recording_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(inference.torch, "load", recording_load)

    # Loading into the real model would fail on shape mismatch; the trust gate is what is under test,
    # so assert the fallback was *permitted* and let the later state_dict load raise if it wants.
    with pytest.raises(Exception):
        inference.load_checkpoint(inference.build_model(), str(trusted))

    assert calls == [True, False], (
        f"a checkpoint matching the trusted digest must be allowed to fall back; got {calls}"
    )


@pytest.mark.unit
def test_an_explicit_opt_in_allows_a_users_own_checkpoint(tmp_path, monkeypatch):
    """`README.md` documents `--checkpoint` for user-supplied weights, so the gate must not make
    training your own model impossible — only make trusting it a deliberate act."""
    own = _write_checkpoint_that_fails_the_safe_load(tmp_path / "my_own.pth")
    monkeypatch.setenv("MOLDETR_ALLOW_UNTRUSTED_CHECKPOINT", "1")

    calls: list[bool] = []
    real_load = torch.load

    def recording_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(inference.torch, "load", recording_load)

    with pytest.raises(Exception):
        inference.load_checkpoint(inference.build_model(), str(own))

    assert False in calls, "the documented opt-in must permit the fallback"


@pytest.mark.unit
def test_a_safe_loadable_checkpoint_never_reaches_the_fallback(tmp_path, monkeypatch):
    """A pure-tensor checkpoint loads under `weights_only=True`, so the unsafe path must not run at
    all — no hashing, no fallback, regardless of trust."""
    plain = tmp_path / "plain.pth"
    torch.save({"model": {"w": torch.zeros(1)}}, plain)

    calls: list[bool] = []
    real_load = torch.load

    def recording_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(inference.torch, "load", recording_load)

    with pytest.raises(Exception):  # state_dict mismatch, not a trust failure
        inference.load_checkpoint(inference.build_model(), str(plain))

    assert calls == [True], f"the safe load succeeded, so nothing else should run; got {calls}"


@pytest.mark.unit
def test_the_security_policy_names_the_digest_the_code_enforces():
    """`SECURITY.md` ↔ code contract. The policy points users at a specific Zenodo DOI as the only
    trusted source; the downloader must pin that same record, or the policy is describing a file
    nobody fetches."""
    policy = (Path(__file__).resolve().parent.parent / ".github" / "SECURITY.md").read_text(
        encoding="utf-8"
    )
    from scripts.download_weights import EXPECTED_MD5, ZENODO_URL

    assert "10.5281/zenodo.21217102" in policy, "SECURITY.md no longer names the trusted record"
    assert "21217102" in ZENODO_URL, f"the downloader fetches a different record: {ZENODO_URL}"
    assert EXPECTED_MD5 == inference.TRUSTED_CHECKPOINT_MD5, (
        "the downloader and the loader disagree about which checkpoint is trusted"
    )


@pytest.mark.model
def test_trusted_digest_matches_the_published_checkpoint():
    """The one claim the fast lane cannot make: the constant really is the published file's digest.

    Requires the 974 MB checkpoint, so it is `model`-marked and runs only where that exists. Without
    this, every other test here proves the gate works against a digest that might be wrong.
    """
    ckpt = (
        Path(inference.__file__).resolve().parent / "model" / "model_spin_system_ABCDEFG_exp2.pth"
    )
    if not ckpt.exists():
        pytest.skip(f"checkpoint not present at {ckpt}")
    assert inference._md5(ckpt) == inference.TRUSTED_CHECKPOINT_MD5


@pytest.mark.unit
def test_unpickling_error_is_what_torch_actually_raises(tmp_path):
    """Pins the premise the whole design rests on.

    If a future torch stops raising `UnpicklingError` here, the reasoning in this module's docstring
    silently stops applying. Better to learn that from a red test than from a security review.
    """
    hostile = _write_checkpoint_that_fails_the_safe_load(tmp_path / "premise.pth")
    with pytest.raises(pickle.UnpicklingError):
        torch.load(hostile, map_location="cpu", weights_only=True)
