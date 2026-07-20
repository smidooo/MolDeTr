"""Config smoke: every committed Hydra YAML parses, and config_big carries its expected top-level sections.

Guards against a malformed config file or a production section silently dropped — the CLAUDE.md 'keep the
dataclass and YAML in step' rule. Uses OmegaConf directly (no Hydra CWD/compose dance needed).
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
CONF = ROOT / "conf"
YAMLS = sorted(CONF.rglob("*.yaml"))

EXPECTED_SECTIONS = {
    "data_augmentation",
    "device",
    "lognames",
    "model_params",
    "mult_class_indices",
    "optim_params",
    "paths",
    "plotting",
    "pretrained",
    "reg_param_indices",
    "weighting",
}


def test_conf_directory_has_yaml_files():
    assert YAMLS, "no conf/**/*.yaml found"


@pytest.mark.parametrize("path", YAMLS, ids=lambda p: str(p.relative_to(CONF)))
def test_every_conf_yaml_parses(path):
    assert OmegaConf.load(path) is not None  # raises on a malformed YAML


def test_config_big_has_the_expected_top_level_sections():
    cfg = OmegaConf.load(CONF / "config_big.yaml")
    missing = EXPECTED_SECTIONS - set(cfg.keys())
    assert not missing, f"config_big.yaml dropped section(s): {sorted(missing)}"
