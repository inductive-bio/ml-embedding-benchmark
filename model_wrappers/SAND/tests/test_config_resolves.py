"""Resolve every ``class_path`` referenced by the shipped LightningCLI configs.

ch3ml-sand uses jsonargparse / LightningCLI (not Hydra), so a ``class_path``
string is just a dotted ``module.Class`` path. This test parses the YAML,
collects every ``class_path`` recursively, and asserts each one imports and
resolves to an attribute. It guards against the structural refactor leaving a
config pointing at a moved/deleted symbol.
"""

import importlib
from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "sand" / "config"
CONFIG_FILES = ["compression.yml", "no_compression.yml"]


def _collect_class_paths(node, acc):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "class_path" and isinstance(value, str):
                acc.append(value)
            else:
                _collect_class_paths(value, acc)
    elif isinstance(node, list):
        for item in node:
            _collect_class_paths(item, acc)


def _class_paths_for(config_name):
    with open(CONFIG_DIR / config_name) as fh:
        cfg = yaml.safe_load(fh)
    acc = []
    _collect_class_paths(cfg, acc)
    return acc


def _all_class_paths():
    pairs = []
    for name in CONFIG_FILES:
        for cp in _class_paths_for(name):
            pairs.append((name, cp))
    return pairs


@pytest.mark.parametrize("config_name,class_path", _all_class_paths())
def test_class_path_resolves(config_name, class_path):
    module_name, _, attr = class_path.rpartition(".")
    module = importlib.import_module(module_name)
    assert hasattr(module, attr), f"{class_path} (from {config_name}) does not resolve"
