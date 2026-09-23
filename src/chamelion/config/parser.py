# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# NOTE: This module was contributed by Markus Pielmeier on PR #63
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from chamelion.config.config import DataConfig, MOSConfig, PathsConfig, TrainingConfig


class ChamelionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="chamelion_")
    out_dir: str = "results"
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    mos: MOSConfig = MOSConfig()
    training: TrainingConfig = TrainingConfig()


def _yaml_source(config_file: Optional[Path]) -> Dict[str, Any]:
    if config_file is None:
        return {}
    try:
        yaml = importlib.import_module("yaml")
    except ModuleNotFoundError:
        print(
            "Custom configuration file specified but PyYAML is not installed on your system,"
            " run `pip install PyYAML`. You can also modify the config.py if your "
            "system does not support PyYaml "
        )
        sys.exit(1)

    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def load_yaml(path: Path, stack: tuple[Path, ...]) -> Dict[str, Any]:
        path = path.resolve()
        if path in stack:
            chain = " -> ".join(str(item) for item in (*stack, path))
            raise ValueError(f"Circular config inheritance: {chain}")
        with path.open() as cfg_file:
            data = yaml.safe_load(cfg_file) or {}
        training = data.get("training", {})
        for split in ("train", "val"):
            split_file = training.pop(f"{split}_split", None)
            if split_file is not None:
                if split in training:
                    raise ValueError(f"Specify either {split} or {split}_split in {path}")
                split_path = Path(split_file)
                if not split_path.is_absolute():
                    split_path = path.parent / split_path
                entries = [
                    line.split("#", 1)[0].strip() for line in split_path.read_text().splitlines()
                ]
                entries = [entry for entry in entries if entry]
                if not entries or len(entries) != len(set(entries)):
                    raise ValueError(f"Empty or duplicate entries in {split_path}")
                if any(Path(entry).is_absolute() or ".." in Path(entry).parts for entry in entries):
                    raise ValueError(f"Split entries must be relative sequence paths: {split_path}")
                training[split] = entries
        parent = data.pop("extends", None)
        if parent is None:
            return data
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        return deep_merge(load_yaml(parent_path, (*stack, path)), data)

    return load_yaml(Path(config_file), ())


def load_config(config_file: Optional[Path]) -> ChamelionConfig:
    """Load configuration from an Optional yaml file."""
    config = ChamelionConfig(**_yaml_source(config_file))
    if set(config.training.train) & set(config.training.val):
        raise ValueError("Training and validation splits overlap")
    if config.training.dataloader == "cham":
        train_sessions = {entry.split("/", 1)[0] for entry in config.training.train}
        val_sessions = {entry.split("/", 1)[0] for entry in config.training.val}
        if train_sessions & val_sessions:
            raise ValueError("Training and validation source sessions overlap")
    return config


def write_config(config: ChamelionConfig, filename: str):
    with open(filename, "w") as outfile:
        try:
            yaml = importlib.import_module("yaml")
            yaml.dump(config.model_dump(), outfile, default_flow_style=False)
        except ModuleNotFoundError:
            outfile.write(str(config.model_dump()))
