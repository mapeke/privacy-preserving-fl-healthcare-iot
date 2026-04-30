"""YAML configuration loader with deep-merge support for inheritance.

The repo ships several YAML configs in ``configs/``. ``default.yaml`` is the base; experiment-
specific configs (``experiment_dp.yaml``, ``experiment_secagg.yaml``, ``experiment_full.yaml``)
override only the keys they care about.

Typical usage::

    from src.utils.config import load_config
    cfg = load_config("configs/experiment_dp.yaml")     # already merged with default.yaml
    print(cfg["privacy"]["differential_privacy"]["target_epsilon"])

Or programmatic overrides from CLI flags::

    cfg = load_config("configs/default.yaml", overrides={"federation.num_rounds": 5})
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_NAME = "default.yaml"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``. ``override`` wins on leaves."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_dotted_overrides(cfg: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply ``{'a.b.c': value}``-style overrides into a nested config."""
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        cursor = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise TypeError(
                    f"Cannot apply override {dotted!r}: '{part}' is not a mapping."
                )
        cursor[parts[-1]] = value
    return out


def load_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a YAML config, merging it onto ``default.yaml`` from the same directory.

    Args:
        path: Path to the YAML config file.
        overrides: Optional dotted-key overrides applied last (e.g., from CLI flags).

    Returns:
        A fully merged plain ``dict``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    default_path = path.parent / DEFAULT_CONFIG_NAME
    if path.name != DEFAULT_CONFIG_NAME and default_path.is_file():
        with default_path.open("r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        cfg = _deep_merge(base, cfg)

    if overrides:
        cfg = _apply_dotted_overrides(cfg, overrides)

    return cfg


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    """Persist a (possibly merged) config to YAML for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False, default_flow_style=False)
