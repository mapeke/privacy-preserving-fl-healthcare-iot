"""Tests for the YAML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config import load_config, save_config


def _write(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_load_default_only(tmp_path):
    base = {"a": 1, "b": {"c": 2}}
    _write(tmp_path / "default.yaml", base)
    cfg = load_config(tmp_path / "default.yaml")
    assert cfg == base


def test_experiment_inherits_from_default(tmp_path):
    _write(tmp_path / "default.yaml", {"a": 1, "b": {"c": 2, "d": 3}})
    _write(tmp_path / "exp.yaml", {"b": {"c": 99}, "e": 5})
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}


def test_dotted_overrides(tmp_path):
    _write(tmp_path / "default.yaml", {"a": {"b": {"c": 1}}})
    cfg = load_config(tmp_path / "default.yaml", overrides={"a.b.c": 42, "x.y": "hello"})
    assert cfg["a"]["b"]["c"] == 42
    assert cfg["x"]["y"] == "hello"


def test_save_and_reload_roundtrip(tmp_path):
    payload = {"a": 1, "b": {"c": 2}}
    out_path = tmp_path / "saved.yaml"
    save_config(payload, out_path)
    reloaded = load_config(out_path)
    assert reloaded == payload


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
