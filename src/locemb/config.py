"""Tiny YAML config system with `_base` inheritance and `a.b.c=value` overrides."""

from __future__ import annotations

import copy
import os
from typing import Any, Iterable

import yaml


def deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def get_dotted(cfg: dict, key: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def load_yaml_with_base(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base", None)
    if base:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base)
        cfg = deep_merge(load_yaml_with_base(base_path), cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: Iterable[str] | dict) -> dict:
    cfg = copy.deepcopy(cfg)
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            set_dotted(cfg, k, v)
        return cfg
    for o in overrides:
        if "=" not in o:
            raise ValueError(f"Override must look like key=value, got: {o!r}")
        k, v = o.split("=", 1)
        set_dotted(cfg, k.strip(), yaml.safe_load(v))
    return cfg


def load_config(path: str, overrides: Iterable[str] | dict = ()) -> dict:
    return apply_overrides(load_yaml_with_base(path), overrides)
