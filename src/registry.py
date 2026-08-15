# src/registry.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fn: Callable
    dim: int
    description: str = ""


_REGISTRY: Dict[str, ModelSpec] = {}


def register(name: str, dim: int, description: str = ""):
    """
    Decorator to register a model/rule callable under a name.
    """
    def deco(fn: Callable):
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' already registered.")
        _REGISTRY[name] = ModelSpec(name=name, fn=fn, dim=dim, description=description)
        return fn
    return deco


def get(name: str) -> ModelSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_models() -> Dict[str, ModelSpec]:
    return dict(_REGISTRY)

