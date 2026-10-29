"""Discover local PixText packages without scanning unrelated model subtrees."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

MODEL_NAMES = ("PixText-MeanFlow", "PixText-Teacher")


def discover_packages(
    registered_roots: Iterable[str | Path], models_dir: str | Path
) -> dict[str, Path]:
    """Prefer registered PixText roots, then check ComfyUI's models directory."""
    found: dict[str, Path] = {}
    for directory in (*registered_roots, models_dir):
        root = Path(directory)
        if not root.is_dir():
            continue
        for manifest in sorted(root.glob("*/manifest.json")):
            if (manifest.parent / "model.safetensors").is_file():
                found.setdefault(manifest.parent.name, manifest.parent)
    return found


def discover_models(
    registered_roots: Iterable[str | Path], models_dir: str | Path
) -> dict[str, Path]:
    """Find standalone weights and keep complete-package workflow names working."""
    found: dict[str, Path] = {}
    for directory in (*registered_roots, models_dir):
        root = Path(directory)
        for name, package in discover_packages((), root).items():
            if name in MODEL_NAMES:
                found.setdefault(name + ".safetensors", package)
            found.setdefault(name, package)
        for name in MODEL_NAMES:
            weights = root / (name + ".safetensors")
            if weights.is_file():
                found.setdefault(weights.name, weights)
                found.setdefault(name, weights)
    return found
