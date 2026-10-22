"""Validate and resolve a self-contained, local PixText model package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_FILES = (
    "config.json",
    "inference_config.json",
    "model.safetensors",
    "fonts/AlibabaPuHuiTi-3-55-Regular.ttf",
    "fonts/GoNotoCJKCore-v7.0.ttf",
    "fonts/OFL-GoNoto.txt",
    "fonts/README.md",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Model package path escapes its directory: {name}")
    return path


def read_package(path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(path).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported or invalid PixText manifest")
    for name, expected in manifest["files"].items():
        file = package_file(root, name)
        if not file.is_file():
            raise FileNotFoundError(f"Model package is missing bundled file: {name}")
        if file_sha256(file) != expected:
            raise ValueError(f"Model package SHA-256 mismatch: {name}")
    for name in REQUIRED_FILES:
        if name not in manifest["files"]:
            raise ValueError(f"Model manifest must verify {name}")
    config = json.loads((root / "config.json").read_text())
    inference = json.loads((root / "inference_config.json").read_text())
    if config.get("format_version") != 1 or config.get("variant") not in ("teacher", "meanflow"):
        raise ValueError("Unsupported PixText model configuration")
    model = config.get("model", {})
    if model.get("interval_conditioning") is not (config["variant"] == "meanflow"):
        raise ValueError("Model interval_conditioning must match the release variant")
    if not isinstance(manifest.get("model_id"), str) or not manifest["model_id"]:
        raise ValueError("Model manifest must contain a model_id")
    if inference.get("format_version") != 1:
        raise ValueError("Unsupported PixText inference configuration")
    return root, config, inference, manifest
