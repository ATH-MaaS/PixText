"""Native workflow downloads must produce a usable, verified local model."""

import hashlib
import json
from pathlib import Path

import pytest

from pixtext.loading import REQUIRED_FILES, read_package

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def split_package(tmp_path):
    package = tmp_path / "bundled"
    package.mkdir()
    for name in REQUIRED_FILES:
        if name == "model.safetensors":
            continue
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"bundled asset")
    (package / "config.json").write_text(
        json.dumps(
            {"format_version": 1, "variant": "meanflow", "model": {"interval_conditioning": True}}
        )
    )
    (package / "inference_config.json").write_text('{"format_version": 1}')
    weights = tmp_path / "PixText-MeanFlow.safetensors"
    weights.write_bytes(b"downloaded model weights")
    files = {
        name: hashlib.sha256(
            weights.read_bytes() if name == "model.safetensors" else (package / name).read_bytes()
        ).hexdigest()
        for name in REQUIRED_FILES
    }
    (package / "manifest.json").write_text(
        json.dumps({"format_version": 1, "model_id": "PixText-MeanFlow", "files": files})
    )
    return package, weights


def test_downloaded_weights_use_bundled_sidecars_without_copying_weights(split_package):
    package, weights = split_package
    root, config, _, manifest = read_package(package, weights_path=weights)
    assert root == package
    assert config["variant"] == "meanflow"
    assert manifest["model_id"] == "PixText-MeanFlow"
    assert not (package / "model.safetensors").exists()


def test_wrong_or_incomplete_download_fails_checksum(split_package):
    package, weights = split_package
    weights.write_bytes(b"wrong model")
    with pytest.raises(ValueError, match="SHA-256 mismatch: model.safetensors"):
        read_package(package, weights_path=weights)


def test_bundled_sidecars_are_still_checked(split_package):
    package, weights = split_package
    (package / "fonts/GoNotoCJKCore-v7.0.ttf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch: fonts/"):
        read_package(package, weights_path=weights)


def test_weight_override_does_not_disable_manifest_path_checks(split_package):
    package, weights = split_package
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["files"]["../outside"] = "0" * 64
    (package / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        read_package(package, weights_path=weights)


def test_workflows_declare_the_selected_weight_and_complete_ocr_download_note():
    index = json.loads((ROOT / "model_index.json").read_text())
    expected = {model["model_id"]: model["weights_sha256"] for model in index["models"]}
    for path in (ROOT / "example_workflows").glob("*.json"):
        workflow = json.loads(path.read_text())
        loader = next(node for node in workflow["nodes"] if node["type"] == "PixTextModelLoader")
        model = loader["properties"]["models"][0]
        selected = loader["widgets_values"][0]
        assert model["name"] == selected
        assert selected.endswith(".safetensors")
        variant = selected.removesuffix(".safetensors")
        assert model["hash"] == expected[variant]
        assert model["directory"] == "diffusion_models"
        assert model["url"] == (
            "https://huggingface.co/ATH-MaaS/PixText/resolve/main/models/"
            + variant
            + "/model.safetensors"
        )
        assets = ROOT / "comfyui/model_assets" / variant
        manifest = json.loads((assets / "manifest.json").read_text())
        assert manifest["files"]["model.safetensors"] == expected[variant]
        for name, sha256 in manifest["files"].items():
            if name != "model.safetensors":
                assert hashlib.sha256((assets / name).read_bytes()).hexdigest() == sha256
        if "English" in path.name:
            note = next(
                node for node in workflow["nodes"] if node.get("title") == "OCR model downloads"
            )
            text = note["widgets_values"][0]
            for filename in ("craft_mlt_25k.zip", "inference.onnx", "inference.yml"):
                assert filename in text
            assert "ComfyUI/models/pixtext_ocr/PP-OCRv6_medium_rec_onnx/" in text
