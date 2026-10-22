import json

import pytest

from pixtext import PixTextPipeline


def test_model_package_reports_missing_bundled_font_before_loading_weights(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"format_version": 1, "variant": "teacher"}))
    (tmp_path / "inference_config.json").write_text("{}")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "files": {"fonts/primary.ttf": "0" * 64},
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="fonts/primary.ttf"):
        PixTextPipeline.from_pretrained(tmp_path, device="cpu")


def test_model_package_rejects_files_outside_package_directory(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "files": {"../private-file": "0" * 64},
            }
        )
    )
    with pytest.raises(ValueError, match="escapes"):
        PixTextPipeline.from_pretrained(tmp_path, device="cpu")


def test_model_package_rejects_tampered_configuration(tmp_path):
    (tmp_path / "config.json").write_text('{"variant": "teacher"}')
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "files": {"config.json": "0" * 64},
            }
        )
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch: config.json"):
        PixTextPipeline.from_pretrained(tmp_path, device="cpu")


def test_model_package_requires_verified_weights_and_fonts(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"format_version": 1, "files": {}}))
    with pytest.raises(ValueError, match="manifest must verify"):
        PixTextPipeline.from_pretrained(tmp_path, device="cpu")


def test_meanflow_package_cannot_silently_load_teacher_architecture(tmp_path):
    import hashlib

    from pixtext.loading import REQUIRED_FILES

    config = {"format_version": 1, "variant": "meanflow", "model": {"interval_conditioning": False}}
    for name in REQUIRED_FILES:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"placeholder")
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "inference_config.json").write_text("{}")
    hashes = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in REQUIRED_FILES
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"format_version": 1, "model_id": "invalid", "files": hashes})
    )
    with pytest.raises(ValueError, match="interval_conditioning"):
        PixTextPipeline.from_pretrained(tmp_path, device="cpu")


def test_model_package_requires_verified_font_license_notice(tmp_path):
    import hashlib

    names = [
        "config.json",
        "inference_config.json",
        "model.safetensors",
        "fonts/AlibabaPuHuiTi-3-55-Regular.ttf",
        "fonts/GoNotoCJKCore-v7.0.ttf",
        "fonts/README.md",
    ]
    for name in names:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"placeholder")
    hashes = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in names}
    (tmp_path / "manifest.json").write_text(
        json.dumps({"format_version": 1, "model_id": "invalid", "files": hashes})
    )
    with pytest.raises(ValueError, match="fonts/OFL-GoNoto.txt"):
        PixTextPipeline.from_pretrained(tmp_path, device="cpu")
