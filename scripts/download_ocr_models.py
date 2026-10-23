"""Install the optional OCR models from pinned public releases, with SHA-256 checks."""

from __future__ import annotations

import argparse
import hashlib
import io
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REVISION = "50c7eacafc52fa7bcf4194e8cd08e46f8558504b"
BASE = f"https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/{REVISION}"
FILES = [
    (
        "craft_mlt_25k.pth",
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
        "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
    ),
    (
        "PP-OCRv6_medium_rec_onnx/inference.onnx",
        BASE + "/inference.onnx",
        "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba",
    ),
    (
        "PP-OCRv6_medium_rec_onnx/inference.yml",
        BASE + "/inference.yml",
        "991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="ComfyUI/models/pixtext_ocr")
    args = parser.parse_args()
    for name, url, expected in FILES:
        destination = args.output / name
        if (
            destination.is_file()
            and hashlib.sha256(destination.read_bytes()).hexdigest() == expected
        ):
            print("Verified", name)
            continue
        print("Downloading", name, flush=True)
        with urllib.request.urlopen(url, timeout=180) as response:
            data = response.read()
        if name.endswith(".pth"):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                data = archive.read("craft_mlt_25k.pth")
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Checksum mismatch: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
        print("Installed", name)


if __name__ == "__main__":
    main()
