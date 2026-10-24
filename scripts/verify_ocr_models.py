"""Check the locally installed optional OCR models against release checksums."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

FILES = {
    "craft_mlt_25k.pth": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
    "PP-OCRv6_medium_rec_onnx/inference.onnx": "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba",
    "PP-OCRv6_medium_rec_onnx/inference.yml": "991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129",
}


def verify(root: Path) -> bool:
    valid = True
    for name, expected in FILES.items():
        path = root / name
        if not path.is_file():
            print("Missing:", name)
            valid = False
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            print("Checksum mismatch:", name)
            valid = False
        else:
            print("Verified:", name)
    return valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path, help="ComfyUI/models/pixtext_ocr")
    args = parser.parse_args()
    return 0 if verify(args.models) else 1


if __name__ == "__main__":
    raise SystemExit(main())
