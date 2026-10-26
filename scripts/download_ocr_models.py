"""Prepare optional OCR models using the same verified installer as the ComfyUI button."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfyui.ocr_models import OCRModelPreparation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="ComfyUI/models/pixtext_ocr")
    args = parser.parse_args()
    installer = OCRModelPreparation(args.output)
    state = installer.prepare()
    previous = None
    while True:
        message = (state["message"], state["progress"])
        if message != previous:
            print(f"{message[1]:3d}% {message[0]}", flush=True)
            previous = message
        if not state["busy"]:
            return 0 if state["state"] == "ready" else 1
        time.sleep(0.2)
        state = installer.snapshot()


if __name__ == "__main__":
    raise SystemExit(main())
