"""Local, optional CRAFT word detection and PP-OCRv6 CTC recognition.

Models are installed explicitly; inference never downloads files or calls an OCR service.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


def model_files(root: Path) -> tuple[Path, Path, Path]:
    paths = (
        root / "craft_mlt_25k.pth",
        root / "PP-OCRv6_medium_rec_onnx/inference.onnx",
        root / "PP-OCRv6_medium_rec_onnx/inference.yml",
    )
    missing = [str(p.name) for p in paths if not p.is_file()]
    if missing:
        raise ValueError(
            "Install the optional OCR models in models/pixtext_ocr first. Missing: "
            + ", ".join(missing)
        )
    return paths


def model_identity(root: Path):
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in model_files(root))


def decode_ctc(probabilities: np.ndarray, vocabulary: list[str], valid_steps: int):
    """Collapse repeats before removing blanks, preserving repeated letters across blanks."""
    sequence = probabilities[:valid_steps]
    ids, scores = sequence.argmax(-1), sequence.max(-1)
    text, confidence, previous = [], [], -1
    for index, score in zip(ids, scores, strict=True):
        index = int(index)
        if index and index != previous:
            if index >= len(vocabulary):
                raise ValueError("OCR dictionary does not match the recognition model")
            text.append(vocabulary[index])
            confidence.append(float(score))
        previous = index
    return "".join(text), float(np.mean(confidence)) if confidence else 0.0


def crop_quad(rgb: np.ndarray, quad: np.ndarray):
    import cv2

    width = max(2, round(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))))
    height = max(
        2, round(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
    )
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    return cv2.warpPerspective(rgb, matrix, (width, height))


class WordOCR:
    def __init__(self, identity):
        try:
            import onnxruntime as ort
            import torch
            import yaml
            from easyocr.craft import CRAFT
            from easyocr.detection import copyStateDict
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements-ocr.txt into ComfyUI's Python environment to use automatic OCR."
            ) from exc
        craft_path, rec_path, dictionary = [Path(row[0]) for row in identity]
        self.detector = CRAFT()
        self.detector.load_state_dict(
            copyStateDict(torch.load(craft_path, map_location="cpu", weights_only=True))
        )
        self.detector.eval()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.recognizer = ort.InferenceSession(
            str(rec_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        config = yaml.safe_load(dictionary.read_text(encoding="utf-8"))
        chars = config["PostProcess"]["character_dict"]
        self.vocabulary = (
            ["blank"] + chars + ([" "] if config["PostProcess"].get("use_space_char", True) else [])
        )
        self.input_name = self.recognizer.get_inputs()[0].name

    def recognize(self, crops):
        import cv2

        results = []
        # Small batches bound RAM for the large multilingual output vocabulary.
        for start in range(0, len(crops), 8):
            group = crops[start : start + 8]
            widths = [min(3200, max(1, math.ceil(48 * c.shape[1] / c.shape[0]))) for c in group]
            width = max(320, max(widths))
            batch = np.zeros((len(group), 3, 48, width), dtype=np.float32)
            for i, (crop, valid_width) in enumerate(zip(group, widths, strict=True)):
                # Official PP-OCR preprocessing uses BGR, not RGB.
                resized = cv2.resize(crop[:, :, ::-1], (valid_width, 48))
                batch[i, :, :, :valid_width] = (
                    (resized.astype(np.float32) / 255 - 0.5) / 0.5
                ).transpose(2, 0, 1)
            predictions = self.recognizer.run(None, {self.input_name: batch})[0]
            if predictions.shape[-1] != len(self.vocabulary):
                raise ValueError("OCR model output and dictionary have different vocabulary sizes")
            for pred, valid_width in zip(predictions, widths, strict=True):
                results.append(
                    decode_ctc(
                        pred,
                        self.vocabulary,
                        min(len(pred), math.ceil(len(pred) * valid_width / width)),
                    )
                )
        return results

    def detect(self, image: Image.Image, text_threshold: float, link_threshold: float):
        from easyocr.detection import test_net

        rgb = np.array(image.convert("RGB"))
        boxes, _ = test_net(
            1280, 1.5, self.detector, rgb, text_threshold, link_threshold, 0.4, False, "cpu"
        )
        quads = list(boxes[0])
        if len(quads) > 256:
            raise ValueError(
                "More than 256 words detected. Crop the image or raise the text threshold."
            )
        recognition = self.recognize([crop_quad(rgb, q) for q in quads])
        rows = []
        for quad, (text, score) in zip(quads, recognition, strict=True):
            if not text.strip():
                continue
            low = np.maximum(0, quad.min(axis=0))
            high = np.minimum([image.width, image.height], quad.max(axis=0))
            if np.any(high <= low):
                continue
            bbox = [round(float(x), 2) for x in [*low, *high]]
            key = json.dumps([bbox, text], ensure_ascii=False)
            rows.append(
                {
                    "id": "ocr-" + hashlib.sha256(key.encode()).hexdigest()[:16],
                    "bbox": bbox,
                    "text": text,
                    "recognized_text": text,
                    "confidence": round(score, 4),
                    "enabled": False,
                    "origin": "ocr",
                }
            )
        rows.sort(key=lambda row: (round(row["bbox"][1] / 12), row["bbox"][0]))
        return rows


@lru_cache(maxsize=1)
def load_ocr(identity):
    return WordOCR(identity)


def detect_words(image: Image.Image, root: Path, text_threshold=0.5, link_threshold=0.2):
    identity = model_identity(root)
    rows = load_ocr(identity).detect(image, text_threshold, link_threshold)
    source_id = "pixels:" + hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    document = {
        "version": 1,
        "source_id": source_id,
        "size": list(image.size),
        "regions": rows,
        "engine": "CRAFT + PP-OCRv6 medium (CPU)",
    }
    document["detection_id"] = hashlib.sha256(
        json.dumps(document, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return document
