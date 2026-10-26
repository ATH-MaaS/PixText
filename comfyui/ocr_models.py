"""Explicit, checksum-verified OCR setup shared by the UI and command line.

Only a user-requested preparation job downloads files. Status checks and inference
are local. Every node and browser tab shares one job per ComfyUI process.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REVISION = "50c7eacafc52fa7bcf4194e8cd08e46f8558504b"
BASE = f"https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/{REVISION}"
CHUNK = 1024 * 1024
RETRY_DELAYS = (1, 2)


@dataclass(frozen=True)
class ModelFile:
    name: str
    label: str
    url: str
    sha256: str
    size: int
    archive_member: str | None = None


FILES = (
    ModelFile(
        "craft_mlt_25k.pth",
        "CRAFT · word detection",
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
        "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
        83152330,
        "craft_mlt_25k.pth",
    ),
    ModelFile(
        "PP-OCRv6_medium_rec_onnx/inference.onnx",
        "PP-OCRv6 · recognition",
        BASE + "/inference.onnx",
        "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba",
        76554979,
    ),
    ModelFile(
        "PP-OCRv6_medium_rec_onnx/inference.yml",
        "Character dictionary",
        BASE + "/inference.yml",
        "991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129",
        150580,
    ),
)


def file_matches(path: Path, model: ModelFile) -> bool:
    if not path.is_file() or path.stat().st_size != model.size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest() == model.sha256


def _install(root: Path, model: ModelFile, progress) -> None:
    destination = root / model.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Temporary files stay on the destination filesystem for atomic replacement.
    with tempfile.TemporaryDirectory(prefix=".pixtext-ocr-", dir=destination.parent) as temporary:
        source = Path(temporary) / "download"
        received = 0
        request = urllib.request.Request(model.url, headers={"User-Agent": "PixText-OCR-Setup"})
        with urllib.request.urlopen(request, timeout=60) as response, source.open("wb") as stream:
            total = int(response.headers.get("Content-Length", 0))
            # Bound downloads even when the server omits Content-Length.
            limit = model.size * 2 + CHUNK
            if total > limit:
                raise ValueError(f"Unexpected download size: {model.name}")
            while chunk := response.read(CHUNK):
                received += len(chunk)
                if received > limit:
                    raise ValueError(f"Unexpected download size: {model.name}")
                stream.write(chunk)
                progress(received, total)
        if total and received != total:
            raise ValueError(f"Incomplete download: {model.name}")
        candidate = source
        if model.archive_member:
            candidate = Path(temporary) / "model"
            with zipfile.ZipFile(source) as archive:
                info = archive.getinfo(model.archive_member)
                if info.file_size != model.size:
                    raise ValueError(f"Unexpected model size: {model.name}")
                # Read only the named member, never extract arbitrary archive paths.
                with archive.open(info) as incoming, candidate.open("wb") as outgoing:
                    for chunk in iter(lambda: incoming.read(CHUNK), b""):
                        outgoing.write(chunk)
        progress(received, total, verifying=True)
        if not file_matches(candidate, model):
            raise ValueError(f"Checksum mismatch: {model.name}. Retry to download a fresh copy.")
        candidate.replace(destination)


class OCRModelPreparation:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._active = False
        self._download_requested = False
        self._signature = None
        self._state = {
            "state": "idle",
            "job_id": 0,
            "progress": 0,
            "message": "Check OCR models",
            "received": 0,
            "total": 0,
            "error_details": "",
            "files": [{"name": f.name, "label": f.label, "state": "pending"} for f in FILES],
        }

    def _fingerprint(self):
        result = []
        for model in FILES:
            try:
                stat = (self.root / model.name).stat()
                result.append((stat.st_size, stat.st_mtime_ns))
            except OSError as exc:
                result.append(("unavailable", exc.errno))
        return tuple(result)

    def snapshot(self):
        with self._lock:
            result = copy.deepcopy(self._state)
            result["busy"] = self._active
        result["directory"] = str(self.root)
        result["model_bytes"] = sum(f.size for f in FILES)
        result["missing_dependencies"] = [
            name
            for name in ("easyocr", "onnxruntime", "yaml")
            if importlib.util.find_spec(name) is None
        ]
        return result

    def status(self):
        self._start(download=False)
        return self.snapshot()

    def prepare(self):
        self._start(download=True)
        return self.snapshot()

    def _start(self, *, download):
        with self._lock:
            if self._active:
                self._download_requested |= download
                return
            unchanged = self._signature == self._fingerprint()
            if unchanged and (not download or self._state["state"] == "ready"):
                return
            self._active = True
            self._download_requested = download
            self._state.update(
                state="checking",
                job_id=self._state["job_id"] + 1,
                progress=0,
                message="Checking local files…",
                received=0,
                total=0,
                error_details="",
            )
            for row in self._state["files"]:
                row["state"] = "pending"
            threading.Thread(target=self._run, args=(download,), daemon=True).start()

    def _update(self, index=None, file_state=None, **state):
        with self._lock:
            self._state.update(state)
            if index is not None:
                self._state["files"][index]["state"] = file_state

    def _prepare_files(self, download):
        missing = False
        for index, model in enumerate(FILES):
            self._update(index, "checking", state="checking", message=f"Checking {model.label}")
            if file_matches(self.root / model.name, model):
                self._update(index, "verified", progress=round((index + 1) / len(FILES) * 100))
                continue
            if not download:
                missing = True
                self._update(index, "missing")
                continue

            def progress(received, total, *, verifying=False):
                fraction = min(received / (total or model.size), 1) * 0.9
                self._update(
                    index,
                    "verifying" if verifying else "downloading",
                    state="verifying" if verifying else "downloading",
                    progress=round((index + fraction) / len(FILES) * 100),
                    message=f"{'Verifying' if verifying else 'Downloading'} {model.label}",
                    received=received,
                    total=total,
                )

            self._update(
                index,
                "downloading",
                state="downloading",
                message=f"Downloading {model.label}",
                received=0,
                total=0,
            )
            for attempt in range(len(RETRY_DELAYS) + 1):
                try:
                    _install(self.root, model, progress)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if attempt == len(RETRY_DELAYS):
                        raise
                    self._update(
                        index,
                        "downloading",
                        state="retrying",
                        message=f"Connection interrupted · retrying {attempt + 2}/{len(RETRY_DELAYS) + 1}",
                        received=0,
                        total=0,
                    )
                    time.sleep(RETRY_DELAYS[attempt])
            self._update(index, "verified", progress=round((index + 1) / len(FILES) * 100))
        return missing

    def _run(self, download):
        try:
            while True:
                missing = self._prepare_files(download)
                with self._lock:
                    # Upgrade a status-only check when a click arrives mid-check.
                    # This and the terminal transition share the start lock.
                    if missing and not download and self._download_requested:
                        download = True
                        continue
                    self._state.update(
                        state="missing" if missing else "ready",
                        progress=0 if missing else 100,
                        message="Prepare models once to detect words"
                        if missing
                        else "All 3 files verified · ready to detect",
                        received=0,
                        total=0,
                    )
                    self._signature = self._fingerprint()
                    self._active = False
                    return
        except Exception as exc:
            with self._lock:
                index = next(
                    (
                        i
                        for i, row in enumerate(self._state["files"])
                        if row["state"] in {"checking", "downloading", "verifying"}
                    ),
                    0,
                )
                self._state["files"][index]["state"] = "error"
                message = str(exc)
                if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
                    message = "Download connection failed. Check this server’s network, then retry."
                self._state.update(
                    state="error", message=message, error_details=str(exc), received=0, total=0
                )
                self._signature = self._fingerprint()
                self._active = False
