"""User-triggered installation, concurrent clicks and failed-download recovery."""

import hashlib
import io
import threading
import time
import urllib.error
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from comfyui import ocr_models as setup


def finished(manager):
    deadline = time.monotonic() + 5
    while manager.snapshot()["busy"] and time.monotonic() < deadline:
        time.sleep(0.005)
    result = manager.snapshot()
    assert not result["busy"], result
    return result


@pytest.fixture
def models(monkeypatch):
    payloads = (b"craft weights", b"recognition weights", b"dictionary")
    files = tuple(
        setup.ModelFile(
            name,
            name,
            "https://example.test/" + str(i),
            hashlib.sha256(data).hexdigest(),
            len(data),
            "craft.pth" if i == 0 else None,
        )
        for i, (name, data) in enumerate(
            zip(("craft.pth", "rec/inference.onnx", "rec/inference.yml"), payloads)
        )
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("craft.pth", payloads[0])
        z.writestr("../../outside", b"must not be extracted")
    downloads = [archive.getvalue(), *payloads[1:]]
    calls = []

    def open_url(request, timeout):
        calls.append(request.full_url)
        data = downloads[int(request.full_url.rsplit("/", 1)[1])]
        response = io.BytesIO(data)
        response.headers = {"Content-Length": str(len(data))}
        return response

    monkeypatch.setattr(setup, "FILES", files)
    monkeypatch.setattr(setup.urllib.request, "urlopen", open_url)
    return files, downloads, calls, open_url


def test_status_is_local_and_installer_verifies_before_reusing_files(tmp_path, models, monkeypatch):
    files, _, calls, _ = models
    manager = setup.OCRModelPreparation(tmp_path)
    manager.status()
    assert finished(manager)["state"] == "missing"
    assert calls == []
    events = []
    update = manager._update

    def capture(*args, **kwargs):
        events.append(kwargs.get("state"))
        update(*args, **kwargs)

    monkeypatch.setattr(manager, "_update", capture)
    manager.prepare()
    ready = finished(manager)
    assert ready["state"] == "ready" and ready["progress"] == 100
    assert len(calls) == len(files)
    assert "downloading" in events and "verifying" in events
    assert all(setup.file_matches(tmp_path / f.name, f) for f in files)
    assert not (tmp_path.parent / "outside").exists()
    assert not list(tmp_path.rglob(".pixtext-ocr-*"))
    for _ in range(20):
        assert manager.prepare()["job_id"] == ready["job_id"]
    assert len(calls) == len(files)
    # A fresh server validates the files again but does not download them.
    another = setup.OCRModelPreparation(tmp_path)
    another.prepare()
    assert finished(another)["state"] == "ready"
    assert len(calls) == len(files)


def test_simultaneous_clicks_share_one_job(tmp_path, models, monkeypatch):
    files, _, calls, opener = models
    entered, resume = threading.Event(), threading.Event()

    def delayed(*args, **kwargs):
        entered.set()
        assert resume.wait(5)
        return opener(*args, **kwargs)

    monkeypatch.setattr(setup.urllib.request, "urlopen", delayed)
    manager = setup.OCRModelPreparation(tmp_path)
    manager.prepare()
    assert entered.wait(5)
    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            statuses = list(pool.map(lambda _: manager.prepare(), range(40)))
        assert {s["job_id"] for s in statuses} == {1}
        assert all(s["busy"] for s in statuses)
    finally:
        resume.set()
    assert finished(manager)["state"] == "ready"
    assert len(calls) == len(files)


def test_failed_checksum_preserves_existing_file_and_retry_downloads_only_invalid(tmp_path, models):
    files, downloads, calls, _ = models
    manager = setup.OCRModelPreparation(tmp_path)
    manager.prepare()
    assert finished(manager)["state"] == "ready"
    bad = tmp_path / files[1].name
    bad.write_bytes(b"x" * files[1].size)
    good_download = downloads[1]
    downloads[1] = b"y" * files[1].size
    calls.clear()
    manager.prepare()
    error = finished(manager)
    assert error["state"] == "error" and "Checksum mismatch" in error["message"]
    assert bad.read_bytes() == b"x" * files[1].size
    assert len(calls) == 1 and not list(tmp_path.rglob(".pixtext-ocr-*"))
    downloads[1] = good_download
    manager.prepare()
    assert finished(manager)["state"] == "ready"
    assert len(calls) == 2


def test_click_during_initial_check_upgrades_same_job(tmp_path, models, monkeypatch):
    entered, resume = threading.Event(), threading.Event()
    check = setup.file_matches

    def delayed(path, model):
        if not entered.is_set():
            entered.set()
            assert resume.wait(5)
        return check(path, model)

    monkeypatch.setattr(setup, "file_matches", delayed)
    manager = setup.OCRModelPreparation(tmp_path)
    manager.status()
    assert entered.wait(5)
    assert manager.prepare()["job_id"] == 1
    resume.set()
    assert finished(manager)["state"] == "ready"
    assert len(models[2]) == 3


def test_truncated_download_never_installs_a_partial_file(tmp_path, models, monkeypatch):
    _, _, _, opener = models

    def truncated(*args, **kwargs):
        response = opener(*args, **kwargs)
        response.headers["Content-Length"] = str(len(response.getvalue()) + 1)
        return response

    monkeypatch.setattr(setup.urllib.request, "urlopen", truncated)
    manager = setup.OCRModelPreparation(tmp_path)
    manager.prepare()
    result = finished(manager)
    assert result["state"] == "error" and "Incomplete download" in result["message"]
    assert not (tmp_path / models[0][0].name).exists()
    assert not list(tmp_path.rglob(".pixtext-ocr-*"))


def test_transient_connection_errors_retry_in_the_same_job(tmp_path, models, monkeypatch):
    opener = models[3]
    attempts = []

    def intermittent(request, **kwargs):
        attempts.append(request.full_url)
        if request.full_url.endswith("/1") and attempts.count(request.full_url) < 3:
            raise urllib.error.URLError("TLS connection interrupted")
        return opener(request, **kwargs)

    monkeypatch.setattr(setup, "RETRY_DELAYS", (0, 0))
    monkeypatch.setattr(setup.urllib.request, "urlopen", intermittent)
    manager = setup.OCRModelPreparation(tmp_path)
    manager.prepare()
    result = finished(manager)
    assert result["state"] == "ready" and result["job_id"] == 1
    assert attempts.count(models[0][0].url) == 1
    assert attempts.count(models[0][1].url) == 3
    assert not list(tmp_path.rglob(".pixtext-ocr-*"))


def test_connection_retries_are_bounded_and_report_a_readable_error(tmp_path, models, monkeypatch):
    attempts = []

    def unavailable(request, **kwargs):
        attempts.append(request.full_url)
        raise urllib.error.URLError("TLS connection interrupted")

    monkeypatch.setattr(setup, "RETRY_DELAYS", (0, 0))
    monkeypatch.setattr(setup.urllib.request, "urlopen", unavailable)
    manager = setup.OCRModelPreparation(tmp_path)
    manager.prepare()
    result = finished(manager)
    assert len(attempts) == 3
    assert result["state"] == "error" and "Check this server" in result["message"]
    assert "TLS" in result["error_details"]
