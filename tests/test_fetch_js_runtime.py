"""Tests for scripts/fetch_js_runtime.py."""

import hashlib
import io

import scripts.fetch_js_runtime as fetch_script


def test_fetch_skips_when_existing_matches_sha256(tmp_path, monkeypatch):
    target = tmp_path / "qjs.exe"
    payload = b"quickjs-binary-bytes"
    target.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    called = []
    monkeypatch.setattr(
        fetch_script.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: called.append(True),
    )

    result = fetch_script.fetch_js_runtime(
        target, url="https://example.invalid/qjs.exe", expected_sha256=digest,
    )
    assert result == target
    assert not called


def test_fetch_downloads_atomically_and_verifies_sha256(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "qjs.exe"
    payload = b"quickjs-downloaded-bytes"
    digest = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(
        fetch_script.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )
    monkeypatch.setattr(fetch_script, "QUICKJS_SHA256", digest)

    rc = fetch_script.main(["--output", str(target)])
    assert rc == 0
    assert target.read_bytes() == payload
    assert list(target.parent.glob("*.part")) == []


def test_fetch_rejects_bad_sha256_and_cleans_part_file(tmp_path, monkeypatch):
    target = tmp_path / "qjs.exe"
    payload = b"tampered-bytes"

    monkeypatch.setattr(
        fetch_script.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )
    monkeypatch.setattr(fetch_script, "QUICKJS_SHA256", "0" * 64)

    rc = fetch_script.main(["--output", str(target)])
    assert rc != 0
    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []
