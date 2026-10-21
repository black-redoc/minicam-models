"""Pinned downloads are verified before use."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from minicam_models.catalog import PinnedFile
from minicam_models.fetch import ChecksumMismatchError, fetch_pinned


def _source(tmp_path: Path, content: bytes) -> str:
    path = tmp_path / "src" / "weights.pt"
    path.parent.mkdir()
    path.write_bytes(content)
    return path.as_uri()


def test_downloads_and_caches_matching_file(tmp_path: Path) -> None:
    url = _source(tmp_path, b"weights")
    pinned = PinnedFile(url, hashlib.sha256(b"weights").hexdigest())
    first = fetch_pinned(pinned, tmp_path / "cache")
    assert first.read_bytes() == b"weights"
    assert fetch_pinned(pinned, tmp_path / "cache") == first


def test_rejects_mismatch_and_leaves_nothing_behind(tmp_path: Path) -> None:
    url = _source(tmp_path, b"tampered")
    pinned = PinnedFile(url, hashlib.sha256(b"weights").hexdigest())
    with pytest.raises(ChecksumMismatchError):
        fetch_pinned(pinned, tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []


def test_redownloads_corrupted_cache(tmp_path: Path) -> None:
    url = _source(tmp_path, b"weights")
    pinned = PinnedFile(url, hashlib.sha256(b"weights").hexdigest())
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "weights.pt").write_bytes(b"corrupt")
    assert fetch_pinned(pinned, tmp_path / "cache").read_bytes() == b"weights"
