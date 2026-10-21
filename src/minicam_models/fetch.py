"""Downloads pinned source files and refuses anything whose sha256 differs."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

from minicam_models.catalog import PinnedFile

_CHUNK_BYTES = 1024 * 1024
_TIMEOUT_SECONDS = 60


class ChecksumMismatchError(RuntimeError):
    """A downloaded or cached file does not match its pinned sha256."""


def sha256_of(path: Path) -> str:
    """Hex sha256 digest of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_pinned(pinned: PinnedFile, cache_dir: Path) -> Path:
    """Return a verified local copy of `pinned`, downloading it if needed.

    The download lands in a temporary file and is only moved into the cache
    after its digest matches, so a tampered or truncated file is never used.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / Path(pinned.url).name
    if destination.is_file() and sha256_of(destination) == pinned.sha256:
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    with (
        urllib.request.urlopen(pinned.url, timeout=_TIMEOUT_SECONDS) as response,  # noqa: S310 - https-only, enforced by the catalog
        partial.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, _CHUNK_BYTES)

    actual = sha256_of(partial)
    if actual != pinned.sha256:
        partial.unlink()
        raise ChecksumMismatchError(
            f"{pinned.url}: expected sha256 {pinned.sha256}, got {actual}"
        )
    partial.replace(destination)
    return destination
