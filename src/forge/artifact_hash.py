"""Content hashing for model artifacts. Every variant in models/MANIFEST.json
gets a hash so a silently-different checkpoint (Ollama's convenience download,
a re-run that picked up a stray file) is detectable, not assumed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory(path: Path) -> str:
    """Order-independent hash over every file's relative path + content hash,
    so it's stable regardless of filesystem iteration order but still changes
    if any file is added, removed, renamed, or its content changes."""
    entries = []
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file.relative_to(path).as_posix()
        entries.append(f"{rel}:{hash_file(file)}")
    combined = hashlib.sha256("\n".join(entries).encode("utf-8"))
    return combined.hexdigest()


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
