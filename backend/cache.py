"""
Caching — recommended ON per spec's Open Decisions.

Key = (reference_image_hash, organism_type, style_mode, backend). With local
inference taking real minutes per job (not just an API round-trip), a cache
hit saves wall-clock time, not just money.
"""

import hashlib
import json
from pathlib import Path

CACHE_INDEX_PATH = Path(__file__).parent / "storage" / "cache_index.json"


def hash_image(image_path: Path) -> str:
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(image_hash: str, organism_type: str, style_mode: str, backend: str) -> str:
    return f"{image_hash}:{organism_type}:{style_mode}:{backend}"


def _load_index() -> dict:
    if CACHE_INDEX_PATH.exists():
        with open(CACHE_INDEX_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_index(index: dict) -> None:
    CACHE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)


def lookup(key: str) -> str | None:
    """Returns cached zip path if present and still on disk, else None."""
    index = _load_index()
    entry = index.get(key)
    if entry and Path(entry["zip_path"]).exists():
        return entry["zip_path"]
    return None


def store(key: str, zip_path: str) -> None:
    index = _load_index()
    index[key] = {"zip_path": zip_path}
    _save_index(index)
