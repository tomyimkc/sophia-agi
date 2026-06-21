"""Cache-first persistence for legal-citation resolutions.

The cache is the snapshot: a verified lookup is stored so later runs (and CI) are
deterministic and the free public services aren't hammered. Fail-closed by design
— a miss returns ``None``, so the registry treats "not cached + offline" as
UNVERIFIED rather than passing it through.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.config import MEMORY_DIR
from agent.legal_citations import normalize_citation
from agent.legal_sources.base import Resolution

DEFAULT_CACHE = MEMORY_DIR / "legal_cache.json"


class ResolutionCache:
    def __init__(self, path: "Path | None" = None) -> None:
        self.path = Path(path) if path else DEFAULT_CACHE
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, citation: str, *, max_age_days: "int | None" = None) -> "Resolution | None":
        key = normalize_citation(citation)
        entry = self._data.get(key)
        if not entry:
            return None
        if max_age_days is not None and not _fresh(entry.get("retrievedAt", ""), max_age_days):
            return None
        return Resolution(**{**_defaults(), **entry})

    def put(self, resolution: Resolution) -> None:
        self._data[normalize_citation(resolution.citation)] = resolution.to_dict()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._data)


def _defaults() -> dict:
    return {"citation": "", "verified": False, "status": "", "provider": ""}


def _fresh(retrieved_at: str, max_age_days: int) -> bool:
    try:
        ts = datetime.fromisoformat(retrieved_at)
    except (ValueError, TypeError):
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).days <= max_age_days
