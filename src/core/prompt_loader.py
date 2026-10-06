"""Load versioned prompt packs from ``src/core/prompts/*.json``.

Change generation behavior by editing or adding a pack file and setting
``PROMPT_PACK_ID`` (or a tenant's ``prompt_pack_id`` in Postgres). Bump the
``version`` field inside the JSON when you ship a new pack so logs and cache
scopes stay traceable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class PromptPack:
    pack_id: str
    version: str
    grounding_rules: str
    persona: dict[str, str]
    response_style: dict[str, str]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_prompt_pack(pack_id: str) -> PromptPack:
    """Load a prompt pack by id (filename without ``.json``)."""
    safe_id = pack_id.strip() or "default"
    path = _PROMPTS_DIR / f"{safe_id}.json"
    if not path.is_file():
        logger.warning("Prompt pack %r not found at %s — falling back to default.", safe_id, path)
        path = _PROMPTS_DIR / "default.json"
    data = _load_json(path)
    return PromptPack(
        pack_id=path.stem,
        version=str(data.get("version", "0.0.0")),
        grounding_rules=str(data["grounding_rules"]),
        persona={str(k): str(v) for k, v in data["persona"].items()},
        response_style={str(k): str(v) for k, v in data["response_style"].items()},
    )


@lru_cache(maxsize=16)
def get_prompt_pack(pack_id: str) -> PromptPack:
    """Cached loader — call ``get_prompt_pack.cache_clear()`` after hot-reload in tests."""
    pack = load_prompt_pack(pack_id)
    logger.debug("Loaded prompt pack id=%s version=%s", pack.pack_id, pack.version)
    return pack


def resolve_pack_id(*, tenant_pack_id: str | None, default_pack_id: str) -> str:
    """Tenant override wins when set."""
    if tenant_pack_id and tenant_pack_id.strip():
        return tenant_pack_id.strip()
    return default_pack_id.strip() or "default"
