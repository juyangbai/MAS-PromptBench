"""Environment knobs of the run protocol, read in one place.

Every value is read when :meth:`ProtocolSettings.from_env` is called, which
the protocol does at the moment it needs the value (so a job sees the
endpoints it configured). Method knobs live in each method's own settings
dataclass; the README lists all of them in one table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .config import REFLECTION_MODEL_ID

# Port convention (see models/): the reflection model is served on 8200.
DEFAULT_REFLECTION_BASE_URL = "http://localhost:8200/v1"


def _items(raw: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (raw or "").split(",") if item.strip())


@dataclass(frozen=True)
class ProtocolSettings:
    """Endpoints, models, credentials and context limits taken from the environment."""

    task_endpoints_env: tuple[str, ...]  # TASK_ENDPOINTS (comma-separated task-model endpoints)
    vllm_base_url: str | None  # VLLM_BASE_URL (single task endpoint fallback)
    model_id: str | None  # MODEL_ID (task model of the current job)
    task_model: str | None  # TASK_MODEL (task-model fallback)
    reflection_model: str  # REFLECTION_MODEL_ID (default: the protocol reflection model)
    reflection_base_url: str | None  # REFLECTION_MODEL_BASE_URL (else DEFAULT_REFLECTION_BASE_URL)
    api_key: str  # OPENAI_API_KEY (default "EMPTY")
    reflection_context_limit: int | None  # REFLECTION_CONTEXT_LIMIT (served reflection window, tokens)
    tokenizer_cache_dirs: tuple[str, ...]  # TOKENIZER_CACHE_DIRS (colon-separated extra tokenizer caches)

    @classmethod
    def from_env(cls) -> ProtocolSettings:
        """Current values of every protocol environment variable."""
        limit = os.environ.get("REFLECTION_CONTEXT_LIMIT")
        return cls(
            task_endpoints_env=_items(os.environ.get("TASK_ENDPOINTS")),
            vllm_base_url=os.environ.get("VLLM_BASE_URL"),
            model_id=os.environ.get("MODEL_ID"),
            task_model=os.environ.get("TASK_MODEL"),
            reflection_model=os.environ.get("REFLECTION_MODEL_ID", REFLECTION_MODEL_ID),
            reflection_base_url=os.environ.get("REFLECTION_MODEL_BASE_URL") or None,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            reflection_context_limit=int(limit) if limit and limit.strip().isdigit() else None,
            tokenizer_cache_dirs=tuple(d for d in os.environ.get("TOKENIZER_CACHE_DIRS", "").split(":") if d),
        )

    @property
    def task_endpoints(self) -> tuple[str, ...]:
        """Task-model endpoints: ``TASK_ENDPOINTS``, else ``VLLM_BASE_URL`` (comma-separated)."""
        return self.task_endpoints_env or _items(self.vllm_base_url)

    def reflection_endpoint(self) -> str:
        """``REFLECTION_MODEL_BASE_URL``, else :data:`DEFAULT_REFLECTION_BASE_URL`."""
        return self.reflection_base_url or DEFAULT_REFLECTION_BASE_URL


__all__ = ["DEFAULT_REFLECTION_BASE_URL", "ProtocolSettings"]
