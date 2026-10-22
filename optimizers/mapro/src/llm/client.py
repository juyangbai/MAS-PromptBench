"""The chat surface the MAP core is written against: a generation config and the ``chat_text`` protocol.

``GenerationConfig`` carries a call site's native sampling (the protocol shims
apply the common reflection policy over it); Qwen thinking stays off unless
a config enables it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationConfig:
    """Native sampling of one call site."""

    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 1.0
    enable_thinking: bool = False
    stop: tuple[str, ...] | None = None


class ChatModel(Protocol):
    """What the MAP core calls: one single-turn chat completion (the protocol supplies request-logged shims)."""

    async def chat_text(self, prompt: str, system: str | None = None, cfg: GenerationConfig | None = None) -> str: ...
