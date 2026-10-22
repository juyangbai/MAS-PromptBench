"""Reasoning removal: what a model wrote after its last ``</think>`` tag."""

from __future__ import annotations


def strip_thinking(text: str) -> str:
    """Cut everything up to the last ``</think>`` tag."""
    index = text.lower().rfind("</think>")
    if index >= 0:
        text = text[index + len("</think>") :]
    return text.strip()


def strip_ai_thinking(messages: list) -> None:
    """Strip reasoning from the text content of every AI message, in place."""
    for msg in messages:
        if msg.type == "ai" and isinstance(msg.content, str):
            msg.content = strip_thinking(msg.content)
