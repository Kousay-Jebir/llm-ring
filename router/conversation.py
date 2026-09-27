"""Helpers to identify conversations and estimate their size."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from .hashing import stable_hash

_VALID_ID = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")

# Rough rule of thumb for English text with BPE tokenizers: ~4 characters per token.
# The router only needs relative sizes (which requests are heavier), not exact counts.
CHARS_PER_TOKEN = 4
TOKENS_PER_MESSAGE_OVERHEAD = 4


def message_text(message: dict[str, Any]) -> str:
    """Return the text of a chat message, supporting both string and list content."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(p for p in parts if isinstance(p, str))
    return ""


def estimate_tokens(messages: Sequence[dict[str, Any]]) -> int:
    chars = sum(len(message_text(m)) for m in messages)
    return max(1, chars // CHARS_PER_TOKEN + TOKENS_PER_MESSAGE_OVERHEAD * len(messages))


def conversation_id(header_value: str | None, messages: Sequence[dict[str, Any]]) -> str:
    """Return the conversation's routing key.

    Prefer the client-provided ``X-Conversation-ID``. Without it, derive a key from
    the start of the conversation (first system message + first user message), which
    stays the same on every turn. Two different users who start with exactly the same
    messages would share a key; that is an accepted limitation of the fallback.
    """
    if header_value:
        value = header_value.strip()
        if not _VALID_ID.match(value):
            raise ValueError("X-Conversation-ID must be 1-128 chars of [A-Za-z0-9._:-]")
        return value
    first_system = next((message_text(m) for m in messages if m.get("role") == "system"), "")
    first_user = next((message_text(m) for m in messages if m.get("role") == "user"), "")
    fingerprint = json.dumps([first_system, first_user], ensure_ascii=False)
    return f"auto-{stable_hash(fingerprint):016x}"
