"""Working Layer — current conversation context.

This is equivalent to the in-memory message list used by agents.
It's the shortest-lived memory layer — conversation turns only.
"""

from typing import Any, Dict, List, Optional
from loguru import logger


class WorkingLayer:
    """In-memory conversation context (same as agent's self._memory)."""

    def __init__(self):
        self._messages: List[Dict[str, Any]] = []

    def add(self, role: str, content: str, **metadata) -> None:
        """Add a message to working memory."""
        msg = {"role": role, "content": content, **metadata}
        self._messages.append(msg)

    def get_messages(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from working memory."""
        if last_n:
            return self._messages[-last_n:]
        return self._messages.copy()

    def clear(self) -> None:
        """Clear working memory."""
        self._messages.clear()

    def load_from_history(self, messages: List[Dict[str, Any]]) -> None:
        """Load from chat history (replaces current content)."""
        self._messages = messages.copy()

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def get_last_exchange(self) -> Optional[Dict[str, Any]]:
        """Get the last user-assistant exchange."""
        if len(self._messages) < 2:
            return None
        return {
            "user": self._messages[-2] if self._messages[-2].get("role") == "user" else None,
            "assistant": self._messages[-1] if self._messages[-1].get("role") == "assistant" else None,
        }
