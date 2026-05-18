"""Episodic Layer — conversation events with emotional annotations.

Stores discrete conversational episodes:
- timestamp, user_input, ai_response
- emotion (happy, frustrated, curious, etc.)
- topics/tags
- importance score

Persisted to SQLite for cross-session durability.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from ..storage.sqlite_store import SQLiteStore


class EpisodicLayer:
    """Persistent episodic memory with emotional annotations."""

    TABLE = "episodes"

    def __init__(self, store: SQLiteStore):
        self._store = store

    async def store_episode(
        self,
        user_input: str,
        ai_response: str,
        emotion: str = "neutral",
        topics: Optional[List[str]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a conversation episode."""
        episode = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response,
            "emotion": emotion,
            "topics": topics or [],
            "importance": importance,
            "metadata": metadata or {},
        }
        record_id = await self._store.store(self.TABLE, episode)
        logger.debug(f"Episodic: stored episode {record_id[:8]}... (emotion={emotion})")
        return record_id

    async def search_episodes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for episodes matching a query."""
        return await self._store.search(self.TABLE, query, top_k)

    async def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent episodes."""
        return await self._store.list_all(self.TABLE, limit)

    async def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific episode."""
        return await self._store.retrieve(self.TABLE, episode_id)

    async def count(self) -> int:
        """Count total episodes."""
        return await self._store.count(self.TABLE)
