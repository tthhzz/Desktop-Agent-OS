"""Semantic Layer — long-term knowledge extracted from episodes.

Stores facts, preferences, and concepts:
- "User likes cats"
- "User is a programmer"
- "User prefers Chinese language"

Persisted to SQLite with BM25 + vector search.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from ..storage.sqlite_store import SQLiteStore


class SemanticLayer:
    """Persistent semantic memory with hybrid retrieval."""

    TABLE = "semantics"

    def __init__(self, store: SQLiteStore):
        self._store = store

    async def store_knowledge(
        self,
        content: str,
        source: str = "reflection",
        categories: Optional[List[str]] = None,
        importance: float = 0.7,
        confidence: float = 1.0,
    ) -> str:
        """Store a piece of knowledge."""
        knowledge = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "content": content,
            "source": source,
            "categories": categories or [],
            "importance": importance,
            "confidence": confidence,
            "access_count": 0,
        }
        record_id = await self._store.store(self.TABLE, knowledge)
        logger.debug(f"Semantic: stored knowledge {record_id[:8]}... ({content[:50]})")
        return record_id

    async def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for knowledge matching a query."""
        return await self._store.search(self.TABLE, query, top_k)

    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific knowledge."""
        return await self._store.retrieve(self.TABLE, knowledge_id)

    async def update_knowledge(self, knowledge_id: str, data: Dict[str, Any]) -> bool:
        """Update existing knowledge."""
        return await self._store.update(self.TABLE, knowledge_id, data)

    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """Delete knowledge."""
        return await self._store.delete(self.TABLE, knowledge_id)

    async def list_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all knowledge entries."""
        return await self._store.list_all(self.TABLE, limit)

    async def count(self) -> int:
        """Count total knowledge entries."""
        return await self._store.count(self.TABLE)
