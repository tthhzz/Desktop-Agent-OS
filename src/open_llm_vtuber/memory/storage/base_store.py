"""Base storage interface for memory persistence."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseStore(ABC):
    """Abstract base class for memory storage backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage (create tables, connect, etc.)."""
        pass

    @abstractmethod
    async def store(self, table: str, data: Dict[str, Any]) -> str:
        """Store a record and return its ID."""
        pass

    @abstractmethod
    async def retrieve(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a record by ID."""
        pass

    @abstractmethod
    async def search(self, table: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for records matching a query."""
        pass

    @abstractmethod
    async def update(self, table: str, record_id: str, data: Dict[str, Any]) -> bool:
        """Update a record. Returns True if successful."""
        pass

    @abstractmethod
    async def delete(self, table: str, record_id: str) -> bool:
        """Delete a record. Returns True if successful."""
        pass

    @abstractmethod
    async def list_all(self, table: str, limit: int = 100) -> List[Dict[str, Any]]:
        """List all records in a table."""
        pass

    @abstractmethod
    async def count(self, table: str) -> int:
        """Count records in a table."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage connection."""
        pass
