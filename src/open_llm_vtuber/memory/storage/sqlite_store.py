"""SQLite-backed persistent storage for episodic and semantic memory layers.

Uses aiosqlite for async operations. Each memory layer gets its own table.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional
from loguru import logger

from .base_store import BaseStore


class SQLiteStore(BaseStore):
    """SQLite storage backend for structured memory records.

    Schema per table:
    - id TEXT PRIMARY KEY
    - data TEXT (JSON-encoded record)
    - created_at TEXT (ISO timestamp)
    - updated_at TEXT (ISO timestamp)
    - tags TEXT (comma-separated for simple search)
    - importance REAL (0.0-1.0, for priority retrieval)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    async def initialize(self) -> None:
        """Create the database directory and tables."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        logger.info(f"SQLiteStore initialized: {self.db_path}")

    def _ensure_table(self, table: str):
        """Create table if it doesn't exist."""
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                tags TEXT DEFAULT '',
                importance REAL DEFAULT 0.5
            )
        """)
        # Create FTS index for text search
        try:
            self._conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {table}_fts USING fts5(
                    id, data, tags,
                    content={table}, content_rowid=rowid
                )
            """)
        except sqlite3.OperationalError:
            pass  # FTS table already exists
        self._conn.commit()

    async def store(self, table: str, data: Dict[str, Any]) -> str:
        self._ensure_table(table)
        record_id = data.get("id", str(hash(json.dumps(data, sort_keys=True, default=str))))
        data_json = json.dumps(data, default=str, ensure_ascii=False)
        tags = data.get("tags", "")
        if isinstance(tags, list):
            tags = ",".join(str(t) for t in tags)
        importance = data.get("importance", 0.5)

        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} (id, data, tags, importance) VALUES (?, ?, ?, ?)",
            (record_id, data_json, str(tags), importance),
        )
        try:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table}_fts (id, data, tags) VALUES (?, ?, ?)",
                (record_id, data_json, str(tags)),
            )
        except sqlite3.OperationalError:
            pass
        self._conn.commit()
        return record_id

    async def retrieve(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_table(table)
        cursor = self._conn.execute(f"SELECT data FROM {table} WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    async def search(self, table: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self._ensure_table(table)
        # Try FTS search first
        try:
            cursor = self._conn.execute(f"""
                SELECT {table}.data, {table}.importance FROM {table}
                JOIN {table}_fts ON {table}.id = {table}_fts.id
                WHERE {table}_fts MATCH ?
                ORDER BY {table}.importance DESC
                LIMIT ?
            """, (query, top_k))
            results = []
            for row in cursor.fetchall():
                record = json.loads(row[0])
                record["_importance"] = row[1]
                results.append(record)
            if results:
                return results
        except sqlite3.OperationalError:
            pass

        # Fallback: LIKE search
        cursor = self._conn.execute(f"""
            SELECT data FROM {table}
            WHERE data LIKE ? OR tags LIKE ?
            ORDER BY importance DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", top_k))
        return [json.loads(row[0]) for row in cursor.fetchall()]

    async def update(self, table: str, record_id: str, data: Dict[str, Any]) -> bool:
        self._ensure_table(table)
        data_json = json.dumps(data, default=str, ensure_ascii=False)
        tags = data.get("tags", "")
        if isinstance(tags, list):
            tags = ",".join(str(t) for t in tags)
        importance = data.get("importance", 0.5)

        cursor = self._conn.execute(
            f"UPDATE {table} SET data=?, tags=?, importance=?, updated_at=datetime('now') WHERE id=?",
            (data_json, str(tags), importance, record_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    async def delete(self, table: str, record_id: str) -> bool:
        self._ensure_table(table)
        cursor = self._conn.execute(f"DELETE FROM {table} WHERE id = ?", (record_id,))
        try:
            self._conn.execute(f"DELETE FROM {table}_fts WHERE id = ?", (record_id,))
        except sqlite3.OperationalError:
            pass
        self._conn.commit()
        return cursor.rowcount > 0

    async def list_all(self, table: str, limit: int = 100) -> List[Dict[str, Any]]:
        self._ensure_table(table)
        cursor = self._conn.execute(
            f"SELECT data FROM {table} ORDER BY importance DESC, created_at DESC LIMIT ?",
            (limit,),
        )
        return [json.loads(row[0]) for row in cursor.fetchall()]

    async def count(self, table: str) -> int:
        self._ensure_table(table)
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
