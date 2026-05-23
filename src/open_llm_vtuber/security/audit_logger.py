"""Audit Logger — records all tool executions for security review.

Logs are stored in a SQLite database with:
- timestamp, tool_name, arguments, result_summary
- permission_level, was_approved, session_id
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger


class AuditLogger:
    """SQLite-backed audit log for tool executions."""

    def __init__(self, db_path: str = "memory_data/audit.db"):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Create the audit database and table."""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments TEXT,
                result_summary TEXT,
                permission_level TEXT,
                was_approved INTEGER DEFAULT 0,
                was_blocked INTEGER DEFAULT 0,
                session_id TEXT,
                duration_ms INTEGER
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_tool ON audit_log(tool_name)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(timestamp)"
        )
        self._conn.commit()
        logger.info(f"AuditLogger initialized: {self._db_path}")

    def log(
        self,
        tool_name: str,
        arguments: str = "",
        result_summary: str = "",
        permission_level: str = "",
        was_approved: bool = False,
        was_blocked: bool = False,
        session_id: str = "",
        duration_ms: int = 0,
    ) -> str:
        """Record a tool execution in the audit log."""
        if not self._conn:
            return ""

        entry_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        try:
            self._conn.execute(
                """INSERT INTO audit_log
                   (id, timestamp, tool_name, arguments, result_summary,
                    permission_level, was_approved, was_blocked, session_id, duration_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry_id,
                    timestamp,
                    tool_name,
                    arguments[:500],  # Truncate long arguments
                    result_summary[:500],
                    permission_level,
                    1 if was_approved else 0,
                    1 if was_blocked else 0,
                    session_id,
                    duration_ms,
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.error(f"AuditLogger write error: {e}")

        return entry_id

    def query(
        self,
        tool_name: Optional[str] = None,
        since: Optional[str] = None,
        blocked_only: bool = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query the audit log with filters."""
        if not self._conn:
            return []

        conditions = []
        params = []

        if tool_name:
            conditions.append("tool_name = ?")
            params.append(tool_name)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if blocked_only:
            conditions.append("was_blocked = 1")

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"SELECT * FROM audit_log{where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            cursor = self._conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"AuditLogger query error: {e}")
            return []

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
