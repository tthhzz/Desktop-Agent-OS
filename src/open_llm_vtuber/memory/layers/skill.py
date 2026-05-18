"""Skill Layer — reusable tool-chain patterns auto-generated from usage.

A Skill captures a recurring tool invocation pattern:
- name: "daily_weather_check"
- description: "Check weather and provide a morning briefing"
- tools: ["ddg-search__search", "time__get_current_time"]
- trigger: "when user asks about weather in the morning"
- frequency: how often this pattern has been used
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from ..storage.sqlite_store import SQLiteStore


class SkillLayer:
    """Persistent skill memory for reusable tool-chain patterns."""

    TABLE = "skills"

    def __init__(self, store: SQLiteStore):
        self._store = store

    async def register_skill(
        self,
        name: str,
        description: str,
        tools: List[str],
        trigger_condition: str,
        example_input: str = "",
        example_output: str = "",
        frequency: int = 1,
    ) -> str:
        """Register a new skill."""
        skill = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "tools": tools,
            "trigger_condition": trigger_condition,
            "example_input": example_input,
            "example_output": example_output,
            "frequency": frequency,
            "created_at": datetime.utcnow().isoformat(),
            "last_used_at": None,
        }
        record_id = await self._store.store(self.TABLE, skill)
        logger.info(f"Skill: registered '{name}' ({len(tools)} tools)")
        return record_id

    async def find_skill(self, task_description: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find skills that match a task description."""
        return await self._store.search(self.TABLE, task_description, top_k)

    async def get_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific skill."""
        return await self._store.retrieve(self.TABLE, skill_id)

    async def increment_usage(self, skill_id: str) -> bool:
        """Increment the usage count of a skill."""
        skill = await self.get_skill(skill_id)
        if not skill:
            return False
        skill["frequency"] = skill.get("frequency", 0) + 1
        skill["last_used_at"] = datetime.utcnow().isoformat()
        return await self._store.update(self.TABLE, skill_id, skill)

    async def list_skills(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all skills, sorted by frequency."""
        return await self._store.list_all(self.TABLE, limit)

    async def count(self) -> int:
        """Count total skills."""
        return await self._store.count(self.TABLE)

    async def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill."""
        return await self._store.delete(self.TABLE, skill_id)
