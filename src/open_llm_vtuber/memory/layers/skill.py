"""Skill Layer — reusable tool-chain patterns auto-generated from usage.

A Skill captures a recurring tool invocation pattern:
- name: "daily_weather_check"
- description: "Check weather and provide a morning briefing"
- tools: ["ddg-search__search", "time__get_current_time"]
- trigger: "when user asks about weather in the morning"
- frequency: how often this pattern has been used

Phase 5.3 enhancements:
- Parameterized templates with typed slots
- Skill validation (dry-run check)
- Self-evolution via Designer (merge, prune, optimize)
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
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
        # Phase 5.3: parameterized template fields
        params: Optional[List[Dict[str, Any]]] = None,
        template: Optional[str] = None,
        version: int = 1,
        validated: bool = False,
    ) -> str:
        """Register a new skill (with optional parameterized template).

        Args:
            params: Parameter definitions, e.g. [{"name": "topic", "type": "string", "required": True}]
            template: Prompt template with slots, e.g. "搜索{topic}相关信息"
            version: Skill version (incremented on evolution).
            validated: Whether this skill has been validated.
        """
        skill = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "tools": tools,
            "trigger_condition": trigger_condition,
            "example_input": example_input,
            "example_output": example_output,
            "frequency": frequency,
            "params": params or [],
            "template": template or description,
            "version": version,
            "validated": validated,
            "created_at": datetime.utcnow().isoformat(),
            "last_used_at": None,
        }
        record_id = await self._store.store(self.TABLE, skill)
        logger.info(f"Skill: registered '{name}' v{version} ({len(tools)} tools, {len(params or [])} params)")
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

    # ── Phase 5.3: Parameterized Skills ──────────────────────

    async def validate_skill(self, skill_id: str) -> Tuple[bool, str]:
        """Validate a skill by checking its structure and parameter definitions.

        Checks:
        1. Required fields are present (name, description, tools)
        2. Params are well-formed (name, type, required flag)
        3. Template placeholders match param names
        4. Tools list is not empty
        5. No duplicate param names

        Returns:
            (is_valid, reason) — reason explains what's wrong if invalid.
        """
        skill = await self.get_skill(skill_id)
        if not skill:
            return False, f"Skill {skill_id} not found"

        # Check required fields
        if not skill.get("name"):
            return False, "Skill missing 'name'"
        if not skill.get("description"):
            return False, "Skill missing 'description'"
        if not skill.get("tools"):
            return False, "Skill has empty tools list"

        # Check params
        params = skill.get("params", [])
        param_names = set()
        for p in params:
            if not isinstance(p, dict):
                return False, f"Invalid param definition: {p}"
            if not p.get("name"):
                return False, "Param missing 'name'"
            if p["name"] in param_names:
                return False, f"Duplicate param name: {p['name']}"
            param_names.add(p["name"])

            # Validate type
            valid_types = {"string", "int", "float", "bool", "list", "dict"}
            if p.get("type", "string") not in valid_types:
                return False, f"Invalid param type: {p.get('type')}"

        # Check template placeholders match params
        template = skill.get("template", "")
        if template and param_names:
            import re
            placeholders = set(re.findall(r"\{(\w+)\}", template))
            # Template can have extra placeholders (from context), but all params must be used
            missing_in_template = param_names - placeholders
            if missing_in_template:
                return False, f"Params not used in template: {missing_in_template}"

        # Mark as validated
        skill["validated"] = True
        await self._store.update(self.TABLE, skill_id, skill)

        logger.info(f"Skill: validated '{skill['name']}'")
        return True, "Valid"

    async def instantiate_skill(
        self, skill_id: str, param_values: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Instantiate a parameterized skill with concrete values.

        Returns the skill with its template filled in and tools configured,
        or None if validation fails.
        """
        skill = await self.get_skill(skill_id)
        if not skill:
            return None

        # Fill template
        template = skill.get("template", skill.get("description", ""))
        try:
            filled = template.format(**param_values)
        except KeyError as e:
            logger.error(f"Skill instantiate: missing param {e}")
            return None

        return {
            "name": skill["name"],
            "description": filled,
            "tools": skill["tools"],
            "params": param_values,
            "trigger_condition": skill.get("trigger_condition", ""),
        }

    async def find_similar_skills(self, skill_id: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Find skills that are similar to the given one (for merging).

        Uses tool overlap as a simple similarity metric.
        """
        skill = await self.get_skill(skill_id)
        if not skill:
            return []

        target_tools = set(skill.get("tools", []))
        if not target_tools:
            return []

        all_skills = await self.list_skills(limit=100)
        similar = []

        for other in all_skills:
            if other.get("id") == skill_id:
                continue
            other_tools = set(other.get("tools", []))
            if not other_tools:
                continue

            # Jaccard similarity
            intersection = len(target_tools & other_tools)
            union = len(target_tools | other_tools)
            similarity = intersection / union if union > 0 else 0

            if similarity >= threshold:
                similar.append({**other, "_similarity": similarity})

        similar.sort(key=lambda s: s["_similarity"], reverse=True)
        return similar

    async def merge_skills(self, skill_id_1: str, skill_id_2: str) -> Optional[str]:
        """Merge two similar skills into one, keeping the more general version."""
        skill1 = await self.get_skill(skill_id_1)
        skill2 = await self.get_skill(skill_id_2)

        if not skill1 or not skill2:
            return None

        # Keep the higher-frequency skill as base
        if skill1.get("frequency", 0) >= skill2.get("frequency", 0):
            base, other = skill1, skill2
        else:
            base, other = skill2, skill1

        # Merge tools (union)
        merged_tools = list(dict.fromkeys(base.get("tools", []) + other.get("tools", [])))

        # Merge params (union, preferring base)
        base_params = {p["name"]: p for p in base.get("params", [])}
        for p in other.get("params", []):
            if p["name"] not in base_params:
                base_params[p["name"]] = p

        merged_skill = {
            **base,
            "tools": merged_tools,
            "params": list(base_params.values()),
            "description": f"{base['description']} (merged with {other['name']})",
            "frequency": base.get("frequency", 0) + other.get("frequency", 0),
            "version": base.get("version", 1) + 1,
            "validated": False,
        }

        # Delete the other skill
        await self.delete_skill(skill_id_2)

        # Update the base skill
        await self._store.update(self.TABLE, skill_id_1, merged_skill)
        logger.info(f"Skill: merged '{other['name']}' into '{base['name']}'")

        return skill_id_1

    async def prune_low_usage(self, min_frequency: int = 2, max_age_days: int = 30) -> int:
        """Delete skills that haven't been used enough and are old.

        Returns the number of skills pruned.
        """
        skills = await self.list_skills(limit=200)
        pruned = 0

        for skill in skills:
            freq = skill.get("frequency", 0)
            if freq >= min_frequency:
                continue

            # Check age
            created = skill.get("created_at", "")
            if created:
                try:
                    created_dt = datetime.fromisoformat(created)
                    age_days = (datetime.utcnow() - created_dt).days
                    if age_days < max_age_days:
                        continue
                except (ValueError, TypeError):
                    pass

            await self.delete_skill(skill["id"])
            pruned += 1

        if pruned:
            logger.info(f"Skill: pruned {pruned} low-usage skills")
        return pruned
