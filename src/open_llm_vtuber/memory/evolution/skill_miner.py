"""Skill Miner — discovers recurring tool-chain patterns and auto-generates Skills.

Monitors tool usage across conversations. When a pattern of tool calls
occurs frequently (e.g., search → summarize → send), it automatically
creates a Skill that can be reused.
"""

import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from ..layers.episodic import EpisodicLayer
from ..layers.skill import SkillLayer


SKILL_GENERATION_PROMPT = """Based on the following recurring tool usage pattern, generate a reusable skill definition.

Tool sequence: {tool_sequence}
Frequency: {frequency} occurrences
Example inputs: {examples}

Generate a skill with:
- name: Short snake_case identifier (e.g., "daily_weather_check")
- description: What this skill does in one sentence
- tools: List of tool names in order
- trigger_condition: When should this skill be activated?

Respond with JSON only:
{{"name": "...", "description": "...", "tools": [...], "trigger_condition": "..."}}"""


class SkillMiner:
    """Mines recurring tool-chain patterns and auto-generates Skills."""

    def __init__(self, episodic: EpisodicLayer, skill: SkillLayer, llm=None):
        self._episodic = episodic
        self._skill = skill
        self._llm = llm
        self._tool_history: List[List[str]] = []  # Recent tool call sequences

    def record_tool_sequence(self, tools: List[str]) -> None:
        """Record a sequence of tool calls from a conversation turn."""
        if tools:
            self._tool_history.append(tools)
            logger.debug(f"SkillMiner: recorded tool sequence {tools}")

    async def mine_skills(self, frequency_threshold: int = 3) -> List[Dict[str, Any]]:
        """Analyze tool history and auto-generate Skills for frequent patterns.

        Args:
            frequency_threshold: Minimum occurrences before creating a Skill.

        Returns:
            List of newly created skills.
        """
        if not self._tool_history:
            logger.debug("SkillMiner: no tool history to mine")
            return []

        # Count tool sequence patterns
        pattern_counts: Counter = Counter()
        pattern_examples: Dict[str, List[str]] = {}

        for seq in self._tool_history:
            # Use the full sequence as a pattern key
            key = " → ".join(seq)
            pattern_counts[key] += 1
            if key not in pattern_examples:
                pattern_examples[key] = []
            if len(pattern_examples[key]) < 3:
                pattern_examples[key].append(key)

        # Find frequent patterns
        new_skills = []
        for pattern, count in pattern_counts.most_common(10):
            if count < frequency_threshold:
                continue

            # Check if skill already exists
            existing = await self._skill.find_skill(pattern, top_k=1)
            if existing and any(
                e.get("tools", []) == pattern.split(" → ") for e in existing
            ):
                logger.debug(f"SkillMiner: skill for '{pattern}' already exists")
                continue

            # Generate skill
            tools = pattern.split(" → ")
            if self._llm:
                skill_data = await self._generate_with_llm(
                    tools, count, pattern_examples[pattern]
                )
            else:
                skill_data = self._generate_rule_based(tools, count)

            if skill_data:
                record_id = await self._skill.register_skill(**skill_data)
                new_skills.append(skill_data)
                logger.info(f"SkillMiner: auto-generated skill '{skill_data['name']}'")

        return new_skills

    def _generate_rule_based(self, tools: List[str], frequency: int) -> Dict[str, Any]:
        """Generate a skill definition using simple rules."""
        name = "_".join(tools[0].split("__")[-1] if "__" in tools[0] else tools[0])
        if len(tools) > 1:
            name += f"_to_{'_'.join(t.split('__')[-1] for t in tools[1:])}"

        return {
            "name": name,
            "description": f"Automated workflow: {' → '.join(tools)} (used {frequency} times)",
            "tools": tools,
            "trigger_condition": f"When user request matches the pattern: {' → '.join(tools)}",
            "frequency": frequency,
        }

    async def _generate_with_llm(
        self, tools: List[str], frequency: int, examples: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Generate a skill definition using LLM."""
        prompt = SKILL_GENERATION_PROMPT.format(
            tool_sequence=" → ".join(tools),
            frequency=frequency,
            examples=json.dumps(examples[:3]),
        )

        try:
            from litellm import completion
            response = completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            content = response.choices[0].message.content.strip()

            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            skill_data = json.loads(content)
            skill_data["frequency"] = frequency
            return skill_data

        except Exception as e:
            logger.error(f"SkillMiner LLM error: {e}")
            return self._generate_rule_based(tools, frequency)
