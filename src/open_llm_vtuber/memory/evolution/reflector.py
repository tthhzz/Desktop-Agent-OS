"""Reflector — extracts semantic knowledge from episodic memories.

Periodically reviews recent episodes and uses LLM to extract
long-term facts, preferences, and patterns.
"""

import json
from typing import Any, Dict, List, Optional
from loguru import logger

from ..layers.episodic import EpisodicLayer
from ..layers.semantic import SemanticLayer


REFLECTION_PROMPT = """Analyze the following conversation episodes and extract long-term knowledge about the user.
Focus on: preferences, habits, facts, personality traits, recurring topics, and relationships.

For each piece of knowledge, provide:
- content: A clear statement of the knowledge (e.g., "User prefers dark mode in code editors")
- categories: Relevant categories (e.g., ["preference", "coding"])
- importance: 0.0-1.0 (how important is this knowledge for future interactions?)
- confidence: 0.0-1.0 (how confident are you in this extraction?)

Episodes:
{episodes}

Respond with a JSON array of knowledge items. Example:
[
  {{"content": "User likes cats", "categories": ["preference", "pets"], "importance": 0.7, "confidence": 0.9}},
  {{"content": "User is a Python developer", "categories": ["fact", "career"], "importance": 0.8, "confidence": 0.95}}
]

If no meaningful knowledge can be extracted, respond with an empty array: []"""


class Reflector:
    """Extracts semantic knowledge from episodic memories."""

    def __init__(self, episodic: EpisodicLayer, semantic: SemanticLayer, llm=None):
        self._episodic = episodic
        self._semantic = semantic
        self._llm = llm  # Optional: if None, uses rule-based extraction

    async def reflect(self, recent_count: int = 10) -> List[Dict[str, Any]]:
        """Reflect on recent episodes and extract knowledge.

        Args:
            recent_count: Number of recent episodes to analyze.

        Returns:
            List of newly created knowledge entries.
        """
        episodes = await self._episodic.get_recent_episodes(recent_count)

        if not episodes:
            logger.debug("Reflector: no episodes to reflect on")
            return []

        logger.info(f"Reflector: analyzing {len(episodes)} episodes")

        if self._llm:
            return await self._reflect_with_llm(episodes)
        else:
            return await self._reflect_rule_based(episodes)

    async def _reflect_with_llm(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to extract knowledge from episodes."""
        episode_text = "\n".join(
            f"- User: {ep.get('user_input', '')} | AI: {ep.get('ai_response', '')} | Emotion: {ep.get('emotion', 'neutral')}"
            for ep in episodes
        )

        prompt = REFLECTION_PROMPT.format(episodes=episode_text)

        try:
            from litellm import completion
            response = completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON from response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            knowledge_items = json.loads(content)
            if not isinstance(knowledge_items, list):
                knowledge_items = [knowledge_items]

        except Exception as e:
            logger.error(f"Reflector LLM error: {e}")
            knowledge_items = []

        # Store extracted knowledge
        new_entries = []
        for item in knowledge_items:
            if isinstance(item, dict) and item.get("content"):
                record_id = await self._semantic.store_knowledge(
                    content=item["content"],
                    source="reflection",
                    categories=item.get("categories", []),
                    importance=item.get("importance", 0.5),
                    confidence=item.get("confidence", 0.8),
                )
                new_entries.append(item)

        logger.info(f"Reflector: extracted {len(new_entries)} knowledge items")
        return new_entries

    async def _reflect_rule_based(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple rule-based reflection as fallback when no LLM available.

        Extracts basic patterns: topics mentioned frequently,
        emotional patterns, and explicit user statements.
        """
        new_entries = []
        topic_counts: Dict[str, int] = {}
        emotion_counts: Dict[str, int] = {}

        for ep in episodes:
            # Count topics
            for topic in ep.get("topics", []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

            # Count emotions
            emotion = ep.get("emotion", "neutral")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Create knowledge from frequent topics
        for topic, count in topic_counts.items():
            if count >= 2:
                record_id = await self._semantic.store_knowledge(
                    content=f"User frequently discusses {topic}",
                    source="rule_reflection",
                    categories=["interest"],
                    importance=0.5,
                    confidence=0.7,
                )
                new_entries.append({"content": f"Frequent topic: {topic}", "id": record_id})

        # Create knowledge from dominant emotions
        if emotion_counts:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            if dominant_emotion != "neutral" and emotion_counts[dominant_emotion] >= 2:
                record_id = await self._semantic.store_knowledge(
                    content=f"User's conversations tend to be {dominant_emotion}",
                    source="rule_reflection",
                    categories=["emotion_pattern"],
                    importance=0.4,
                    confidence=0.6,
                )
                new_entries.append({"content": f"Emotional pattern: {dominant_emotion}", "id": record_id})

        logger.info(f"Reflector (rule-based): extracted {len(new_entries)} knowledge items")
        return new_entries
