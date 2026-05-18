"""MemorySystem — unified 5-layer memory manager.

Provides a single interface to all memory layers and evolution operations.

Layers:
1. Sensory  — short-lived perceptual buffer (camera/screen frames)
2. Working  — current conversation context
3. Episodic — conversation events with emotional annotations
4. Semantic — long-term knowledge (facts, preferences)
5. Skill    — reusable tool-chain patterns

Evolution:
- Reflector: episodic → semantic (knowledge extraction)
- Consolidator: semantic merge + decay
- SkillMiner: tool patterns → skills
"""

import os
from typing import Any, Dict, List, Optional
from loguru import logger

from .config import MemoryConfig
from .storage.sqlite_store import SQLiteStore
from .layers import (
    SensoryLayer,
    WorkingLayer,
    EpisodicLayer,
    SemanticLayer,
    SkillLayer,
)
from .evolution import Reflector, Consolidator, SkillMiner


class MemorySystem:
    """Unified 5-layer memory system for the autonomous desktop pet."""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self._config = config or MemoryConfig()
        self._initialized = False

        # Layers (initialized lazily)
        self.sensory: Optional[SensoryLayer] = None
        self.working: Optional[WorkingLayer] = None
        self.episodic: Optional[EpisodicLayer] = None
        self.semantic: Optional[SemanticLayer] = None
        self.skill: Optional[SkillLayer] = None

        # Evolution
        self.reflector: Optional[Reflector] = None
        self.consolidator: Optional[Consolidator] = None
        self.skill_miner: Optional[SkillMiner] = None

        # Internal tracking
        self._turn_count = 0

    async def initialize(self) -> None:
        """Initialize all memory layers and evolution components."""
        if self._initialized:
            return

        # Create data directory
        os.makedirs(os.path.dirname(self._config.episodic_db_path) or ".", exist_ok=True)

        # Sensory & Working — in-memory, no persistence
        self.sensory = SensoryLayer(buffer_size=self._config.sensory_buffer_size)
        self.working = WorkingLayer()

        # Episodic — SQLite
        epi_store = SQLiteStore(self._config.episodic_db_path)
        await epi_store.initialize()
        self.episodic = EpisodicLayer(epi_store)

        # Semantic — SQLite
        sem_store = SQLiteStore(self._config.semantic_db_path)
        await sem_store.initialize()
        self.semantic = SemanticLayer(sem_store)

        # Skill — SQLite
        skill_store = SQLiteStore(self._config.skill_db_path)
        await skill_store.initialize()
        self.skill = SkillLayer(skill_store)

        # Evolution
        self.reflector = Reflector(self.episodic, self.semantic)
        self.consolidator = Consolidator(self.semantic)
        self.skill_miner = SkillMiner(self.episodic, self.skill)

        self._initialized = True
        logger.info("MemorySystem: 5-layer memory initialized")

    async def on_conversation_turn(
        self,
        user_input: str,
        ai_response: str,
        emotion: str = "neutral",
        topics: Optional[List[str]] = None,
        tools_used: Optional[List[str]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Called after each conversation turn to update memory.

        This is the main entry point for feeding data into the memory system.
        """
        if not self._initialized:
            await self.initialize()

        self._turn_count += 1

        # 1. Update working memory
        self.working.add("user", user_input)
        self.working.add("assistant", ai_response)

        # 2. Store in episodic
        importance = 0.5
        if tools_used:
            importance = 0.7  # Tool-using conversations are more important
        if any(kw in user_input.lower() for kw in ["记住", "remind", "important", "喜欢", "讨厌"]):
            importance = 0.9  # Explicit memory requests

        await self.episodic.store_episode(
            user_input=user_input,
            ai_response=ai_response,
            emotion=emotion,
            topics=topics or [],
            importance=importance,
        )

        # 3. Record tool usage for skill mining
        if tools_used:
            self.skill_miner.record_tool_sequence(tools_used)

        # 4. Add sensory frames
        if images:
            for img in images:
                self.sensory.add_frame(
                    frame_data=img.get("data", ""),
                    source=img.get("source", "camera"),
                    mime_type=img.get("mime_type", "image/jpeg"),
                )

        # 5. Run evolution at intervals
        if self._turn_count % self._config.reflection_interval == 0:
            logger.info(f"MemorySystem: running reflection at turn {self._turn_count}")
            await self.reflector.reflect()

        if self._turn_count % self._config.consolidation_interval == 0:
            logger.info(f"MemorySystem: running consolidation at turn {self._turn_count}")
            await self.consolidator.consolidate()

    async def retrieve_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve relevant context from all memory layers for a query.

        Returns a dict with:
        - episodic: List of relevant episodes
        - semantic: List of relevant knowledge
        - skills: List of matching skills
        """
        if not self._initialized:
            await self.initialize()

        context = {}

        # Search episodic
        context["episodic"] = await self.episodic.search_episodes(query, top_k)

        # Search semantic
        context["semantic"] = await self.semantic.search_knowledge(query, top_k)

        # Search skills
        context["skills"] = await self.skill.find_skill(query, top_k=3)

        return context

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        if not self._initialized:
            return {"status": "not_initialized"}

        return {
            "sensory_frames": self.sensory.frame_count,
            "working_messages": self.working.message_count,
            "episodic_count": await self.episodic.count(),
            "semantic_count": await self.semantic.count(),
            "skill_count": await self.skill.count(),
            "turn_count": self._turn_count,
        }

    async def close(self) -> None:
        """Close all storage connections."""
        if not self._initialized:
            return

        # Close SQLite stores
        if self.episodic and self.episodic._store:
            await self.episodic._store.close()
        if self.semantic and self.semantic._store:
            await self.semantic._store.close()
        if self.skill and self.skill._store:
            await self.skill._store.close()

        self._initialized = False
        logger.info("MemorySystem: closed")
