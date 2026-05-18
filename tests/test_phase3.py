"""Tests for Phase 3: 5-layer Memory System."""

import asyncio
import os
import tempfile
import pytest

from open_llm_vtuber.memory.memory_system import MemorySystem
from open_llm_vtuber.memory.config import MemoryConfig
from open_llm_vtuber.memory.layers import SensoryLayer, WorkingLayer
from open_llm_vtuber.memory.storage.sqlite_store import SQLiteStore


# ── Test 1: Sensory Layer ─────────────────────────────────────


def test_sensory_layer():
    """Test the sensory ring buffer."""
    layer = SensoryLayer(buffer_size=3)

    layer.add_frame("frame1", source="camera")
    layer.add_frame("frame2", source="camera")
    layer.add_frame("frame3", source="screen")
    assert layer.frame_count == 3

    layer.add_frame("frame4", source="camera")
    assert layer.frame_count == 3  # Ring buffer

    recent = layer.get_recent_frames(2)
    assert len(recent) == 2
    assert recent[0]["data"] == "frame4"


# ── Test 2: Working Layer ─────────────────────────────────────


def test_working_layer():
    """Test the working memory."""
    layer = WorkingLayer()
    layer.add("user", "Hello")
    layer.add("assistant", "Hi there!")
    assert layer.message_count == 2

    last = layer.get_messages(last_n=1)
    assert last[0]["content"] == "Hi there!"


# ── Test 3: Episodic Layer ────────────────────────────────────


@pytest.mark.asyncio
async def test_episodic_layer():
    """Test episodic memory storage and retrieval."""
    tmpdir = tempfile.mkdtemp()
    try:
        store = SQLiteStore(os.path.join(tmpdir, "test.db"))
        await store.initialize()

        from open_llm_vtuber.memory.layers.episodic import EpisodicLayer
        layer = EpisodicLayer(store)

        eid = await layer.store_episode(
            user_input="What's the weather?",
            ai_response="It's sunny today.",
            emotion="happy",
            topics=["weather"],
            importance=0.6,
        )
        assert eid is not None

        episode = await layer.get_episode(eid)
        assert episode is not None
        assert episode["user_input"] == "What's the weather?"
        assert episode["emotion"] == "happy"

        count = await layer.count()
        assert count == 1

        results = await layer.search_episodes("weather", top_k=5)
        assert len(results) >= 1

        await store.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test 4: Semantic Layer ─────────────────────────────────────


@pytest.mark.asyncio
async def test_semantic_layer():
    """Test semantic memory storage and retrieval."""
    tmpdir = tempfile.mkdtemp()
    try:
        store = SQLiteStore(os.path.join(tmpdir, "test.db"))
        await store.initialize()

        from open_llm_vtuber.memory.layers.semantic import SemanticLayer
        layer = SemanticLayer(store)

        kid = await layer.store_knowledge(
            content="User likes cats",
            source="reflection",
            categories=["preference", "pets"],
            importance=0.8,
        )
        assert kid is not None

        knowledge = await layer.get_knowledge(kid)
        assert knowledge is not None
        assert knowledge["content"] == "User likes cats"

        results = await layer.search_knowledge("cats", top_k=5)
        assert len(results) >= 1

        count = await layer.count()
        assert count == 1

        await store.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test 5: Skill Layer ───────────────────────────────────────


@pytest.mark.asyncio
async def test_skill_layer():
    """Test skill memory registration and search."""
    tmpdir = tempfile.mkdtemp()
    try:
        store = SQLiteStore(os.path.join(tmpdir, "test.db"))
        await store.initialize()

        from open_llm_vtuber.memory.layers.skill import SkillLayer
        layer = SkillLayer(store)

        sid = await layer.register_skill(
            name="daily_weather_check",
            description="Check weather and provide morning briefing",
            tools=["ddg-search__search", "time__get_current_time"],
            trigger_condition="when user asks about weather in the morning",
            frequency=3,
        )
        assert sid is not None

        skill = await layer.get_skill(sid)
        assert skill is not None
        assert skill["name"] == "daily_weather_check"
        assert len(skill["tools"]) == 2

        results = await layer.find_skill("weather briefing")
        assert len(results) >= 1

        await layer.increment_usage(sid)
        skill = await layer.get_skill(sid)
        assert skill["frequency"] == 4

        await store.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test 6: Full MemorySystem conversation turn ────────────────


@pytest.mark.asyncio
async def test_conversation_turn():
    """Test the full on_conversation_turn flow."""
    tmpdir = tempfile.mkdtemp()
    try:
        config = MemoryConfig(
            episodic_db_path=os.path.join(tmpdir, "epi.db"),
            semantic_db_path=os.path.join(tmpdir, "sem.db"),
            skill_db_path=os.path.join(tmpdir, "skill.db"),
            reflection_interval=2,
            consolidation_interval=100,
        )
        system = MemorySystem(config)
        await system.initialize()

        await system.on_conversation_turn(
            user_input="你好，我叫小明",
            ai_response="你好小明！很高兴认识你。",
            emotion="happy",
            topics=["greeting"],
        )

        await system.on_conversation_turn(
            user_input="我喜欢猫",
            ai_response="猫很可爱！你养了几只猫？",
            emotion="happy",
            topics=["pets", "cats"],
        )

        assert system.working.message_count == 4

        epi_count = await system.episodic.count()
        assert epi_count == 2

        stats = await system.get_stats()
        assert stats["episodic_count"] == 2
        assert stats["working_messages"] == 4

        await system.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test 7: Cross-session persistence ─────────────────────────


@pytest.mark.asyncio
async def test_cross_session_persistence():
    """Test that memory persists across sessions."""
    tmpdir = tempfile.mkdtemp()
    try:
        config = MemoryConfig(
            episodic_db_path=os.path.join(tmpdir, "epi.db"),
            semantic_db_path=os.path.join(tmpdir, "sem.db"),
            skill_db_path=os.path.join(tmpdir, "skill.db"),
        )

        system1 = MemorySystem(config)
        await system1.initialize()
        await system1.on_conversation_turn(
            user_input="我喜欢Python编程",
            ai_response="Python是一门很棒的语言！",
            topics=["programming", "python"],
        )
        await system1.close()

        # Session 2
        system2 = MemorySystem(config)
        await system2.initialize()
        count = await system2.episodic.count()
        assert count == 1
        await system2.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test 8: Skill mining ──────────────────────────────────────


@pytest.mark.asyncio
async def test_skill_mining():
    """Test skill mining from tool usage patterns."""
    tmpdir = tempfile.mkdtemp()
    try:
        config = MemoryConfig(
            episodic_db_path=os.path.join(tmpdir, "epi.db"),
            semantic_db_path=os.path.join(tmpdir, "sem.db"),
            skill_db_path=os.path.join(tmpdir, "skill.db"),
        )
        system = MemorySystem(config)
        await system.initialize()

        for _ in range(3):
            system.skill_miner.record_tool_sequence(
                ["ddg-search__search", "time__get_current_time"]
            )

        new_skills = await system.skill_miner.mine_skills(frequency_threshold=3)
        assert len(new_skills) >= 1
        assert new_skills[0]["name"]
        assert len(new_skills[0]["tools"]) == 2

        await system.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test 9: Rule-based reflection ────────────────────────────


@pytest.mark.asyncio
async def test_rule_based_reflection():
    """Test the rule-based reflector (no LLM needed)."""
    tmpdir = tempfile.mkdtemp()
    try:
        config = MemoryConfig(
            episodic_db_path=os.path.join(tmpdir, "epi.db"),
            semantic_db_path=os.path.join(tmpdir, "sem.db"),
            skill_db_path=os.path.join(tmpdir, "skill.db"),
        )
        system = MemorySystem(config)
        await system.initialize()

        for _ in range(3):
            await system.episodic.store_episode(
                user_input="今天天气怎么样？",
                ai_response="今天天气很好！",
                emotion="happy",
                topics=["weather"],
            )

        results = await system.reflector.reflect()
        assert len(results) >= 1

        sem_count = await system.semantic.count()
        assert sem_count >= 1

        await system.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
