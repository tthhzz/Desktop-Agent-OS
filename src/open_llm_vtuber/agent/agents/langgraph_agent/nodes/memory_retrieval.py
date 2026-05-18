"""Memory retrieval node — fetches relevant context from the 5-layer memory system.

Runs before the supervisor so that all downstream nodes have access to
retrieved memories and skill matches.
"""

from typing import Dict, Any, Optional
from loguru import logger

from ..state import AgentState


async def memory_retrieval_node(
    state: AgentState,
    memory_system=None,
) -> Dict[str, Any]:
    """Retrieve relevant memories and skills based on the latest user message.

    Updates:
        state["retrieved_memories"] — dict with episodic/semantic/skills lists
        state["skill_match"] — the best matching skill (or None)
    """
    if not memory_system:
        logger.debug("[Memory] no MemorySystem, skipping retrieval")
        return {
            "retrieved_memories": None,
            "skill_match": None,
        }

    # Extract query from the last human message
    messages = state.get("messages", [])
    query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not query:
        logger.debug("[Memory] no user query found, skipping retrieval")
        return {
            "retrieved_memories": None,
            "skill_match": None,
        }

    try:
        context = await memory_system.retrieve_context(query, top_k=5)
    except Exception as e:
        logger.error(f"[Memory] retrieval error: {e}")
        return {
            "retrieved_memories": None,
            "skill_match": None,
        }

    # Log what we found
    n_epi = len(context.get("episodic", []))
    n_sem = len(context.get("semantic", []))
    n_skill = len(context.get("skills", []))
    logger.info(f"[Memory] retrieved: {n_epi} episodic, {n_sem} semantic, {n_skill} skills")

    # Pick the best skill match
    skill_match = None
    skills = context.get("skills", [])
    if skills:
        # First skill is the best match (already sorted by relevance)
        skill_match = skills[0] if isinstance(skills[0], dict) else {"name": str(skills[0])}
        logger.info(f"[Memory] skill match: {skill_match.get('name', 'unknown')}")

    return {
        "retrieved_memories": context,
        "skill_match": skill_match,
    }
