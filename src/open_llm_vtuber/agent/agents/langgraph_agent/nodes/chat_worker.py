"""Chat worker node — generates conversational responses via LLM."""

from typing import Dict, Any, AsyncIterator, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from loguru import logger

from ..state import AgentState


def _build_memory_block(
    retrieved_memories: Optional[Dict[str, Any]],
    skill_match: Optional[Dict[str, Any]],
) -> str:
    """Build a memory context block to inject into the LLM prompt."""
    if not retrieved_memories and not skill_match:
        return ""

    parts = []

    # Episodic memories
    episodic = retrieved_memories.get("episodic", []) if retrieved_memories else []
    if episodic:
        parts.append("[相关记忆]")
        for ep in episodic[:5]:
            if isinstance(ep, dict):
                user = ep.get("user_input", "")
                ai = ep.get("ai_response", "")
                emotion = ep.get("emotion", "")
                label = f" ({emotion})" if emotion else ""
                parts.append(f"- 用户: {user} | AI: {ai}{label}")
            else:
                parts.append(f"- {ep}")

    # Semantic knowledge
    semantic = retrieved_memories.get("semantic", []) if retrieved_memories else []
    if semantic:
        parts.append("[已知知识]")
        for kn in semantic[:5]:
            if isinstance(kn, dict):
                cat = kn.get("category", "")
                fact = kn.get("content", kn.get("fact", str(kn)))
                conf = kn.get("confidence", "")
                label = f" [{cat}]" if cat else ""
                conf_label = f" (置信度: {conf})" if conf else ""
                parts.append(f"- {fact}{label}{conf_label}")
            else:
                parts.append(f"- {kn}")

    # Skill match
    if skill_match:
        parts.append("[匹配技能]")
        name = skill_match.get("name", "unknown")
        desc = skill_match.get("description", "")
        tools = skill_match.get("tools", [])
        parts.append(f"- {name}: {desc} (工具链: {' → '.join(tools)})")

    return "\n".join(parts) if parts else ""


async def chat_node(
    state: AgentState,
    llm: BaseChatModel,
    system_prompt: str,
) -> Dict[str, Any]:
    """Generate a conversational response using the LLM.

    This worker handles general chat, Q&A, and formulating responses
    after tool results are available.
    """
    messages = state.get("messages", [])

    # Build the message list for the LLM
    llm_messages = [SystemMessage(content=system_prompt)]

    # Inject memory context if available
    memories = state.get("retrieved_memories")
    skill_match = state.get("skill_match")
    memory_block = _build_memory_block(memories, skill_match)
    if memory_block:
        llm_messages.append(SystemMessage(content=memory_block))
        logger.debug(f"[ChatWorker] injected memory block ({len(memory_block)} chars)")

    for msg in messages:
        # Skip system messages (we already added our own)
        if msg.type == "system":
            continue
        llm_messages.append(msg)

    try:
        # Use streaming for real-time token generation
        response_chunks = []
        full_response = ""

        async for chunk in llm.astream(llm_messages):
            content = chunk.content
            if content:
                full_response += content
                response_chunks.append(content)

        logger.info(f"[ChatWorker] ✓ response ({len(full_response)} chars)")
        return {
            "messages": [AIMessage(content=full_response)],
            "next_worker": "__end__",
        }

    except Exception as e:
        logger.error(f"[ChatWorker] ✗ error: {e}")
        error_msg = f"[Error generating response: {e}]"
        return {
            "messages": [AIMessage(content=error_msg)],
            "next_worker": "__end__",
        }


async def chat_stream_node(
    state: AgentState,
    llm: BaseChatModel,
    system_prompt: str,
) -> AsyncIterator[str]:
    """Stream tokens from the chat worker for real-time output.

    Yields individual text chunks as they arrive from the LLM.
    """
    messages = state.get("messages", [])

    llm_messages = [SystemMessage(content=system_prompt)]
    for msg in messages:
        if msg.type == "system":
            continue
        llm_messages.append(msg)

    async for chunk in llm.astream(llm_messages):
        content = chunk.content
        if content:
            yield content
