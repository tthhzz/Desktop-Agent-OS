"""Context Compressor — reduces token usage for long conversations.

When the conversation history exceeds a threshold, this module:
1. Keeps the task goal + key decisions intact
2. Compresses early conversation rounds into a summary
3. Preserves the most recent N rounds verbatim

This is inspired by Hermes's 4-layer Token optimization (600K→30K).
"""

from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from loguru import logger

# Configuration
MAX_RECENT_MESSAGES = 10  # Keep last N messages verbatim
MIN_MESSAGES_TO_COMPRESS = 20  # Only compress if total exceeds this
MAX_SUMMARY_LENGTH = 500  # Max chars for compressed summary


COMPRESS_PROMPT = """Summarize the following conversation segment in a concise paragraph.
Focus on: task goals, key decisions, important facts, and outcomes.
Omit: pleasantries, repetitions, and tool execution details (keep only results).

Conversation:
{conversation_text}

Concise summary:"""


async def compress_messages(
    messages: list,
    llm: Optional[BaseChatModel] = None,
) -> list:
    """Compress a message list by summarizing early messages.

    Strategy:
    - If messages < MIN_MESSAGES_TO_COMPRESS: return as-is (no compression needed)
    - Otherwise: summarize messages[0:-MAX_RECENT_MESSAGES] into one SystemMessage,
      keep last MAX_RECENT_MESSAGES verbatim

    Returns a new (potentially shorter) message list.
    """
    if len(messages) < MIN_MESSAGES_TO_COMPRESS:
        return messages

    logger.info(f"[Compressor] compressing {len(messages)} messages")

    # Split into old (to compress) and recent (to keep)
    old_messages = messages[:-MAX_RECENT_MESSAGES]
    recent_messages = messages[-MAX_RECENT_MESSAGES:]

    # Build text from old messages
    conversation_parts = []
    for msg in old_messages:
        if isinstance(msg, SystemMessage):
            continue  # Skip system messages in compression
        elif isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            conversation_parts.append(f"User: {content[:200]}")
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content:
                conversation_parts.append(f"AI: {content[:200]}")
            # Note tool calls but don't include full details
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_names = [tc.get("name", "?") for tc in msg.tool_calls]
                conversation_parts.append(f"  [Called tools: {', '.join(tool_names)}]")
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            conversation_parts.append(f"Tool result: {content[:100]}")

    conversation_text = "\n".join(conversation_parts)

    # If LLM available, use it for intelligent compression
    if llm and len(conversation_text) > 500:
        try:
            summary = await _compress_with_llm(llm, conversation_text)
        except Exception as e:
            logger.error(f"[Compressor] LLM error: {e}, using truncation")
            summary = _compress_truncation(conversation_text)
    else:
        summary = _compress_truncation(conversation_text)

    # Build compressed message list
    compressed = [
        SystemMessage(content=f"[Earlier conversation summary: {summary}]"),
    ] + recent_messages

    original_tokens = sum(len(str(m.content)) for m in messages if hasattr(m, 'content'))
    compressed_tokens = sum(len(str(m.content)) for m in compressed if hasattr(m, 'content'))
    logger.info(
        f"[Compressor] {len(messages)} → {len(compressed)} messages, "
        f"~{original_tokens} → ~{compressed_tokens} chars"
    )

    return compressed


async def _compress_with_llm(llm: BaseChatModel, text: str) -> str:
    """Use LLM to generate a concise summary of the conversation."""
    prompt = COMPRESS_PROMPT.format(conversation_text=text[:3000])  # Cap input

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    summary = response.content.strip()

    if len(summary) > MAX_SUMMARY_LENGTH:
        summary = summary[:MAX_SUMMARY_LENGTH] + "..."

    return summary


def _compress_truncation(text: str) -> str:
    """Simple truncation-based compression (no LLM needed)."""
    if len(text) <= MAX_SUMMARY_LENGTH:
        return text

    # Take first and last portions
    half = MAX_SUMMARY_LENGTH // 2 - 3
    return text[:half] + " ... " + text[-half:]
