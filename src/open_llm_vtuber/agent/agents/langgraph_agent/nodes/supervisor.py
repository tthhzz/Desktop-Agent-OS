"""Supervisor node — analyses the conversation and decides which worker to route to."""

from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

from ..state import AgentState


def _build_tool_inventory(tool_schemas: List[Dict[str, Any]]) -> str:
    """Build a human-readable tool inventory from OpenAI-format tool schemas."""
    if not tool_schemas:
        return "No tools available."
    lines = []
    for tool in tool_schemas:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "No description")
        # Truncate long descriptions
        if len(desc) > 120:
            desc = desc[:117] + "..."
        lines.append(f"- **{name}**: {desc}")
    return "\n".join(lines)


SUPERVISOR_SYSTEM_TEMPLATE = """You are a supervisor managing a multi-agent system. Your job is to decide the next action based on the conversation.

You have these workers available:
- **chat**: For general conversation, Q&A, and tasks that don't require tools.
- **tools**: For tasks that require using MCP tools.

You have the following tools available:
{tool_inventory}

Routing rules:
- If the user's request can be answered from general knowledge without tools → route to "chat"
- If the user's request matches any tool's capability (e.g., asking time → time tool, searching → search tool, asking about files → terminal tool, etc.) → route to "tools"
- If a matching skill is found in the memory context → route to "tools"
- If tool results came back and no more tools needed → route to "chat" to formulate the response
- If the conversation is complete → route to "__end__"

CRITICAL: When the user asks for real-time information, factual lookups, file operations, web searches, or any task that a tool can handle, you MUST route to "tools". Do NOT route to "chat" when a tool is available for the task.

Respond with ONLY the worker name: "chat", "tools", or "__end__"
"""


async def supervisor_node(
    state: AgentState,
    llm: BaseChatModel,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Determine the next worker based on conversation state.

    Returns a partial state update with ``next_worker`` set.
    """
    messages = state.get("messages", [])

    # Build the system prompt with tool inventory
    tool_inventory = _build_tool_inventory(tool_schemas or [])
    system_content = SUPERVISOR_SYSTEM_TEMPLATE.format(tool_inventory=tool_inventory)

    # Build the prompt for the supervisor
    prompt_messages = [
        SystemMessage(content=system_content),
    ]

    # Inject memory context if available
    memories = state.get("retrieved_memories")
    skill_match = state.get("skill_match")
    if memories or skill_match:
        memory_parts = []
        if memories:
            semantic = memories.get("semantic", [])
            if semantic:
                memory_parts.append("[已知知识]")
                for kn in semantic[:3]:
                    if isinstance(kn, dict):
                        memory_parts.append(f"- {kn.get('content', kn.get('fact', str(kn)))}")
                    else:
                        memory_parts.append(f"- {kn}")
        if skill_match:
            memory_parts.append(f"[匹配技能: {skill_match.get('name', 'unknown')} — {skill_match.get('description', '')}]")
        if memory_parts:
            prompt_messages.append(SystemMessage(content="\n".join(memory_parts)))

    # Include recent conversation context (last 10 messages to stay within token limits)
    recent = messages[-10:] if len(messages) > 10 else messages
    for msg in recent:
        # Handle multimodal ToolMessage content
        if isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, list):
                # Multimodal: extract text summary, skip image blocks for routing
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                text = " ".join(text_parts) if text_parts else "[Screenshot captured]"
                prompt_messages.append(HumanMessage(content=f"[Tool result: {text}]"))
            elif isinstance(content, str) and content.startswith("data:image/"):
                prompt_messages.append(HumanMessage(content="[Tool result: Screenshot captured]"))
            else:
                prompt_messages.append(msg)
        else:
            prompt_messages.append(msg)

    # Add a human message asking for the routing decision
    prompt_messages.append(
        HumanMessage(
            content="Based on the conversation above, which worker should handle the next step? "
            "Respond with ONLY: chat, tools, or __end__"
        )
    )

    try:
        response = await llm.ainvoke(prompt_messages)
        decision = response.content.strip().lower()

        # Normalize the decision
        if "chat" in decision and "tool" not in decision:
            next_worker = "chat"
        elif "tool" in decision:
            next_worker = "tools"
        elif "end" in decision:
            next_worker = "__end__"
        else:
            # Default to chat if ambiguous
            logger.warning(f"[Supervisor] ambiguous '{decision}' → defaulting to 'chat'")
            next_worker = "chat"

        logger.info(f"[Supervisor] route → {next_worker} (raw: '{decision}')")
        return {"next_worker": next_worker}

    except Exception as e:
        logger.error(f"Supervisor error: {e}")
        return {"next_worker": "chat"}  # Fallback to chat on error
