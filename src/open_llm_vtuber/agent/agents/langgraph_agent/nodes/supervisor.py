"""Supervisor node — analyses the conversation and decides which worker to route to."""

from typing import Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

from ..state import AgentState

SUPERVISOR_SYSTEM = """You are a supervisor managing a multi-agent system. Your job is to decide the next action based on the conversation.

You have these workers available:
- **chat**: For general conversation, Q&A, and tasks that don't require tools.
- **tools**: For tasks that require using MCP tools (searching, browsing, file operations, etc.).

Analyze the latest user message, any tool results, and the memory context, then decide:
- If the user is just chatting or asking questions that you can answer directly → route to "chat"
- If the user's request requires using a tool, or if there are tool results that need follow-up → route to "tools"
- If a matching skill is found in the memory context and the user's request matches its trigger condition → route to "tools" to execute the skill
- If the conversation is complete and no further action is needed → route to "__end__"

Important rules:
- After tool results come back, you should usually route to "chat" so the response can be formulated, unless more tools are needed.
- If the user explicitly asks to use a tool (search, browse, etc.), route to "tools".
- If a matching skill's trigger condition aligns with the user's request, prefer routing to "tools" to leverage the skill.
- Use the retrieved memory context to provide more informed and personalized routing decisions.
- Respond with ONLY the worker name, nothing else.

Valid responses: "chat", "tools", "__end__"
"""


async def supervisor_node(
    state: AgentState,
    llm: BaseChatModel,
) -> Dict[str, Any]:
    """Determine the next worker based on conversation state.

    Returns a partial state update with ``next_worker`` set.
    """
    messages = state.get("messages", [])

    # Build the prompt for the supervisor
    prompt_messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
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
