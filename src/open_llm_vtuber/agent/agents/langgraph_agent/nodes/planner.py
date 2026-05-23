"""Planner node — decomposes complex tasks into executable sub-steps.

When the supervisor identifies a task that requires multiple tool calls,
the planner breaks it down into ordered steps. Each step has:
- goal: what this step accomplishes
- tool_hint: suggested tool to use
- success_criteria: how to verify the step succeeded
"""

import json
from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from loguru import logger

from ..state import AgentState

PLANNER_SYSTEM = """You are a task planner for an AI agent. Given a user request and available tools, break it down into concrete executable steps.

For each step, provide:
- goal: What this step accomplishes (short phrase)
- tool_hint: Which tool category to use (e.g., "search", "computer", "terminal", "browser", "memory")
- success_criteria: How to verify the step succeeded (one sentence)

Rules:
- Keep plans concise (2-5 steps max for efficiency)
- Each step should be independently executable
- Prefer fewer steps — combine related operations
- Only plan steps that require tools; conversation steps don't need planning

Respond with JSON only:
{"steps": [{"goal": "...", "tool_hint": "...", "success_criteria": "..."}, ...]}

If the task is simple (single tool call), respond with:
{"steps": [{"goal": "complete the task", "tool_hint": "auto", "success_criteria": "task completed"}]}
"""


async def planner_node(
    state: AgentState,
    llm: BaseChatModel,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Decompose a complex task into planned steps.

    Only activates when there's no existing plan. If a plan exists,
    passes through to the next step.
    """
    # If plan already exists, skip re-planning
    existing_plan = state.get("plan_steps")
    if existing_plan and len(existing_plan) > 0:
        current = state.get("current_step", 0)
        if current < len(existing_plan):
            logger.info(f"[Planner] continuing plan: step {current + 1}/{len(existing_plan)}")
            return {"next_worker": "tools"}  # Continue executing

    messages = state.get("messages", [])

    # Build tool summary for context
    tool_summary = ""
    if tool_schemas:
        tool_names = []
        for tool in tool_schemas:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            tool_names.append(name)
        tool_summary = f"Available tools: {', '.join(tool_names[:20])}..."

    prompt_messages = [
        SystemMessage(content=PLANNER_SYSTEM),
    ]

    if tool_summary:
        prompt_messages.append(SystemMessage(content=tool_summary))

    # Include recent conversation (last 6 messages for efficiency)
    recent = [m for m in messages if m.type != "system"][-6:]
    for msg in recent:
        if isinstance(msg, ToolMessage):
            prompt_messages.append(
                HumanMessage(content=f"[Tool result: {msg.content[:200]}]")
            )
        else:
            prompt_messages.append(msg)

    prompt_messages.append(
        HumanMessage(
            content="Break down the user's request into execution steps. "
            "Respond with JSON only."
        )
    )

    try:
        response = await llm.ainvoke(prompt_messages)
        content = response.content.strip()

        # Parse JSON from response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        plan_data = json.loads(content)
        steps = plan_data.get("steps", [])

        if not steps:
            # Fallback: single-step plan
            steps = [{"goal": "complete the task", "tool_hint": "auto", "success_criteria": "task completed"}]

        logger.info(f"[Planner] ▶ plan: {len(steps)} steps")
        for i, step in enumerate(steps):
            logger.info(f"  Step {i + 1}: {step.get('goal', '?')} (tool: {step.get('tool_hint', '?')})")

        return {
            "plan_steps": steps,
            "current_step": 0,
            "retry_count": 0,
            "next_worker": "tools",
        }

    except json.JSONDecodeError as e:
        logger.error(f"[Planner] JSON parse error: {e}")
        # Fallback: treat as single-step
        return {
            "plan_steps": [{"goal": "complete the task", "tool_hint": "auto", "success_criteria": "task completed"}],
            "current_step": 0,
            "retry_count": 0,
            "next_worker": "tools",
        }
    except Exception as e:
        logger.error(f"[Planner] error: {e}")
        return {
            "plan_steps": [{"goal": "complete the task", "tool_hint": "auto", "success_criteria": "task completed"}],
            "current_step": 0,
            "retry_count": 0,
            "next_worker": "tools",
        }
