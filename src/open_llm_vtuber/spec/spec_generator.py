"""Spec Generator — generates structured specs from natural language tasks.

Uses LLM to transform a user request into an OpenSpec-compliant specification
with goals, constraints, interfaces, acceptance criteria, and decomposition.
"""

import json
from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from loguru import logger

from .schema import Spec

SPEC_GENERATION_SYSTEM = """You are a specification engineer. Given a user task and available tools, generate a structured task specification (Spec).

The Spec must include:
1. **goal**: Clear statement of what the task accomplishes
2. **constraints**: Boundaries the agent must respect (e.g., "no external API calls", "response < 500 chars")
3. **interfaces**: Input/output contracts (what goes in, what comes out)
4. **acceptance_criteria**: Verifiable conditions for success (2-5 criteria)
5. **decomposition**: Ordered steps to execute (2-5 steps, each with action and expected output)

Available tools: {tool_summary}

Rules:
- Keep specs concise and practical
- Each acceptance criterion must be verifiable (not subjective)
- Decomposition steps should map to specific tools
- Constraints should prevent harmful or wasteful actions
- If the task is simple, use a minimal 1-2 step decomposition

Respond with JSON only:
{{
  "spec_version": "1.0",
  "task": {{
    "goal": "...",
    "constraints": ["..."],
    "interfaces": {{
      "input": {{"type": "object", "properties": {{}}, "required": []}},
      "output": {{"type": "object", "properties": {{}}, "required": []}}
    }},
    "acceptance_criteria": [
      {{"description": "...", "check_type": "contains|not_contains|returns_success", "expected": "...", "critical": true}}
    ],
    "decomposition": [
      {{"step": 1, "action": "tool_name", "description": "...", "with_args": {{}}, "expected_output": "...", "on_failure": "retry"}}
    ]
  }}
}}"""


async def spec_generator_node(
    state: dict,
    llm: BaseChatModel,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate a structured Spec from the user's task.

    This replaces the simple Planner for complex tasks. The Spec includes
    acceptance criteria that the Harness can verify after each step.

    Args:
        state: Agent state dict (compatible with AgentState).

    Returns state updates:
    - plan_steps: from spec decomposition
    - metadata: includes current_spec for Harness verification
    - next_worker: "tools" to begin execution
    """
    # If there's already a spec/plan being executed, skip re-generation
    existing_plan = state.get("plan_steps")
    if existing_plan and len(existing_plan) > 0:
        current = state.get("current_step", 0)
        if current < len(existing_plan):
            logger.info(f"[SpecGen] continuing existing plan: step {current + 1}/{len(existing_plan)}")
            return {"next_worker": "tools"}

    messages = state.get("messages", [])

    # Build tool summary
    tool_summary = "No tools available."
    if tool_schemas:
        tool_names = []
        for tool in tool_schemas:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            tool_names.append(f"{name}: {desc[:60]}")
        tool_summary = "\n".join(tool_names[:25])

    system_content = SPEC_GENERATION_SYSTEM.format(tool_summary=tool_summary)

    prompt_messages = [
        SystemMessage(content=system_content),
    ]

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
            content="Generate a structured specification for this task. "
            "Include acceptance criteria that can be verified after execution. "
            "Respond with JSON only."
        )
    )

    try:
        response = await llm.ainvoke(prompt_messages)
        content = response.content.strip()

        # Parse JSON
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        spec_data = json.loads(content)
        spec = Spec.from_dict(spec_data)

        logger.info(f"[SpecGen] ▶ spec generated: goal='{spec.task.goal[:80]}'")
        logger.info(f"  Constraints: {spec.task.constraints}")
        logger.info(f"  Acceptance: {len(spec.task.acceptance_criteria)} criteria")
        logger.info(f"  Steps: {len(spec.task.decomposition)}")

        # Convert decomposition to plan_steps for compatibility with existing nodes
        plan_steps = []
        for step in spec.task.decomposition:
            plan_steps.append({
                "goal": step.description,
                "tool_hint": step.action,
                "success_criteria": step.expected_output,
                "on_failure": step.on_failure,
            })

        return {
            "plan_steps": plan_steps,
            "current_step": 0,
            "retry_count": 0,
            "next_worker": "tools",
            "metadata": {
                **(state.get("metadata") or {}),
                "current_spec": spec.to_dict(),
            },
        }

    except json.JSONDecodeError as e:
        logger.error(f"[SpecGen] JSON parse error: {e}")
        # Fallback: create minimal spec
        return _fallback_spec(state)
    except Exception as e:
        logger.error(f"[SpecGen] error: {e}")
        return _fallback_spec(state)


def _fallback_spec(state: dict) -> Dict[str, Any]:
    """Create a minimal fallback spec when LLM generation fails."""
    return {
        "plan_steps": [{
            "goal": "complete the task",
            "tool_hint": "auto",
            "success_criteria": "task completed",
            "on_failure": "retry",
        }],
        "current_step": 0,
        "retry_count": 0,
        "next_worker": "tools",
    }
