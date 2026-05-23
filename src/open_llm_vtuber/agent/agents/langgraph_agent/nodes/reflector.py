"""Reflector node — verifies tool execution results and decides whether to retry.

After the tool worker executes, the reflector checks:
1. Did the tool execution succeed?
2. Does the result meet the acceptance criteria? (via Harness when Spec available)
3. Should we retry with a different approach?

This enables the autonomous agent loop: execute → verify → retry/continue.
"""

import json
from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from loguru import logger

from ..state import AgentState

MAX_RETRIES = 3

REFLECTOR_SYSTEM = """You are a reflection agent. Evaluate the tool execution result and decide the next action.

Current task step: {current_step_goal}
Success criteria: {success_criteria}

Possible decisions:
1. "success" — Step completed, move to next step
2. "retry" — Step failed, retry with different approach (max {max_retries} retries)
3. "abort" — Cannot complete this step, abort the task
4. "chat" — Need user clarification or task is complete, return to chat

Respond with JSON only:
{{"decision": "success|retry|abort|chat", "reason": "...", "adjusted_approach": "..."}}"""


async def reflector_node(
    state: AgentState,
    llm: BaseChatModel,
) -> Dict[str, Any]:
    """Reflect on tool execution results and decide next action.

    Returns state updates with:
    - next_worker: "tools" (retry), "chat" (success/abort/chat), or "__end__"
    - Updated plan progress
    """
    messages = state.get("messages", [])
    plan_steps = state.get("plan_steps") or []
    current_step = state.get("current_step", 0)
    retry_count = state.get("retry_count", 0)

    # Get current step info
    current_goal = "unknown"
    success_criteria = "task completed"
    if plan_steps and current_step < len(plan_steps):
        step = plan_steps[current_step]
        current_goal = step.get("goal", "unknown")
        success_criteria = step.get("success_criteria", "task completed")

    # Gather recent tool results for reflection
    recent_results = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # Extract text content, skipping image data
            content = msg.content
            if isinstance(content, list):
                # Multimodal: extract text blocks, skip image_url blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = " ".join(text_parts) if text_parts else "[Screenshot captured]"
            elif isinstance(content, str) and content.startswith("data:image/"):
                content = "[Screenshot captured]"
            recent_results.append(str(content)[:300])
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            break
        if len(recent_results) >= 3:
            break

    # If no tool results, just pass through
    if not recent_results:
        logger.debug("[Reflector] no tool results to reflect on, continuing")
        return _advance_plan(state)

    # Quick rule-based check first (avoid unnecessary LLM calls)
    result_text = " ".join(recent_results).lower()

    # Check for obvious errors
    has_error = any(kw in result_text for kw in ["error", "failed", "timeout", "blocked"])
    # Check for obvious success
    has_success = any(kw in result_text for kw in ["success", "clicked", "wrote", "stored", "found"])

    # SDD Harness verification: if a Spec exists, run acceptance criteria
    harness_result = None
    metadata = state.get("metadata") or {}
    spec_data = metadata.get("current_spec")
    if spec_data and recent_results:
        try:
            from ....spec.harness import Harness
            from ....spec.schema import Spec
            spec = Spec.from_dict(spec_data)
            harness = Harness()
            harness_result = harness.verify(spec, recent_results)
            logger.info(
                f"[Reflector] Harness: {harness_result.pass_rate:.0%} "
                f"passed ({len(harness_result.critical_failures)} critical failures)"
            )
            # If Harness says all passed, trust it over keyword matching
            if harness_result.all_passed:
                has_success = True
                has_error = False
            elif harness_result.critical_failures:
                has_error = True
        except Exception as e:
            logger.error(f"[Reflector] Harness error: {e}")

    # If it's the last step or no plan, go to chat
    if not plan_steps or current_step >= len(plan_steps) - 1:
        if has_error and retry_count < MAX_RETRIES:
            logger.info(f"[Reflector] last step errored, retry {retry_count + 1}/{MAX_RETRIES}")
            return _retry_step(state)
        logger.info("[Reflector] plan complete, routing to chat")
        return {
            "next_worker": "chat",
            "task_complete": True,
            "current_step": current_step + 1,
        }

    # For middle steps with obvious success, advance without LLM call
    if has_success and not has_error:
        logger.info(f"[Reflector] step {current_step + 1} succeeded, advancing")
        return _advance_plan(state)

    # For ambiguous cases, use LLM reflection (but limit calls for efficiency)
    if retry_count >= MAX_RETRIES:
        logger.warning(f"[Reflector] max retries ({MAX_RETRIES}) reached, aborting step")
        return {
            "next_worker": "chat",
            "task_complete": False,
            "current_step": current_step + 1,
        }

    # LLM reflection
    system_content = REFLECTOR_SYSTEM.format(
        current_step_goal=current_goal,
        success_criteria=success_criteria,
        max_retries=MAX_RETRIES,
    )

    prompt_messages = [
        SystemMessage(content=system_content),
        HumanMessage(
            content=f"Tool results:\n"
            + "\n".join(f"- {r}" for r in recent_results)
            + f"\n\nRetry count: {retry_count}/{MAX_RETRIES}"
        ),
    ]

    try:
        response = await llm.ainvoke(prompt_messages)
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        decision_data = json.loads(content)
        decision = decision_data.get("decision", "success").lower()
        reason = decision_data.get("reason", "")

        logger.info(f"[Reflector] decision: {decision} (reason: {reason})")

        if decision == "success":
            return _advance_plan(state)
        elif decision == "retry":
            return _retry_step(state)
        elif decision == "abort":
            return {
                "next_worker": "chat",
                "task_complete": False,
                "current_step": current_step + 1,
            }
        else:  # "chat"
            return {
                "next_worker": "chat",
                "task_complete": True,
                "current_step": current_step + 1,
            }

    except Exception as e:
        logger.error(f"[Reflector] error: {e}")
        # Default: advance plan
        return _advance_plan(state)


def _advance_plan(state: AgentState) -> Dict[str, Any]:
    """Advance to the next step in the plan."""
    plan_steps = state.get("plan_steps") or []
    current_step = state.get("current_step", 0)

    next_step = current_step + 1

    if next_step >= len(plan_steps):
        # All steps complete
        return {
            "next_worker": "chat",
            "task_complete": True,
            "current_step": next_step,
            "retry_count": 0,
        }

    # More steps to go
    next_goal = plan_steps[next_step].get("goal", "?") if next_step < len(plan_steps) else "?"
    logger.info(f"[Reflector] advancing to step {next_step + 1}/{len(plan_steps)}: {next_goal}")
    return {
        "next_worker": "supervisor",
        "current_step": next_step,
        "retry_count": 0,
    }


def _retry_step(state: AgentState) -> Dict[str, Any]:
    """Retry the current step with incremented retry count."""
    retry_count = state.get("retry_count", 0)
    current_step = state.get("current_step", 0)
    plan_steps = state.get("plan_steps") or []

    logger.info(f"[Reflector] retrying step {current_step + 1} (attempt {retry_count + 1}/{MAX_RETRIES})")
    return {
        "next_worker": "tools",
        "retry_count": retry_count + 1,
        "current_step": current_step,
    }
