"""Tool worker node — executes MCP tool calls and manages approval flow."""

import json
from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger

from ..state import AgentState


def _parse_tool_calls_from_ai(state: AgentState) -> list:
    """Extract pending tool calls from the last AI message.

    Looks at the last message in the state. If it contains tool_calls
    (LangChain format), return them; otherwise return an empty list.
    """
    messages = state.get("messages", [])
    if not messages:
        return []

    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return last_msg.tool_calls

    return []


async def tool_node(
    state: AgentState,
    tool_executor=None,
    tool_manager=None,
    high_risk_tools: list = None,
    human_in_the_loop: bool = True,
) -> Dict[str, Any]:
    """Execute tool calls requested by the LLM.

    This worker:
    1. Extracts tool calls from the last AI message
    2. Checks if any tool requires human approval
    3. If approval needed and not yet given → returns ``pending_approval``
    4. Executes tools via the ToolExecutor
    5. Returns tool results as messages
    """
    tool_calls = _parse_tool_calls_from_ai(state)

    if not tool_calls:
        logger.warning("[ToolWorker] invoked but no tool_calls found in state")
        return {"next_worker": "chat"}

    logger.info(f"[ToolWorker] ▶ executing {len(tool_calls)} tool call(s): {[tc['name'] for tc in tool_calls]}")

    high_risk = set(high_risk_tools or [])
    pending = state.get("pending_approval")

    # ── Human-in-the-Loop check ──────────────────────────────
    if human_in_the_loop:
        needs_approval = any(
            tc["name"] in high_risk for tc in tool_calls
        )

        if needs_approval and not pending:
            # First encounter with a risky tool → request approval
            risky_calls = [tc for tc in tool_calls if tc["name"] in high_risk]
            logger.info(f"[ToolWorker] ⏸ approval needed for: {[tc['name'] for tc in risky_calls]}")
            return {
                "pending_approval": {
                    "tool_calls": risky_calls,
                    "description": f"AI requests to use: {', '.join(tc['name'] for tc in risky_calls)}",
                },
                "next_worker": "tools",  # Stay in tools node, waiting for approval
            }

        if pending:
            approval = state.get("approval_response")
            if approval == "rejected":
                logger.info("[ToolWorker] ✗ user rejected tool execution")
                result_msgs = [
                    ToolMessage(
                        content=f"User rejected the tool call: {tc['name']}",
                        tool_call_id=tc["id"],
                    )
                    for tc in tool_calls
                    if tc["name"] in high_risk
                ]
                # Execute non-risky tools still
                safe_calls = [tc for tc in tool_calls if tc["name"] not in high_risk]
                if safe_calls and tool_executor:
                    safe_results = await _execute_tool_calls(safe_calls, tool_executor, tool_manager)
                    result_msgs.extend(safe_results)

                return {
                    "messages": result_msgs,
                    "pending_approval": None,
                    "approval_response": None,
                    "next_worker": "chat",
                }
            # approved → proceed to execution

    # ── Execute tools ─────────────────────────────────────────
    if not tool_executor:
        error_msgs = [
            ToolMessage(
                content="Error: ToolExecutor not available",
                tool_call_id=tc.get("id", "unknown"),
            )
            for tc in tool_calls
        ]
        return {
            "messages": error_msgs,
            "next_worker": "chat",
        }

    result_messages = await _execute_tool_calls(tool_calls, tool_executor, tool_manager)

    return {
        "messages": result_messages,
        "pending_approval": None,
        "approval_response": None,
        "next_worker": "supervisor",
    }


async def _execute_tool_calls(
    tool_calls: list,
    tool_executor,
    tool_manager,
) -> list:
    """Execute a list of tool calls and return ToolMessage results."""
    from .....mcpp.types import ToolCallObject, ToolCallFunctionObject

    result_messages = []

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_id = tc.get("id", f"tool_{tool_name}")
        tool_input = tc.get("args", tc.get("input", {}))
        logger.info(f"[ToolWorker] ⚙ {tool_name} | args: {json.dumps(tool_input, ensure_ascii=False)[:200]}")

        # Convert to ToolCallObject format expected by ToolExecutor
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                tool_input = {}

        tool_call_obj = ToolCallObject(
            id=tool_id,
            type="function",
            function=ToolCallFunctionObject(
                name=tool_name,
                arguments=json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input),
            ),
        )

        logger.info(f"Executing tool: {tool_name} (ID: {tool_id})")

        try:
            result_iter = tool_executor.execute_tools(
                tool_calls=[tool_call_obj],
                caller_mode="OpenAI",
            )
            final_result = None
            async for update in result_iter:
                if update.get("type") == "final_tool_results":
                    final_result = update
                # Individual tool_call_status updates are yielded separately

            if final_result and final_result.get("results"):
                for res in final_result["results"]:
                    content = res.get("content", "Tool executed with no output")
                    result_messages.append(
                        ToolMessage(
                            content=str(content),
                            tool_call_id=tool_id,
                        )
                    )
            else:
                result_messages.append(
                    ToolMessage(
                        content="Tool executed with no output",
                        tool_call_id=tool_id,
                    )
                )

        except Exception as e:
            logger.error(f"[ToolWorker] ✗ {tool_name} error: {e}")
            result_messages.append(
                ToolMessage(
                    content=f"Error executing tool {tool_name}: {e}",
                    tool_call_id=tool_id,
                )
            )

    return result_messages
