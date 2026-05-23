"""Tool worker node — executes MCP tool calls and manages approval flow."""

import json
from typing import Dict, Any, Optional, List, Union
from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger

from ..state import AgentState


def _is_base64_image(content: str) -> bool:
    """Check if a string is a base64-encoded image (data URI or raw base64 PNG/JPEG)."""
    if not isinstance(content, str):
        return False
    return content.startswith("data:image/") or content.startswith("iVBOR") or content.startswith("/9j/")


def _extract_image_from_json(content: str) -> tuple:
    """Try to extract a base64 image from a JSON tool result.

    Some tools (like screen_capture_and_parse) return JSON with an
    "annotated_image" field containing base64 data. We extract it so
    the VLM can see the image.

    Returns:
        (text_summary, image_url) tuple, or (None, None) if no image found.
    """
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            return None, None

        # Check common image field names
        image_b64 = None
        text_parts = []

        for key in ("annotated_image", "image", "screenshot", "base64_image"):
            if key in data and isinstance(data[key], str) and _is_base64_image(data[key]):
                image_b64 = data[key]
                # Remove the image from the dict to create a text summary
                summary_data = {k: v for k, v in data.items() if k != key}
                text_parts.append(json.dumps(summary_data, ensure_ascii=False)[:500])
                break

        if image_b64:
            image_url = image_b64 if image_b64.startswith("data:image/") else f"data:image/png;base64,{image_b64}"
            text_summary = text_parts[0] if text_parts else "[Screenshot with annotations]"
            return text_summary, image_url

    except (json.JSONDecodeError, TypeError):
        pass

    return None, None


def _build_tool_message_content(content: str, tool_name: str = "") -> Union[str, List[dict]]:
    """Build ToolMessage content that preserves image data for downstream nodes.

    If the content contains base64 image data (direct or embedded in JSON),
    returns a structured content list with both a text summary and the
    image_url, so downstream nodes can pass the image to a VLM.

    Returns:
        Either a plain string (no image) or a list of content blocks.
    """
    # Case 1: Content is directly a base64 image
    if _is_base64_image(content):
        image_url = content if content.startswith("data:image/") else f"data:image/png;base64,{content}"
        return [
            {"type": "text", "text": f"[Screenshot captured via {tool_name}]"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    # Case 2: Content is JSON with embedded base64 image (e.g. screen_capture_and_parse)
    text_summary, image_url = _extract_image_from_json(content)
    if image_url:
        return [
            {"type": "text", "text": text_summary},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    # Case 3: Plain text, no image
    return content


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
                    raw_content = res.get("content", "Tool executed with no output")
                    # Build multimodal-aware content (preserves base64 images)
                    msg_content = _build_tool_message_content(str(raw_content), tool_name)
                    result_messages.append(
                        ToolMessage(
                            content=msg_content,
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
