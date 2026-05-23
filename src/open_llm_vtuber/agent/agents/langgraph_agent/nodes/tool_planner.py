"""Tool Planner node — asks the LLM to decide which tools to call using bind_tools().

This node sits between the supervisor and the tool_worker. When the supervisor
routes to "tools", the tool_planner uses the LLM with bound tools to generate
actual tool calls (AIMessage with tool_calls). The tool_worker then executes them.

Supports multimodal tool results: screenshots and other base64 images are passed
to the VLM as image_url content blocks instead of being truncated as text.
"""

from typing import Dict, Any, List, Union
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from loguru import logger

from ..state import AgentState

TOOL_PLANNER_SYSTEM = """You are a tool planner. Given the conversation and the available tools, decide which tool(s) to call and with what arguments.

Rules:
- Only call tools that are relevant to the user's request.
- Provide appropriate arguments for each tool call.
- If no tool is relevant, respond with text saying you cannot help with that specific task.
- For time-related questions, use the time tool.
- For search questions, use the search tool.
- For file/terminal operations, use the terminal tool.
- For screen/computer operations, use the computer tool.
- For browser automation, use the playwright tool.
- For memory operations (remembering/recalling), use the memory tool.
- When you receive a screenshot, ANALYZE IT VISUALLY to understand what's on screen, then decide the next tool action based on the visual content.
"""


def _tool_message_to_human_content(msg: ToolMessage) -> Union[str, List[dict]]:
    """Convert a ToolMessage to content suitable for HumanMessage.

    If the ToolMessage contains multimodal content (e.g. a screenshot),
    preserve the image_url blocks so the VLM can see the image.
    For plain text results, return a truncated text summary.
    """
    content = msg.content

    # Multimodal content (list of content blocks with image_url)
    if isinstance(content, list):
        # Reconstruct: prepend tool context, keep image blocks
        result = [{"type": "text", "text": f"[Tool result from {msg.name or 'tool'}]"}]
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "image_url":
                    result.append(block)
                elif block.get("type") == "text":
                    result.append(block)
        return result

    # Plain string content — check for embedded base64 image
    if isinstance(content, str) and (
        content.startswith("data:image/") or content.startswith("iVBOR")
    ):
        # This is an image that wasn't wrapped into multimodal format
        image_url = content if content.startswith("data:image/") else f"data:image/png;base64,{content}"
        return [
            {"type": "text", "text": f"[Tool result: Screenshot captured via {msg.name or 'tool'}]"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    # Regular text — truncate to avoid token bloat
    if isinstance(content, str):
        text = content[:500] + ("..." if len(content) > 500 else "")
        return f"[Tool result: {text}]"

    # Fallback
    return f"[Tool result: {str(content)[:300]}]"


async def tool_planner_node(
    state: AgentState,
    llm: BaseChatModel,
    lc_tools: List,
) -> Dict[str, Any]:
    """Ask the LLM (with bound tools) to generate tool calls.

    Returns an update with an AIMessage containing tool_calls if the LLM
    decided to call tools, or a plain text response if not.
    """
    messages = state.get("messages", [])

    # Build prompt messages
    prompt_messages = [SystemMessage(content=TOOL_PLANNER_SYSTEM)]

    # Include recent messages (skip system messages, keep last 10)
    recent = [m for m in messages if m.type != "system"][-10:]
    for msg in recent:
        # Convert ToolMessage results to VLM-friendly format
        if isinstance(msg, ToolMessage):
            human_content = _tool_message_to_human_content(msg)
            prompt_messages.append(HumanMessage(content=human_content))
        else:
            prompt_messages.append(msg)

    if not lc_tools:
        logger.warning("[ToolPlanner] No tools available, falling back to chat")
        return {
            "messages": [AIMessage(content="I don't have any tools available to help with that.")],
            "next_worker": "__end__",
        }

    # Bind tools to the LLM and invoke
    llm_with_tools = llm.bind_tools(lc_tools)

    try:
        response = await llm_with_tools.ainvoke(prompt_messages)

        # Check if the LLM generated tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [tc.get("name", "?") for tc in response.tool_calls]
            logger.info(f"[ToolPlanner] ▶ planned tools: {tool_names}")
            return {
                "messages": [response],
                "next_worker": "tools",
            }
        else:
            # LLM decided not to call any tools — generate a text response instead
            text = response.content or "I couldn't find a suitable tool for that request."
            logger.info(f"[ToolPlanner] ✗ no tool calls, text fallback ({len(text)} chars)")
            return {
                "messages": [AIMessage(content=text)],
                "next_worker": "__end__",
            }

    except Exception as e:
        logger.error(f"[ToolPlanner] error: {e}")
        return {
            "messages": [AIMessage(content=f"[Error planning tools: {e}]")],
            "next_worker": "__end__",
        }
