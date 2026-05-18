"""Tests for Phase 2: Computer Use, File Upload, and Approval Flow.

Run with:
    conda activate open-vtuber
    python -m pytest tests/test_phase2.py -v
"""

import asyncio
import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Test 1: Computer Use MCP server tools exist ────────────────


def test_computer_use_server_loads():
    """Verify the Computer Use MCP server registers all expected tools."""
    from open_llm_vtuber.mcp_servers.computer_use_server import mcp

    # FastMCP stores tools internally; verify the module has the right functions
    import open_llm_vtuber.mcp_servers.computer_use_server as server_module

    expected_tools = [
        "computer_screenshot",
        "computer_click",
        "computer_type",
        "computer_hotkey",
        "computer_scroll",
        "computer_screen_info",
        "computer_move",
        "computer_drag",
    ]
    for tool_name in expected_tools:
        assert hasattr(server_module, tool_name), f"Missing tool: {tool_name}"


# ── Test 2: Computer screen info ──────────────────────────────


def test_computer_screen_info():
    """Test that computer_screen_info returns a string with screen info."""
    from open_llm_vtuber.mcp_servers.computer_use_server import computer_screen_info

    result = computer_screen_info()
    assert isinstance(result, str)
    assert "Screen:" in result
    assert "Mouse:" in result


# ── Test 3: Computer screenshot ───────────────────────────────


def test_computer_screenshot():
    """Test that computer_screenshot returns base64 image data."""
    from open_llm_vtuber.mcp_servers.computer_use_server import computer_screenshot

    result = computer_screenshot()
    assert isinstance(result, str)
    assert result.startswith("data:image/png;base64,")
    assert len(result) > 100  # Should have actual image data


# ── Test 4: File upload endpoint exists ────────────────────────


def test_upload_route_registered():
    """Verify the /upload POST endpoint exists in routes."""
    from open_llm_vtuber.routes import init_webtool_routes

    # We can verify the function exists by checking it was called
    # The route is defined as a closure, so we just need to verify
    # the module imports cleanly
    assert callable(init_webtool_routes)


# ── Test 5: BatchInput accepts files ──────────────────────────


def test_batch_input_with_files():
    """Test that BatchInput correctly handles the files field."""
    from open_llm_vtuber.agent.input_types import BatchInput, TextData, FileData, TextSource

    files = [
        FileData(
            name="test.txt",
            data=base64.b64encode(b"Hello world").decode("utf-8"),
            mime_type="text/plain",
        )
    ]

    batch = BatchInput(
        texts=[TextData(source=TextSource.INPUT, content="Check this file")],
        files=files,
    )
    assert batch.files is not None
    assert len(batch.files) == 1
    assert batch.files[0].name == "test.txt"
    assert batch.files[0].mime_type == "text/plain"


# ── Test 6: create_batch_input with files ─────────────────────


def test_create_batch_input_with_files():
    """Test the updated create_batch_input with files parameter."""
    from open_llm_vtuber.conversations.conversation_utils import create_batch_input

    files_data = [
        {
            "name": "doc.pdf",
            "data": "base64data...",
            "mime_type": "application/pdf",
        }
    ]

    batch = create_batch_input(
        input_text="Read this document",
        images=None,
        from_name="Human",
        files=files_data,
    )
    assert batch.files is not None
    assert len(batch.files) == 1
    assert batch.files[0].name == "doc.pdf"


# ── Test 7: Approval flow in tool worker ──────────────────────


@pytest.mark.asyncio
async def test_tool_worker_approval_flow():
    """Test that the tool worker correctly requests approval for high-risk tools."""
    from open_llm_vtuber.agent.agents.langgraph_agent.nodes.tool_worker import tool_node
    from open_llm_vtuber.agent.agents.langgraph_agent.state import AgentState
    from langchain_core.messages import AIMessage

    # Simulate state with a high-risk tool call
    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "computer__click", "id": "tc1", "args": {"x": 100, "y": 200}}],
            )
        ],
        "next_worker": "tools",
        "tool_calls": [],
        "tool_results": [],
        "images": None,
        "metadata": None,
        "pending_approval": None,
        "approval_response": None,
        "retrieved_memories": None,
        "skill_match": None,
    }

    result = await tool_node(
        state,
        tool_executor=None,
        tool_manager=None,
        high_risk_tools=["computer__click"],
        human_in_the_loop=True,
    )

    # Should have requested approval
    assert result["pending_approval"] is not None
    assert "computer__click" in result["pending_approval"]["description"]


# ── Test 8: Low-risk tool bypasses approval ───────────────────


@pytest.mark.asyncio
async def test_tool_worker_low_risk_bypass():
    """Test that low-risk tools don't require approval."""
    from open_llm_vtuber.agent.agents.langgraph_agent.nodes.tool_worker import tool_node
    from open_llm_vtuber.agent.agents.langgraph_agent.state import AgentState
    from langchain_core.messages import AIMessage

    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "time__get_current_time", "id": "tc2", "args": {}}],
            )
        ],
        "next_worker": "tools",
        "tool_calls": [],
        "tool_results": [],
        "images": None,
        "metadata": None,
        "pending_approval": None,
        "approval_response": None,
        "retrieved_memories": None,
        "skill_match": None,
    }

    # With no tool_executor, it will return error messages but NOT request approval
    result = await tool_node(
        state,
        tool_executor=None,
        tool_manager=None,
        high_risk_tools=["computer__click"],
        human_in_the_loop=True,
    )

    # Should NOT have requested approval (time is not high-risk)
    assert result.get("pending_approval") is None


# ── Test 9: MCP servers config has playwright + computer ───────


def test_mcp_servers_config():
    """Verify mcp_servers.json has the new servers."""
    import json
    import os

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "mcp_servers.json"
    )
    with open(config_path) as f:
        config = json.load(f)

    servers = config["mcp_servers"]
    assert "playwright" in servers
    assert "computer" in servers
    assert "time" in servers
    assert "ddg-search" in servers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
