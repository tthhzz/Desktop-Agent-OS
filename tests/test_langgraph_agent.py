"""Tests for Phase 1: LangGraph multi-agent system.

Run with:
    conda activate open-vtuber
    python -m pytest tests/test_langgraph_agent.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Test 1: State definition ──────────────────────────────────


def test_agent_state_fields():
    """Verify AgentState has all required fields."""
    from open_llm_vtuber.agent.agents.langgraph_agent.state import AgentState

    state: AgentState = {
        "messages": [],
        "next_worker": "",
        "tool_calls": [],
        "tool_results": [],
        "images": None,
        "metadata": None,
        "pending_approval": None,
        "approval_response": None,
        "retrieved_memories": None,
        "skill_match": None,
    }
    assert state["messages"] == []
    assert state["next_worker"] == ""
    assert state["pending_approval"] is None


# ── Test 2: Supervisor routing logic ──────────────────────────


def test_route_supervisor():
    """Test the route_supervisor function returns correct worker names."""
    from open_llm_vtuber.agent.agents.langgraph_agent.routers.route import route_supervisor
    from open_llm_vtuber.agent.agents.langgraph_agent.state import AgentState

    assert route_supervisor({"next_worker": "chat"}) == "chat"
    assert route_supervisor({"next_worker": "tools"}) == "tools"
    assert route_supervisor({"next_worker": "__end__"}) == "__end__"

    # Unknown worker defaults to chat
    assert route_supervisor({"next_worker": "unknown"}) == "chat"


# ── Test 3: Graph compilation ─────────────────────────────────


def test_graph_builds():
    """Verify the LangGraph compiles without errors."""
    from open_llm_vtuber.agent.agents.langgraph_agent.graph import build_graph
    from langchain_openai import ChatOpenAI

    # Use a mock LLM (won't actually call the API)
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="sk-test", base_url="http://localhost:9999")
    graph = build_graph(
        llm=llm,
        system_prompt="You are a test assistant.",
    )
    assert graph is not None
    # Verify it's a compiled graph
    assert hasattr(graph, "ainvoke")


# ── Test 4: Config validation ──────────────────────────────────


def test_langgraph_agent_config():
    """Verify LangGraphAgentConfig parses correctly."""
    from open_llm_vtuber.config_manager.agent import LangGraphAgentConfig

    config = LangGraphAgentConfig(
        llm_provider="openai_compatible_llm",
        faster_first_response=True,
        segment_method="pysbd",
        use_mcpp=True,
        mcp_enabled_servers=["time", "ddg-search"],
        human_in_the_loop=True,
        high_risk_tools=["computer__click"],
        max_tool_rounds=5,
    )
    assert config.llm_provider == "openai_compatible_llm"
    assert config.human_in_the_loop is True
    assert config.max_tool_rounds == 5


# ── Test 5: AgentSettings with langgraph_agent ────────────────


def test_agent_settings_langgraph():
    """Verify AgentSettings accepts langgraph_agent config."""
    from open_llm_vtuber.config_manager.agent import (
        AgentSettings,
        LangGraphAgentConfig,
    )

    settings = AgentSettings(
        langgraph_agent=LangGraphAgentConfig(
            llm_provider="openai_compatible_llm",
            use_mcpp=True,
            mcp_enabled_servers=["time"],
        )
    )
    assert settings.langgraph_agent is not None
    assert settings.langgraph_agent.llm_provider == "openai_compatible_llm"


# ── Test 6: AgentConfig with langgraph_agent choice ────────────


def test_agent_config_langgraph_choice():
    """Verify AgentConfig accepts 'langgraph_agent' as conversation_agent_choice."""
    from open_llm_vtuber.config_manager.agent import (
        AgentConfig,
        AgentSettings,
        LangGraphAgentConfig,
    )
    from open_llm_vtuber.config_manager.stateless_llm import StatelessLLMConfigs

    config = AgentConfig(
        conversation_agent_choice="langgraph_agent",
        agent_settings=AgentSettings(
            langgraph_agent=LangGraphAgentConfig(
                llm_provider="openai_compatible_llm",
            )
        ),
        llm_configs=StatelessLLMConfigs(
            openai_compatible_llm={
                "base_url": "https://test.com/v1",
                "llm_api_key": "sk-test",
                "model": "test-model",
                "temperature": 1.0,
                "interrupt_method": "user",
            }
        ),
    )
    assert config.conversation_agent_choice == "langgraph_agent"


# ── Test 7: LangGraphAgent instantiation ───────────────────────


def test_langgraph_agent_init():
    """Verify LangGraphAgent can be instantiated with mock dependencies."""
    from open_llm_vtuber.agent.agents.langgraph_agent import LangGraphAgent

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.base_url = "http://localhost:9999/v1"
    mock_llm.temperature = 1.0
    mock_llm.client = MagicMock()
    mock_llm.client.api_key = "sk-test"

    agent = LangGraphAgent(
        llm=mock_llm,
        system="You are a test assistant.",
        live2d_model=None,
        use_mcpp=False,
    )
    assert agent is not None
    assert agent._system.startswith("You are a test assistant.")


# ── Test 8: Memory management ──────────────────────────────────


def test_memory_management():
    """Test LangGraphAgent memory add and history loading."""
    from open_llm_vtuber.agent.agents.langgraph_agent import LangGraphAgent

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.base_url = "http://localhost:9999/v1"
    mock_llm.temperature = 1.0
    mock_llm.client = MagicMock()
    mock_llm.client.api_key = "sk-test"

    agent = LangGraphAgent(
        llm=mock_llm,
        system="You are a test assistant.",
        live2d_model=None,
        use_mcpp=False,
    )

    # Add messages
    agent._add_message("Hello", "user")
    agent._add_message("Hi there!", "assistant")
    assert len(agent._memory) == 2
    assert agent._memory[0]["role"] == "user"
    assert agent._memory[1]["role"] == "assistant"

    # Skip memory
    agent._add_message("Skip this", "user", skip_memory=True)
    assert len(agent._memory) == 2

    # Don't add empty assistant messages
    agent._add_message("", "assistant")
    assert len(agent._memory) == 2


# ── Test 9: Interrupt handling ─────────────────────────────────


def test_interrupt_handling():
    """Test interrupt handling preserves partial response."""
    from open_llm_vtuber.agent.agents.langgraph_agent import LangGraphAgent

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.base_url = "http://localhost:9999/v1"
    mock_llm.temperature = 1.0
    mock_llm.client = MagicMock()
    mock_llm.client.api_key = "sk-test"

    agent = LangGraphAgent(
        llm=mock_llm,
        system="You are a test assistant.",
        live2d_model=None,
        use_mcpp=False,
        interrupt_method="user",
    )

    agent._add_message("Hello", "user")
    agent._add_message("I was saying that...", "assistant")
    agent.handle_interrupt("I was saying that")
    assert agent._memory[-1]["role"] == "user"
    assert "[Interrupted by user]" in agent._memory[-1]["content"]

    # Double interrupt should be ignored
    agent.handle_interrupt("Another interrupt")
    assert agent._interrupt_handled is True

    # Reset
    agent.reset_interrupt()
    assert agent._interrupt_handled is False


# ── Test 10: Input conversion ──────────────────────────────────


def test_input_to_state():
    """Test BatchInput → AgentState conversion."""
    from open_llm_vtuber.agent.agents.langgraph_agent import LangGraphAgent
    from open_llm_vtuber.agent.input_types import BatchInput, TextData, TextSource

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.base_url = "http://localhost:9999/v1"
    mock_llm.temperature = 1.0
    mock_llm.client = MagicMock()
    mock_llm.client.api_key = "sk-test"

    agent = LangGraphAgent(
        llm=mock_llm,
        system="You are a test assistant.",
        live2d_model=None,
        use_mcpp=False,
    )

    input_data = BatchInput(
        texts=[TextData(source=TextSource.INPUT, content="What's the weather?")],
    )

    state = agent._input_to_state(input_data)
    assert "messages" in state
    assert state["images"] is None
    assert state["pending_approval"] is None

    # The last message should be the user's input
    from langchain_core.messages import HumanMessage
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    assert any("weather" in m.content for m in user_msgs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
