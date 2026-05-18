"""LangGraph StateGraph definition for the Memory-Aware Supervisor + Worker architecture.

Graph topology::

    memory_retrieval → supervisor ──→ chat_worker ──→ __end__
                         │
                         ├──→ tool_worker ──→ supervisor
                         │        │
                         │        └──→ [pending_approval] ──→ tool_worker
                         │
                         └──→ __end__

    ENTRY: memory_retrieval
"""

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel
from loguru import logger

from .state import AgentState
from .nodes.memory_retrieval import memory_retrieval_node
from .nodes.supervisor import supervisor_node
from .nodes.chat_worker import chat_node
from .nodes.tool_worker import tool_node
from .routers.route import route_supervisor


def build_graph(
    llm: BaseChatModel,
    system_prompt: str,
    tool_executor=None,
    tool_manager=None,
    high_risk_tools: list = None,
    human_in_the_loop: bool = True,
    memory_system=None,
) -> StateGraph:
    """Build and compile the multi-agent LangGraph.

    Args:
        llm: LangChain-compatible chat model (BaseChatModel).
        system_prompt: System prompt for the chat worker.
        tool_executor: MCP ToolExecutor instance (or None to disable tools).
        tool_manager: MCP ToolManager instance.
        high_risk_tools: List of tool names that require human approval.
        human_in_the_loop: Whether to enable approval flow.
        memory_system: MemorySystem instance for memory retrieval.

    Returns:
        A compiled ``StateGraph`` ready for invocation.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ───────────────────────────────────────
    graph.add_node(
        "memory_retrieval",
        _make_async_node(_run_memory_retrieval, memory_system=memory_system),
    )
    graph.add_node(
        "supervisor",
        _make_async_node(_run_supervisor, llm=llm),
    )
    graph.add_node(
        "chat",
        _make_async_node(_run_chat, llm=llm, system_prompt=system_prompt),
    )
    graph.add_node(
        "tools",
        _make_async_node(_run_tools, tool_executor=tool_executor, tool_manager=tool_manager,
                         high_risk_tools=high_risk_tools, human_in_the_loop=human_in_the_loop),
    )

    # ── Entry point ──────────────────────────────────────────
    graph.set_entry_point("memory_retrieval")

    # ── Memory retrieval → supervisor ────────────────────────
    graph.add_edge("memory_retrieval", "supervisor")

    # ── Conditional edge from supervisor ─────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "chat": "chat",
            "tools": "tools",
            "__end__": END,
        },
    )

    # ── Chat worker → END ────────────────────────────────────
    graph.add_edge("chat", END)

    # ── Tool worker → supervisor (loop back for next decision) ─
    graph.add_edge("tools", "supervisor")

    compiled = graph.compile()
    logger.info("LangGraph memory-aware graph compiled successfully")
    return compiled


# ── Adapter wrappers (inject closures for dependencies) ──────────


def _make_async_node(func, **kwargs):
    """Wrap an async node function with extra kwargs so LangGraph can call it correctly.

    LangGraph expects node functions to accept a single ``state`` argument.
    This wrapper creates an async function that partially applies the extra
    kwargs and properly awaits the result.
    """
    async def _node(state: AgentState) -> dict:
        return await func(state, **kwargs)
    return _node


async def _run_memory_retrieval(state: AgentState, memory_system=None) -> dict:
    return await memory_retrieval_node(state, memory_system=memory_system)


async def _run_supervisor(state: AgentState, llm: BaseChatModel) -> dict:
    return await supervisor_node(state, llm)


async def _run_chat(state: AgentState, llm: BaseChatModel, system_prompt: str) -> dict:
    return await chat_node(state, llm, system_prompt)


async def _run_tools(
    state: AgentState,
    tool_executor,
    tool_manager,
    high_risk_tools: list,
    human_in_the_loop: bool,
) -> dict:
    return await tool_node(
        state,
        tool_executor=tool_executor,
        tool_manager=tool_manager,
        high_risk_tools=high_risk_tools,
        human_in_the_loop=human_in_the_loop,
    )
