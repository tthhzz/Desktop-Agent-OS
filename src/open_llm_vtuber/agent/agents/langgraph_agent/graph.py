"""LangGraph StateGraph definition for the Memory-Aware Supervisor + Worker architecture.

Graph topology (Phase 5.2 — with Planner + Reflector)::

    memory_retrieval → supervisor ──→ chat_worker ──→ __end__
                         │
                         ├──→ planner ──→ tool_planner ──→ tool_worker ──→ reflector
                         │                                              │
                         │                              ┌─────────────────┘
                         │                              ↓
                         │                         supervisor (loop)
                         │                              ↑
                         │                     ┌─── retry (max 3)
                         │                     ↓
                         │                  reflector
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
from .nodes.tool_planner import tool_planner_node
from .nodes.tool_worker import tool_node
from .nodes.planner import planner_node
from .nodes.reflector import reflector_node
from .routers.route import route_supervisor
from .utils.status_emitter import get_status_emitter


def build_graph(
    llm: BaseChatModel,
    system_prompt: str,
    tool_executor=None,
    tool_manager=None,
    high_risk_tools: list = None,
    human_in_the_loop: bool = True,
    memory_system=None,
    lc_tools: list = None,
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
        lc_tools: OpenAI-format tool schemas for bind_tools().

    Returns:
        A compiled ``StateGraph`` ready for invocation.
    """
    lc_tools = lc_tools or []
    graph = StateGraph(AgentState)

    # ── Register nodes ───────────────────────────────────────
    graph.add_node(
        "memory_retrieval",
        _make_async_node(_run_memory_retrieval, memory_system=memory_system),
    )
    graph.add_node(
        "supervisor",
        _make_async_node(_run_supervisor, llm=llm, tool_schemas=lc_tools),
    )
    graph.add_node(
        "chat",
        _make_async_node(_run_chat, llm=llm, system_prompt=system_prompt),
    )
    graph.add_node(
        "planner",
        _make_async_node(_run_spec_generator, llm=llm, tool_schemas=lc_tools),
    )
    graph.add_node(
        "tool_planner",
        _make_async_node(_run_tool_planner, llm=llm, lc_tools=lc_tools),
    )
    graph.add_node(
        "tools",
        _make_async_node(_run_tools, tool_executor=tool_executor, tool_manager=tool_manager,
                         high_risk_tools=high_risk_tools, human_in_the_loop=human_in_the_loop),
    )
    graph.add_node(
        "reflector",
        _make_async_node(_run_reflector, llm=llm),
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
            "tools": "planner",
            "__end__": END,
        },
    )

    # ── Chat worker → END ────────────────────────────────────
    graph.add_edge("chat", END)

    # ── Planner → tool_planner ───────────────────────────────
    graph.add_edge("planner", "tool_planner")

    # ── Tool planner → tool worker ───────────────────────────
    graph.add_edge("tool_planner", "tools")

    # ── Tool worker → reflector ──────────────────────────────
    graph.add_edge("tools", "reflector")

    # ── Reflector → supervisor (loop back for next decision) ─
    graph.add_edge("reflector", "supervisor")

    compiled = graph.compile()
    logger.info("LangGraph memory-aware graph compiled (with planner + reflector)")
    return compiled


# ── Adapter wrappers (inject closures for dependencies) ──────────


def _make_async_node(func, **kwargs):
    """Wrap an async node function with extra kwargs so LangGraph can call it correctly.

    LangGraph expects node functions to accept a single ``state`` argument.
    This wrapper creates an async function that partially applies the extra
    kwargs and properly awaits the result.

    Also emits status updates before and after each node execution.
    """
    node_name = kwargs.get("_node_name", func.__name__.replace("_run_", ""))

    async def _node(state: AgentState) -> dict:
        emitter = get_status_emitter()
        emitter.emit(node=node_name, status="executing")

        try:
            result = await func(state, **{k: v for k, v in kwargs.items() if k != "_node_name"})

            # Emit step progress if we have plan info
            plan_steps = state.get("plan_steps") or []
            current_step = state.get("current_step", 0)
            if plan_steps:
                emitter.emit(
                    node=node_name,
                    status="completed",
                    step=current_step + 1,
                    total_steps=len(plan_steps),
                )
            else:
                emitter.emit(node=node_name, status="completed")

            return result
        except Exception as e:
            emitter.emit(node=node_name, status="failed", detail=str(e))
            raise

    return _node


async def _run_memory_retrieval(state: AgentState, memory_system=None) -> dict:
    return await memory_retrieval_node(state, memory_system=memory_system)


async def _run_supervisor(state: AgentState, llm: BaseChatModel, tool_schemas: list = None) -> dict:
    return await supervisor_node(state, llm, tool_schemas=tool_schemas)


async def _run_chat(state: AgentState, llm: BaseChatModel, system_prompt: str) -> dict:
    return await chat_node(state, llm, system_prompt)


async def _run_planner(state: AgentState, llm: BaseChatModel, tool_schemas: list = None) -> dict:
    # Use SDD Spec Generator instead of simple planner
    from ...spec.spec_generator import spec_generator_node
    return await spec_generator_node(state, llm, tool_schemas=tool_schemas)


async def _run_tool_planner(state: AgentState, llm: BaseChatModel, lc_tools: list = None) -> dict:
    return await tool_planner_node(state, llm, lc_tools=lc_tools or [])


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


async def _run_reflector(state: AgentState, llm: BaseChatModel) -> dict:
    return await reflector_node(state, llm=llm)
