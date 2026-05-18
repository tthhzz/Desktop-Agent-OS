"""AgentState definition for the LangGraph multi-agent graph."""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State flowing through the LangGraph supervisor-worker graph.

    Key design choices:
    - ``messages`` uses LangGraph's ``add_messages`` reducer so that each
      node can append messages without overwriting.
    - ``next_worker`` is set by the Supervisor node and read by the
      conditional edge to route to the correct worker.
    - ``tool_calls`` / ``tool_results`` carry tool invocation data between
      the tool worker and the supervisor.
    - ``pending_approval`` holds an action awaiting user confirmation
      (Human-in-the-Loop).  When non-None, the graph yields an approval
      request and waits for the caller to provide ``approval_response``.
    """

    # ── Conversation ──────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Routing ──────────────────────────────────────────────
    next_worker: str

    # ── Tool calling ─────────────────────────────────────────
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    # ── Multi-modal input ────────────────────────────────────
    images: Optional[List[Dict[str, Any]]]

    # ── Metadata ─────────────────────────────────────────────
    metadata: Optional[Dict[str, Any]]

    # ── Human-in-the-Loop ────────────────────────────────────
    pending_approval: Optional[Dict[str, Any]]
    approval_response: Optional[str]  # "approved" | "rejected"

    # ── Memory (Phase 3 integration) ─────────────────────────
    retrieved_memories: Optional[List[Dict[str, Any]]]
    skill_match: Optional[Dict[str, Any]]
