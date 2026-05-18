"""Routing logic for the Supervisor conditional edge."""

from typing import Literal
from loguru import logger

from ..state import AgentState


def route_supervisor(state: AgentState) -> Literal["chat", "tools", "__end__"]:
    """Determine which worker the supervisor wants to route to.

    This reads ``state["next_worker"]`` set by the supervisor node.
    """
    next_worker = state.get("next_worker", "__end__")

    if next_worker in ("chat", "tools", "__end__"):
        return next_worker

    logger.warning(f"Unknown next_worker '{next_worker}', falling back to 'chat'")
    return "chat"
