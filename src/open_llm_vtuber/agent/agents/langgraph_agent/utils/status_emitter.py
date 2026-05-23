"""Agent Status Emitter — pushes execution state updates via callback.

Each graph node can emit status updates (current node, plan progress,
tool status) that get forwarded to the frontend via WebSocket.

This is the backend for Feature 5.5 (Real-time Agent Dashboard).
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional
from loguru import logger


class AgentStatusEmitter:
    """Emits agent execution status updates via a callback.

    The callback typically sends a WebSocket message with type
    "agent_state_update" to the frontend for real-time visualization.
    """

    def __init__(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self._callback = callback
        self._start_time: Optional[float] = None
        self._current_plan: Optional[List[Dict[str, Any]]] = None

    def set_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set or replace the status callback."""
        self._callback = callback

    def emit(
        self,
        node: str,
        status: str = "executing",
        step: int = 0,
        total_steps: int = 0,
        plan: Optional[List[str]] = None,
        current_tool: str = "",
        detail: str = "",
    ) -> None:
        """Emit a status update.

        Args:
            node: Current graph node name (e.g., "supervisor", "tool_planner").
            status: Node status — "executing", "completed", "failed", "waiting".
            step: Current step number (for multi-step plans).
            total_steps: Total number of steps in the plan.
            plan: List of step descriptions (sent after planning).
            current_tool: Name of the tool currently being executed.
            detail: Additional detail message.
        """
        if not self._callback:
            return

        elapsed = 0.0
        if self._start_time:
            elapsed = round(time.monotonic() - self._start_time, 1)

        update = {
            "type": "agent_state_update",
            "node": node,
            "status": status,
            "step": step,
            "total_steps": total_steps,
            "current_tool": current_tool,
            "detail": detail,
            "elapsed_seconds": elapsed,
        }

        if plan:
            update["plan"] = plan
            self._current_plan = [{"description": s} for s in plan] if isinstance(plan[0], str) else plan

        try:
            self._callback(update)
        except Exception as e:
            logger.debug(f"[StatusEmitter] callback error: {e}")

    def start_session(self):
        """Mark the start of an agent execution session."""
        self._start_time = time.monotonic()

    def end_session(self):
        """Mark the end of an agent execution session."""
        self._start_time = None
        self._current_plan = None


# Global emitter instance — shared across graph nodes
_global_emitter = AgentStatusEmitter()


def get_status_emitter() -> AgentStatusEmitter:
    """Get the global status emitter instance."""
    return _global_emitter
