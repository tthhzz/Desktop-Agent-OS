"""Task Decomposer — spec-based task decomposition replacing simple Planner steps.

Converts a Spec's decomposition into execution-ready step definitions
that are compatible with the existing tool_planner + tool_worker pipeline.
"""

from typing import Any, Dict, List
from loguru import logger

from .schema import Spec, DecompositionStep


def decompose_from_spec(spec: Spec) -> List[Dict[str, Any]]:
    """Convert a Spec's decomposition into plan_steps for the agent.

    Each step becomes a dict compatible with the planner's output format:
    - goal: step description
    - tool_hint: which tool to use
    - success_criteria: expected output description
    - on_failure: what to do if this step fails

    Args:
        spec: The Spec containing the decomposition.

    Returns:
        List of step dicts for the agent's plan_steps state.
    """
    if not spec.task.decomposition:
        # No decomposition = single step
        logger.debug("[Decomposer] no decomposition in spec, using single step")
        return [{
            "goal": spec.task.goal,
            "tool_hint": "auto",
            "success_criteria": _criteria_to_string(spec.task.acceptance_criteria),
            "on_failure": "retry",
        }]

    steps = []
    for i, decomp_step in enumerate(spec.task.decomposition):
        # Map acceptance criteria relevant to this step
        step_criteria = _match_criteria_to_step(decomp_step, spec.task.acceptance_criteria)

        step_dict = {
            "goal": decomp_step.description,
            "tool_hint": decomp_step.action,
            "success_criteria": decomp_step.expected_output or step_criteria,
            "on_failure": decomp_step.on_failure,
            "with_args": decomp_step.with_args,
        }
        steps.append(step_dict)

    logger.info(f"[Decomposer] {len(steps)} steps from spec")
    for i, s in enumerate(steps):
        logger.info(f"  Step {i + 1}: {s['goal'][:60]} (tool: {s['tool_hint']})")

    return steps


def _match_criteria_to_step(
    step: DecompositionStep,
    criteria: list,
) -> str:
    """Try to match acceptance criteria to a specific step.

    Simple heuristic: if criterion description mentions the step's action,
    it's relevant to that step.
    """
    relevant = []
    step_keywords = set(step.action.lower().split("_"))
    step_keywords.update(step.description.lower().split()[:5])

    for c in criteria:
        c_lower = c.description.lower()
        if any(kw in c_lower for kw in step_keywords if len(kw) > 3):
            relevant.append(c.description)

    return "; ".join(relevant) if relevant else "step completed successfully"


def _criteria_to_string(criteria: list) -> str:
    """Convert acceptance criteria to a summary string."""
    if not criteria:
        return "task completed"
    return "; ".join(c.description for c in criteria[:3])
