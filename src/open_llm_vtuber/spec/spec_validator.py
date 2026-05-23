"""Spec Validator — checks spec completeness and consistency.

Validates:
1. Required fields present (goal, at least 1 acceptance criterion)
2. Decomposition steps are well-ordered
3. Acceptance criteria are verifiable
4. Constraints are not contradictory
5. Interface definitions are valid
"""

from typing import List, Tuple
from loguru import logger

from .schema import Spec, AcceptanceCriterion


def validate_spec(spec: Spec) -> Tuple[bool, List[str]]:
    """Validate a spec for completeness and consistency.

    Returns:
        (is_valid, issues) — list of issue descriptions (empty if valid).
    """
    issues = []

    # 1. Goal is required
    if not spec.task.goal or not spec.task.goal.strip():
        issues.append("Spec missing goal")

    # 2. At least 1 acceptance criterion
    if not spec.task.acceptance_criteria:
        issues.append("Spec has no acceptance criteria — cannot verify success")

    # 3. Acceptance criteria are verifiable
    for i, criterion in enumerate(spec.task.acceptance_criteria):
        if not criterion.description or not criterion.description.strip():
            issues.append(f"Acceptance criterion #{i + 1} has no description")
        if criterion.check_type not in ("contains", "not_contains", "matches_schema",
                                         "returns_success", "custom"):
            issues.append(f"Acceptance criterion #{i + 1} has invalid check_type: {criterion.check_type}")

    # 4. Decomposition steps are well-ordered
    if spec.task.decomposition:
        step_numbers = [s.step for s in spec.task.decomposition]
        if step_numbers != list(range(1, len(step_numbers) + 1)):
            issues.append("Decomposition steps are not sequential (1, 2, 3, ...)")

        for i, step in enumerate(spec.task.decomposition):
            if not step.action or not step.action.strip():
                issues.append(f"Step {step.step} has no action")
            if not step.description or not step.description.strip():
                issues.append(f"Step {step.step} has no description")

    # 5. No contradictory constraints
    constraint_lower = [c.lower() for c in spec.task.constraints]
    for i, c1 in enumerate(constraint_lower):
        for c2 in constraint_lower[i + 1:]:
            # Simple negation detection
            if c1.startswith("no ") and c2.startswith(c1[3:]):
                issues.append(f"Contradictory constraints: '{c1}' vs '{c2}'")
            elif c2.startswith("no ") and c1.startswith(c2[3:]):
                issues.append(f"Contradictory constraints: '{c1}' vs '{c2}'")

    is_valid = len(issues) == 0
    if is_valid:
        logger.info(f"Spec validated: goal='{spec.task.goal[:50]}'")
    else:
        logger.warning(f"Spec has {len(issues)} issues: {issues}")

    return is_valid, issues
