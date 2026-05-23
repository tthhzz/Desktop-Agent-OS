"""Spec module — Spec-Driven Development (SDD) engine for autonomous agents.

Implements:
- OpenSpec schema: structured task specification with goals, constraints, interfaces, and acceptance criteria
- Spec Generator: LLM-powered spec generation from natural language tasks
- Spec Validator: checks spec completeness and consistency
- Harness: acceptance criteria verification against execution results
- Task Decomposer: spec-based decomposition replacing simple planner steps
"""

from .schema import Spec, TaskSpec, InterfaceSpec, AcceptanceCriterion
from .spec_generator import spec_generator_node
from .spec_validator import validate_spec
from .harness import Harness, HarnessResult, CriterionResult
from .task_decomposer import decompose_from_spec

__all__ = [
    "Spec",
    "TaskSpec",
    "InterfaceSpec",
    "AcceptanceCriterion",
    "spec_generator_node",
    "validate_spec",
    "Harness",
    "HarnessResult",
    "CriterionResult",
    "decompose_from_spec",
]
