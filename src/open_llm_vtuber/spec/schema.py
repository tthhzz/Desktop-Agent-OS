"""OpenSpec Schema — structured task specification for SDD.

A Spec defines:
- goal: What the task should accomplish
- constraints: Boundaries the agent must respect
- interfaces: Input/output contracts
- acceptance_criteria: Verifiable conditions for task completion
- decomposition: Ordered steps with tool hints and expected outcomes
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InterfaceSpec:
    """Input/output interface contract."""
    type: str = "object"
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)


@dataclass
class AcceptanceCriterion:
    """A verifiable acceptance condition."""
    description: str
    check_type: str = "contains"  # contains | not_contains | matches_schema | returns_success | custom
    expected: Any = None
    critical: bool = True  # If True, failing this criterion = task failure


@dataclass
class DecompositionStep:
    """A single step in the task decomposition."""
    step: int
    action: str  # Tool name or "llm_extract" / "respond"
    description: str  # What this step does
    with_args: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""  # Brief description of expected result
    on_failure: str = "retry"  # retry | skip | abort


@dataclass
class TaskSpec:
    """The core task specification."""
    goal: str
    constraints: List[str] = field(default_factory=list)
    interfaces: Dict[str, InterfaceSpec] = field(default_factory=dict)
    acceptance_criteria: List[AcceptanceCriterion] = field(default_factory=list)
    decomposition: List[DecompositionStep] = field(default_factory=list)


@dataclass
class Spec:
    """Complete specification for a task.

    This is the SDD "contract" that drives agent behavior:
    - The agent generates a Spec before executing complex tasks
    - Each step is executed against the Spec's decomposition
    - The Harness verifies results against acceptance_criteria
    - The Spec can be stored as a Skill template for reuse
    """
    spec_version: str = "1.0"
    task: TaskSpec = field(default_factory=TaskSpec)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage/transmission."""
        return {
            "spec_version": self.spec_version,
            "task": {
                "goal": self.task.goal,
                "constraints": self.task.constraints,
                "interfaces": {
                    k: {
                        "type": v.type,
                        "properties": v.properties,
                        "required": v.required,
                    }
                    for k, v in self.task.interfaces.items()
                },
                "acceptance_criteria": [
                    {
                        "description": c.description,
                        "check_type": c.check_type,
                        "expected": c.expected,
                        "critical": c.critical,
                    }
                    for c in self.task.acceptance_criteria
                ],
                "decomposition": [
                    {
                        "step": s.step,
                        "action": s.action,
                        "description": s.description,
                        "with_args": s.with_args,
                        "expected_output": s.expected_output,
                        "on_failure": s.on_failure,
                    }
                    for s in self.task.decomposition
                ],
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Spec":
        """Deserialize from dict."""
        task_data = data.get("task", {})

        interfaces = {}
        for k, v in task_data.get("interfaces", {}).items():
            interfaces[k] = InterfaceSpec(
                type=v.get("type", "object"),
                properties=v.get("properties", {}),
                required=v.get("required", []),
            )

        criteria = []
        for c in task_data.get("acceptance_criteria", []):
            criteria.append(AcceptanceCriterion(
                description=c.get("description", ""),
                check_type=c.get("check_type", "contains"),
                expected=c.get("expected"),
                critical=c.get("critical", True),
            ))

        steps = []
        for s in task_data.get("decomposition", []):
            steps.append(DecompositionStep(
                step=s.get("step", 0),
                action=s.get("action", ""),
                description=s.get("description", ""),
                with_args=s.get("with_args", {}),
                expected_output=s.get("expected_output", ""),
                on_failure=s.get("on_failure", "retry"),
            ))

        return cls(
            spec_version=data.get("spec_version", "1.0"),
            task=TaskSpec(
                goal=task_data.get("goal", ""),
                constraints=task_data.get("constraints", []),
                interfaces=interfaces,
                acceptance_criteria=criteria,
                decomposition=steps,
            ),
            metadata=data.get("metadata", {}),
        )
