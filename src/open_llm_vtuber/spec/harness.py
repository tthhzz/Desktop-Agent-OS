"""Harness — acceptance criteria verification against execution results.

The Harness is the SDD "test runner" for agent tasks. After each step
(or after all steps), it verifies whether the execution results satisfy
the spec's acceptance criteria.

This replaces the Reflector's simple error/success keyword matching with
structured, spec-driven verification.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from .schema import Spec, AcceptanceCriterion


@dataclass
class CriterionResult:
    """Result of checking a single acceptance criterion."""
    criterion: str  # The criterion description
    passed: bool
    check_type: str
    detail: str = ""  # Why it passed/failed


@dataclass
class HarnessResult:
    """Result of running the full Harness against execution results."""
    all_passed: bool
    results: List[CriterionResult] = field(default_factory=list)
    critical_failures: List[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)


class Harness:
    """Verifies execution results against Spec acceptance criteria.

    Usage:
        harness = Harness()
        result = harness.verify(spec, tool_results)
        if not result.all_passed:
            # Handle failures
    """

    def verify(
        self,
        spec: Spec,
        tool_results: List[str],
    ) -> HarnessResult:
        """Run all acceptance criteria against the execution results.

        Args:
            spec: The specification containing acceptance criteria.
            tool_results: List of tool result strings from execution.

        Returns:
            HarnessResult with pass/fail for each criterion.
        """
        if not spec.task.acceptance_criteria:
            # No criteria = nothing to verify, treat as passed
            return HarnessResult(all_passed=True, results=[])

        combined_result = "\n".join(tool_results).lower()
        criterion_results = []
        critical_failures = []

        for criterion in spec.task.acceptance_criteria:
            result = self._check_criterion(criterion, combined_result, tool_results)
            criterion_results.append(result)

            if not result.passed and criterion.critical:
                critical_failures.append(criterion.description)

        all_passed = all(r.passed for r in criterion_results)

        harness_result = HarnessResult(
            all_passed=all_passed,
            results=criterion_results,
            critical_failures=critical_failures,
        )

        logger.info(
            f"[Harness] {'✓ ALL PASSED' if all_passed else '✗ FAILED'} "
            f"({harness_result.pass_rate:.0%}) "
            f"| critical failures: {critical_failures}"
        )

        return harness_result

    def _check_criterion(
        self,
        criterion: AcceptanceCriterion,
        combined_result: str,
        tool_results: List[str],
    ) -> CriterionResult:
        """Check a single acceptance criterion against results."""
        check_type = criterion.check_type
        desc = criterion.description.lower()
        expected = str(criterion.expected).lower() if criterion.expected else ""

        if check_type == "contains":
            # Check if expected text appears in results
            if expected:
                passed = expected in combined_result
                return CriterionResult(
                    criterion=criterion.description,
                    passed=passed,
                    check_type=check_type,
                    detail=f"Expected '{expected}' {'found' if passed else 'NOT found'} in results",
                )
            # If no expected, check description keywords
            keywords = [w for w in desc.split() if len(w) > 3]
            found = any(kw in combined_result for kw in keywords[:3])
            return CriterionResult(
                criterion=criterion.description,
                passed=found,
                check_type=check_type,
                detail=f"Keywords from criterion {'found' if found else 'NOT found'}",
            )

        elif check_type == "not_contains":
            # Check that expected text does NOT appear
            if expected:
                passed = expected not in combined_result
                return CriterionResult(
                    criterion=criterion.description,
                    passed=passed,
                    check_type=check_type,
                    detail=f"'{expected}' {'NOT present (OK)' if passed else 'FOUND (FAIL)'}",
                )
            # Check for common dangerous patterns
            dangerous = ["error", "failed", "denied", "blocked", "timeout"]
            found_any = any(d in combined_result for d in dangerous)
            return CriterionResult(
                criterion=criterion.description,
                passed=not found_any,
                check_type=check_type,
                detail=f"Dangerous patterns {'NOT found (OK)' if not found_any else 'found (FAIL)'}",
            )

        elif check_type == "returns_success":
            # Check that tool execution didn't error
            error_keywords = ["error", "failed", "exception", "timeout", "blocked"]
            has_error = any(ek in combined_result for ek in error_keywords)
            success_keywords = ["success", "completed", "clicked", "wrote", "stored", "found"]
            has_success = any(sk in combined_result for sk in success_keywords)

            passed = has_success and not has_error
            return CriterionResult(
                criterion=criterion.description,
                passed=passed,
                check_type=check_type,
                detail=f"Success indicators: {has_success}, Error indicators: {has_error}",
            )

        elif check_type == "matches_schema":
            # Check if result contains structured data (simplified)
            # For now, just check if result is non-empty and not an error
            passed = bool(combined_result.strip()) and "error" not in combined_result[:50].lower()
            return CriterionResult(
                criterion=criterion.description,
                passed=passed,
                check_type=check_type,
                detail=f"Result is {'valid' if passed else 'empty or error'}",
            )

        else:  # "custom" or unknown
            # Fallback: check for error-free execution
            passed = "error" not in combined_result[:100].lower()
            return CriterionResult(
                criterion=criterion.description,
                passed=passed,
                check_type=check_type,
                detail=f"Custom check: {'no obvious errors' if passed else 'potential error detected'}",
            )
