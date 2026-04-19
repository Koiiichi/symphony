"""Reliability Evaluator — first-class pass/fail gate for Symphony runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from symphony.flow.executor import FlowResult


# ------------------------------------------------------------------
# Report types
# ------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class FailureReason:
    id: str
    severity: Severity
    message: str
    node_id: str | None = None
    evidence_path: str | None = None


@dataclass
class AssertionResult:
    assertion_id: str
    passed: bool
    message: str = ""
    node_id: str | None = None


@dataclass
class Artifact:
    type: str  # "screenshot", "dom_snapshot", "network_trace", "log"
    path: str
    node_id: str | None = None


@dataclass
class EvalReport:
    """Machine-readable evaluation report."""

    status: str  # "pass" or "fail"
    failing_reasons: list[FailureReason] = field(default_factory=list)
    assertion_results: list[AssertionResult] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    planner_confidence: float | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failing_reasons": [
                {"id": f.id, "severity": f.severity.value,
                 "message": f.message, "node_id": f.node_id,
                 "evidence_path": f.evidence_path}
                for f in self.failing_reasons
            ],
            "assertion_results": [
                {"assertion_id": a.assertion_id, "passed": a.passed,
                 "message": a.message, "node_id": a.node_id}
                for a in self.assertion_results
            ],
            "artifacts": [
                {"type": a.type, "path": a.path, "node_id": a.node_id}
                for a in self.artifacts
            ],
            "token_usage": self.token_usage,
            "planner_confidence": self.planner_confidence,
        }


# ------------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------------

class ReliabilityEvaluator:
    """Evaluates a Symphony run against acceptance criteria."""

    def __init__(
        self,
        *,
        require_all_assertions: bool = True,
        require_http_expectations: bool = True,
        require_accessibility: bool = False,
    ):
        self._require_all_assertions = require_all_assertions
        self._require_http = require_http_expectations
        self._require_a11y = require_accessibility

    def evaluate(
        self,
        flow_results: list[FlowResult],
        *,
        expected_http_statuses: dict[str, int] | None = None,
        accessibility_checks: list[dict[str, Any]] | None = None,
        token_usage: dict[str, int] | None = None,
        planner_confidence: float | None = None,
    ) -> EvalReport:
        """Run all evaluation checks and produce a report."""
        failures: list[FailureReason] = []
        assertion_results: list[AssertionResult] = []
        artifacts: list[Artifact] = []

        # ---- Flow assertion checks ----
        for fr in flow_results:
            # Collect artifacts
            for ss in fr.evidence.screenshots:
                artifacts.append(Artifact(
                    type="screenshot", path=str(ss), node_id=fr.script_name
                ))
            for dom in fr.evidence.dom_snapshots:
                artifacts.append(Artifact(
                    type="dom_snapshot", path=dom[:100], node_id=fr.script_name
                ))

            for ar in fr.results:
                a_id = f"{fr.script_name}:{ar.action.action.value}:{ar.action.selector or 'global'}"
                assertion_results.append(AssertionResult(
                    assertion_id=a_id,
                    passed=ar.success,
                    message=ar.message,
                    node_id=fr.script_name,
                ))
                if not ar.success:
                    failures.append(FailureReason(
                        id=a_id,
                        severity=Severity.CRITICAL,
                        message=ar.message,
                        node_id=fr.script_name,
                    ))

        # ---- HTTP status checks ----
        if self._require_http and expected_http_statuses:
            for url_pattern, expected_status in expected_http_statuses.items():
                found = False
                for fr in flow_results:
                    for ar in fr.results:
                        if (
                            ar.action.action.value == "assert_http_status"
                            and ar.success
                        ):
                            found = True
                            break
                if not found:
                    fid = f"http_check:{url_pattern}:{expected_status}"
                    failures.append(FailureReason(
                        id=fid,
                        severity=Severity.CRITICAL,
                        message=f"Expected HTTP {expected_status} for {url_pattern} not verified",
                    ))
                    assertion_results.append(AssertionResult(
                        assertion_id=fid, passed=False,
                        message=f"HTTP {expected_status} not found for {url_pattern}",
                    ))

        # ---- Accessibility checks ----
        if self._require_a11y and accessibility_checks:
            for check in accessibility_checks:
                check_id = check.get("id", "a11y_check")
                passed = check.get("passed", False)
                assertion_results.append(AssertionResult(
                    assertion_id=check_id,
                    passed=passed,
                    message=check.get("message", ""),
                ))
                if not passed:
                    failures.append(FailureReason(
                        id=check_id,
                        severity=Severity.WARNING
                        if not check.get("critical", False)
                        else Severity.CRITICAL,
                        message=check.get("message", "Accessibility check failed"),
                    ))

        # ---- Evidence completeness ----
        for fr in flow_results:
            if not fr.evidence.screenshots:
                failures.append(FailureReason(
                    id=f"evidence_missing:{fr.script_name}",
                    severity=Severity.WARNING,
                    message=f"No screenshots captured for flow '{fr.script_name}'",
                    node_id=fr.script_name,
                ))

        # ---- Determine overall status ----
        critical_failures = [
            f for f in failures if f.severity == Severity.CRITICAL
        ]
        status = "fail" if critical_failures else "pass"

        return EvalReport(
            status=status,
            failing_reasons=failures,
            assertion_results=assertion_results,
            artifacts=artifacts,
            token_usage=token_usage or {},
            planner_confidence=planner_confidence,
        )
