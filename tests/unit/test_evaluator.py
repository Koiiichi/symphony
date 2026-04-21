"""Unit tests for Reliability Evaluator."""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import List, Optional, Tuple

from symphony.evaluator.evaluator import (
    Artifact,
    AssertionResult,
    EvalReport,
    FailureReason,
    ReliabilityEvaluator,
    RunStatus,
    Severity,
)
from symphony.flow.dsl import ActionType, FlowAction
from symphony.flow.executor import ActionResult, Evidence, FlowResult


def _make_flow_result(
    name: str = "test_flow",
    passed: bool = True,
    actions: Optional[List[Tuple[str, bool, str]]] = None,
) -> FlowResult:
    """Helper to create FlowResult for testing.

    actions: list of (action_type, success, message) tuples.
    """
    results = []
    if actions:
        for action_type, success, message in actions:
            results.append(ActionResult(
                action=FlowAction(
                    action=ActionType(action_type),
                    value="http://localhost" if action_type == "navigate" else "test",
                    selector="#el" if action_type not in ("navigate", "press", "assert_http_status", "scroll") else None,
                ),
                success=success,
                message=message,
            ))
    return FlowResult(
        script_name=name,
        passed=passed,
        results=results,
        evidence=Evidence(screenshots=[Path("/tmp/test.png")] if passed else []),
    )


class TestReliabilityEvaluator:
    def test_all_pass(self):
        ev = ReliabilityEvaluator()
        flow = _make_flow_result(
            actions=[("navigate", True, "ok")],
        )
        report = ev.evaluate([flow])
        assert report.passed
        assert report.status is RunStatus.PASS
        assert len(report.failing_reasons) == 0

    def test_assertion_failure(self):
        ev = ReliabilityEvaluator()
        flow = _make_flow_result(
            passed=False,
            actions=[
                ("navigate", True, "ok"),
                ("assert_text", False, "Expected 'Welcome' not found"),
            ],
        )
        report = ev.evaluate([flow])
        assert not report.passed
        assert report.status is RunStatus.FAIL
        critical = [f for f in report.failing_reasons if f.severity == Severity.CRITICAL]
        assert len(critical) >= 1

    def test_missing_evidence_warning(self):
        ev = ReliabilityEvaluator()
        flow = FlowResult(
            script_name="bare_flow",
            passed=True,
            results=[],
            evidence=Evidence(),  # no screenshots
        )
        report = ev.evaluate([flow])
        warnings = [f for f in report.failing_reasons if f.severity == Severity.WARNING]
        assert any("No screenshots" in w.message for w in warnings)

    def test_http_expectation_failure(self):
        ev = ReliabilityEvaluator(require_http_expectations=True)
        flow = _make_flow_result(actions=[("navigate", True, "ok")])
        report = ev.evaluate(
            [flow],
            expected_http_statuses={"/api/contact": 200},
        )
        assert not report.passed

    def test_accessibility_checks(self):
        ev = ReliabilityEvaluator(require_accessibility=True)
        flow = _make_flow_result(actions=[("navigate", True, "ok")])
        report = ev.evaluate(
            [flow],
            accessibility_checks=[
                {"id": "tab_nav", "passed": False, "message": "Tab order broken", "critical": True},
            ],
        )
        assert not report.passed
        assert any("Tab order" in f.message for f in report.failing_reasons)

    def test_accessibility_non_critical_still_passes(self):
        ev = ReliabilityEvaluator(require_accessibility=True)
        flow = _make_flow_result(actions=[("navigate", True, "ok")])
        report = ev.evaluate(
            [flow],
            accessibility_checks=[
                {"id": "contrast", "passed": False, "message": "Low contrast", "critical": False},
            ],
        )
        # Non-critical a11y failure = warning, not critical → still passes
        assert report.passed

    def test_token_usage_in_report(self):
        ev = ReliabilityEvaluator()
        flow = _make_flow_result(actions=[("navigate", True, "ok")])
        report = ev.evaluate([flow], token_usage={"total": 5000})
        assert report.token_usage == {"total": 5000}

    def test_planner_confidence_in_report(self):
        ev = ReliabilityEvaluator()
        flow = _make_flow_result(actions=[("navigate", True, "ok")])
        report = ev.evaluate([flow], planner_confidence=0.85)
        assert report.planner_confidence == 0.85

    def test_network_trace_artifact_is_reported(self):
        ev = ReliabilityEvaluator()
        flow = FlowResult(
            script_name="trace_flow",
            passed=True,
            results=[ActionResult(
                action=FlowAction(action=ActionType.NAVIGATE, value="http://localhost"),
                success=True,
                message="ok",
            )],
            evidence=Evidence(
                screenshots=[Path("/tmp/test.png")],
                network_traces=[{"path": "/tmp/network_trace.json", "entries": 2}],
            ),
        )
        report = ev.evaluate([flow])
        assert any(
            a.type == "network_trace" and a.path == "/tmp/network_trace.json"
            for a in report.artifacts
        )

    def test_to_dict_roundtrip(self):
        ev = ReliabilityEvaluator()
        flow = _make_flow_result(
            passed=False,
            actions=[("assert_text", False, "fail")],
        )
        report = ev.evaluate([flow])
        d = report.to_dict()
        assert d["status"] == "fail"
        assert isinstance(d["failing_reasons"], list)
        assert isinstance(d["assertion_results"], list)
        assert isinstance(d["artifacts"], list)


class TestEvalReport:
    def test_pass_report(self):
        r = EvalReport(status=RunStatus.PASS)
        assert r.passed is True

    def test_fail_report(self):
        r = EvalReport(status=RunStatus.FAIL)
        assert r.passed is False

    def test_empty_to_dict(self):
        r = EvalReport(status=RunStatus.PASS)
        d = r.to_dict()
        assert d["status"] == "pass"
        assert d["failing_reasons"] == []


class TestFalsePositiveGuardrails:
    """Regression: ensure no false successes when evidence is incomplete."""

    def test_no_results_with_missing_evidence_not_pass_silently(self):
        ev = ReliabilityEvaluator()
        flow = FlowResult(
            script_name="empty",
            passed=True,
            results=[],
            evidence=Evidence(),
        )
        report = ev.evaluate([flow])
        # Should warn about missing evidence
        assert len(report.failing_reasons) > 0

    def test_mixed_pass_fail_actions_fails_overall(self):
        ev = ReliabilityEvaluator()
        flow = _make_flow_result(
            passed=False,
            actions=[
                ("navigate", True, "ok"),
                ("assert_text", True, "found text"),
                ("assert_http_status", False, "Expected 200 not found"),
            ],
        )
        report = ev.evaluate([flow])
        assert not report.passed
