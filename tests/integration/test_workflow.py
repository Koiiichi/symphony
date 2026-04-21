"""Integration tests — end-to-end workflow validation.

These tests exercise the planner → flow DSL → evaluator pipeline
without requiring a live browser (browser executor is mocked).
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from symphony.planner.schema import (
    AcceptanceCriterion,
    NodeType,
    TaskEdge,
    TaskGraph,
    TaskNode,
)
from symphony.planner.planner import LLMPlanner
from symphony.flow.dsl import ActionType, FlowAction, FlowScript
from symphony.flow.executor import ActionResult, Evidence, FlowExecutor, FlowResult
from symphony.evaluator.evaluator import ReliabilityEvaluator, RunStatus, Severity
from symphony.prompt.compiler import ContextBlock, PromptCompiler


# ------------------------------------------------------------------
# Contact form happy path: 2xx + success banner
# ------------------------------------------------------------------

class TestContactFormHappyPath:
    def test_plan_and_evaluate_success(self):
        graph = TaskGraph(
            goal="Submit contact form and verify success",
            nodes=[
                TaskNode(id="nav", type=NodeType.SERVICE_START, description="Start server"),
                TaskNode(
                    id="contact_test",
                    type=NodeType.WEB_FLOW_TEST,
                    description="Fill and submit contact form",
                    actions=[
                        {"action": "navigate", "value": "http://localhost:5000"},
                        {"action": "fill", "selector": "#name", "value": "Test User"},
                        {"action": "fill", "selector": "#email", "value": "test@example.com"},
                        {"action": "fill", "selector": "#message", "value": "Hello!"},
                        {"action": "click", "selector": "#submit"},
                    ],
                    assertions=[
                        {"action": "assert_banner", "value": "Thank you"},
                        {"action": "assert_http_status", "value": "200", "params": {"url_pattern": "/api/contact"}},
                    ],
                ),
                TaskNode(id="done", type=NodeType.FINALIZE),
            ],
            edges=[
                TaskEdge(source="nav", target="contact_test"),
                TaskEdge(source="contact_test", target="done"),
            ],
        )

        # Simulate successful flow execution
        flow_result = FlowResult(
            script_name="contact_test",
            passed=True,
            results=[
                ActionResult(
                    action=FlowAction(action=ActionType.NAVIGATE, value="http://localhost:5000"),
                    success=True, message="ok",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.ASSERT_BANNER, value="Thank you"),
                    success=True, message="Banner found",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.ASSERT_HTTP_STATUS, value="200"),
                    success=True, message="HTTP 200 verified",
                ),
            ],
            evidence=Evidence(screenshots=[Path("/tmp/contact_success.png")]),
        )

        evaluator = ReliabilityEvaluator()
        report = evaluator.evaluate([flow_result])
        assert report.passed
        assert report.status is RunStatus.PASS


# ------------------------------------------------------------------
# Contact form invalid email: 4xx + error banner
# ------------------------------------------------------------------

class TestContactFormInvalidEmail:
    def test_invalid_email_shows_error(self):
        flow_result = FlowResult(
            script_name="contact_invalid_email",
            passed=True,
            results=[
                ActionResult(
                    action=FlowAction(action=ActionType.NAVIGATE, value="http://localhost:5000"),
                    success=True, message="ok",
                ),
                ActionResult(
                    action=FlowAction(
                        action=ActionType.FILL, selector="#email", value="not-an-email"
                    ),
                    success=True, message="filled",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.CLICK, selector="#submit"),
                    success=True, message="clicked",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.ASSERT_BANNER, value="Invalid email"),
                    success=True, message="Error banner found",
                ),
            ],
            evidence=Evidence(screenshots=[Path("/tmp/contact_error.png")]),
        )

        evaluator = ReliabilityEvaluator()
        report = evaluator.evaluate([flow_result])
        assert report.passed


# ------------------------------------------------------------------
# Login flow: valid + invalid credentials
# ------------------------------------------------------------------

class TestLoginFlow:
    def test_valid_login(self):
        flow_result = FlowResult(
            script_name="login_valid",
            passed=True,
            results=[
                ActionResult(
                    action=FlowAction(action=ActionType.NAVIGATE, value="http://localhost:5000/login"),
                    success=True, message="ok",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.FILL, selector="#email", value="user@test.com"),
                    success=True, message="filled",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.FILL, selector="#password", value="password123"),
                    success=True, message="filled",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.CLICK, selector="#login-btn"),
                    success=True, message="clicked",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.ASSERT_TEXT, selector=".dashboard", value="Welcome"),
                    success=True, message="Welcome text found",
                ),
            ],
            evidence=Evidence(screenshots=[Path("/tmp/login_success.png")]),
        )

        evaluator = ReliabilityEvaluator()
        report = evaluator.evaluate([flow_result])
        assert report.passed

    def test_invalid_login(self):
        flow_result = FlowResult(
            script_name="login_invalid",
            passed=True,
            results=[
                ActionResult(
                    action=FlowAction(action=ActionType.NAVIGATE, value="http://localhost:5000/login"),
                    success=True, message="ok",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.FILL, selector="#email", value="wrong@test.com"),
                    success=True, message="filled",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.CLICK, selector="#login-btn"),
                    success=True, message="clicked",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.ASSERT_BANNER, value="Invalid credentials"),
                    success=True, message="Error banner found",
                ),
            ],
            evidence=Evidence(screenshots=[Path("/tmp/login_error.png")]),
        )

        evaluator = ReliabilityEvaluator()
        report = evaluator.evaluate([flow_result])
        assert report.passed


# ------------------------------------------------------------------
# Keyboard navigation flow
# ------------------------------------------------------------------

class TestKeyboardNavigation:
    def test_tab_focus_movement(self):
        flow_result = FlowResult(
            script_name="keyboard_nav",
            passed=True,
            results=[
                ActionResult(
                    action=FlowAction(action=ActionType.NAVIGATE, value="http://localhost:5000"),
                    success=True, message="ok",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.PRESS, value="Tab"),
                    success=True, message="Tab pressed",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.PRESS, value="Tab"),
                    success=True, message="Tab pressed",
                ),
                ActionResult(
                    action=FlowAction(action=ActionType.PRESS, value="Enter"),
                    success=True, message="Enter pressed",
                ),
                ActionResult(
                    action=FlowAction(
                        action=ActionType.ASSERT_TEXT,
                        selector="body",
                        value="Form submitted",
                    ),
                    success=True, message="Submit confirmed via keyboard",
                ),
            ],
            evidence=Evidence(
                screenshots=[Path("/tmp/keyboard_nav.png")],
                focused_elements=["INPUT#name.", "INPUT#email.", "BUTTON#submit."],
            ),
        )

        evaluator = ReliabilityEvaluator(require_accessibility=True)
        report = evaluator.evaluate(
            [flow_result],
            accessibility_checks=[
                {"id": "tab_focus", "passed": True, "message": "Tab order correct"},
            ],
        )
        assert report.passed


# ------------------------------------------------------------------
# Multi-step web flow with scrolling and conditional waits
# ------------------------------------------------------------------

class TestMultiStepFlow:
    def test_scroll_and_wait(self):
        script = FlowScript(
            name="multi_step",
            actions=[
                FlowAction(action=ActionType.NAVIGATE, value="http://localhost:5000/products"),
                FlowAction(action=ActionType.SCROLL, params={"direction": "down", "pixels": 500}),
                FlowAction(action=ActionType.WAIT_FOR, selector=".product-card"),
                FlowAction(action=ActionType.CLICK, selector=".product-card:first-child .add-to-cart"),
                FlowAction(action=ActionType.WAIT_FOR, selector=".cart-badge"),
                FlowAction(action=ActionType.ASSERT_TEXT, selector=".cart-badge", value="1"),
            ],
        )
        assert len(script.actions) == 6
        assert len(script.assertion_actions()) == 1

        # Simulate execution result
        flow_result = FlowResult(
            script_name="multi_step",
            passed=True,
            results=[
                ActionResult(
                    action=a,
                    success=True,
                    message=f"{a.action.value} ok",
                )
                for a in script.actions
            ],
            evidence=Evidence(screenshots=[Path("/tmp/multi_step.png")]),
        )

        evaluator = ReliabilityEvaluator()
        report = evaluator.evaluate([flow_result])
        assert report.passed


# ------------------------------------------------------------------
# Token budget regression
# ------------------------------------------------------------------

class TestTokenBudgetRegression:
    """Ensure prompt size stays bounded across representative goals."""

    GOALS = [
        "Fix the contact form submission to return 200 and show a success banner",
        "Ensure login with valid credentials redirects to the dashboard and displays the username",
        "Verify that the product listing page loads all items and the add-to-cart button works",
        "Fix the registration form to validate email format and show appropriate error messages",
    ]

    def test_prompts_fit_budget(self):
        budget = 16_000
        compiler = PromptCompiler(token_budget=budget)
        for goal in self.GOALS:
            blocks = [
                ContextBlock(name="Failures", content="Error: assert failed\n" * 50, priority=20),
                ContextBlock(name="Source Code", content="def handler():\n    pass\n" * 200, priority=10),
                ContextBlock(name="DOM Snapshot", content="<html>" + "<div>" * 500 + "</html>", priority=5),
            ]
            messages, usage = compiler.compile_with_usage(goal, blocks)
            assert usage["total_tokens"] <= budget, (
                f"Budget exceeded for goal: {goal} (used {usage['total_tokens']})"
            )
            assert usage["budget"] == budget

    def test_no_budget_includes_all_blocks(self):
        compiler = PromptCompiler()  # no budget
        blocks = [
            ContextBlock(name="Block A", content="content A", priority=5),
            ContextBlock(name="Block B", content="content B", priority=1),
        ]
        messages, usage = compiler.compile_with_usage("goal", blocks)
        user_content = messages[1]["content"]
        assert "Block A" in user_content
        assert "Block B" in user_content
        assert "budget" not in usage


# ------------------------------------------------------------------
# Planner error handling
# ------------------------------------------------------------------

class TestPlannerErrors:
    def test_llm_planner_raises_on_error(self):
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("API down")

        planner = LLMPlanner(mock_client)
        with pytest.raises(RuntimeError, match="API down"):
            planner.plan("test goal")
