"""Unit tests for Flow DSL validation."""

import pytest
from pydantic import ValidationError

from symphony.flow.dsl import ActionType, FlowAction, FlowScript


class TestFlowAction:
    def test_navigate_requires_value(self):
        with pytest.raises(ValidationError, match="navigate requires"):
            FlowAction(action=ActionType.NAVIGATE)

    def test_navigate_valid(self):
        a = FlowAction(action=ActionType.NAVIGATE, value="http://localhost:3000")
        assert a.action == ActionType.NAVIGATE
        assert a.value == "http://localhost:3000"

    def test_click_requires_selector(self):
        with pytest.raises(ValidationError, match="click requires"):
            FlowAction(action=ActionType.CLICK)

    def test_click_valid(self):
        a = FlowAction(action=ActionType.CLICK, selector="#submit-btn")
        assert a.selector == "#submit-btn"

    def test_fill_requires_both(self):
        with pytest.raises(ValidationError, match="fill requires"):
            FlowAction(action=ActionType.FILL, selector="#email")
        with pytest.raises(ValidationError, match="fill requires"):
            FlowAction(action=ActionType.FILL, value="test@example.com")

    def test_fill_valid(self):
        a = FlowAction(
            action=ActionType.FILL,
            selector="#email",
            value="test@example.com",
        )
        assert a.value == "test@example.com"

    def test_press_requires_value(self):
        with pytest.raises(ValidationError, match="press requires"):
            FlowAction(action=ActionType.PRESS)

    def test_press_valid(self):
        a = FlowAction(action=ActionType.PRESS, value="Tab")
        assert a.value == "Tab"

    def test_wait_for_requires_selector(self):
        with pytest.raises(ValidationError, match="wait_for requires"):
            FlowAction(action=ActionType.WAIT_FOR)

    def test_assert_text_requires_both(self):
        with pytest.raises(ValidationError, match="assert_text requires"):
            FlowAction(action=ActionType.ASSERT_TEXT, selector="body")

    def test_assert_http_status_requires_value(self):
        with pytest.raises(ValidationError, match="assert_http_status requires"):
            FlowAction(action=ActionType.ASSERT_HTTP_STATUS)

    def test_assert_http_status_valid(self):
        a = FlowAction(
            action=ActionType.ASSERT_HTTP_STATUS,
            value="200",
            params={"url_pattern": "/api/contact"},
        )
        assert a.value == "200"

    def test_assert_banner_requires_value(self):
        with pytest.raises(ValidationError, match="assert_banner requires"):
            FlowAction(action=ActionType.ASSERT_BANNER)

    def test_other_requires_type(self):
        with pytest.raises(ValidationError, match="other_action_type"):
            FlowAction(action=ActionType.OTHER)

    def test_other_valid(self):
        a = FlowAction(
            action=ActionType.OTHER,
            other_action_type="drag_and_drop",
            selector="#item",
            params={"target": "#dropzone"},
        )
        assert a.other_action_type == "drag_and_drop"

    def test_scroll_no_requirements(self):
        a = FlowAction(action=ActionType.SCROLL, params={"direction": "down"})
        assert a.action == ActionType.SCROLL

    def test_default_timeout(self):
        a = FlowAction(action=ActionType.NAVIGATE, value="http://localhost")
        assert a.timeout_ms == 10_000


class TestFlowScript:
    def test_requires_at_least_one_action(self):
        with pytest.raises(ValidationError):
            FlowScript(name="empty", actions=[])

    def test_valid_script(self):
        script = FlowScript(
            name="login",
            actions=[
                FlowAction(action=ActionType.NAVIGATE, value="http://localhost/login"),
                FlowAction(action=ActionType.FILL, selector="#email", value="a@b.com"),
                FlowAction(action=ActionType.CLICK, selector="#submit"),
                FlowAction(action=ActionType.ASSERT_TEXT, selector=".welcome", value="Welcome"),
            ],
        )
        assert len(script.actions) == 4
        assert len(script.assertion_actions()) == 1

    def test_assertion_actions_filter(self):
        script = FlowScript(
            name="test",
            actions=[
                FlowAction(action=ActionType.NAVIGATE, value="http://localhost"),
                FlowAction(action=ActionType.ASSERT_TEXT, selector="body", value="hello"),
                FlowAction(action=ActionType.ASSERT_HTTP_STATUS, value="200"),
                FlowAction(action=ActionType.ASSERT_BANNER, value="Success"),
            ],
        )
        assert len(script.assertion_actions()) == 3
