"""Flow DSL — structured browser actions the LLM can emit."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    NAVIGATE = "navigate"
    SCROLL = "scroll"
    CLICK = "click"
    FILL = "fill"
    PRESS = "press"
    WAIT_FOR = "wait_for"
    ASSERT_TEXT = "assert_text"
    ASSERT_HTTP_STATUS = "assert_http_status"
    ASSERT_BANNER = "assert_banner"
    OTHER = "other"


class FlowAction(BaseModel):
    """A single browser action in the Flow DSL."""

    action: ActionType
    selector: Optional[str] = None
    value: Optional[str] = None
    timeout_ms: int = Field(default=10_000, ge=0)
    params: Dict[str, Any] = Field(default_factory=dict)

    # For OTHER action type — caller describes what this does
    other_action_type: Optional[str] = None

    @model_validator(mode="after")
    def _validate_required_fields(self) -> "FlowAction":
        a = self.action
        if a == ActionType.NAVIGATE:
            if not self.value:
                raise ValueError("navigate requires 'value' (URL)")
        elif a == ActionType.CLICK:
            if not self.selector:
                raise ValueError("click requires 'selector'")
        elif a == ActionType.FILL:
            if not self.selector or self.value is None:
                raise ValueError("fill requires 'selector' and 'value'")
        elif a == ActionType.PRESS:
            if not self.value:
                raise ValueError("press requires 'value' (key name)")
        elif a == ActionType.WAIT_FOR:
            if not self.selector:
                raise ValueError("wait_for requires 'selector'")
        elif a == ActionType.ASSERT_TEXT:
            if not self.selector or not self.value:
                raise ValueError(
                    "assert_text requires 'selector' and 'value'"
                )
        elif a == ActionType.ASSERT_HTTP_STATUS:
            if self.value is None:
                raise ValueError(
                    "assert_http_status requires 'value' (status code)"
                )
        elif a == ActionType.ASSERT_BANNER:
            if not self.value:
                raise ValueError(
                    "assert_banner requires 'value' (expected text)"
                )
        elif a == ActionType.OTHER:
            if not self.other_action_type:
                raise ValueError(
                    "other action must specify other_action_type"
                )
        return self


class FlowScript(BaseModel):
    """An ordered sequence of FlowActions constituting a browser test flow."""

    name: str
    description: str = ""
    actions: List[FlowAction] = Field(min_length=1)

    def assertion_actions(self) -> List[FlowAction]:
        """Return only the assertion-type actions."""
        return [
            a
            for a in self.actions
            if a.action
            in {
                ActionType.ASSERT_TEXT,
                ActionType.ASSERT_HTTP_STATUS,
                ActionType.ASSERT_BANNER,
            }
        ]
