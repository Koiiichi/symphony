"""LLM Planner — emits a typed TaskGraph from a goal description."""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

from symphony.planner.schema import (
    TaskGraph,
)
from symphony.prompt.compiler import ContextBlock, PromptCompiler

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Planner prompt
# ------------------------------------------------------------------

PLANNER_SYSTEM = """\
You are a task planner for Symphony, a web reliability tool.
Given a goal, emit a JSON TaskGraph that orchestrates the work.

TaskGraph JSON schema:
{
  "version": "2.0",
  "goal": "<string>",
  "constraints": ["<string>", ...],
  "nodes": [
    {
      "id": "<unique_id>",
      "type": "stack_detect|service_start|ui_discovery|web_flow_test|api_check|code_patch|retest|finalize|other",
      "description": "<what this step does>",
      "config": {},
      "retry": {"max_attempts": 3, "backoff_seconds": 1.0},
      "preconditions": ["<description>", ...],
      "actions": [<FlowAction objects for web_flow_test nodes — see FlowAction schema below>],
      "assertions": [<assertion FlowAction objects for web_flow_test nodes — see FlowAction schema below>],
      "evidence_requirements": {"screenshot": true, "network_trace": false, "dom_snapshot": false, "focused_element_trace": false},
      "other_task_type": "<required if type is 'other'>"
    }
  ],
  "edges": [{"source": "<node_id>", "target": "<node_id>", "condition": "<optional>"}],
  "acceptance_criteria": [{"id": "<id>", "description": "<text>", "required": true}]
}

FlowAction schema (use for actions and assertions arrays):
{
  "action": "<navigate|scroll|click|fill|press|wait_for|assert_text|assert_http_status|assert_banner|other>",
  "selector": "<CSS selector — required for click, fill, wait_for, assert_text>",
  "value": "<string — required for navigate (URL), fill (text), press (key), assert_text (expected text), assert_http_status (status code as string), assert_banner (expected text)>",
  "timeout_ms": 10000,
  "params": {},
  "other_action_type": "<required only when action is 'other'>"
}

IMPORTANT: Use "action" (not "type") as the field name for the action type.

Node type guide:
- stack_detect: identify project stack/framework
- service_start: start backend/frontend servers
- ui_discovery: explore the UI to find elements/pages
- web_flow_test: execute browser flow with actions and assertions
- api_check: verify API endpoint behavior
- code_patch: modify source code to fix a failure
- retest: re-run a previous test after a patch
- finalize: cleanup and report generation
- other: any task that doesn't fit the above — MUST include "other_task_type"

Rules:
- Emit ONLY valid JSON. No markdown, no commentary.
- Every web_flow_test MUST have actions and assertions.
- Use edges to express dependencies between nodes.
- Prefer smaller, focused nodes over large monolithic ones.
- Use the exact CSS selectors from the provided HTML context (ids, names, classes). Do NOT guess selectors.
"""


# ------------------------------------------------------------------
# LLM Planner
# ------------------------------------------------------------------

class LLMPlanner:
    """Plans task graphs using an LLM."""

    def __init__(
        self,
        llm_client: "LLMClient",  # noqa: F821 — avoids circular import
        *,
        token_budget: Optional[int] = None,
    ):
        self._client = llm_client
        self._compiler = PromptCompiler(
            token_budget=token_budget, system_prompt=PLANNER_SYSTEM
        )

    def plan(
        self,
        goal: str,
        *,
        project_context: str = "",
        prior_failures: str = "",
        project_path: Optional[str] = None,
    ) -> Tuple[TaskGraph, Optional[float], int]:
        """Generate a TaskGraph for *goal*.

        Returns (graph, confidence, token_estimate) where confidence is
        the model's self-reported confidence (0-1) or None, and
        token_estimate is the estimated prompt tokens used.

        Raises on LLM or parsing failure — no silent fallback.
        """
        blocks: list = []
        if project_context:
            blocks.append(ContextBlock(
                name="Project Context", content=project_context, priority=10
            ))
        if prior_failures:
            blocks.append(ContextBlock(
                name="Prior Failures", content=prior_failures, priority=20
            ))

        messages, usage = self._compiler.compile_with_usage(goal, blocks)
        prompt_tokens = usage.get("total_tokens", 0)
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        user_messages = [m for m in messages if m["role"] != "system"]

        raw = self._client.complete(
            user_messages, system=system, response_schema=TaskGraph
        ).strip()
        # Strip markdown fences if present (fallback for providers that ignore schema)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        data = json.loads(raw)
        graph = TaskGraph.model_validate(data)
        confidence = data.get("confidence")
        logger.info("LLM planner produced graph with %d nodes", len(graph.nodes))
        return graph, confidence, prompt_tokens
