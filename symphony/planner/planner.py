"""LLM Planner — emits a typed TaskGraph from a goal description."""

from __future__ import annotations

import json
import logging
from typing import Any

from symphony.planner.schema import (
    AcceptanceCriterion,
    NodeType,
    TaskEdge,
    TaskGraph,
    TaskNode,
    RetryPolicy,
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
      "actions": [<FlowAction objects for web_flow_test nodes>],
      "assertions": [<assertion objects for web_flow_test nodes>],
      "evidence_requirements": {"screenshot": true, "network_trace": false, "dom_snapshot": false, "focused_element_trace": false},
      "other_task_type": "<required if type is 'other'>"
    }
  ],
  "edges": [{"source": "<node_id>", "target": "<node_id>", "condition": "<optional>"}],
  "acceptance_criteria": [{"id": "<id>", "description": "<text>", "required": true}]
}

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
"""


# ------------------------------------------------------------------
# Heuristic fallback planner
# ------------------------------------------------------------------

def _heuristic_plan(goal: str, project_path: str | None = None) -> TaskGraph:
    """Minimal fallback plan when LLM planner fails."""
    nodes = [
        TaskNode(
            id="detect",
            type=NodeType.STACK_DETECT,
            description="Detect project stack and framework",
        ),
        TaskNode(
            id="start",
            type=NodeType.SERVICE_START,
            description="Start application services",
        ),
        TaskNode(
            id="discover",
            type=NodeType.UI_DISCOVERY,
            description="Discover UI structure and available pages",
        ),
        TaskNode(
            id="test",
            type=NodeType.WEB_FLOW_TEST,
            description=f"Test web flow for: {goal}",
            actions=[{"action": "navigate", "value": "http://localhost:3000"}],
            assertions=[{"action": "assert_text", "selector": "body", "value": ""}],
            evidence_requirements={"screenshot": True},
        ),
        TaskNode(
            id="finalize",
            type=NodeType.FINALIZE,
            description="Generate report",
        ),
    ]
    edges = [
        TaskEdge(source="detect", target="start"),
        TaskEdge(source="start", target="discover"),
        TaskEdge(source="discover", target="test"),
        TaskEdge(source="test", target="finalize"),
    ]
    return TaskGraph(
        goal=goal,
        nodes=nodes,
        edges=edges,
        acceptance_criteria=[
            AcceptanceCriterion(
                id="goal_met",
                description=f"Goal achieved: {goal}",
            )
        ],
    )


# ------------------------------------------------------------------
# LLM Planner
# ------------------------------------------------------------------

class LLMPlanner:
    """Plans task graphs using an LLM, with heuristic fallback."""

    def __init__(
        self,
        llm_client: Any,
        *,
        model: str = "claude-sonnet-4-20250514",
        token_budget: int = 8_000,
    ):
        self._client = llm_client
        self._model = model
        self._compiler = PromptCompiler(
            token_budget=token_budget, system_prompt=PLANNER_SYSTEM
        )

    def plan(
        self,
        goal: str,
        *,
        project_context: str = "",
        prior_failures: str = "",
        project_path: str | None = None,
    ) -> tuple[TaskGraph, float | None]:
        """Generate a TaskGraph for *goal*.

        Returns (graph, confidence) where confidence is the model's
        self-reported confidence (0-1) or None.
        """
        blocks: list[ContextBlock] = []
        if project_context:
            blocks.append(ContextBlock(
                name="Project Context", content=project_context, priority=10
            ))
        if prior_failures:
            blocks.append(ContextBlock(
                name="Prior Failures", content=prior_failures, priority=20
            ))

        messages = self._compiler.compile(goal, blocks)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                messages=[m for m in messages if m["role"] != "system"],
                system=messages[0]["content"] if messages[0]["role"] == "system" else "",
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            data = json.loads(raw)
            graph = TaskGraph.model_validate(data)
            confidence = data.get("confidence")
            logger.info("LLM planner produced graph with %d nodes", len(graph.nodes))
            return graph, confidence
        except Exception as exc:
            logger.warning("LLM planner failed (%s), using heuristic fallback", exc)
            return _heuristic_plan(goal, project_path), None

    def plan_heuristic(
        self, goal: str, *, project_path: str | None = None
    ) -> TaskGraph:
        """Direct access to the heuristic fallback planner."""
        return _heuristic_plan(goal, project_path)
