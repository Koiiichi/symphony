"""Unit tests for TaskGraph schema validation."""

import pytest
from pydantic import ValidationError

from symphony.planner.schema import (
    AcceptanceCriterion,
    EvidenceRequirements,
    NodeType,
    RetryPolicy,
    TaskEdge,
    TaskGraph,
    TaskNode,
)


# ------------------------------------------------------------------
# TaskNode validation
# ------------------------------------------------------------------

class TestTaskNode:
    def test_basic_node(self):
        node = TaskNode(id="n1", type=NodeType.STACK_DETECT, description="detect stack")
        assert node.id == "n1"
        assert node.type == NodeType.STACK_DETECT

    def test_web_flow_test_requires_actions(self):
        with pytest.raises(ValidationError, match="must declare actions"):
            TaskNode(
                id="n1",
                type=NodeType.WEB_FLOW_TEST,
                description="test",
                assertions=[{"action": "assert_text", "selector": "body", "value": "ok"}],
            )

    def test_web_flow_test_requires_assertions(self):
        with pytest.raises(ValidationError, match="must declare assertions"):
            TaskNode(
                id="n1",
                type=NodeType.WEB_FLOW_TEST,
                description="test",
                actions=[{"action": "click", "selector": "#btn"}],
            )

    def test_web_flow_test_valid(self):
        node = TaskNode(
            id="n1",
            type=NodeType.WEB_FLOW_TEST,
            description="test",
            actions=[{"action": "click", "selector": "#btn"}],
            assertions=[{"action": "assert_text", "selector": "body", "value": "ok"}],
        )
        assert node.type == NodeType.WEB_FLOW_TEST
        assert len(node.actions) == 1

    def test_other_requires_task_type(self):
        with pytest.raises(ValidationError, match="other_task_type"):
            TaskNode(id="n1", type=NodeType.OTHER, description="custom")

    def test_other_with_task_type(self):
        node = TaskNode(
            id="n1", type=NodeType.OTHER,
            description="custom", other_task_type="database_migration",
        )
        assert node.other_task_type == "database_migration"

    def test_retry_policy_defaults(self):
        node = TaskNode(id="n1", type=NodeType.FINALIZE)
        assert node.retry.max_attempts == 3
        assert node.retry.backoff_seconds == 1.0

    def test_custom_retry(self):
        node = TaskNode(
            id="n1", type=NodeType.API_CHECK,
            retry=RetryPolicy(max_attempts=5, backoff_seconds=2.0),
        )
        assert node.retry.max_attempts == 5

    def test_evidence_requirements_defaults(self):
        node = TaskNode(id="n1", type=NodeType.FINALIZE)
        assert node.evidence_requirements.screenshot is True
        assert node.evidence_requirements.network_trace is False


# ------------------------------------------------------------------
# TaskGraph validation
# ------------------------------------------------------------------

class TestTaskGraph:
    def _simple_graph(self) -> TaskGraph:
        return TaskGraph(
            goal="test goal",
            nodes=[
                TaskNode(id="a", type=NodeType.STACK_DETECT),
                TaskNode(id="b", type=NodeType.FINALIZE),
            ],
            edges=[TaskEdge(source="a", target="b")],
        )

    def test_basic_graph(self):
        g = self._simple_graph()
        assert g.goal == "test goal"
        assert len(g.nodes) == 2
        assert len(g.edges) == 1

    def test_requires_at_least_one_node(self):
        with pytest.raises(ValidationError):
            TaskGraph(goal="test", nodes=[])

    def test_invalid_edge_source(self):
        with pytest.raises(ValidationError, match="Edge source 'x' not found"):
            TaskGraph(
                goal="test",
                nodes=[TaskNode(id="a", type=NodeType.FINALIZE)],
                edges=[TaskEdge(source="x", target="a")],
            )

    def test_invalid_edge_target(self):
        with pytest.raises(ValidationError, match="Edge target 'x' not found"):
            TaskGraph(
                goal="test",
                nodes=[TaskNode(id="a", type=NodeType.FINALIZE)],
                edges=[TaskEdge(source="a", target="x")],
            )

    def test_topo_order(self):
        g = self._simple_graph()
        order = g.topo_order()
        assert order.index("a") < order.index("b")

    def test_topo_order_cycle_detection(self):
        with pytest.raises(ValueError, match="cycle"):
            TaskGraph(
                goal="test",
                nodes=[
                    TaskNode(id="a", type=NodeType.FINALIZE),
                    TaskNode(id="b", type=NodeType.FINALIZE),
                ],
                edges=[
                    TaskEdge(source="a", target="b"),
                    TaskEdge(source="b", target="a"),
                ],
            ).topo_order()

    def test_get_node(self):
        g = self._simple_graph()
        assert g.get_node("a").id == "a"

    def test_get_node_missing(self):
        g = self._simple_graph()
        with pytest.raises(KeyError):
            g.get_node("z")

    def test_acceptance_criteria(self):
        g = TaskGraph(
            goal="test",
            nodes=[TaskNode(id="a", type=NodeType.FINALIZE)],
            acceptance_criteria=[
                AcceptanceCriterion(id="ac1", description="Must pass"),
            ],
        )
        assert g.acceptance_criteria[0].required is True

    def test_json_roundtrip(self):
        g = self._simple_graph()
        data = g.model_dump()
        g2 = TaskGraph.model_validate(data)
        assert g2.goal == g.goal
        assert len(g2.nodes) == len(g.nodes)


# ------------------------------------------------------------------
# Heuristic fallback
# ------------------------------------------------------------------

class TestPlannerRequiresLLM:
    def test_planner_raises_without_llm(self):
        from unittest.mock import MagicMock
        import pytest
        from symphony.planner.planner import LLMPlanner

        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("no key")
        planner = LLMPlanner(mock_client)
        with pytest.raises(RuntimeError):
            planner.plan("fix contact form")
