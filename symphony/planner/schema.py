"""TaskGraph schema — the typed plan the LLM Planner emits."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ------------------------------------------------------------------
# Node types
# ------------------------------------------------------------------

class NodeType(str, Enum):
    STACK_DETECT = "stack_detect"
    SERVICE_START = "service_start"
    UI_DISCOVERY = "ui_discovery"
    WEB_FLOW_TEST = "web_flow_test"
    API_CHECK = "api_check"
    CODE_PATCH = "code_patch"
    RETEST = "retest"
    FINALIZE = "finalize"
    OTHER = "other"


# ------------------------------------------------------------------
# Retry policy
# ------------------------------------------------------------------

class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    backoff_seconds: float = Field(default=1.0, ge=0)


# ------------------------------------------------------------------
# Evidence requirements (for web_flow_test nodes)
# ------------------------------------------------------------------

class EvidenceRequirements(BaseModel):
    screenshot: bool = True
    network_trace: bool = False
    dom_snapshot: bool = False
    focused_element_trace: bool = False


# ------------------------------------------------------------------
# Task node
# ------------------------------------------------------------------

class TaskNode(BaseModel):
    id: str
    type: NodeType
    description: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)

    # Required for web_flow_test nodes
    preconditions: list[str] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    assertions: list[dict[str, Any]] = Field(default_factory=list)
    evidence_requirements: EvidenceRequirements = Field(
        default_factory=EvidenceRequirements
    )

    # For OTHER node type — caller describes the task type
    other_task_type: str | None = None

    @model_validator(mode="after")
    def _validate_web_flow_fields(self) -> "TaskNode":
        if self.type == NodeType.WEB_FLOW_TEST:
            if not self.actions:
                raise ValueError("web_flow_test node must declare actions")
            if not self.assertions:
                raise ValueError("web_flow_test node must declare assertions")
        if self.type == NodeType.OTHER and not self.other_task_type:
            raise ValueError(
                "other node must specify other_task_type describing the task"
            )
        return self


# ------------------------------------------------------------------
# Edge
# ------------------------------------------------------------------

class TaskEdge(BaseModel):
    source: str
    target: str
    condition: str | None = None  # optional guard expression


# ------------------------------------------------------------------
# Acceptance criterion
# ------------------------------------------------------------------

class AcceptanceCriterion(BaseModel):
    id: str
    description: str
    required: bool = True


# ------------------------------------------------------------------
# Top-level TaskGraph
# ------------------------------------------------------------------

class TaskGraph(BaseModel):
    version: str = "2.0"
    goal: str
    constraints: list[str] = Field(default_factory=list)
    nodes: list[TaskNode] = Field(min_length=1)
    edges: list[TaskEdge] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_edge_refs(self) -> "TaskGraph":
        node_ids = {n.id for n in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(
                    f"Edge source '{edge.source}' not found in nodes"
                )
            if edge.target not in node_ids:
                raise ValueError(
                    f"Edge target '{edge.target}' not found in nodes"
                )
        return self

    def topo_order(self) -> list[str]:
        """Return node IDs in topological order (Kahn's algorithm)."""
        from collections import deque

        in_degree: dict[str, int] = {n.id: 0 for n in self.nodes}
        adj: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            adj[e.source].append(e.target)
            in_degree[e.target] += 1

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for child in adj[nid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("TaskGraph contains a cycle")
        return order

    def get_node(self, node_id: str) -> TaskNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"Node '{node_id}' not found")
