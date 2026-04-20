"""TaskGraph schema — the typed plan the LLM Planner emits."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from symphony.flow.dsl import FlowAction


# Shared config: suppress additionalProperties from JSON schema output.
# Gemini's API rejects schemas that contain this keyword.
_no_extras = ConfigDict(json_schema_extra={"additionalProperties": False})


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
    model_config = ConfigDict(json_schema_extra={})

    max_attempts: int = Field(
        default=3, ge=1,
        description="Maximum number of retry attempts before giving up.",
    )
    backoff_seconds: float = Field(
        default=1.0, ge=0,
        description="Seconds to wait between retry attempts.",
    )


# ------------------------------------------------------------------
# Evidence requirements (for web_flow_test nodes)
# ------------------------------------------------------------------

class EvidenceRequirements(BaseModel):
    model_config = ConfigDict(json_schema_extra={})

    screenshot: bool = Field(
        default=True,
        description="Capture a screenshot after each action.",
    )
    network_trace: bool = Field(
        default=False,
        description="Record HTTP network traffic during the flow.",
    )
    dom_snapshot: bool = Field(
        default=False,
        description="Capture a full DOM snapshot after each action.",
    )
    focused_element_trace: bool = Field(
        default=False,
        description="Track which element has focus after each action.",
    )


# ------------------------------------------------------------------
# Task node
# ------------------------------------------------------------------

class TaskNode(BaseModel):
    model_config = ConfigDict(json_schema_extra={})

    id: str = Field(description="Unique identifier for this node, e.g. 'login_test'.")
    type: NodeType = Field(description="Category of work this node performs.")
    description: str = Field(
        default="",
        description="Human-readable summary of what this step does.",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary key-value config for the node. "
            "For service_start, use {'command': 'npm start', 'port': 3000}. "
            "For api_check, use {'url': '...', 'method': 'GET', 'expected_status': 200}. "
            "For code_patch, use {'target_files': ['src/app.py']}."
        ),
    )
    retry: RetryPolicy = Field(
        default_factory=RetryPolicy,
        description="Retry policy applied if this node fails.",
    )
    preconditions: list[str] = Field(
        default_factory=list,
        description="Human-readable conditions that must hold before this node runs.",
    )
    actions: list[FlowAction] = Field(
        default_factory=list,
        description=(
            "Ordered list of browser actions for web_flow_test nodes. "
            "Use navigate, click, fill, press, wait_for action types."
        ),
    )
    assertions: list[FlowAction] = Field(
        default_factory=list,
        description=(
            "Ordered list of assertion actions for web_flow_test nodes. "
            "Use assert_text, assert_http_status, or assert_banner action types."
        ),
    )
    evidence_requirements: EvidenceRequirements = Field(
        default_factory=EvidenceRequirements,
        description="What evidence to collect while executing this node.",
    )
    other_task_type: str | None = Field(
        default=None,
        description="Required when type is 'other'. Describes the custom task type.",
    )

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
    model_config = ConfigDict(json_schema_extra={})

    source: str = Field(description="ID of the upstream node.")
    target: str = Field(description="ID of the downstream node.")
    condition: str | None = Field(
        default=None,
        description="Optional guard expression; omit for unconditional edges.",
    )


# ------------------------------------------------------------------
# Acceptance criterion
# ------------------------------------------------------------------

class AcceptanceCriterion(BaseModel):
    model_config = ConfigDict(json_schema_extra={})

    id: str = Field(description="Short unique identifier, e.g. 'login_401'.")
    description: str = Field(description="What must be true for this criterion to pass.")
    required: bool = Field(
        default=True,
        description="If true, failure here fails the entire run.",
    )


# ------------------------------------------------------------------
# Top-level TaskGraph
# ------------------------------------------------------------------

class TaskGraph(BaseModel):
    model_config = ConfigDict(json_schema_extra={})

    version: str = Field(default="2.0", description="Schema version, always '2.0'.")
    goal: str = Field(description="The original goal this graph was planned to achieve.")
    constraints: list[str] = Field(
        default_factory=list,
        description="Global constraints that apply to all nodes, e.g. 'do not modify the database'.",
    )
    nodes: list[TaskNode] = Field(
        min_length=1,
        description="Ordered list of task nodes. Must contain at least one node.",
    )
    edges: list[TaskEdge] = Field(
        default_factory=list,
        description="Directed edges expressing dependencies between nodes.",
    )
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        default_factory=list,
        description="Criteria that must all pass for the run to be considered successful.",
    )

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
