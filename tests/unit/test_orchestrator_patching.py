"""Unit tests for orchestrator patch policy integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from symphony.flow.dsl import ActionType, FlowAction
from symphony.flow.executor import ActionResult, FlowResult
from symphony.orchestrator import Orchestrator
from symphony.planner.schema import NodeType, TaskNode


class _FakeLLM:
    def __init__(self, patch_payload: dict):
        self._patch_payload = patch_payload
        self.model = "test-model"

    def complete(self, *_args, **_kwargs) -> str:
        return json.dumps(self._patch_payload)


class _FakeCompiler:
    def compile_with_usage(self, *_args, **_kwargs):
        return (
            [{"role": "system", "content": ""}, {"role": "user", "content": "patch"}],
            {"total_tokens": 0},
        )


def _failing_flow() -> list[FlowResult]:
    return [
        FlowResult(
            script_name="test_flow",
            passed=False,
            results=[
                ActionResult(
                    action=FlowAction(action=ActionType.ASSERT_TEXT, selector="body", value="ok"),
                    success=False,
                    message="Assertion failed",
                )
            ],
        )
    ]


def test_auto_mode_applies_and_restarts(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "app.py"
    target.write_text("status = 'old'\n")

    llm = _FakeLLM(
        {"patches": [{"file": "app.py", "search": "old", "replace": "new"}]}
    )
    orch = Orchestrator(project_path=project, llm=llm, edit_mode="auto")
    orch._service_commands = [{"cmd": "npm start", "cwd": str(project)}]
    orch._restart_services = MagicMock()

    node = TaskNode(id="code_patch_1", type=NodeType.CODE_PATCH, description="patch")
    orch._execute_code_patch(node, _FakeCompiler(), "fix issue", _failing_flow())

    assert target.read_text() == "status = 'new'\n"
    orch._restart_services.assert_called_once()
    assert orch._patch_records[0]["decision"] == "applied"


def test_suggest_mode_does_not_apply_or_restart(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "app.py"
    target.write_text("status = 'old'\n")

    llm = _FakeLLM(
        {"patches": [{"file": "app.py", "search": "old", "replace": "new"}]}
    )
    orch = Orchestrator(project_path=project, llm=llm, edit_mode="suggest")
    orch._service_commands = [{"cmd": "npm start", "cwd": str(project)}]
    orch._restart_services = MagicMock()

    node = TaskNode(id="code_patch_1", type=NodeType.CODE_PATCH, description="patch")
    orch._execute_code_patch(node, _FakeCompiler(), "fix issue", _failing_flow())

    assert target.read_text() == "status = 'old'\n"
    orch._restart_services.assert_not_called()
    assert orch._patch_records[0]["decision"] == "suggested"
    assert orch._patch_records[0]["blocked_reason"] == "edit_mode_suggest"


def test_manual_approval_blocked_flag_is_set(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "app.py"
    target.write_text("status = 'old'\n")

    llm = _FakeLLM(
        {"patches": [{"file": "app.py", "search": "old", "replace": "new"}]}
    )
    orch = Orchestrator(
        project_path=project,
        llm=llm,
        edit_mode="suggest",
        require_manual_approval=True,
    )
    node = TaskNode(id="code_patch_1", type=NodeType.CODE_PATCH, description="patch")
    orch._execute_code_patch(node, _FakeCompiler(), "fix issue", _failing_flow())

    assert orch._manual_approval_blocked is True
    assert orch._patch_records[0]["blocked_reason"] == "manual_approval_required"
    assert target.read_text() == "status = 'old'\n"
