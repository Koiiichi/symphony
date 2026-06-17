"""Unit tests for perception-first planning additions (no live browser)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from symphony.flow.dsl import ActionType
from symphony.orchestrator import Orchestrator
from symphony.planner.schema import NodeType, TaskNode


class _FakeLLM:
    model = "test-model"

    def complete(self, *_args, **_kwargs) -> str:
        return "{}"


def _orch(tmp_path: Path) -> Orchestrator:
    project = tmp_path / "project"
    project.mkdir()
    return Orchestrator(project_path=project, llm=_FakeLLM(), edit_mode="auto")


# ---- Service command detection (node-independent) ----

def test_detect_service_commands_node_project(tmp_path: Path):
    orch = _orch(tmp_path)
    (orch._project / "package.json").write_text("{}")
    specs = orch._detect_service_commands()
    assert any(s["cmd"] == "npm start" for s in specs)


def test_detect_service_commands_python_project(tmp_path: Path):
    orch = _orch(tmp_path)
    (orch._project / "requirements.txt").write_text("flask\n")
    (orch._project / "app.py").write_text("# app\n")
    specs = orch._detect_service_commands()
    assert any("app.py" in s["cmd"] for s in specs)


def test_detect_service_commands_empty(tmp_path: Path):
    orch = _orch(tmp_path)
    assert orch._detect_service_commands() == []


def test_ensure_service_started_skips_when_running(tmp_path: Path):
    orch = _orch(tmp_path)
    (orch._project / "package.json").write_text("{}")
    orch._base_url = "http://localhost:3000"
    # Simulate a live process so it should NOT restart.
    proc = MagicMock()
    proc.poll.return_value = None
    orch._service_processes = [proc]
    orch._start_services_from_specs = MagicMock()
    assert orch._ensure_service_started() == "http://localhost:3000"
    orch._start_services_from_specs.assert_not_called()


def test_ensure_service_started_derives_base_url(tmp_path: Path):
    orch = _orch(tmp_path)
    (orch._project / "package.json").write_text("{}")
    orch._start_services_from_specs = MagicMock()
    base = orch._ensure_service_started()
    assert base == "http://localhost:3000"
    orch._start_services_from_specs.assert_called_once()


# ---- service_start node idempotency ----

def test_service_start_node_noop_when_running(tmp_path: Path):
    orch = _orch(tmp_path)
    (orch._project / "package.json").write_text("{}")
    proc = MagicMock()
    proc.poll.return_value = None
    orch._service_processes = [proc]
    orch._start_services_from_specs = MagicMock()
    node = TaskNode(id="svc", type=NodeType.SERVICE_START)
    orch._execute_service_start(node)
    orch._start_services_from_specs.assert_not_called()


# ---- HTML page discovery (public-dir aware) ----

def test_discover_html_pages_strips_public(tmp_path: Path):
    orch = _orch(tmp_path)
    public = orch._project / "public"
    public.mkdir()
    (public / "index.html").write_text("<html></html>")
    (public / "events.html").write_text("<html></html>")
    pages = orch._discover_html_pages()
    assert "index.html" in pages
    assert "events.html" in pages
    # No 'public/' prefix should leak into web paths.
    assert all(not p.startswith("public/") for p in pages)


# ---- UI map formatting ----

def test_format_ui_map_renders_selectors():
    ui = {
        "inputs": [
            {"tag": "input", "attrs": {"id": "email", "type": "email"}, "text": ""},
            {"tag": "input", "attrs": {"name": "q", "placeholder": "search"}, "text": ""},
        ],
        "buttons": [
            {"tag": "button", "attrs": {"data-book-id": "e_1"}, "text": "Book now"},
        ],
        "links": [{"tag": "a", "attrs": {"href": "/events.html"}, "text": "Events"}],
        "ids": ["email", "historyStatus"],
    }
    out = Orchestrator._format_ui_map("index.html", "http://localhost:3000/index.html", ui)
    assert "#email" in out
    assert 'input[name="q"]' in out
    assert 'button[data-book-id="e_1"]' in out
    assert 'a[href="/events.html"]' in out
    assert "#historyStatus" in out


def test_run_perception_returns_empty_without_base_url(tmp_path: Path):
    orch = _orch(tmp_path)
    assert orch._run_perception("") == ""


# ---- Auto-navigate key variants ----

def _exec_navigate_value(orch: Orchestrator, node: TaskNode) -> str:
    """Capture the FlowScript handed to the executor and return first action value."""
    captured = {}

    class _FakeExecutor:
        def __init__(self, *_a, **_k):
            pass

        def execute(self, script):
            captured["actions"] = script.actions
            from symphony.flow.executor import FlowResult
            return FlowResult(script_name=script.name, passed=True, results=[])

    import symphony.orchestrator as orch_mod
    orig = orch_mod.FlowExecutor
    orch_mod.FlowExecutor = _FakeExecutor
    orch._get_shared_driver = MagicMock(return_value=MagicMock())
    try:
        orch._execute_web_flow(node)
    finally:
        orch_mod.FlowExecutor = orig
    first = captured["actions"][0]
    return first.action, first.value


def test_auto_navigate_start_url(tmp_path: Path):
    orch = _orch(tmp_path)
    orch._base_url = "http://localhost:3000"
    node = TaskNode(
        id="flow",
        type=NodeType.WEB_FLOW_TEST,
        config={"start_url": "http://localhost:3000/index.html"},
        actions=[{"action": "fill", "selector": "#email", "value": "x"}],
        assertions=[{"action": "assert_text", "selector": "#email", "value": "x",
                     "params": {"claim_id": "c"}}],
    )
    action, value = _exec_navigate_value(orch, node)
    assert action == ActionType.NAVIGATE
    assert value == "http://localhost:3000/index.html"


def test_auto_navigate_relative_base_url(tmp_path: Path):
    orch = _orch(tmp_path)
    orch._base_url = "http://localhost:3000"
    node = TaskNode(
        id="flow",
        type=NodeType.WEB_FLOW_TEST,
        config={"start_url": "/support.html"},
        actions=[{"action": "fill", "selector": "#supportEmail", "value": "x"}],
        assertions=[{"action": "assert_text", "selector": "#supportStatus", "value": "x",
                     "params": {"claim_id": "c"}}],
    )
    action, value = _exec_navigate_value(orch, node)
    assert action == ActionType.NAVIGATE
    assert value == "http://localhost:3000/support.html"
