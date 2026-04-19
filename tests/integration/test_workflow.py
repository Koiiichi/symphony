import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
pytest.importorskip("rich")

from core.types import StartCommand, WorkflowConfig
from orchestrator import run_workflow


class FakeHooks:
    def __init__(self):
        self.events = []
        self._resolved = False

    def run_brain(self, instructions: str, *, pass_index: int):
        self.events.append(("brain", pass_index, instructions))
        return {"status": "ok"}

    def run_vision(self, url: str, expectations, *, pass_index: int, mode: str):
        self.events.append(("vision", pass_index, url, mode))
        if not self._resolved:
            self._resolved = True
            return {
                "vision_scores": {"alignment": 0.6, "spacing": 0.6, "contrast": 0.6},
                "elements": {},
                "interactions": {},
            }
        return {
            "vision_scores": {"alignment": 0.95, "spacing": 0.94, "contrast": 0.92},
            "elements": {},
            "interactions": {},
        }

    def consume_brain_log(self, pass_index: int):  # pragma: no cover - interface fulfilment
        return None


class AlwaysPassingVision(FakeHooks):
    def run_vision(self, url: str, expectations, *, pass_index: int, mode: str):
        self.events.append(("vision", pass_index, url, mode))
        return {
            "vision_scores": {"alignment": 0.95, "spacing": 0.95, "contrast": 0.95},
            "elements": {},
            "interactions": {},
        }


@pytest.fixture(autouse=True)
def server_manager_stub(monkeypatch):
    class StubServerManager:
        def __init__(self, stack, project_root=None):
            self.stack = stack

        def start_all(self, preferred_kind=None):
            return {"frontend": "http://localhost:5173"}

        def stop_all(self):
            pass

        class _Selection:
            def __init__(self, url: str):
                self.url = url
                self.probe = None
                self.fallback_used = False
                self.message = None
                self.artifacts = {}

        def resolve_preview_surface(self, run_id, preferred_kind=None, hints=None):
            return self._Selection("http://localhost:5173")

    monkeypatch.setattr("orchestrator.ServerManager", StubServerManager)
    monkeypatch.setenv("SYMPHONY_BRAIN_API_KEY", "test")
    monkeypatch.setenv("SYMPHONY_VISION_API_KEY", "test")


def _prepare_project(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "package.json").write_text(json.dumps({"name": "demo", "scripts": {"dev": "vite"}}))


def test_refine_ui_flow(tmp_path):
    project = tmp_path / "app"
    _prepare_project(project)

    config = WorkflowConfig(project_path=project, goal="Improve UI spacing", max_passes=3, open_browser=False)
    hooks = FakeHooks()

    summary = run_workflow(config, hooks=hooks)

    assert summary.status in {"success", "stalled", "max_passes"}
    assert any(evt[0] == "vision" for evt in hooks.events)
    assert any(evt[0] == "brain" for evt in hooks.events)


def test_create_flow_runs_brain_first(tmp_path, monkeypatch):
    project = tmp_path / "new"
    project.mkdir()
    monkeypatch.setattr(
        "orchestrator.prompt_for_start_command",
        lambda goal, root: StartCommand(
            command=["npm", "run", "dev"],
            cwd=root,
            kind="frontend",
            url="http://localhost:5173",
        ),
    )

    config = WorkflowConfig(
        project_path=project,
        goal="Create a landing page with beautiful UI",
        max_passes=2,
        open_browser=False,
    )
    hooks = AlwaysPassingVision()

    summary = run_workflow(config, hooks=hooks)

    assert hooks.events[0][0] == "brain"
    assert summary.passes
    assert summary.status in {"success", "max_passes", "stalled"}
