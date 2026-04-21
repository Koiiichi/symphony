"""CLI tests for patch approval UX and mode fallback."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from symphony.cli.main import cli


class _DummyLLMClient:
    def __init__(self, provider=None, model=None):
        self.provider = provider or "gemini"
        self.model = model or "gemini-test"


class _DummyOrchestrator:
    last_init: dict | None = None

    def __init__(
        self,
        *,
        project_path: Path,
        llm,
        token_budget,
        artifact_dir,
        edit_mode,
        write_scope,
        require_manual_approval,
        request_patch_approval,
    ):
        _DummyOrchestrator.last_init = {
            "project_path": project_path,
            "edit_mode": edit_mode,
            "write_scope": write_scope,
            "require_manual_approval": require_manual_approval,
            "request_patch_approval": request_patch_approval,
        }
        self.project_path = project_path
        self.request_patch_approval = request_patch_approval

    def run(self, *, goal, profile, max_passes, on_progress):
        diff_path = self.project_path / "dummy.diff"
        diff_path.write_text(
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-status = 'old'\n"
            "+status = 'new'\n"
        )
        payload = {
            "proposal_id": "proposal_1",
            "summary": {"files": 1, "added": 1, "removed": 1},
            "files": [
                {
                    "file": "app.py",
                    "added": 1,
                    "removed": 1,
                    "preview": ["@@ -1 +1 @@", "-status = 'old'", "+status = 'new'"],
                }
            ],
        }
        on_progress("patch_proposed", json.dumps(payload))

        approved = True
        if self.request_patch_approval:
            approved = self.request_patch_approval(payload)

        if approved:
            on_progress("patch_decision", json.dumps({"decision": "applied"}))
            on_progress(
                "patch_applied",
                json.dumps({"proposal_id": "proposal_1", "diff_path": str(diff_path)}),
            )
        else:
            on_progress(
                "patch_blocked",
                json.dumps({"blocked_reason": "user_rejected", "decision": "rejected"}),
            )
        on_progress("done", "pass")
        return {
            "status": "pass",
            "failing_reasons": [],
            "assertion_results": [],
            "token_usage": {"total": 1},
            "patches": [],
        }


def test_cli_non_interactive_ask_falls_back_to_suggest(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("symphony.llm.LLMClient", _DummyLLMClient)
    monkeypatch.setattr("symphony.orchestrator.Orchestrator", _DummyOrchestrator)

    runner = CliRunner()
    project = tmp_path / "project"
    project.mkdir()
    result = runner.invoke(
        cli,
        [
            "run",
            "--project",
            str(project),
            "--goal",
            "test",
            "--provider",
            "gemini",
            "--model",
            "gemini-test",
        ],
    )
    assert result.exit_code == 0
    assert "falling back from ask -> suggest mode" in result.output
    assert _DummyOrchestrator.last_init is not None
    assert _DummyOrchestrator.last_init["edit_mode"] == "suggest"
    assert _DummyOrchestrator.last_init["require_manual_approval"] is True


def test_cli_prints_preview_and_full_diff(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("symphony.llm.LLMClient", _DummyLLMClient)
    monkeypatch.setattr("symphony.orchestrator.Orchestrator", _DummyOrchestrator)
    monkeypatch.setattr("click.confirm", lambda *_args, **_kwargs: True)

    runner = CliRunner()
    project = tmp_path / "project"
    project.mkdir()
    result = runner.invoke(
        cli,
        [
            "run",
            "--project",
            str(project),
            "--goal",
            "test",
            "--provider",
            "gemini",
            "--model",
            "gemini-test",
            "--edit-mode",
            "auto",
        ],
    )
    assert result.exit_code == 0
    assert "Patch proposed:" in result.output
    assert "-status = 'old'" in result.output
    assert "+status = 'new'" in result.output
    assert "Applied Diff" in result.output
