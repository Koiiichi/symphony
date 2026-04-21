"""Unit tests for patch policy and safe-apply behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from symphony.patching.policy import EditMode, PatchDecision, PatchPolicy


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_rejects_path_traversal(tmp_path: Path):
    project = tmp_path / "project"
    _write(project / "app.py", "print('hello')\n")
    _write(tmp_path / "outside.py", "print('outside')\n")
    policy = PatchPolicy(project_root=project, mode=EditMode.AUTO)
    with pytest.raises(ValueError, match="escapes project root"):
        policy.build_proposal(
            node_id="n1",
            patches=[{"file": "../outside.py", "search": "outside", "replace": "safe"}],
            artifact_dir=tmp_path / "artifacts",
        )


def test_rejects_out_of_write_scope(tmp_path: Path):
    project = tmp_path / "project"
    _write(project / "src/app.py", "status = 'old'\n")
    _write(project / "other/app.py", "status = 'old'\n")
    policy = PatchPolicy(
        project_root=project,
        mode=EditMode.AUTO,
        write_scopes=[project / "src"],
    )
    with pytest.raises(PermissionError, match="outside write scope"):
        policy.build_proposal(
            node_id="n1",
            patches=[{"file": "other/app.py", "search": "old", "replace": "new"}],
            artifact_dir=tmp_path / "artifacts",
        )


def test_rejects_missing_search_string(tmp_path: Path):
    project = tmp_path / "project"
    target = project / "app.py"
    _write(target, "status = 'ok'\n")
    policy = PatchPolicy(project_root=project, mode=EditMode.AUTO)
    with pytest.raises(ValueError, match="search string not found"):
        policy.build_proposal(
            node_id="n1",
            patches=[{"file": "app.py", "search": "missing", "replace": "new"}],
            artifact_dir=tmp_path / "artifacts",
        )


def test_suggest_mode_never_writes(tmp_path: Path):
    project = tmp_path / "project"
    target = project / "app.py"
    _write(target, "status = 'old'\n")
    policy = PatchPolicy(project_root=project, mode=EditMode.SUGGEST)
    proposal = policy.build_proposal(
        node_id="n1",
        patches=[{"file": "app.py", "search": "old", "replace": "new"}],
        artifact_dir=tmp_path / "artifacts",
    )
    outcome = policy.decide_and_apply(proposal)
    assert outcome.decision is PatchDecision.SUGGESTED
    assert outcome.blocked_reason == "edit_mode_suggest"
    assert target.read_text() == "status = 'old'\n"


def test_ask_mode_rejects_without_writing(tmp_path: Path):
    project = tmp_path / "project"
    target = project / "app.py"
    _write(target, "status = 'old'\n")
    policy = PatchPolicy(project_root=project, mode=EditMode.ASK)
    proposal = policy.build_proposal(
        node_id="n1",
        patches=[{"file": "app.py", "search": "old", "replace": "new"}],
        artifact_dir=tmp_path / "artifacts",
    )
    outcome = policy.decide_and_apply(proposal, approval_callback=lambda _payload: False)
    assert outcome.decision is PatchDecision.REJECTED
    assert outcome.blocked_reason == "user_rejected"
    assert target.read_text() == "status = 'old'\n"


def test_auto_mode_applies_changes(tmp_path: Path):
    project = tmp_path / "project"
    target = project / "app.py"
    _write(target, "status = 'old'\n")
    policy = PatchPolicy(project_root=project, mode=EditMode.AUTO)
    proposal = policy.build_proposal(
        node_id="n1",
        patches=[{"file": "app.py", "search": "old", "replace": "new"}],
        artifact_dir=tmp_path / "artifacts",
    )
    outcome = policy.decide_and_apply(proposal)
    assert outcome.decision is PatchDecision.APPLIED
    assert outcome.applied_files == ["app.py"]
    assert target.read_text() == "status = 'new'\n"


def test_manual_approval_required_reason(tmp_path: Path):
    project = tmp_path / "project"
    target = project / "app.py"
    _write(target, "status = 'old'\n")
    policy = PatchPolicy(
        project_root=project,
        mode=EditMode.SUGGEST,
        require_manual_approval=True,
    )
    proposal = policy.build_proposal(
        node_id="n1",
        patches=[{"file": "app.py", "search": "old", "replace": "new"}],
        artifact_dir=tmp_path / "artifacts",
    )
    outcome = policy.decide_and_apply(proposal)
    assert outcome.decision is PatchDecision.SUGGESTED
    assert outcome.blocked_reason == "manual_approval_required"
    assert target.read_text() == "status = 'old'\n"
