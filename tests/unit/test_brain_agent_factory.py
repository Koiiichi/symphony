from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agents.brain_agent_factory import (
    BrainConfig,
    _is_disallowed_new_file,
    create_brain_agent,
)


def test_create_brain_agent_missing_litellm(monkeypatch, tmp_path: Path) -> None:
    """A friendly RuntimeError is raised when litellm is unavailable."""

    def fake_load_model(*_args, **_kwargs):  # pragma: no cover - exercised via wrapper
        raise ModuleNotFoundError("No module named 'litellm'")

    class DummyAgent:  # pragma: no cover - instantiation not reached in test
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr("agents.brain_agent_factory.load_model", fake_load_model)
    monkeypatch.setattr("agents.brain_agent_factory.CodeAgent", DummyAgent)

    config = BrainConfig(model_type="LiteLLMModel", model_id="gpt-4o-mini")

    with pytest.raises(RuntimeError) as excinfo:
        create_brain_agent(tmp_path, config=config, run_id="test")

    message = str(excinfo.value)
    assert "litellm" in message
    assert "pip install litellm" in message
    assert f"\"{sys.executable}\"" in message


def test_blocks_new_root_level_python_scripts(tmp_path: Path) -> None:
    blocked = tmp_path / "test_contact_form.py"
    assert _is_disallowed_new_file(tmp_path, blocked) is True


def test_allows_new_nested_python_scripts(tmp_path: Path) -> None:
    allowed = tmp_path / "tests" / "test_contact_form.py"
    assert _is_disallowed_new_file(tmp_path, allowed) is False


def test_allows_known_root_entrypoints(tmp_path: Path) -> None:
    allowed = tmp_path / "app.py"
    assert _is_disallowed_new_file(tmp_path, allowed) is False
