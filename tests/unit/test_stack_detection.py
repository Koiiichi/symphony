import json
from pathlib import Path

from core.stack import analyze_project


def test_detects_vite_frontend(tmp_path):
    project = tmp_path
    pkg = {
        "name": "vite-app",
        "scripts": {"dev": "vite"},
        "dependencies": {"vite": "^5.0.0", "react": "^18.0.0"},
    }
    (project / "package.json").write_text(json.dumps(pkg))

    info = analyze_project(project)

    assert info.frontend == "vite"
    assert any(cmd.command[:3] == ["npm", "run", "dev"] for cmd in info.start_commands)
    assert info.frontend_url == "http://localhost:5173"


def test_detects_python_backend(tmp_path):
    project = tmp_path
    (project / "requirements.txt").write_text("flask==2.0.0")
    (project / "app.py").write_text("print('hello')")

    info = analyze_project(project)

    assert info.backend == "python"
    assert any(
        cmd.command[-1].endswith("app.py")
        or ("--app" in cmd.command and "app.py" in cmd.command)
        for cmd in info.start_commands
    )
