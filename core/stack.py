from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config_store import load_config, update_section
from .types import StackInfo, StartCommand

_DEFAULT_FRONTEND_PORTS = {
    "vite": 5173,
    "next": 3000,
    "react-scripts": 3000,
    "nuxt": 3000,
    "astro": 4321,
    "webpack": 8080,
}

_DEFAULT_BACKEND_PORTS = {
    "flask": 5000,
    "fastapi": 8000,
    "django": 8000,
    "express": 3000,
}

def _detect_package_json(root: Path) -> Optional[Dict[str, object]]:
    package_file = root / "package.json"
    if package_file.exists():
        try:
            return json.loads(package_file.read_text())
        except json.JSONDecodeError:
            return None
    return None


_MARKER_FILES = [
    "package.json",
    "vite.config.js",
    "vite.config.ts",
    "next.config.js",
    "astro.config.mjs",
    "tsconfig.json",
    "requirements.txt",
    "pyproject.toml",
    "pom.xml",
    "build.gradle",
    "go.mod",
    "Cargo.toml",
    "docker-compose.yml",
    "Dockerfile",
    "manage.py",
    "index.html",
]

_EXCLUDE_DIRS = {"node_modules", ".git", "artifacts", "venv", ".venv", "__pycache__", "drivers", ".idea"}


def _collect_files(root: Path) -> List[Path]:
    interesting: List[Path] = []
    seen: set = set()

    # Root-level markers
    for name in _MARKER_FILES:
        path = root / name
        if path.exists() and path not in seen:
            interesting.append(path)
            seen.add(path)

    # Scan 1–3 levels of subdirectories for marker files
    for depth in range(1, 4):
        pattern = "/".join(["*"] * depth)
        try:
            for sub in root.glob(pattern):
                if not sub.is_dir():
                    continue
                if any(part in _EXCLUDE_DIRS for part in sub.relative_to(root).parts):
                    continue
                for marker in _MARKER_FILES:
                    candidate = sub / marker
                    if candidate.exists() and candidate not in seen:
                        interesting.append(candidate)
                        seen.add(candidate)
        except OSError:
            continue

    return interesting


def analyze_project(root: Path) -> StackInfo:
    root = root.resolve()
    detected_files = _collect_files(root)
    has_code = any(path.is_file() for path in detected_files)

    frameworks: List[str] = []
    package_managers: List[str] = []
    start_commands: List[StartCommand] = []
    frontend = None
    backend = None
    frontend_url = None
    backend_url = None
    notes: List[str] = []

    config = load_config(root)

    # Node/Frontend detection
    node_roots: List[Path] = []
    package_json_paths = [p for p in detected_files if p.name == "package.json"]
    for pkg_path in package_json_paths:
        pkg_dir = pkg_path.parent
        node_roots.append(pkg_dir)
        try:
            pkg = json.loads(pkg_path.read_text())
        except json.JSONDecodeError:
            notes.append(f"package.json unreadable at {pkg_path.relative_to(root)}")
            continue

        scripts = pkg.get("scripts", {}) if isinstance(pkg.get("scripts"), dict) else {}
        deps = {}
        if isinstance(pkg.get("dependencies"), dict):
            deps.update(pkg["dependencies"])
        if isinstance(pkg.get("devDependencies"), dict):
            deps.update(pkg["devDependencies"])

        for framework, port in _DEFAULT_FRONTEND_PORTS.items():
            if framework in deps or any(framework in s for s in scripts.values() if isinstance(s, str)):
                if framework not in frameworks:
                    frameworks.append(framework)
                if framework in {"vite", "next", "react-scripts", "nuxt", "astro"}:
                    frontend = framework
                    frontend_url = frontend_url or f"http://localhost:{port}"
        if scripts:
            runner = "npm"
            if (pkg_dir / "pnpm-lock.yaml").exists():
                runner = "pnpm"
            elif (pkg_dir / "yarn.lock").exists():
                runner = "yarn"
            package_managers.append(runner)

            if "dev" in scripts:
                start_commands.append(
                    StartCommand(
                        command=[runner, "run", "dev"],
                        cwd=pkg_dir,
                        kind="frontend",
                        port=_guess_port(frameworks, scripts.get("dev")),
                        url=_guess_url(frontend_url, scripts.get("dev")),
                        description=f"{runner} run dev ({pkg_dir.relative_to(root)})",
                    )
                )
            elif "start" in scripts:
                start_commands.append(
                    StartCommand(
                        command=[runner, "run", "start"],
                        cwd=pkg_dir,
                        kind="frontend",
                        port=_guess_port(frameworks, scripts.get("start")),
                        url=_guess_url(frontend_url, scripts.get("start")),
                        description=f"{runner} run start ({pkg_dir.relative_to(root)})",
                    )
                )

    # Python backend detection
    if (root / "requirements.txt").exists() or (root / "pyproject.toml").exists():
        backend = "python"
        if (root / "requirements.txt").exists():
            requirements_text = (root / "requirements.txt").read_text().lower()
        else:
            requirements_text = (root / "pyproject.toml").read_text().lower()
        for framework, port in _DEFAULT_BACKEND_PORTS.items():
            if framework in requirements_text:
                frameworks.append(framework)
                backend_url = backend_url or f"http://localhost:{port}"
        app_candidates = [
            root / "app.py",
            root / "main.py",
            root / "server.py",
            root / "manage.py",
        ]
        for candidate in app_candidates:
            if candidate.exists():
                start_commands.append(
                    StartCommand(
                        command=[_python_executable(), str(candidate)],
                        cwd=candidate.parent,
                        kind="backend",
                        port=_guess_backend_port(frameworks),
                        url=backend_url,
                        description=f"python {candidate.name}",
                    )
                )
                break

    # Fallback to config overrides
    for key, entry in config.get("start_commands", {}).items():
        cmd = entry.get("command")
        if cmd:
            start_commands.append(
                StartCommand(
                    command=cmd,
                    cwd=root / entry.get("cwd", "."),
                    kind=entry.get("kind", "frontend"),
                    port=entry.get("port"),
                    url=entry.get("url"),
                    description=entry.get("description"),
                )
            )

    # Deduplicate commands preserving order
    unique_commands: List[StartCommand] = []
    seen_keys = set()
    for cmd in start_commands:
        key = (tuple(cmd.command), cmd.cwd)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_commands.append(cmd)
    start_commands = unique_commands

    has_code = has_code or bool(start_commands)

    return StackInfo(
        root=root,
        has_code=has_code,
        detected_files=detected_files,
        frameworks=frameworks,
        package_managers=package_managers,
        frontend=frontend,
        backend=backend,
        start_commands=start_commands,
        frontend_url=frontend_url,
        backend_url=backend_url,
        notes=notes,
    )


def ensure_config_override(root: Path, command: StartCommand) -> None:
    config = load_config(root)
    start_cfg = config.setdefault("start_commands", {})
    key = command.description or command.kind
    start_cfg[key] = {
        "command": command.command,
        "cwd": str(command.cwd.relative_to(root)),
        "kind": command.kind,
        "port": command.port,
        "url": command.url,
        "description": command.description,
    }
    update_section(root, "start_commands", start_cfg)


def _python_executable() -> str:
    return os.environ.get("PYTHON", "python")


def _guess_port(frameworks: List[str], script: Optional[str]) -> Optional[int]:
    for framework in frameworks:
        if framework in _DEFAULT_FRONTEND_PORTS:
            return _DEFAULT_FRONTEND_PORTS[framework]
    if script:
        match = re.search(r"--port\s+(\d+)", script)
        if match:
            return int(match.group(1))
    return None


def _guess_url(default_url: Optional[str], script: Optional[str]) -> Optional[str]:
    if default_url:
        return default_url
    if script:
        match = re.search(r"http://[\w.:]+", script)
        if match:
            return match.group(0)
    return None


def _guess_backend_port(frameworks: List[str]) -> Optional[int]:
    for framework in frameworks:
        if framework in _DEFAULT_BACKEND_PORTS:
            return _DEFAULT_BACKEND_PORTS[framework]
    return None
