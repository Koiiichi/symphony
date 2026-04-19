"""Brain Agent Factory - Creates project-scoped agent instances.

This module provides a factory to create Brain agents bound to specific project roots
with configurable models. No global state, full isolation between concurrent runs.
"""

import os
import pathlib
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from smolagents import CodeAgent, tool
    from smolagents.cli import load_model
except ModuleNotFoundError:  # pragma: no cover - fallback used in tests
    def tool(func: Callable) -> Callable:
        """Fallback decorator when smolagents is unavailable."""

        return func


    class CodeAgent:  # type: ignore[misc]
        """Minimal stub that mimics the interface required by tests."""

        def __init__(self, *_, **__):
            raise ModuleNotFoundError(
                "smolagents is required to create a real CodeAgent. Install smolagents to enable agent execution."
            )


    def load_model(*_, **__):  # type: ignore[unused-argument]
        raise ModuleNotFoundError(
            "smolagents is required to load agent models. Install smolagents to enable agent execution."
        )


_ALLOWED_COMMANDS = frozenset({
    "python", "python3", "pip", "pip3",
    "npm", "npx", "node", "yarn", "pnpm",
    "cat", "ls", "echo", "mkdir", "cp", "mv", "rm", "touch",
    "git", "find", "grep", "head", "tail", "wc", "sort", "uniq",
    "curl", "wget",
    "pytest", "ruff", "eslint", "tsc", "prettier",
})

_ALLOWED_NEW_ROOT_PY_FILES = frozenset({
    "app.py",
    "main.py",
    "server.py",
    "manage.py",
    "wsgi.py",
    "asgi.py",
    "setup.py",
})


def _sanitize_command(cmd: str, project_root: pathlib.Path) -> Tuple[List[str], Optional[str]]:
    """Parse and validate a shell command string.

    Returns (cmd_list, error).  error is ``None`` when the command is safe.
    """
    try:
        parts = shlex.split(cmd)
    except ValueError as exc:
        return [], f"Failed to parse command: {exc}"

    if not parts:
        return [], "Empty command"

    base = pathlib.Path(parts[0]).name.lower()
    # Strip .cmd/.exe suffix for Windows compat
    for suffix in (".cmd", ".exe"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]

    if base not in _ALLOWED_COMMANDS:
        return [], (
            f"Command '{base}' is not in the allow-list.  "
            f"Permitted: {', '.join(sorted(_ALLOWED_COMMANDS))}"
        )

    # Reject obvious shell metacharacters that slipped through shlex
    dangerous = {"&&", "||", ";", "|", "`", "$(", "${"}
    for part in parts:
        for pattern in dangerous:
            if pattern in part:
                return [], f"Shell metacharacter '{pattern}' is not allowed in arguments"

    return parts, None


@dataclass
class BrainConfig:
    """Configuration for Brain agent instances."""
    model_type: str = "LiteLLMModel"
    model_id: str = "gpt-4o-mini"  # Changed from gpt-5-nano (doesn't exist)
    provider: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    max_steps: int = 15
    temperature: float = 0.7
    verbosity: int = 1
    timeout: int = 180  # seconds for operations


@dataclass
class SensoryConfig:
    """Configuration for Sensory agent."""
    model_id: str = "gpt-4o"
    headless: bool = True
    timeout: int = 120


def validate_path_safety(project_root: pathlib.Path, target_path: str) -> pathlib.Path:
    """Ensure target path is within project root to prevent directory traversal.
    
    Args:
        project_root: The project root directory
        target_path: The target path to validate (relative or absolute)
        
    Returns:
        Resolved absolute path within project root
        
    Raises:
        ValueError: If path escapes project root
    """
    # Resolve project root to absolute path
    project_root = project_root.resolve()
    
    # Handle absolute paths by making them relative to project root
    if pathlib.Path(target_path).is_absolute():
        target_path = pathlib.Path(target_path).name
    
    # Resolve target path relative to project root
    resolved = (project_root / target_path).resolve()
    
    # Ensure resolved path is within project root
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise ValueError(
            f"Security: Path '{target_path}' escapes project root '{project_root}'"
        )
    
    return resolved


def _is_disallowed_new_file(project_root: pathlib.Path, full_path: pathlib.Path) -> bool:
    """Guardrail against ad-hoc script creation in project root."""
    if full_path.exists():
        return False
    try:
        rel = full_path.relative_to(project_root)
    except ValueError:
        return False

    # Only enforce for root-level Python files.
    if len(rel.parts) != 1 or full_path.suffix.lower() != ".py":
        return False

    return rel.name.lower() not in _ALLOWED_NEW_ROOT_PY_FILES


def create_brain_agent(
    project_root: str | pathlib.Path,
    config: Optional[BrainConfig] = None,
    run_id: Optional[str] = None
) -> CodeAgent:
    """Create a Brain agent instance bound to a specific project.
    
    Args:
        project_root: Absolute path to the project directory
        config: Brain configuration (uses defaults if None)
        run_id: Unique identifier for this run (for artifacts/logs)
        
    Returns:
        CodeAgent instance with project-scoped tools
    """
    if config is None:
        config = BrainConfig()
    
    # Ensure project root is absolute
    project_root = pathlib.Path(project_root).resolve()
    if not project_root.exists():
        project_root.mkdir(parents=True, exist_ok=True)
    
    # Store run_id for artifact naming
    _run_id = run_id or "default"
    
    # Create closures that capture project_root
    
    @tool
    def write_code(path: str, content: str) -> str:
        """Write code to a file within the project.
        
        Args:
            path: File path relative to project root
            content: Code content to write
            
        Returns:
            Success message with absolute path
        """
        try:
            full_path = validate_path_safety(project_root, path)
            if _is_disallowed_new_file(project_root, full_path):
                rel = full_path.relative_to(project_root)
                return (
                    "Blocked: Creating new root-level Python files is disabled by default. "
                    f"Path '{rel}' was blocked. "
                    "Use existing app files or place helpers under tests/ or scripts/."
                )
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} bytes to {full_path.relative_to(project_root)}"
        except Exception as e:
            return f"Error writing {path}: {e}"
    
    @tool
    def read_existing_code(path: str) -> str:
        """Read existing code from a file within the project.
        
        Args:
            path: File path relative to project root
            
        Returns:
            File content or error message
        """
        try:
            full_path = validate_path_safety(project_root, path)
            
            if not full_path.exists():
                return f"File not found: {path}"
            
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return f"Content of {path} ({len(content)} bytes):\n{content}"
        except Exception as e:
            return f"Error reading {path}: {e}"
    
    @tool
    def list_project_files(pattern: str = "**/*") -> str:
        """List files in the project directory matching a pattern.
        
        Args:
            pattern: Glob pattern (default: all files)
            
        Returns:
            Newline-separated list of relative file paths
        """
        try:
            # Handle simple patterns that need expansion
            if pattern == "*":
                pattern = "*.*"
            elif pattern == "**":
                pattern = "**/*.*"
            
            files = []
            # Exclude common directories
            exclude_patterns = {'.git', 'venv', 'node_modules', '__pycache__', '.venv', 'artifacts', 'drivers'}
            
            # Use rglob for recursive patterns
            if pattern.startswith("**/"):
                search_pattern = pattern[3:]  # Remove "**/"
                for path in project_root.rglob(search_pattern):
                    if path.is_file() and not any(part in exclude_patterns for part in path.parts):
                        rel_path = path.relative_to(project_root)
                        files.append(str(rel_path))
            else:
                for path in project_root.glob(pattern):
                    if path.is_file() and not any(part in exclude_patterns for part in path.parts):
                        rel_path = path.relative_to(project_root)
                        files.append(str(rel_path))
            
            if not files:
                return f"No files found matching pattern: {pattern}"
            
            return f"Project files ({len(files)} total):\n" + "\n".join(sorted(files))
        except Exception as e:
            return f"Error listing files: {e}"
    
    @tool
    def run_command(cmd: str, cwd: str = ".") -> str:
        """Run a shell command within the project sandbox.

        Only commands in the allow-list may be executed (python, npm, git, etc.).

        Args:
            cmd: Command to execute
            cwd: Working directory relative to project root

        Returns:
            Command output (stdout + stderr)
        """
        try:
            # Validate working directory
            work_dir = validate_path_safety(project_root, cwd)

            # Parse and validate
            cmd_list, error = _sanitize_command(cmd, project_root)
            if error:
                return f"Blocked: {error}"

            # Use current interpreter to avoid venv mismatches
            base = pathlib.Path(cmd_list[0]).name.lower().removesuffix(".cmd").removesuffix(".exe")
            if base in ("python", "python3"):
                cmd_list[0] = sys.executable
            elif base in ("pip", "pip3"):
                cmd_list = [sys.executable, "-m", "pip"] + cmd_list[1:]

            # Execute with timeout – shell=False for safety
            result = subprocess.run(
                cmd_list,
                cwd=work_dir,
                shell=False,
                capture_output=True,
                text=True,
                timeout=config.timeout,
            )

            output = result.stdout or result.stderr or "(no output)"
            return f"Command: {cmd}\nExit code: {result.returncode}\nOutput:\n{output}"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {config.timeout}s: {cmd}"
        except Exception as e:
            return f"Error executing command: {e}"
    
    # Load model with configuration
    model_kwargs = {}
    if config.provider:
        model_kwargs["provider"] = config.provider
    if config.api_base:
        model_kwargs["api_base"] = config.api_base
    
    # Use API key from config, or fall back to environment variable
    api_key = config.api_key or os.getenv("SYMPHONY_BRAIN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        model_kwargs["api_key"] = api_key
    
    try:
        model = load_model(config.model_type, config.model_id, **model_kwargs)
    except ModuleNotFoundError as exc:
        missing = exc.name
        if not missing and "'" in str(exc):
            missing = str(exc).split("'")[1]

        dependency = missing or "required dependency"
        if dependency == "litellm":
            python_cmd = f"\"{sys.executable}\" -m pip install litellm"
            install_hint = (
                f"{python_cmd} (or reinstall smolagents with the litellm extra inside this interpreter)"
            )
        elif missing:
            install_hint = f"\"{sys.executable}\" -m pip install {dependency}"
        else:
            install_hint = f"\"{sys.executable}\" -m pip install <package>"

        raise RuntimeError(
            "Brain agent setup requires additional packages. "
            f"Install the '{dependency}' dependency in this environment with {install_hint} and rerun your request."
        ) from None
    except Exception as exc:  # pragma: no cover - surface friendly error message
        raise RuntimeError(
            f"Failed to initialize Brain model '{config.model_id}': {exc}"
        ) from exc
    
    # Create agent with project-scoped tools
    # Agent name must be a valid Python identifier (no hyphens)
    agent_name = f"BrainAgent_{_run_id.replace('-', '_')}"
    agent = CodeAgent(
        tools=[write_code, read_existing_code, list_project_files, run_command],
        model=model,
        name=agent_name,
        max_steps=config.max_steps
    )
    
    return agent


def detect_existing_stack(project_root: pathlib.Path) -> Dict[str, Any]:
    """Detect existing stack in the project directory.
    
    Args:
        project_root: Project root directory
        
    Returns:
        Dictionary with stack information
    """
    stack = {
        "frontend": None,
        "backend": None,
        "package_managers": [],
        "frameworks": [],
        "has_content": False
    }
    
    # Check for frontend indicators
    if (project_root / "frontend").exists():
        stack["has_content"] = True
        
        if (project_root / "frontend" / "package.json").exists():
            stack["frontend"] = "node"
            stack["package_managers"].append("npm")
            
            # Check for framework indicators in package.json
            try:
                import json
                with open(project_root / "frontend" / "package.json") as f:
                    pkg = json.load(f)
                    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                    
                    if "react" in deps:
                        stack["frameworks"].append("react")
                    if "vue" in deps:
                        stack["frameworks"].append("vue")
                    if "vite" in deps:
                        stack["frameworks"].append("vite")
            except:
                pass
        
        elif (project_root / "frontend" / "index.html").exists():
            stack["frontend"] = "static"
    
    # Check for backend indicators
    if (project_root / "backend").exists():
        stack["has_content"] = True
        
        if (project_root / "backend" / "requirements.txt").exists():
            stack["backend"] = "python"
            stack["package_managers"].append("pip")
            
            # Check for framework indicators
            try:
                with open(project_root / "backend" / "requirements.txt") as f:
                    reqs = f.read().lower()
                    if "flask" in reqs:
                        stack["frameworks"].append("flask")
                    if "fastapi" in reqs:
                        stack["frameworks"].append("fastapi")
                    if "django" in reqs:
                        stack["frameworks"].append("django")
            except:
                pass
        
        elif (project_root / "backend" / "package.json").exists():
            stack["backend"] = "node"
            stack["package_managers"].append("npm")
    
    return stack
