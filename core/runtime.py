from __future__ import annotations

import atexit
import json as _json
import os
import signal
import socket
import subprocess
import time
import contextlib
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .stack import ensure_config_override
from .types import StackInfo, StartCommand


_LOCKFILE_NAME = ".symphony_servers.lock"


def _read_lockfile(lockfile: Path) -> List[int]:
    """Read PIDs from the lockfile, ignoring malformed entries."""
    if not lockfile.exists():
        return []
    try:
        data = _json.loads(lockfile.read_text())
        return [int(pid) for pid in data.get("pids", []) if isinstance(pid, int)]
    except Exception:
        return []


def _write_lockfile(lockfile: Path, pids: List[int]) -> None:
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    lockfile.write_text(_json.dumps({"pids": pids}))


def _remove_lockfile(lockfile: Path) -> None:
    try:
        lockfile.unlink(missing_ok=True)
    except OSError:
        pass


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _pid_cwd(pid: int) -> Optional[Path]:
    """Best-effort process cwd lookup for ownership checks."""
    if os.name == "nt":
        return None
    try:
        # lsof output with -Fn prefixes "n" on name paths.
        result = subprocess.run(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if line.startswith("n"):
                cwd = line[1:].strip()
                if cwd:
                    return Path(cwd)
    except Exception:
        return None
    return None


def _pid_belongs_to_project(pid: int, project_root: Path) -> bool:
    """Return True only when the process cwd lives under this project root."""
    cwd = _pid_cwd(pid)
    if cwd is None:
        return False
    try:
        root = project_root.resolve()
        cwd = cwd.resolve()
        return cwd == root or root in cwd.parents
    except Exception:
        return False


def _kill_stale_pids(pids: List[int], project_root: Path) -> List[int]:
    """Terminate stale PIDs that belong to this project."""
    killed = []
    for pid in pids:
        if not _is_pid_alive(pid):
            continue
        if not _pid_belongs_to_project(pid, project_root):
            continue
        try:
            if os.name != "nt":
                pgid = os.getpgid(pid)
                if pgid == pid:
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    os.kill(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except (OSError, ProcessLookupError):
            pass
    return killed


@dataclass
class RunningProcess:
    command: StartCommand
    process: subprocess.Popen


@dataclass
class ServerProbe:
    url: str
    kind: str
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    is_blank: bool = False
    node_count: int = 0
    body: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ServerSelection:
    url: Optional[str]
    probe: Optional[ServerProbe]
    fallback_used: bool = False
    message: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)


class ServerManager:
    """Start and stop project services based on detected commands."""

    def __init__(self, stack: StackInfo, project_root: Optional[Path] = None) -> None:
        self.stack = stack
        self.running: List[RunningProcess] = []
        self._project_root = project_root or stack.root
        self._lockfile = self._project_root / _LOCKFILE_NAME
        self._cleanup_stale_processes()
        atexit.register(self._atexit_cleanup)

    def _cleanup_stale_processes(self) -> None:
        """Kill any orphaned servers from a previous crashed run."""
        stale = _read_lockfile(self._lockfile)
        if stale:
            killed = _kill_stale_pids(stale, self._project_root)
            if killed:
                time.sleep(0.5)
            _remove_lockfile(self._lockfile)

    def _atexit_cleanup(self) -> None:
        """Safety net: terminate any still-running servers on interpreter exit."""
        self.stop_all()

    def _save_pids(self) -> None:
        pids = [rp.process.pid for rp in self.running if rp.process.poll() is None]
        if pids:
            _write_lockfile(self._lockfile, pids)

    def plan(self) -> List[StartCommand]:
        return self.stack.start_commands

    def _ensure_dependencies(self, command: StartCommand) -> None:
        """Ensure project dependencies are installed before starting servers."""
        cwd = command.cwd
        
        # Check for npm project
        package_json = cwd / "package.json"
        node_modules = cwd / "node_modules"
        
        if package_json.exists() and not node_modules.exists():
            print(f" Installing npm dependencies in {cwd.name}...")
            try:
                # On Windows, use shell=True or call npm.cmd directly
                npm_cmd = ["npm.cmd" if os.name == "nt" else "npm", "install"]
                result = subprocess.run(
                    npm_cmd,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes timeout
                    shell=(os.name == "nt"),  # Use shell on Windows
                )
                if result.returncode != 0:
                    print(f" Warning: npm install failed: {result.stderr}")
                else:
                    print(f"✓ Dependencies installed successfully")
            except subprocess.TimeoutExpired:
                print(f" Warning: npm install timed out")
            except FileNotFoundError:
                print(f" Warning: npm not found in PATH. Please install Node.js")
        
        # Check for Python project
        requirements_txt = cwd / "requirements.txt"
        venv_dir = cwd / "venv"
        
        if requirements_txt.exists() and not venv_dir.exists():
            print(f" Creating Python virtual environment in {cwd.name}...")
            try:
                # Create venv
                subprocess.run(
                    ["python", "-m", "venv", "venv"],
                    cwd=str(cwd),
                    capture_output=True,
                    timeout=60,
                )
                # Install requirements
                pip_path = venv_dir / "Scripts" / "pip.exe" if os.name == "nt" else venv_dir / "bin" / "pip"
                if pip_path.exists():
                    print(f" Installing Python dependencies...")
                    result = subprocess.run(
                        [str(pip_path), "install", "-r", "requirements.txt"],
                        cwd=str(cwd),
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if result.returncode != 0:
                        print(f" Warning: pip install failed: {result.stderr}")
                    else:
                        print(f"✓ Python dependencies installed")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f" Warning: Failed to setup Python environment: {e}")

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except (OSError, socket.timeout):
            return False

    def start_all(self, *, timeout: int = 60, preferred_kind: Optional[str] = None) -> Dict[str, str]:
        urls: Dict[str, str] = {}

        # Check for port conflicts before starting anything
        port_conflicts = []
        for command in self.stack.start_commands:
            if command.port:
                if self._is_port_in_use(command.port):
                    port_conflicts.append((command.port, command.description or command.kind))
        
        if port_conflicts:
            error_msg = "Port conflicts detected:\n"
            for port, desc in port_conflicts:
                error_msg += f"  - Port {port} ({desc}) is already in use\n"
            error_msg += "\nTo fix this, run:\n"
            for port, _ in port_conflicts:
                error_msg += f"  lsof -ti:{port} | xargs kill -9\n"
            raise RuntimeError(error_msg.strip())

        # First, ensure all dependencies are installed
        for command in self.stack.start_commands:
            self._ensure_dependencies(command)

        commands = list(self.stack.start_commands)
        if preferred_kind:
            commands.sort(
                key=lambda cmd: (cmd.kind != preferred_kind, cmd.kind != "frontend")
            )

        for command in commands:
            env = os.environ.copy()
            env.update(command.env)
            
            # On Windows, npm/node commands need special handling
            cmd_list = list(command.command)  # Make a copy
            
            # If command starts with 'python', check for venv
            if cmd_list[0] == "python":
                venv_dir = command.cwd / "venv"
                venv_python = venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"
                if venv_python.exists():
                    cmd_list[0] = str(venv_python)
            
            # npm/node commands need .cmd extension on Windows
            if os.name == "nt" and cmd_list[0] in ["npm", "node", "npx"]:
                cmd_list = [f"{cmd_list[0]}.cmd"] + cmd_list[1:]
            
            popen_kwargs: Dict = dict(
                cwd=str(command.cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if os.name != "nt":
                popen_kwargs["start_new_session"] = True

            proc = subprocess.Popen(cmd_list, **popen_kwargs)
            self.running.append(RunningProcess(command=command, process=proc))
            if command.url:
                urls[command.kind] = command.url
            elif command.port:
                urls[command.kind] = f"http://localhost:{command.port}"
            if command.port:
                try:
                    self._wait_for_port(
                        command.port,
                        timeout=timeout,
                        process=proc,
                        description=command.description or "Service",
                    )
                except Exception:
                    self.stop_all()
                    raise
        self._save_pids()
        return urls

    def stop_all(self) -> None:
        for proc in self.running:
            if proc.process.poll() is None:
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(proc.process.pid), signal.SIGTERM)
                    else:
                        proc.process.terminate()
                except (OSError, ProcessLookupError):
                    proc.process.terminate()
                try:
                    proc.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.process.kill()
        self.running.clear()
        _remove_lockfile(self._lockfile)

    def resolve_preview_surface(
        self,
        *,
        run_id: str,
        preferred_kind: Optional[str] = None,
        hints: Optional[Dict[str, List[str]]] = None,
    ) -> ServerSelection:
        """Select the best preview surface among running services."""

        hints = hints or {}
        selectors = hints.get("selectors", [])
        keywords = hints.get("keywords", [])

        candidates: List[Tuple[str, str]] = []
        for command in self.stack.start_commands:
            url = command.url or (command.port and f"http://localhost:{command.port}")
            if url:
                candidates.append((command.kind, url))

        if self.stack.frontend_url and not any(
            url == self.stack.frontend_url for _, url in candidates
        ):
            candidates.append(("frontend", self.stack.frontend_url))
        if self.stack.backend_url and not any(
            url == self.stack.backend_url for _, url in candidates
        ):
            candidates.append(("backend", self.stack.backend_url))

        seen = set()
        unique_candidates: List[Tuple[str, str]] = []
        for kind, url in candidates:
            if url in seen:
                continue
            seen.add(url)
            unique_candidates.append((kind, url))

        probes: List[ServerProbe] = []
        for kind, url in unique_candidates:
            probes.append(self._probe_candidate(kind, url))

        healthy = [probe for probe in probes if probe.status_code and 200 <= probe.status_code < 300 and not probe.error]
        non_blank = [probe for probe in healthy if not probe.is_blank]
        blank = [probe for probe in healthy if probe.is_blank]

        def matches_hints(probe: ServerProbe) -> bool:
            if not selectors and not keywords:
                return True
            body = probe.body or ""
            if selectors:
                if any(selector in body for selector in selectors):
                    return True
            if keywords:
                normalized = body.lower()
                if any(keyword.lower() in normalized for keyword in keywords):
                    return True
            return False

        preferred_candidates = [
            probe for probe in non_blank if preferred_kind and probe.kind == preferred_kind
        ]
        preferred_candidates = [probe for probe in preferred_candidates if matches_hints(probe)]

        chosen: Optional[ServerProbe] = None
        if preferred_candidates:
            chosen = preferred_candidates[0]
        else:
            hint_matches = [probe for probe in non_blank if matches_hints(probe)]
            if hint_matches:
                chosen = hint_matches[0]
            elif non_blank:
                chosen = non_blank[0]

        if chosen:
            return ServerSelection(url=chosen.url, probe=chosen)

        artifacts: Dict[str, str] = {}
        message: Optional[str] = None
        fallback_used = False

        if blank:
            blank_probe = blank[0]
            artifacts = self._capture_blank_artifacts(run_id, blank_probe)
            message = (
                "Primary surface returned a blank DOM. Captured diagnostics and continuing with fallback."
            )
            fallback_used = True
            return ServerSelection(
                url=blank_probe.url,
                probe=blank_probe,
                fallback_used=fallback_used,
                message=message,
                artifacts=artifacts,
            )

        first_error = next((probe for probe in probes if probe.error), None)
        if first_error:
            message = f"Failed to reach {first_error.url}: {first_error.error}"
        else:
            message = "No responsive preview surface detected."

        return ServerSelection(
            url=None,
            probe=None,
            fallback_used=False,
            message=message,
            artifacts=artifacts,
        )

    # Internal helpers -------------------------------------------------

    def _probe_candidate(self, kind: str, url: str) -> ServerProbe:
        request = Request(url, headers={"User-Agent": "SymphonyLite/1.0"})
        try:
            with contextlib.closing(urlopen(request, timeout=5)) as response:
                status = getattr(response, "status", None) or response.getcode()
                content_type = response.headers.get("Content-Type", "")
                body_bytes = response.read()
                try:
                    body = body_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    body = body_bytes.decode("latin-1", errors="ignore")
                parser = _DOMCountingParser()
                try:
                    parser.feed(body)
                except Exception:
                    pass
                stripped = body.strip()
                is_blank = not stripped or parser.node_count <= 1
                return ServerProbe(
                    url=url,
                    kind=kind,
                    status_code=status,
                    content_type=content_type.split(";")[0],
                    is_blank=is_blank,
                    node_count=parser.node_count,
                    body=body,
                )
        except HTTPError as exc:
            return ServerProbe(url=url, kind=kind, status_code=exc.code, error=str(exc))
        except URLError as exc:
            return ServerProbe(url=url, kind=kind, error=str(exc.reason))
        except Exception as exc:
            return ServerProbe(url=url, kind=kind, error=str(exc))

    def _wait_for_port(
        self,
        port: int,
        *,
        timeout: int = 60,
        process: subprocess.Popen | None = None,
        description: str | None = None,
    ) -> None:
        start = time.time()
        while time.time() - start < timeout:
            if process and process.poll() is not None:
                stdout, stderr = process.communicate()
                message = description or f"Service on port {port}"
                detail = (stderr or stdout or "").strip()
                if detail:
                    message += f" failed: {detail.splitlines()[0]}"
                else:
                    message += " exited unexpectedly."
                raise RuntimeError(message)
            try:
                with socket.create_connection(("localhost", port), timeout=2):
                    return
            except OSError:
                pass
            time.sleep(1)
        message = description or f"Service on port {port}"
        if process:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                detail = (stderr or stdout or "").strip()
                if detail:
                    message += f" exited early: {detail.splitlines()[0]}"
                else:
                    message += " exited before becoming ready."
                raise RuntimeError(message)
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            detail = (stderr or stdout or "").strip()
            if detail:
                message += f" timed out. Last output: {detail.splitlines()[0]}"
            else:
                message += " timed out before responding."
            raise TimeoutError(message)

        raise TimeoutError(f"{message} did not become ready in {timeout} seconds")

    def _capture_blank_artifacts(self, run_id: str, probe: ServerProbe) -> Dict[str, str]:
        artifacts_dir = Path("artifacts") / run_id / "servers"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(probe.url)
        html_path = artifacts_dir / f"{slug}_dom.html"
        html_path.write_text(probe.body or "")
        console_path = artifacts_dir / f"{slug}_console.log"
        console_path.write_text("Console logs are unavailable; page rendered blank DOM.\n")
        network_path = artifacts_dir / f"{slug}_network.har"
        network_path.write_text("[]")
        return {
            "dom": str(html_path),
            "console": str(console_path),
            "network": str(network_path),
        }


class _DOMCountingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.node_count = 0

    def handle_starttag(self, tag, attrs):  # pragma: no cover - html parser trivial
        if tag:
            self.node_count += 1


def _slugify(value: str) -> str:
    safe = [ch if ch.isalnum() else "_" for ch in value]
    return "".join(safe)[:60]


def prompt_for_start_command(goal: str, project_root: Path) -> StartCommand:
    """Fallback manual command creation when detection fails."""

    from typer import prompt

    description = prompt("Describe the command (e.g., npm run dev)")
    parts = description.strip().split()
    if not parts:
        raise ValueError("No command entered")
    cwd_input = prompt("Working directory (relative to project root)", default=".")
    kind = prompt("Command type", default="frontend")
    port_input = prompt("Expected port (blank to skip)", default="")
    port = int(port_input) if port_input else None
    url = prompt("URL (blank to auto infer)", default="") or None

    cmd = StartCommand(
        command=parts,
        cwd=project_root / cwd_input,
        kind=kind,
        port=port,
        url=url,
        description=description,
    )
    ensure_config_override(project_root, cmd)
    return cmd
