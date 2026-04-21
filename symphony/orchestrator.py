"""Orchestrator — wires planner, flow executor, evaluator into the
evidence-driven patch cycle: failure → hypothesis → patch → retest."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from symphony.patching.policy import EditMode, PatchDecision, PatchPolicy
from symphony.llm import LLMClient
from symphony.planner.schema import NodeType, TaskNode
from symphony.planner.planner import LLMPlanner
from symphony.flow.dsl import ActionType, FlowAction, FlowScript
from symphony.flow.executor import ActionResult, FlowExecutor, FlowResult
from symphony.evaluator.evaluator import ReliabilityEvaluator
from symphony.prompt.compiler import ContextBlock, PromptCompiler


class CodePatch(BaseModel):
    model_config = ConfigDict(json_schema_extra={})

    file: str = Field(description="Relative path to the file to patch, e.g. 'src/app.py'.")
    search: str = Field(description="Exact string in the file to find and replace.")
    replace: str = Field(description="Replacement string.")


class CodePatchResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={})

    patches: List[CodePatch] = Field(description="List of patches to apply sequentially.")

logger = logging.getLogger(__name__)


# Event callback type: fn(event: str, detail: str)
ProgressCallback = Callable[[str, str], None]
PatchApprovalCallback = Callable[[dict[str, Any]], bool]


def _noop_callback(event: str, detail: str) -> None:
    pass


class Orchestrator:
    """Top-level run coordinator for Symphony v2."""

    def __init__(
        self,
        project_path: Path,
        *,
        llm: Optional[LLMClient] = None,
        token_budget: Optional[int] = None,
        artifact_dir: Optional[Path] = None,
        edit_mode: str = EditMode.ASK.value,
        write_scope: Optional[list[Path]] = None,
        require_manual_approval: bool = False,
        request_patch_approval: Optional[PatchApprovalCallback] = None,
    ):
        self._project = project_path
        self._llm = llm or LLMClient.from_env()
        self._token_budget = token_budget
        self._artifact_dir = artifact_dir or (
            project_path / "artifacts" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        )
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._total_tokens = 0
        # Tracks (cmd, cwd) tuples for every spawned service so they can be
        # restarted after a code patch takes effect.
        self._service_commands: list[dict] = []
        self._service_processes: list[subprocess.Popen] = []
        self._patch_policy = PatchPolicy(
            project_root=self._project,
            mode=EditMode(edit_mode),
            write_scopes=write_scope,
            require_manual_approval=require_manual_approval,
        )
        self._request_patch_approval = request_patch_approval
        self._patch_records: list[dict[str, Any]] = []
        self._manual_approval_blocked = False
        self._emit: ProgressCallback = _noop_callback

    def run(
        self,
        *,
        goal: str,
        profile: str = "web",
        max_passes: int = 1,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Execute the full Symphony pipeline, returning a report dict."""
        emit = on_progress or _noop_callback
        self._emit = emit
        self._patch_records = []
        self._manual_approval_blocked = False
        planner = LLMPlanner(self._llm, token_budget=self._token_budget)
        evaluator = ReliabilityEvaluator()
        compiler = PromptCompiler(token_budget=self._token_budget, model=self._llm.model)
        try:
            # ---- Phase 1: Plan ----
            emit("phase", "Planning")
            emit("detail", "Gathering project context...")
            project_context = self._gather_project_context()
            emit("detail", "Generating task graph via LLM...")
            graph, confidence, planner_tokens = planner.plan(
                goal,
                project_context=project_context,
            )
            self._total_tokens += planner_tokens
            self._save_artifact("taskgraph.json", graph.model_dump_json(indent=2))
            emit("plan_ready", f"{len(graph.nodes)} nodes planned")

            # ---- Phase 2: Execute nodes in topo order ----
            topo = graph.topo_order()
            flow_results: List[FlowResult] = []
            pass_num = 0

            while pass_num < max_passes:
                pass_num += 1
                emit("phase", f"Pass {pass_num}/{max_passes}")

                for i, node_id in enumerate(topo, 1):
                    node = graph.get_node(node_id)
                    logger.info("Executing node: %s (%s)", node.id, node.type.value)
                    emit("node_start", f"[{i}/{len(topo)}] {node.type.value}: {node.description}")

                    nt = node.type
                    if nt == NodeType.STACK_DETECT:
                        self._execute_stack_detect(node)
                    elif nt == NodeType.SERVICE_START:
                        self._execute_service_start(node)
                    elif nt == NodeType.UI_DISCOVERY:
                        self._execute_ui_discovery(node)
                    elif nt == NodeType.WEB_FLOW_TEST:
                        result = self._execute_web_flow(node)
                        if result:
                            flow_results.append(result)
                            status = "pass" if result.passed else "fail"
                            emit("node_result", f"{node.id}: {status} ({len(result.failures)} failures)")
                    elif nt == NodeType.API_CHECK:
                        result = self._execute_api_check(node)
                        if result:
                            flow_results.append(result)
                            status = "pass" if result.passed else "fail"
                            emit("node_result", f"{node.id}: {status}")
                    elif nt == NodeType.CODE_PATCH:
                        self._execute_code_patch(node, compiler, goal, flow_results)
                    elif nt == NodeType.RETEST:
                        ref = node.config.get("retest_node_id")
                        if ref:
                            retest_node = graph.get_node(ref)
                            result = self._execute_web_flow(retest_node)
                            if result:
                                flow_results.append(result)
                                status = "pass" if result.passed else "fail"
                                emit("node_result", f"{node.id}: {status}")
                    elif nt == NodeType.FINALIZE:
                        self._execute_finalize(node)
                    elif nt == NodeType.OTHER:
                        logger.info(
                            "Other node '%s' (type: %s): %s",
                            node.id, node.other_task_type, node.description,
                        )

                # ---- Phase 3: Evaluate ----
                emit("phase", "Evaluating results")
                report = evaluator.evaluate(
                    flow_results,
                    token_usage={"total": self._total_tokens},
                    planner_confidence=confidence,
                )

                if report.passed:
                    logger.info("All checks passed on pass %d", pass_num)
                    break

                if pass_num < max_passes:
                    emit("phase", f"Patch cycle (pass {pass_num} failed)")
                    logger.info("Pass %d failed, attempting auto-patch before retry", pass_num)

                    # Auto-patch: use failure evidence to generate fixes,
                    # regardless of whether the plan included a code_patch node.
                    emit("node_start", "Auto-patching based on failure evidence...")
                    dummy_patch_node = TaskNode(
                        id=f"auto_patch_pass_{pass_num}",
                        type=NodeType.CODE_PATCH,
                        description="Auto-generated patch from failure evidence",
                    )
                    self._execute_code_patch(dummy_patch_node, compiler, goal, flow_results)
                    emit("node_result", f"auto_patch_pass_{pass_num}: done")

                    flow_results.clear()

            # ---- Save report ----
            report_dict = report.to_dict()
            report_dict["patches"] = self._patch_records
            if self._manual_approval_blocked:
                existing_ids = {f.get("id") for f in report_dict.get("failing_reasons", [])}
                if "manual_approval_required" not in existing_ids:
                    report_dict.setdefault("failing_reasons", []).append(
                        {
                            "id": "manual_approval_required",
                            "severity": "critical",
                            "message": (
                                "Patch proposal requires manual approval, but the run is "
                                "non-interactive so edits were not applied."
                            ),
                            "node_id": None,
                            "evidence_path": None,
                        }
                    )
                report_dict["status"] = "fail"
            self._save_artifact("report.json", json.dumps(report_dict, indent=2))
            emit("done", report_dict.get("status", "unknown"))
            return report_dict
        finally:
            self._stop_services()

    # ---- Node executors ----

    def _execute_stack_detect(self, node) -> Dict[str, Any]:
        markers = {
            "package.json": "node",
            "requirements.txt": "python",
            "Cargo.toml": "rust",
            "go.mod": "go",
            "pom.xml": "java",
            "Gemfile": "ruby",
        }
        detected = {}
        for filename, stack in markers.items():
            if (self._project / filename).exists():
                detected[stack] = str(self._project / filename)

        framework_markers = {
            "next.config.js": "nextjs",
            "nuxt.config.ts": "nuxt",
            "angular.json": "angular",
            "vite.config.ts": "vite",
            "webpack.config.js": "webpack",
            "app.py": "flask",
            "manage.py": "django",
        }
        frameworks = []
        for filename, fw in framework_markers.items():
            for _ in self._project.rglob(filename):
                frameworks.append(fw)
                break

        result = {"stacks": detected, "frameworks": frameworks}
        self._save_artifact("stack_detect.json", json.dumps(result, indent=2))
        logger.info("Stack detected: %s, frameworks: %s", detected, frameworks)
        return result

    def _execute_service_start(self, node) -> None:
        commands = node.config.get("commands", [])
        if not commands:
            pkg = self._project / "package.json"
            if pkg.exists():
                commands.append({"cmd": "npm start", "cwd": str(self._project)})
            req = self._project / "requirements.txt"
            if req.exists():
                for entry in ["app.py", "manage.py"]:
                    if (self._project / entry).exists():
                        commands.append({"cmd": f"python {entry}", "cwd": str(self._project)})
                        break
                for sub in self._project.iterdir():
                    if sub.is_dir() and (sub / "app.py").exists():
                        commands.append({"cmd": "python app.py", "cwd": str(sub)})

        specs: list[dict[str, Any]] = []
        for cmd_spec in commands:
            cmd = cmd_spec if isinstance(cmd_spec, str) else cmd_spec.get("cmd", "")
            if not cmd:
                continue
            cwd = (
                cmd_spec.get("cwd", str(self._project))
                if isinstance(cmd_spec, dict)
                else str(self._project)
            )
            health_url = cmd_spec.get("health_url") if isinstance(cmd_spec, dict) else None
            specs.append({"cmd": cmd, "cwd": cwd, "health_url": health_url})

        if not specs:
            logger.warning("No service commands found for service_start node '%s'", node.id)
            return

        self._service_commands = specs
        self._stop_services()
        for spec in self._service_commands:
            proc = self._spawn_service(spec["cmd"], spec["cwd"])
            if proc is not None:
                self._service_processes.append(proc)
        self._wait_for_services(self._service_commands)

    def _spawn_service(self, cmd: str, cwd: str) -> Optional[subprocess.Popen]:
        logger.info("Starting service: %s (cwd=%s)", cmd, cwd)
        try:
            proc = subprocess.Popen(
                cmd, shell=True, cwd=cwd,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return proc
        except Exception as exc:
            logger.warning("Failed to start '%s': %s", cmd, exc)
            return None

    def _restart_services(self) -> None:
        """Restart tracked service commands and wait for readiness."""
        self._stop_services()
        for spec in self._service_commands:
            proc = self._spawn_service(spec["cmd"], spec["cwd"])
            if proc is not None:
                self._service_processes.append(proc)
        self._wait_for_services(self._service_commands)

    def _execute_ui_discovery(self, node) -> Dict[str, Any]:
        logger.info("UI discovery: %s", node.description)
        return {"discovered": True}

    def _execute_web_flow(self, node) -> Optional[FlowResult]:
        if not node.actions:
            logger.warning("Node %s has no actions, skipping", node.id)
            return None

        actions = list(node.actions) + list(node.assertions)

        script = FlowScript(name=node.id, description=node.description, actions=actions)
        try:
            driver = self._get_driver()
            executor = FlowExecutor(
                driver,
                self._artifact_dir / node.id,
                collect_network=getattr(node.evidence_requirements, "network_trace", False),
                collect_dom=getattr(node.evidence_requirements, "dom_snapshot", False),
                collect_focus=getattr(node.evidence_requirements, "focused_element_trace", False),
            )
            result = executor.execute(script)
            logger.info(
                "Flow %s: %s (%d actions, %d failures)",
                node.id, "PASS" if result.passed else "FAIL",
                len(result.results), len(result.failures),
            )
            return result
        except Exception as exc:
            logger.error("Flow execution failed for %s: %s", node.id, exc)
            return None

    def _execute_finalize(self, node) -> None:
        logger.info("Finalize node '%s': stopping tracked services", node.id)
        self._stop_services()

    def _stop_services(self) -> None:
        """Terminate any services that were started by this run."""
        if not self._service_processes:
            return

        for proc in self._service_processes:
            if proc.poll() is not None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning("Could not terminate service process group %s: %s", proc.pid, exc)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            alive = [p for p in self._service_processes if p.poll() is None]
            if not alive:
                break
            time.sleep(0.2)

        for proc in self._service_processes:
            if proc.poll() is not None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning("Could not force-kill service process group %s: %s", proc.pid, exc)

        self._service_processes = []

    def _wait_for_services(self, command_specs: list[dict[str, Any]]) -> None:
        """Poll each service for readiness using health URL or inferred port."""
        for spec in command_specs:
            cmd = spec.get("cmd", "")
            health_url = spec.get("health_url")
            if health_url:
                if not self._wait_for_http_ready(health_url):
                    logger.warning("Service '%s' not ready at %s within timeout", cmd, health_url)
                continue

            port = self._infer_port_from_command(cmd)
            if port is None:
                time.sleep(2)
                continue

            if not self._wait_for_port_ready("127.0.0.1", port):
                logger.warning("Service '%s' did not open localhost:%d within timeout", cmd, port)

    @staticmethod
    def _infer_port_from_command(cmd: str) -> Optional[int]:
        patterns = [
            r"(?:^|\s)PORT=(\d+)(?:\s|$)",
            r"--port(?:=|\s+)(\d+)(?:\s|$)",
            r"-p(?:\s+)(\d+)(?:\s|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, cmd)
            if match:
                return int(match.group(1))
        if any(token in cmd for token in ("npm", "node", "python")):
            return 3000
        return None

    @staticmethod
    def _wait_for_port_ready(host: str, port: int, timeout_s: int = 30) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except OSError:
                time.sleep(0.5)
        return False

    @staticmethod
    def _wait_for_http_ready(url: str, timeout_s: int = 30) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status < 500:
                        return True
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    return True
                time.sleep(0.5)
                continue
            except (urllib.error.URLError, ValueError):
                time.sleep(0.5)
                continue
            except Exception:
                time.sleep(0.5)
                continue
        return False

    def _execute_api_check(self, node) -> Optional[FlowResult]:
        import urllib.request
        import urllib.error

        url = node.config.get("url", "")
        method = node.config.get("method", "GET")
        body = node.config.get("body")
        headers = node.config.get("headers", {})
        expected_status = node.config.get("expected_status", 200)

        if not url:
            logger.warning("API check node %s has no URL", node.id)
            return None

        # Build request with optional JSON body
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8") if isinstance(body, dict) else str(body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

        req = urllib.request.Request(url, method=method, data=data, headers=headers)

        actual_status = None
        response_body = ""
        error_msg = None
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                actual_status = resp.status
                response_body = resp.read().decode("utf-8", errors="replace")[:2000]
        except urllib.error.HTTPError as exc:
            actual_status = exc.code
            response_body = exc.read().decode("utf-8", errors="replace")[:2000]
        except Exception as exc:
            error_msg = str(exc)
            logger.error("API check %s failed: %s", url, exc)

        logger.info("API check %s: %s (expected %d)", url, actual_status, expected_status)

        # Build a synthetic FlowResult so the evaluator sees it
        if error_msg is not None:
            action = FlowAction(action=ActionType.NAVIGATE, value=url)
            ar = ActionResult(
                action=action, success=False,
                message=f"Request failed: {error_msg}",
            )
        else:
            action = FlowAction(
                action=ActionType.ASSERT_HTTP_STATUS,
                value=str(expected_status),
                params={"url_pattern": url},
            )
            passed = actual_status == expected_status
            msg = (
                f"HTTP {actual_status} (expected {expected_status}) for {method} {url}"
                + (f" — body: {response_body[:200]}" if not passed else "")
            )
            ar = ActionResult(action=action, success=passed, message=msg)

        result = FlowResult(
            script_name=node.id,
            passed=ar.success,
            results=[ar],
        )
        return result

    def _execute_code_patch(self, node, compiler, goal, flow_results) -> None:
        failure_evidence = [
            f"[{fr.script_name}] {ar.action.action.value}: {ar.message}"
            for fr in flow_results
            for ar in fr.results
            if not ar.success
        ]
        if not failure_evidence:
            logger.info("No failures to patch")
            return

        blocks = [ContextBlock(name="Failures", content="\n".join(failure_evidence), priority=20)]

        # Use explicitly listed files, or fall back to auto-discovering server-side
        # source files in the project (excluding node_modules, dist, etc.)
        target_files = node.config.get("target_files", [])
        if not target_files:
            target_files = self._discover_source_files()

        for fpath in target_files:
            full = self._project / fpath
            if full.exists():
                blocks.append(ContextBlock(name=f"Source: {fpath}", content=full.read_text()[:10_000], priority=10))

        messages, usage = compiler.compile_with_usage(
            f"Fix the following failures for goal: {goal}",
            blocks,
            user_instruction=(
                "Output a JSON object with 'patches': "
                "[{'file': '<path>', 'search': '<old>', 'replace': '<new>'}]"
            ),
        )
        self._total_tokens += usage.get("total_tokens", 0)

        try:
            system = messages[0]["content"] if messages[0]["role"] == "system" else ""
            user_msgs = [m for m in messages if m["role"] != "system"]
            raw = self._llm.complete(
                user_msgs, system=system, response_schema=CodePatchResponse
            ).strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            llm_patches = json.loads(raw).get("patches", [])
            if not llm_patches:
                logger.info("No LLM patch operations were returned")
                return
        except Exception as exc:
            logger.error("Code patch proposal failed: %s", exc)
            return

        try:
            proposal = self._patch_policy.build_proposal(
                node_id=node.id,
                patches=llm_patches,
                artifact_dir=self._artifact_dir,
            )
        except Exception as exc:
            logger.error("Invalid patch proposal for node %s: %s", node.id, exc)
            blocked_record = {
                "proposal_id": None,
                "node_id": node.id,
                "mode": self._patch_policy.mode.value,
                "decision": PatchDecision.SKIPPED.value,
                "applied_files": [],
                "blocked_reason": f"proposal_invalid: {exc}",
                "proposal_path": None,
                "diff_path": None,
                "summary": {"files": 0, "added": 0, "removed": 0},
            }
            self._patch_records.append(blocked_record)
            self._emit("patch_blocked", json.dumps(blocked_record))
            return

        proposed_payload = proposal.to_event_payload()
        self._emit("patch_proposed", json.dumps(proposed_payload))

        approval_cb = (
            self._request_patch_approval if self._patch_policy.mode == EditMode.ASK else None
        )
        outcome = self._patch_policy.decide_and_apply(
            proposal,
            approval_callback=approval_cb,
        )

        patch_record = {
            "proposal_id": proposal.proposal_id,
            "node_id": node.id,
            "mode": self._patch_policy.mode.value,
            "decision": outcome.decision.value,
            "applied_files": outcome.applied_files,
            "blocked_reason": outcome.blocked_reason,
            "proposal_path": str(proposal.proposal_path),
            "diff_path": str(proposal.diff_path),
            "summary": {
                "files": len(proposal.files),
                "added": proposal.added,
                "removed": proposal.removed,
            },
        }
        self._patch_records.append(patch_record)
        self._emit(
            "patch_decision",
            json.dumps(
                {
                    "proposal_id": proposal.proposal_id,
                    "decision": outcome.decision.value,
                    "blocked_reason": outcome.blocked_reason,
                }
            ),
        )

        if outcome.decision == PatchDecision.APPLIED:
            self._emit(
                "patch_applied",
                json.dumps(
                    {
                        "proposal_id": proposal.proposal_id,
                        "diff_path": str(proposal.diff_path),
                    }
                ),
            )
        else:
            self._emit("patch_blocked", json.dumps(patch_record))

        if outcome.blocked_reason == "manual_approval_required":
            self._manual_approval_blocked = True

        if outcome.applied_files and self._service_commands:
            logger.info("Restarting services after patch")
            self._restart_services()

    def _discover_source_files(self) -> list[str]:
        """Return relative paths of likely server-side source files in the project."""
        skip_dirs = {"node_modules", ".git", "dist", "build", "vendor", ".venv", "__pycache__", "artifacts"}
        extensions = {".js", ".ts", ".jsx", ".tsx", ".mjs", ".py", ".rb", ".go", ".java", ".vue", ".svelte", ".php"}
        priority_keywords = {"server", "app", "auth", "route", "api", "main", "index"}
        found: list[tuple[int, str]] = []
        root = str(self._project)
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skipped directories in-place so os.walk doesn't descend
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fname in filenames:
                ext = os.path.splitext(fname)[1]
                if ext not in extensions:
                    continue
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, root)
                stem = os.path.splitext(fname)[0].lower()
                priority = 0 if any(k in stem for k in priority_keywords) else 1
                found.append((priority, rel))
        found.sort()
        return [rel for _, rel in found[:10]]

    def _get_driver(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        opts = Options()
        opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        return webdriver.Chrome(options=opts)

    def _gather_project_context(self) -> str:
        parts = []
        for marker in ["package.json", "requirements.txt", "pyproject.toml"]:
            p = self._project / marker
            if p.exists():
                parts.append(f"=== {marker} ===\n{p.read_text()[:2000]}")
        files = [f.name for f in self._project.iterdir() if not f.name.startswith(".")]
        parts.append(f"=== Files ===\n{', '.join(sorted(files))}")

        # Include HTML and JS files so the planner can see actual selectors,
        # element IDs, page structure, and the runtime behavior (status text,
        # event handlers, API calls) that determines what assertions to write.
        skip_dirs = {"node_modules", ".git", "dist", "build", "vendor", ".venv", "artifacts"}
        source_exts = {".html", ".js"}
        file_count = 0
        max_files = 15
        for dirpath, dirnames, filenames in os.walk(str(self._project)):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1]
                if ext not in source_exts:
                    continue
                full = Path(dirpath) / fname
                rel = full.relative_to(self._project)
                content = full.read_text(errors="replace")[:4000]
                parts.append(f"=== {rel} ===\n{content}")
                file_count += 1
                if file_count >= max_files:
                    break
            if file_count >= max_files:
                break

        return "\n\n".join(parts)

    def _save_artifact(self, name: str, content: str) -> Path:
        path = self._artifact_dir / name
        path.write_text(content)
        return path
