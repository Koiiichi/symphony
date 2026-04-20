"""Orchestrator — wires planner, flow executor, evaluator into the
evidence-driven patch cycle: failure → hypothesis → patch → retest."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from symphony.llm import LLMClient
from symphony.planner.schema import NodeType
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
    ):
        self._project = project_path
        self._llm = llm or LLMClient.from_env()
        self._token_budget = token_budget
        self._artifact_dir = artifact_dir or (
            project_path / "artifacts" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        )
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._total_tokens = 0

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
        planner = LLMPlanner(self._llm, token_budget=self._token_budget)
        evaluator = ReliabilityEvaluator()
        compiler = PromptCompiler(token_budget=self._token_budget, model=self._llm.model)

        # ---- Phase 1: Plan ----
        emit("phase", "Planning")
        emit("detail", "Gathering project context...")
        project_context = self._gather_project_context()
        emit("detail", "Generating task graph via LLM...")
        graph, confidence, planner_tokens = planner.plan(
            goal,
            project_context=project_context,
            project_path=str(self._project),
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
                    pass
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
                emit("detail", f"Pass {pass_num} failed, retrying...")
                logger.info("Pass %d failed, will attempt fix cycle", pass_num)
                flow_results.clear()

        # ---- Save report ----
        report_dict = report.to_dict()
        self._save_artifact("report.json", json.dumps(report_dict, indent=2))
        emit("done", report_dict.get("status", "unknown"))
        return report_dict

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

        for cmd_spec in commands:
            cmd = cmd_spec if isinstance(cmd_spec, str) else cmd_spec.get("cmd", "")
            cwd = cmd_spec.get("cwd", str(self._project)) if isinstance(cmd_spec, dict) else str(self._project)
            logger.info("Starting service: %s (cwd=%s)", cmd, cwd)
            try:
                subprocess.Popen(
                    cmd, shell=True, cwd=cwd,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception as exc:
                logger.warning("Failed to start '%s': %s", cmd, exc)
        time.sleep(2)

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
        for fpath in node.config.get("target_files", []):
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
            for patch in json.loads(raw).get("patches", []):
                fpath = self._project / patch["file"]
                if fpath.exists():
                    fpath.write_text(fpath.read_text().replace(patch["search"], patch["replace"]))
                    logger.info("Patched %s", patch["file"])
        except Exception as exc:
            logger.error("Code patch failed: %s", exc)

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

        # Include HTML files so the planner can see actual form selectors,
        # element IDs, and page structure.
        html_files = sorted(self._project.rglob("*.html"))
        for hf in html_files[:10]:  # cap to avoid blowing context
            rel = hf.relative_to(self._project)
            # Skip node_modules, vendor dirs, etc.
            if any(p.startswith(".") or p in ("node_modules", "vendor", "dist", "build")
                   for p in rel.parts):
                continue
            content = hf.read_text(errors="replace")[:3000]
            parts.append(f"=== {rel} ===\n{content}")

        return "\n\n".join(parts)

    def _save_artifact(self, name: str, content: str) -> Path:
        path = self._artifact_dir / name
        path.write_text(content)
        return path
