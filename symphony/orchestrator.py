"""Orchestrator — wires planner, flow executor, evaluator into the
evidence-driven patch cycle: failure → hypothesis → patch → retest."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from symphony.planner.schema import NodeType, TaskGraph
from symphony.planner.planner import LLMPlanner
from symphony.flow.dsl import ActionType, FlowAction, FlowScript
from symphony.flow.executor import FlowExecutor, FlowResult
from symphony.evaluator.evaluator import (
    EvalReport,
    ReliabilityEvaluator,
)
from symphony.prompt.compiler import ContextBlock, PromptCompiler

logger = logging.getLogger(__name__)


class Orchestrator:
    """Top-level run coordinator for Symphony v2."""

    def __init__(
        self,
        project_path: Path,
        *,
        model: str = "claude-sonnet-4-20250514",
        token_budget: int = 16_000,
        artifact_dir: Path | None = None,
    ):
        self._project = project_path
        self._model = model
        self._token_budget = token_budget
        self._artifact_dir = artifact_dir or (
            project_path / "artifacts" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        )
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

        self._total_tokens = 0

    def run(
        self, *, goal: str, profile: str = "web", max_passes: int = 1
    ) -> dict[str, Any]:
        """Execute the full Symphony pipeline, returning a report dict."""
        import anthropic

        client = anthropic.Anthropic()
        planner = LLMPlanner(client, model=self._model, token_budget=self._token_budget)
        evaluator = ReliabilityEvaluator()
        compiler = PromptCompiler(token_budget=self._token_budget)

        # ---- Phase 1: Plan ----
        project_context = self._gather_project_context()
        graph, confidence = planner.plan(
            goal,
            project_context=project_context,
            project_path=str(self._project),
        )

        self._save_artifact("taskgraph.json", graph.model_dump_json(indent=2))

        # ---- Phase 2: Execute nodes in topo order ----
        topo = graph.topo_order()
        flow_results: list[FlowResult] = []
        pass_num = 0

        while pass_num < max_passes:
            pass_num += 1
            logger.info("=== Pass %d/%d ===", pass_num, max_passes)

            for node_id in topo:
                node = graph.get_node(node_id)
                logger.info("Executing node: %s (%s)", node.id, node.type.value)

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
                elif nt == NodeType.API_CHECK:
                    self._execute_api_check(node)
                elif nt == NodeType.CODE_PATCH:
                    self._execute_code_patch(node, client, compiler, goal, flow_results)
                elif nt == NodeType.RETEST:
                    ref = node.config.get("retest_node_id")
                    if ref:
                        retest_node = graph.get_node(ref)
                        result = self._execute_web_flow(retest_node)
                        if result:
                            flow_results.append(result)
                elif nt == NodeType.FINALIZE:
                    pass  # handled below
                elif nt == NodeType.OTHER:
                    logger.info(
                        "Other node '%s' (type: %s): %s",
                        node.id, node.other_task_type, node.description,
                    )

            # ---- Phase 3: Evaluate ----
            report = evaluator.evaluate(
                flow_results,
                token_usage={"total": self._total_tokens},
                planner_confidence=confidence,
            )

            if report.passed:
                logger.info("All checks passed on pass %d", pass_num)
                break

            if pass_num < max_passes:
                logger.info("Pass %d failed, will attempt fix cycle", pass_num)
                # Inject a code_patch node dynamically for next pass
                # The next iteration will re-run the graph
                flow_results.clear()

        # ---- Save report ----
        report_dict = report.to_dict()
        self._save_artifact("report.json", json.dumps(report_dict, indent=2))

        return report_dict

    # ---- Node executors ----

    def _execute_stack_detect(self, node) -> dict[str, Any]:
        """Detect project stack from file markers."""
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

        # Check for framework-specific files
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
            for p in self._project.rglob(filename):
                frameworks.append(fw)
                break

        result = {"stacks": detected, "frameworks": frameworks}
        self._save_artifact("stack_detect.json", json.dumps(result, indent=2))
        logger.info("Stack detected: %s, frameworks: %s", detected, frameworks)
        return result

    def _execute_service_start(self, node) -> None:
        """Start services defined in node config or auto-detected scripts."""
        commands = node.config.get("commands", [])
        if not commands:
            # Auto-detect from package.json / common patterns
            pkg = self._project / "package.json"
            if pkg.exists():
                commands.append({"cmd": "npm start", "cwd": str(self._project)})
            req = self._project / "requirements.txt"
            if req.exists():
                # Check for app.py, manage.py
                for entry in ["app.py", "manage.py"]:
                    if (self._project / entry).exists():
                        commands.append({
                            "cmd": f"python {entry}",
                            "cwd": str(self._project),
                        })
                        break
                # Check subdirectories
                for sub in self._project.iterdir():
                    if sub.is_dir() and (sub / "app.py").exists():
                        commands.append({
                            "cmd": "python app.py",
                            "cwd": str(sub),
                        })

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

        # Brief pause for services to initialize
        time.sleep(2)

    def _execute_ui_discovery(self, node) -> dict[str, Any]:
        """Placeholder for UI discovery — would use headless browser."""
        logger.info("UI discovery: %s", node.description)
        return {"discovered": True}

    def _execute_web_flow(self, node) -> FlowResult | None:
        """Execute a web flow test node."""
        if not node.actions:
            logger.warning("Node %s has no actions, skipping", node.id)
            return None

        # Build FlowScript from node
        actions = []
        for a in node.actions:
            if isinstance(a, dict):
                actions.append(FlowAction.model_validate(a))
            elif isinstance(a, FlowAction):
                actions.append(a)

        # Add assertion actions
        for a in node.assertions:
            if isinstance(a, dict):
                actions.append(FlowAction.model_validate(a))
            elif isinstance(a, FlowAction):
                actions.append(a)

        script = FlowScript(
            name=node.id,
            description=node.description,
            actions=actions,
        )

        # Execute with browser
        try:
            driver = self._get_driver()
            executor = FlowExecutor(
                driver,
                self._artifact_dir / node.id,
                collect_dom=node.evidence_requirements.dom_snapshot
                if hasattr(node.evidence_requirements, "dom_snapshot")
                else False,
                collect_focus=node.evidence_requirements.focused_element_trace
                if hasattr(node.evidence_requirements, "focused_element_trace")
                else False,
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

    def _execute_api_check(self, node) -> None:
        """Execute an API check node."""
        import urllib.request
        import urllib.error

        url = node.config.get("url", "")
        method = node.config.get("method", "GET")
        expected_status = node.config.get("expected_status", 200)

        if not url:
            logger.warning("API check node %s has no URL", node.id)
            return

        try:
            req = urllib.request.Request(url, method=method)
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                logger.info("API check %s: %d (expected %d)", url, status, expected_status)
        except urllib.error.HTTPError as exc:
            logger.info("API check %s: %d (expected %d)", url, exc.code, expected_status)
        except Exception as exc:
            logger.error("API check %s failed: %s", url, exc)

    def _execute_code_patch(
        self, node, client, compiler, goal, flow_results
    ) -> None:
        """Use LLM to generate a code patch based on failure evidence."""
        failure_evidence = []
        for fr in flow_results:
            for ar in fr.results:
                if not ar.success:
                    failure_evidence.append(
                        f"[{fr.script_name}] {ar.action.action.value}: {ar.message}"
                    )

        if not failure_evidence:
            logger.info("No failures to patch")
            return

        blocks = [
            ContextBlock(
                name="Failures",
                content="\n".join(failure_evidence),
                priority=20,
            ),
        ]

        # Read relevant source files
        target_files = node.config.get("target_files", [])
        for fpath in target_files:
            full = self._project / fpath
            if full.exists():
                blocks.append(ContextBlock(
                    name=f"Source: {fpath}",
                    content=full.read_text()[:10_000],
                    priority=10,
                ))

        messages, usage = compiler.compile_with_usage(
            f"Fix the following failures for goal: {goal}",
            blocks,
            user_instruction="Output a JSON object with 'patches': [{'file': '<path>', 'search': '<old>', 'replace': '<new>'}]",
        )
        self._total_tokens += usage["total_tokens"]

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=messages[0]["content"],
                messages=[m for m in messages if m["role"] != "system"],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            patches = json.loads(raw).get("patches", [])
            for patch in patches:
                fpath = self._project / patch["file"]
                if fpath.exists():
                    content = fpath.read_text()
                    content = content.replace(patch["search"], patch["replace"])
                    fpath.write_text(content)
                    logger.info("Patched %s", patch["file"])
        except Exception as exc:
            logger.error("Code patch failed: %s", exc)

    # ---- Helpers ----

    def _get_driver(self):
        """Get or create a Selenium WebDriver."""
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        opts = Options()
        opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        return webdriver.Chrome(options=opts)

    def _gather_project_context(self) -> str:
        """Gather minimal project context for the planner."""
        parts = []
        for marker in ["package.json", "requirements.txt", "pyproject.toml"]:
            p = self._project / marker
            if p.exists():
                parts.append(f"=== {marker} ===\n{p.read_text()[:2000]}")
        # List top-level files
        files = [f.name for f in self._project.iterdir() if not f.name.startswith(".")]
        parts.append(f"=== Files ===\n{', '.join(sorted(files))}")
        return "\n\n".join(parts)

    def _save_artifact(self, name: str, content: str) -> Path:
        path = self._artifact_dir / name
        path.write_text(content)
        return path
