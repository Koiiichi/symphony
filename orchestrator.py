from __future__ import annotations

import os
import sys
import time
import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rich.console import Console

from agents.brain_agent_factory import BrainConfig, SensoryConfig, create_brain_agent
from agents.brain_instructions import get_generation_instructions
from agents.goal_interpreter import build_expectations
from agents.sensory_agent import inspect_site
from agents.sensory_contract import SensoryReport
from core.intent import classify_intent
from core.router import build_agent_plan, required_agents
from core.runtime import ServerManager, prompt_for_start_command
from core.stack import analyze_project
from core.tui import SymphonyTUI
from core.types import (
    AgentHooks,
    IntentResult,
    PassOutcome,
    StackInfo,
    WorkflowConfig,
    WorkflowSummary,
)
from gates.engine import evaluate as evaluate_gates, get_fix_instructions as build_gate_fix_instructions, run_build_gate
from core.rollback import create_checkpoint, restore_checkpoint, discard_checkpoint
from core.vision_result import VisionResult, parse_vision_payload, write_raw_payload


console = Console()


def _stack_to_dict(stack: StackInfo) -> Dict[str, object]:
    return {
        "root": str(stack.root),
        "has_content": stack.has_code,
        "frameworks": stack.frameworks,
        "frontend": stack.frontend,
        "backend": stack.backend,
    }


def _summarize_stack(stack: StackInfo) -> str:
    def describe(kind: str, detected: Optional[str]) -> str:
        if detected:
            return f"{kind}: {detected}"
        has_command = any(cmd.kind == kind for cmd in stack.start_commands)
        if has_command:
            return f"{kind}: start command ready"
        return f"{kind}: none"

    parts = [describe("frontend", stack.frontend), describe("backend", stack.backend)]
    return ", ".join(parts)


def _summarize_list(values: Iterable[str], *, limit: int = 4) -> str:
    values = list(values)
    if len(values) <= limit:
        return ", ".join(values)
    clipped = values[:limit]
    return ", ".join(clipped) + ", …"


def _summarize_vision_report(report_data: Dict[str, object]) -> list[str]:
    lines: list[str] = []

    scores_dict = {}
    if isinstance(report_data.get("scores"), dict):
        scores_dict = report_data["scores"]  # type: ignore[assignment]
    elif isinstance(report_data.get("vision_scores"), dict):
        scores_dict = report_data["vision_scores"]  # type: ignore[assignment]
    else:
        for key, label in (
            ("alignment_score", "alignment"),
            ("spacing_score", "spacing"),
            ("contrast_score", "contrast"),
        ):
            value = report_data.get(key)
            if isinstance(value, (int, float)):
                scores_dict[label] = value
    if scores_dict:
        score_bits = []
        for key in ("alignment", "spacing", "contrast"):
            value = scores_dict.get(key)
            if isinstance(value, (int, float)):
                score_bits.append(f"{key}: {value:.2f}")
        if score_bits:
            vision_scores = report_data.get("vision_scores", {})
            source = vision_scores.get("source", "heuristic")
            source_indicator = " (via Vision API)" if source == "vision_api" else " (heuristic)"
            lines.append("Scores – " + ", ".join(score_bits) + source_indicator)

    sections = report_data.get("visible_sections")
    if isinstance(sections, Iterable) and not isinstance(sections, (str, bytes)):
        section_list = [str(section) for section in sections if section]
        if section_list:
            lines.append("Sections in view – " + _summarize_list(section_list))

    visited = report_data.get("visited_urls")
    if isinstance(visited, Iterable) and not isinstance(visited, (str, bytes)):
        visited_list = [str(url) for url in visited if url]
        if visited_list:
            lines.append("Visited – " + _summarize_list(visited_list, limit=3))

    interactions = report_data.get("interactions")
    if isinstance(interactions, dict):
        for key, meta in interactions.items():
            if not isinstance(meta, dict):
                continue
            status = "ok" if meta.get("ok") else "not attempted" if not meta.get("attempted") else "fail"
            selector = meta.get("selector")
            detail = meta.get("notes")
            text = f"Interaction – {key}: {status}"
            if selector:
                text += f" ({selector})"
            if detail:
                text += f" – {detail}"
            lines.append(text)

    a11y = report_data.get("accessibility") or report_data.get("a11y")
    if isinstance(a11y, dict):
        violations = a11y.get("violations")
        if isinstance(violations, int):
            wcag_level = a11y.get("target") or a11y.get("wcag_level")
            text = f"Accessibility – {violations} violation{'s' if violations != 1 else ''}"
            if wcag_level:
                text += f", target: {wcag_level}"
            lines.append(text)

    warnings = report_data.get("warnings")
    if isinstance(warnings, Iterable) and not isinstance(warnings, (str, bytes)):
        warn_list = [str(w) for w in warnings if w]
        if warn_list:
            lines.append("Warnings – " + "; ".join(warn_list))

    issues = report_data.get("issues")
    if isinstance(issues, Iterable):
        for issue in issues:
            if isinstance(issue, dict) and issue.get("detail"):
                lines.append(f"Issue – {issue.get('id', 'unknown')}: {issue['detail']}")

    return lines


def _build_preview_hints(expectations: Dict[str, Any], mode: str) -> Dict[str, List[str]]:
    selectors: List[str] = []
    keywords: List[str] = []
    if mode == "qa":
        for interaction in expectations.get("interactions", []):
            if not isinstance(interaction, dict):
                continue
            selector = interaction.get("selector")
            if selector:
                selectors.append(str(selector))
        if selectors:
            keywords.append("form")
    else:
        keywords.extend(["section", "header", "hero"])
    return {"selectors": selectors, "keywords": keywords}


def _sensory_to_vision_payload(
    report: SensoryReport,
    *,
    mode: str,
    url: str,
) -> Dict[str, Any]:
    data = report.to_dict()
    interactions_payload: List[Dict[str, Any]] = []
    interactions = data.get("interactions", {})
    if isinstance(interactions, dict):
        for key, value in interactions.items():
            if not isinstance(value, dict):
                continue
            attempted = bool(value.get("attempted"))
            success = bool(value.get("success_banner")) and not value.get("error_banner")
            notes = value.get("details") or ""
            errors = value.get("errors")
            if errors and isinstance(errors, list):
                notes = "; ".join(str(err) for err in errors if err)
            interactions_payload.append(
                {
                    "id": key,
                    "action": value.get("action", "form_submit"),
                    "selector": value.get("selector"),
                    "ok": attempted and success,
                    "notes": notes or None,
                    "attempted": attempted,
                    "http_status": value.get("http_status"),
                    "success_banner": bool(value.get("success_banner")),
                    "error_banner": bool(value.get("error_banner")),
                }
            )
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    issues = []
    for warning in warnings or []:
        issues.append({"id": "warning", "status": "warn", "detail": str(warning)})
    for reason in data.get("failing_reasons", []):
        issues.append({"id": "gate_failure", "status": "fail", "detail": str(reason)})

    return {
        "version": "1.0",
        "target_url": url,
        "mode": mode,
        "scores": {
            "alignment": data.get("alignment_score"),
            "spacing": data.get("spacing_score"),
            "contrast": data.get("contrast_score"),
            "source": (
                data.get("vision_scores", {}).get("source")
                if isinstance(data.get("vision_scores"), dict)
                else None
            ),
        },
        "accessibility": {
            "violations": data.get("a11y", {}).get("violations") if isinstance(data.get("a11y"), dict) else None,
            "target": data.get("a11y", {}).get("wcag_level") if isinstance(data.get("a11y"), dict) else "AA",
        },
        "interactions": interactions_payload,
        "issues": issues,
        "suggestions": [],
        "artifacts": {
            "screenshots": [
                screen.get("path")
                for screen in data.get("screens", [])
                if isinstance(screen, dict) and screen.get("path")
            ]
        },
    }


def _format_success_message(goal: str, intent: Optional[IntentResult], summary: WorkflowSummary) -> str:
    """Create a friendly confirmation message for successful runs."""

    passes_run = len(summary.passes)
    changes_made = any(pass_outcome.changes_made for pass_outcome in summary.passes)

    goal_text = goal.strip().rstrip(".")

    if intent and intent.topic == "ui_ux":
        base = f"Done! \"{goal_text}\" is handled and your UI/UX experience is polished."
    else:
        base = f"Done! \"{goal_text}\" is handled."

    detail = ""
    if passes_run:
        detail = f" after {passes_run} pass{'es' if passes_run != 1 else ''}"

    if changes_made:
        tail = "Symphony applied targeted fixes for you to review."
    else:
        tail = "No code changes were needed—the experience already met the expectations."

    return f"{base}{detail}. {tail}"


def _extract_failure_categories(reasons: list[str]) -> set[str]:
    """Extract category keys from gate failure strings.

    ``"alignment_score: 0.70 < 0.90"`` → ``"alignment_score"``
    ``"contact_submit: http_status 501 not 2xx"`` → ``"contact_submit"``
    """
    cats: set[str] = set()
    for reason in reasons:
        key = reason.split(":")[0].strip()
        if key:
            cats.add(key)
    return cats


def _format_stalled_message(failing_reasons: list[str]) -> str:
    if failing_reasons:
        issues = ", ".join(failing_reasons)
        return (
            "Run stopped because no progress was observed across repeated passes: "
            f"{issues}. Address them manually and try again."
        )
    return "Run stopped because progress stalled across multiple passes. Review the latest output and try again."


def _format_max_passes_message(failing_reasons: Optional[list[str]]) -> str:
    if failing_reasons:
        issues = ", ".join(failing_reasons)
        return (
            "Reached the maximum number of passes while the following items still need attention: "
            f"{issues}."
        )
    return "Reached the maximum number of passes without clearing every expectation."


@dataclass
class DefaultAgentHooks(AgentHooks):
    project_path: Path
    brain_config: BrainConfig
    sensory_config: SensoryConfig
    run_id: str
    vision_mode: str
    _brain_logs: Dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._brain_agent = None

    def _ensure_brain(self):
        if self._brain_agent is None:
            self._brain_agent = create_brain_agent(self.project_path, self.brain_config, self.run_id)

    def run_brain(self, instructions: str, *, pass_index: int):
        self._ensure_brain()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = self._brain_agent.run(instructions)

        combined_output = "".join(
            part for part in (stdout_buffer.getvalue(), stderr_buffer.getvalue()) if part
        )

        if combined_output.strip():
            self._brain_logs[pass_index] = combined_output

        return result

    def consume_brain_log(self, pass_index: int) -> Optional[str]:
        return self._brain_logs.pop(pass_index, None)

    def run_vision(self, url: str, expectations: Dict[str, object], *, pass_index: int, mode: str):
        report: SensoryReport = inspect_site(
            url,
            self.run_id,
            {"model_id": self.sensory_config.model_id},
            expectations,
            mode=mode,
        )
        return report


def _require_api_keys(agent_names) -> None:
    mapping = {
        "brain": "SYMPHONY_BRAIN_API_KEY",
        "vision": "SYMPHONY_VISION_API_KEY",
    }
    missing = []
    for agent in agent_names:
        env_key = mapping.get(agent)
        if env_key and not os.environ.get(env_key):
            missing.append(env_key)
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(sorted(set(missing)))}")


def run_workflow(
    config: WorkflowConfig,
    brain_config: Optional[BrainConfig] = None,
    sensory_config: Optional[SensoryConfig] = None,
    hooks: Optional[AgentHooks] = None,
    *,
    stack: Optional[StackInfo] = None,
    intent: Optional[IntentResult] = None,
) -> WorkflowSummary:
    project_path = config.project_path.resolve()
    brain_config = brain_config or BrainConfig()
    sensory_config = sensory_config or SensoryConfig()

    run_id = config.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    summary = WorkflowSummary()

    tui = SymphonyTUI(detailed=config.detailed_log)
    success = False
    keep_servers_running = False
    preview_url: Optional[str] = None
    detected_stack = stack
    detected_intent = intent
    plan = []
    agents_needed = []
    server_manager: Optional[ServerManager] = None
    brain_log_notice_shown = False

    try:
        with tui.live():
            tui.set_header(
                Project=str(project_path),
                Goal=config.goal,
                Mode="detecting…",
                Passes=config.max_passes,
                Run=run_id,
                Vision=config.vision_mode,
            )

            tui.update_status("Stack Detection", "RUNNING")
            tui.start_activity("Scanning project stack…", spinner="pulsing_star")
            if detected_stack is None:
                detected_stack = analyze_project(project_path)
            summary.stack = detected_stack
            tui.stop_activity("Stack: scan complete", icon="[stack]")
            stack_summary = _summarize_stack(detected_stack)
            tui.update_status("Stack Detection", "COMPLETE", detail=stack_summary)

            tui.update_status("Intent", "RUNNING")
            tui.start_activity("Interpreting goal and planning agents…", spinner="orbit")
            if detected_intent is None:
                detected_intent = classify_intent(config.goal, detected_stack)
            plan = build_agent_plan(detected_intent, include_ui_validation=True)
            agents_needed = required_agents(plan)
            summary.intent = detected_intent
            tui.stop_activity("Intent: routing ready", icon="[intent]")
            tui.set_header(
                Project=str(project_path),
                Goal=config.goal,
                Mode=f"{detected_intent.intent} ({detected_intent.topic})",
                Passes=config.max_passes,
                Run=run_id,
                Vision=config.vision_mode,
            )

            tui.add_voice("Analyzing project and classifying intent…")
            tui.add_sub_info(f"Detected intent: {detected_intent.intent}, topic: {detected_intent.topic}")
            tui.update_status("Intent", "COMPLETE")

            if config.dry_run:
                tui.add_voice("Dry-run mode – no files will be written.")
                tui.add_sub_info("Agent plan:")
                for step in plan:
                    tui.add_sub_info(f"{step.agent}: {step.description}")
                tui.add_sub_info(f"Start commands detected: {len(detected_stack.start_commands)}")
                summary.status = "dry_run"
                return summary

            _require_api_keys(agents_needed)

            server_manager = ServerManager(detected_stack, project_root=project_path)

            if not detected_stack.start_commands:
                tui.add_voice("No start command detected. Requesting manual command…")
                with tui.suspend_live():
                    manual = prompt_for_start_command(config.goal, project_path)
                detected_stack.start_commands.append(manual)

            tui.update_status("Servers", "STARTING")
            tui.start_activity("Spooling up local services…", spinner="pulsing_star")
            preferred_kind = None
            if detected_intent and detected_intent.topic == "ui_ux":
                preferred_kind = "frontend"
            try:
                urls = server_manager.start_all(preferred_kind=preferred_kind)
            except TimeoutError as exc:
                tui.stop_activity("Server startup timed out", icon="[warn]")
                summary.status = "error"
                message = str(exc)
                summary.final_message = message
                tui.set_footer(message)
                raise RuntimeError(message) from exc
            except RuntimeError as exc:
                tui.stop_activity("Server startup failed", icon="[warn]")
                summary.status = "error"
                message = str(exc)
                summary.final_message = message
                tui.set_footer(message)
                raise
            else:
                tui.stop_activity("Local services ready", icon="[ready]")
            tui.update_status("Servers", "READY", detail=", ".join(f"{k}: {v}" for k, v in urls.items()))
            summary.urls = urls
            hooks = hooks or DefaultAgentHooks(
                project_path,
                brain_config,
                sensory_config,
                run_id,
                config.vision_mode,
            )

            expectations = build_expectations(
                config.goal,
                page_type_hint=None,
                stack=_stack_to_dict(detected_stack),
                vision_mode=config.vision_mode,
            )
            tui.update_status(
                "Expectations",
                "READY",
                detail=f"{len(expectations.get('capabilities', {}))} capabilities",
            )

            hints = _build_preview_hints(expectations, config.vision_mode)
            selection = server_manager.resolve_preview_surface(
                run_id=run_id,
                preferred_kind=preferred_kind,
                hints=hints,
            )
            if selection.artifacts:
                for key, path in selection.artifacts.items():
                    summary.artifacts[f"server_{Path(path).name}"] = path
            if selection.message:
                tui.add_sub_info(selection.message)

            preview_url = selection.url
            if selection.probe and selection.probe.is_blank:
                summary.status = "error"
                final_message = (
                    "Preview surface rendered a blank document. Review captured diagnostics before retrying."
                )
                summary.final_message = final_message
                tui.set_footer(final_message)
                if selection.artifacts:
                    tui.add_sub_info(
                        "Artifacts: " + ", ".join(selection.artifacts.values())
                    )
                keep_servers_running = False
                return summary

            if not preview_url:
                summary.status = "error"
                final_message = selection.message or "No reachable preview surface after server startup."
                summary.final_message = final_message
                tui.set_footer(final_message)
                keep_servers_running = False
                return summary

            keep_servers_running = config.open_browser and bool(preview_url)

            last_report: Optional[VisionResult] = None
            last_failures: Optional[list[str]] = None
            stagnation_counter = 0
            _active_checkpoint: Optional[str] = None

            brain_in_plan = any(step.agent == "brain" for step in plan)
            pending_handoff: Optional[Tuple[str, str]] = None

            if plan:
                tui.add_voice(
                    "Plan ready: "
                    + " → ".join(f"{step.agent.capitalize()}" for step in plan)
                )

            for index in range(1, config.max_passes + 1):
                tui.update_status("Pass", f"{index}/{config.max_passes}", detail="running")
                changes_made = False
                pass_report: Optional[SensoryReport] = None
                failing_reasons: list[str] = []
                pass_outcome: Optional[PassOutcome] = None

                for step_idx, step in enumerate(plan):
                    remaining_steps = plan[step_idx + 1 :]
                    if step.agent == "vision":
                        if pending_handoff:
                            message, icon = pending_handoff
                            tui.stop_activity(message, icon=icon)
                            pending_handoff = None
                        vision_url = (
                            preview_url or urls.get("frontend") or detected_stack.frontend_url
                        )
                        if not vision_url:
                            raise RuntimeError("Vision agent requires a frontend URL")
                        tui.start_activity(
                            f"Vision: {step.description}…",
                            spinner="pulsing_star",
                        )
                        tui.update_activity_progress(f"mode: {config.vision_mode}")
                        tui.update_activity_progress("viewports: 360 | 768 | 1280")
                        pass_report = None
                        raw_report = hooks.run_vision(
                            vision_url,
                            expectations,
                            pass_index=index,
                            mode=config.vision_mode,
                        )
                        tui.stop_activity("Vision: Audit complete", icon="[vision]")
                        if isinstance(raw_report, SensoryReport):
                            pass_report = raw_report
                            payload = _sensory_to_vision_payload(
                                raw_report, mode=config.vision_mode, url=vision_url
                            )
                        else:
                            payload = raw_report
                        vision_result, parse_warnings = parse_vision_payload(
                            payload,
                            url=vision_url,
                            mode=config.vision_mode,
                        )
                        report_data = vision_result.to_observations()
                        if parse_warnings:
                            report_data.setdefault("warnings", []).extend(parse_warnings)
                            artifact_payload = (
                                payload.to_dict() if hasattr(payload, "to_dict") else payload
                            )
                            artifact_path = write_raw_payload(run_id, index, artifact_payload)
                            summary.artifacts[f"vision_raw_pass_{index}"] = str(artifact_path)
                            tui.add_sub_info(
                                "Vision output invalid; using fallback parser"
                            )
                        for info_line in _summarize_vision_report(report_data or {}):
                            tui.add_sub_info(info_line)
                        gate_result = evaluate_gates(expectations, report_data)
                        failing_reasons = gate_result["failing_reasons"]
                        if failing_reasons:
                            tui.add_sub_info(
                                "Issues: " + ", ".join(failing_reasons)
                            )
                            if brain_in_plan:
                                issue_count = len(failing_reasons)
                                handoff = (
                                    f"Vision => Brain: Sharing {issue_count} finding"
                                    f"{'s' if issue_count != 1 else ''} for fixes."
                                )
                                pending_handoff = (handoff, ">>")
                                tui.start_activity("Hand-off: Vision => Brain…", spinner="pulsing_star")
                        else:
                            tui.add_sub_info("Issues: none")
                            pass_outcome = PassOutcome(
                                index=index,
                                vision_passed=True,
                                changes_made=changes_made,
                                failing_reasons=[],
                                summary={"status": "passed"},
                            )
                            summary.add_pass(pass_outcome)
                            summary.status = "success"
                            final_message = _format_success_message(config.goal, detected_intent, summary)
                            summary.final_message = final_message
                            tui.set_footer(final_message)
                            success = True
                            keep_servers_running = config.open_browser and bool(preview_url)
                            break
                        last_report = vision_result
                    elif step.agent == "brain":
                        if pending_handoff:
                            message, icon = pending_handoff
                            tui.stop_activity(message, icon=icon)
                            pending_handoff = None

                        # Checkpoint before brain writes code
                        checkpoint_label = f"symphony-pre-pass-{index}"
                        _active_checkpoint = create_checkpoint(project_path, checkpoint_label)
                        if not _active_checkpoint:
                            tui.add_sub_info("Checkpoint skipped (requires a clean git subtree)")

                        if pass_report is None and last_report is None:
                            instructions = get_generation_instructions(
                                str(project_path),
                                config.goal,
                                _stack_to_dict(detected_stack),
                            )
                            if preview_url:
                                instructions += (
                                    "\n\n## Runtime Context\n"
                                    f"- Preview URL: {preview_url}\n"
                                    f"- Contact API should be reachable at: {preview_url.rstrip('/')}/api/contact\n"
                                )
                        else:
                            report_for_fix = pass_report or last_report
                            if isinstance(report_for_fix, VisionResult):
                                report_dict = report_for_fix.to_observations()
                            elif isinstance(report_for_fix, SensoryReport):
                                report_dict = _sensory_to_vision_payload(
                                    report_for_fix, mode=config.vision_mode, url=vision_url
                                )
                            else:
                                report_dict = report_for_fix or {}
                            instructions = build_gate_fix_instructions(
                                expectations,
                                report_dict,
                                failing_reasons,
                                preview_url=preview_url,
                            )
                            instructions = (
                                f"You are improving pass {index}.\n"
                                f"Goal: {config.goal}\n\n"
                                f"Preview URL: {preview_url or 'unknown'}\n"
                                f"Expected contact API: {(preview_url.rstrip('/') + '/api/contact') if preview_url else 'unknown'}\n\n"
                                f"{instructions}\n"
                            )
                        tui.start_activity(
                            f"Brain: {step.description}…",
                            spinner="orbit",
                        )
                        tui.update_activity_progress("computing patches")
                        hooks.run_brain(instructions, pass_index=index)
                        brain_log = hooks.consume_brain_log(pass_index=index)
                        if brain_log:
                            summary.artifacts[f"brain_pass_{index}_log"] = brain_log
                            if config.detailed_log:
                                for line in brain_log.strip().splitlines():
                                    tui.add_sub_info(f"[brain-log] {line}")
                            elif not brain_log_notice_shown:
                                tui.add_sub_info(
                                    "Brain agent output captured; rerun with --detailed-log to view the transcript."
                                )
                                brain_log_notice_shown = True
                        tui.stop_activity("Brain: Applied targeted fixes", icon="[brain]")
                        changes_made = True
                        tui.add_sub_info("Applied targeted fixes")

                        # Build gate: catch syntax errors before running vision
                        build_ok, build_errors = run_build_gate(project_path, _stack_to_dict(detected_stack))
                        if not build_ok:
                            for err in build_errors[:5]:
                                tui.add_sub_info(f"Build error: {err}")
                            tui.add_sub_info("Build gate failed — feeding errors back to brain")

                        has_follow_up_vision = any(
                            future_step.agent == "vision" for future_step in remaining_steps
                        )
                        if has_follow_up_vision:
                            pending_handoff = (
                                "Brain => Vision: Updates ready for validation.",
                                "<<",
                            )
                            tui.start_activity("Hand-off: Brain => Vision…", spinner="pulsing_star")

                if pass_outcome is None:
                    pass_outcome = PassOutcome(
                        index=index,
                        vision_passed=not failing_reasons,
                        changes_made=changes_made,
                        failing_reasons=failing_reasons,
                        summary={"issues": failing_reasons},
                    )
                    summary.add_pass(pass_outcome)

                if success:
                    # Accept brain changes — discard checkpoint
                    if _active_checkpoint:
                        discard_checkpoint(project_path, _active_checkpoint)
                        _active_checkpoint = None
                    break

                # If brain made changes but things got worse, roll back
                if (
                    _active_checkpoint
                    and changes_made
                    and last_failures is not None
                    and failing_reasons
                    and len(failing_reasons) > len(last_failures)
                ):
                    restored = restore_checkpoint(project_path, _active_checkpoint)
                    _active_checkpoint = None
                    if restored:
                        tui.add_sub_info("Regression detected — reverted brain changes")
                        failing_reasons = list(last_failures)
                elif _active_checkpoint:
                    # Changes accepted (or no regression) — discard checkpoint
                    discard_checkpoint(project_path, _active_checkpoint)
                    _active_checkpoint = None

                if failing_reasons:
                    current_cats = _extract_failure_categories(failing_reasons)
                    last_cats = _extract_failure_categories(last_failures or [])
                    same_reason_set = set(failing_reasons) == set(last_failures or [])
                    if current_cats == last_cats and same_reason_set:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0
                    if stagnation_counter >= 2:
                        summary.status = "stalled"
                        final_message = _format_stalled_message(failing_reasons)
                        summary.final_message = final_message
                        tui.set_footer(final_message)
                        break
                last_failures = failing_reasons
                last_report = pass_report

            if summary.status == "unknown":
                summary.status = "max_passes"
                final_message = _format_max_passes_message(last_failures or [])
                summary.final_message = final_message
                tui.set_footer(final_message)
    finally:
        if not keep_servers_running and server_manager:
            server_manager.stop_all()

    if keep_servers_running and preview_url and server_manager:
        try:
            import webbrowser

            webbrowser.open(preview_url)
        except Exception:
            pass

        if summary.status == "success":
            preview_line = (
                f"Preview ready at {preview_url}. Symphony will keep the local servers running while you review the changes."
            )
        else:
            preview_line = (
                f"Preview ready at {preview_url}. Symphony will keep the local servers running so you can inspect the current state before retrying."
            )
        console.print(preview_line)
        prompt_text = "Press Enter when you're done previewing to shut everything down…"
        if sys.stdin.isatty():
            try:
                input(prompt_text)
            except EOFError:
                time.sleep(5)
        else:
            time.sleep(5)

        server_manager.stop_all()

    return summary
