"""Symphony v2 CLI — hard-break surface, no legacy compatibility."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from symphony import __version__
from symphony.evaluator.evaluator import RunStatus
from symphony.patching.policy import EditMode

console = Console()


@click.group()
@click.version_option(__version__, prog_name="symphony")
def cli():
    """Symphony v2 — LLM-first web reliability orchestrator."""


@cli.command()
@click.option("--project", "-p", type=click.Path(exists=True), required=True,
              help="Path to the project directory.")
@click.option("--goal", "-g", required=True, help="Goal description.")
@click.option("--profile", default="web", show_default=True,
              help="Execution profile (web, api).")
@click.option("--passes", default=1, show_default=True, type=int,
              help="Number of fix-retest passes.")
@click.option("--provider", default=None,
              type=click.Choice(["openai", "gemini"], case_sensitive=False),
              help="LLM provider. Auto-detected from environment if unset.")
@click.option("--model", default=None,
              help="Model name. Defaults to provider default if unset.")
@click.option("--token-budget", default=None, type=int,
              help="Max token budget for prompts. Unlimited by default.")
@click.option("--artifact-dir", default=None, type=click.Path(),
              help="Directory for run artifacts.")
@click.option(
    "--edit-mode",
    default=EditMode.ASK.value,
    show_default=True,
    type=click.Choice([e.value for e in EditMode], case_sensitive=False),
    help="Patch editing mode: ask for approval, suggest only, or auto apply.",
)
@click.option(
    "--write-scope",
    multiple=True,
    type=click.Path(exists=True),
    help="Restrict editable paths. Can be passed multiple times.",
)
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show detailed log output (INFO level).")
def run(
    project: str,
    goal: str,
    profile: str,
    passes: int,
    provider: Optional[str],
    model: Optional[str],
    token_budget: Optional[int],
    artifact_dir: Optional[str],
    edit_mode: str,
    write_scope: tuple[str, ...],
    verbose: bool,
):
    """Execute a Symphony reliability run."""
    from rich.logging import RichHandler
    from symphony.llm import LLMClient
    from symphony.orchestrator import Orchestrator

    try:
        llm = LLMClient(provider=provider, model=model)
    except RuntimeError as exc:
        console.print(f"[red bold]Error:[/] {exc}")
        sys.exit(1)

    # Print the header first so it's always the first thing the user sees,
    # then configure logging — this prevents verbose log lines from the
    # LLM client init appearing above the header.
    console.print(
        Panel(
            f"[bold]{goal}[/]\n"
            f"[dim]project:[/] {project}  [dim]profile:[/] {profile}  "
            f"[dim]passes:[/] {passes}  [dim]provider:[/] {llm.provider}/{llm.model}",
            title="[bold blue]Symphony[/]",
            border_style="blue",
        )
    )

    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(
            console=console,
            show_path=False,
            rich_tracebacks=True,
            markup=False,
        )],
    )

    project_path = Path(project).resolve()
    art_dir = Path(artifact_dir) if artifact_dir else None
    scope_paths = [Path(p).resolve() for p in write_scope] if write_scope else [project_path]
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    requested_edit_mode = EditMode(edit_mode.lower())
    effective_edit_mode = requested_edit_mode
    require_manual_approval = False
    if requested_edit_mode is EditMode.ASK and not interactive:
        effective_edit_mode = EditMode.SUGGEST
        require_manual_approval = True
        console.print(
            "[yellow]Non-interactive environment detected: "
            "falling back from ask -> suggest mode.[/]"
        )

    status = console.status("", spinner="dots")

    def request_patch_approval(payload: dict) -> bool:
        status.stop()
        summary = payload.get("summary", {})
        console.print(
            "[bold]Approve patch batch?[/] "
            f"{payload.get('proposal_id', 'unknown')} "
            f"(files={summary.get('files', 0)}, "
            f"+{summary.get('added', 0)}/-{summary.get('removed', 0)})"
        )
        approved = click.confirm("Apply this patch batch", default=False)
        status.start()
        return approved

    orch = Orchestrator(
        project_path=project_path,
        llm=llm,
        token_budget=token_budget,
        artifact_dir=art_dir,
        edit_mode=effective_edit_mode.value,
        write_scope=scope_paths,
        require_manual_approval=require_manual_approval,
        request_patch_approval=request_patch_approval if effective_edit_mode is EditMode.ASK else None,
    )

    status.start()

    def on_progress(event: str, detail: str) -> None:
        if event == "phase":
            status.update(f"[bold cyan]{detail}[/]")
        elif event == "detail":
            status.update(f"[dim]{detail}[/]")
        elif event == "plan_ready":
            status.stop()
            console.print(f"  [green]+[/] Plan ready: {detail}")
            status.start()
        elif event == "node_start":
            status.update(f"[cyan]{detail}[/]")
        elif event == "node_result":
            status.stop()
            # Check for actual failure — detail format is "node_id: pass/fail (...)"
            is_fail = ": fail" in detail or detail.startswith("fail")
            if is_fail:
                console.print(f"  [red]-[/] {detail}")
            else:
                console.print(f"  [green]+[/] {detail}")
            status.start()
        elif event == "patch_proposed":
            status.stop()
            payload = _safe_json(detail)
            _print_patch_preview(payload)
            status.start()
        elif event == "patch_decision":
            status.stop()
            payload = _safe_json(detail)
            decision = payload.get("decision", "unknown")
            reason = payload.get("blocked_reason")
            suffix = f" ({reason})" if reason else ""
            console.print(f"  [cyan]~[/] Patch decision: {decision}{suffix}")
            status.start()
        elif event == "patch_applied":
            status.stop()
            payload = _safe_json(detail)
            diff_path = payload.get("diff_path")
            if diff_path and Path(diff_path).exists():
                console.print(f"  [green]+[/] Patch applied: {payload.get('proposal_id', 'unknown')}")
                _print_colored_diff(Path(diff_path).read_text(errors="replace"))
            status.start()
        elif event == "patch_blocked":
            status.stop()
            payload = _safe_json(detail)
            reason = payload.get("blocked_reason", "blocked")
            console.print(f"  [yellow]![/] Patch blocked: {reason}")
            status.start()
        elif event == "done":
            status.stop()

    try:
        report = orch.run(
            goal=goal, profile=profile, max_passes=passes,
            on_progress=on_progress,
        )
    except Exception as exc:
        status.stop()
        console.print(f"\n[red bold]Run failed:[/] {exc}")
        sys.exit(2)

    console.print()
    _print_report(report)
    sys.exit(0 if report.get("status") == RunStatus.PASS.value else 1)


@cli.command()
@click.option("--goal", "-g", required=True, help="Goal description.")
@click.option("--emit-taskgraph", is_flag=True, default=True,
              help="Print the TaskGraph JSON.")
@click.option("--provider", default=None,
              type=click.Choice(["openai", "gemini"], case_sensitive=False),
              help="LLM provider. Auto-detected from environment if unset.")
@click.option("--model", default=None, help="Model name.")
@click.option("--project-context", default="", help="Optional project context.")
def plan(
    goal: str,
    emit_taskgraph: bool,
    provider: Optional[str],
    model: Optional[str],
    project_context: str,
):
    """Generate a TaskGraph plan without executing it."""
    from symphony.llm import LLMClient
    from symphony.planner.planner import LLMPlanner

    try:
        llm = LLMClient(provider=provider, model=model)
    except RuntimeError as exc:
        console.print(f"[red bold]Error:[/] {exc}")
        sys.exit(1)

    planner = LLMPlanner(llm)
    with console.status("[cyan]Generating plan...[/]", spinner="dots"):
        graph, confidence, _tokens = planner.plan(goal, project_context=project_context)

    if emit_taskgraph:
        output = graph.model_dump()
        if confidence is not None:
            output["confidence"] = confidence
        console.print_json(json.dumps(output, indent=2))


@cli.command()
@click.option("--run-id", required=True, help="Run ID to replay.")
@click.option("--artifact-dir", default="artifacts", type=click.Path(),
              help="Artifact directory to search.")
def replay(run_id: str, artifact_dir: str):
    """Replay a previous run's report and artifacts."""
    art_path = Path(artifact_dir) / run_id
    if not art_path.exists():
        console.print(f"[red]Run '{run_id}' not found in {artifact_dir}[/]")
        sys.exit(1)

    report_path = art_path / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        _print_report(report)
    else:
        console.print(f"[red]No report.json in {art_path}[/]")
        sys.exit(1)

    console.print("\n[bold]Artifacts:[/]")
    for f in sorted(art_path.iterdir()):
        console.print(f"  {f.name}")


def _print_report(report: dict):
    status = report.get("status", "unknown")
    if status == RunStatus.PASS.value:
        console.print("[bold green]PASS[/]", highlight=False)
    else:
        console.print("[bold red]FAIL[/]", highlight=False)

    failures = report.get("failing_reasons", [])
    if failures:
        console.print(f"\n[bold]Failures ({len(failures)}):[/]")
        for f in failures:
            sev = f.get("severity", "?")
            node = f.get("node_id", "")
            msg = f.get("message", "")
            if len(msg) > 200:
                msg = msg[:200] + "..."
            prefix = f"  [red][{sev}][/] {node}: " if node else f"  [red][{sev}][/] "
            console.print(f"{prefix}{msg}", highlight=False)

    assertions = report.get("assertion_results", [])
    passed = sum(1 for a in assertions if a.get("passed"))
    console.print(f"\n[bold]Assertions: {passed}/{len(assertions)} passed[/]")
    for a in assertions:
        if a.get("passed"):
            icon = "[green]+[/]"
        else:
            icon = "[red]-[/]"
        a_msg = a.get("message", "")
        if len(a_msg) > 150:
            a_msg = a_msg[:150] + "..."
        console.print(f"  [{icon}] {a.get('assertion_id', '?')}: {a_msg}", highlight=False)

    usage = report.get("token_usage", {})
    if usage:
        total = usage.get("total", 0)
        console.print(f"\n[dim]Tokens used: {total}[/]")


def _safe_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _print_patch_preview(payload: dict) -> None:
    summary = payload.get("summary", {})
    console.print(
        "  [cyan]~[/] Patch proposed: "
        f"{summary.get('files', 0)} file(s), "
        f"+{summary.get('added', 0)}/-{summary.get('removed', 0)} "
        f"[dim]({payload.get('proposal_id', 'unknown')})[/]"
    )
    for file_info in payload.get("files", []):
        console.print(
            f"      [bold]{file_info.get('file', '?')}[/] "
            f"(+{file_info.get('added', 0)}/-{file_info.get('removed', 0)})"
        )
        for line in file_info.get("preview", []):
            styled = Text(f"      {line}")
            if line.startswith("+") and not line.startswith("+++"):
                styled.stylize("green")
            elif line.startswith("-") and not line.startswith("---"):
                styled.stylize("red")
            elif line.startswith("@@"):
                styled.stylize("cyan")
            console.print(styled)


def _print_colored_diff(diff_text: str) -> None:
    if not diff_text.strip():
        return
    console.print("\n[bold]Applied Diff[/]")
    for line in diff_text.splitlines():
        styled = Text(line)
        if line.startswith("+") and not line.startswith("+++"):
            styled.stylize("green")
        elif line.startswith("-") and not line.startswith("---"):
            styled.stylize("red")
        elif line.startswith("@@"):
            styled.stylize("cyan")
        elif line.startswith(("---", "+++")):
            styled.stylize("bold")
        console.print(styled)


if __name__ == "__main__":
    cli()
