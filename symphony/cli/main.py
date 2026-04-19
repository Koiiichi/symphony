"""Symphony v2 CLI — hard-break surface, no legacy compatibility."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from symphony import __version__


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
@click.option("--model", default="claude-sonnet-4-20250514",
              help="LLM model for planning/patching.")
@click.option("--token-budget", default=16000, type=int,
              help="Max token budget for prompts.")
@click.option("--artifact-dir", default=None, type=click.Path(),
              help="Directory for run artifacts.")
def run(project: str, goal: str, profile: str, passes: int,
        model: str, token_budget: int, artifact_dir: str | None):
    """Execute a Symphony reliability run."""
    from symphony.orchestrator import Orchestrator

    project_path = Path(project).resolve()
    art_dir = Path(artifact_dir) if artifact_dir else None

    orch = Orchestrator(
        project_path=project_path,
        model=model,
        token_budget=token_budget,
        artifact_dir=art_dir,
    )

    report = orch.run(goal=goal, profile=profile, max_passes=passes)
    _print_report(report)
    sys.exit(0 if report.get("status") == "pass" else 1)


@cli.command()
@click.option("--goal", "-g", required=True, help="Goal description.")
@click.option("--emit-taskgraph", is_flag=True, default=True,
              help="Print the TaskGraph JSON.")
@click.option("--model", default="claude-sonnet-4-20250514",
              help="LLM model for planning.")
@click.option("--project-context", default="", help="Optional project context.")
def plan(goal: str, emit_taskgraph: bool, model: str, project_context: str):
    """Generate a TaskGraph plan without executing it."""
    import anthropic
    from symphony.planner.planner import LLMPlanner

    client = anthropic.Anthropic()
    planner = LLMPlanner(client, model=model)
    graph, confidence = planner.plan(goal, project_context=project_context)

    if emit_taskgraph:
        output = graph.model_dump()
        if confidence is not None:
            output["confidence"] = confidence
        click.echo(json.dumps(output, indent=2))


@cli.command()
@click.option("--run-id", required=True, help="Run ID to replay.")
@click.option("--artifact-dir", default="artifacts", type=click.Path(),
              help="Artifact directory to search.")
def replay(run_id: str, artifact_dir: str):
    """Replay a previous run's report and artifacts."""
    art_path = Path(artifact_dir) / run_id
    if not art_path.exists():
        click.echo(f"Run '{run_id}' not found in {artifact_dir}", err=True)
        sys.exit(1)

    report_path = art_path / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        _print_report(report)
    else:
        click.echo(f"No report.json in {art_path}", err=True)
        sys.exit(1)

    # List artifacts
    click.echo("\nArtifacts:")
    for f in sorted(art_path.iterdir()):
        click.echo(f"  {f.name}")


def _print_report(report: dict):
    """Pretty-print a run report."""
    status = report.get("status", "unknown")
    icon = "PASS" if status == "pass" else "FAIL"
    click.echo(f"\n[{icon}] Run status: {status}")

    failures = report.get("failing_reasons", [])
    if failures:
        click.echo(f"\nFailures ({len(failures)}):")
        for f in failures:
            sev = f.get("severity", "?")
            click.echo(f"  [{sev}] {f.get('message', '')}")

    assertions = report.get("assertion_results", [])
    passed = sum(1 for a in assertions if a.get("passed"))
    click.echo(f"\nAssertions: {passed}/{len(assertions)} passed")

    usage = report.get("token_usage", {})
    if usage:
        click.echo(f"Token usage: {usage}")


if __name__ == "__main__":
    cli()
