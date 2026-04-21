<p align="center">
  <img src="music-stand.svg" width="80" alt="Symphony" />
</p>

<h1 align="center">Symphony</h1>

<p align="center">LLM-first web reliability orchestrator.</p>

---

Symphony runs goal-driven browser flows against your web application, evaluates results with a first-class reliability gate, and drives an evidence-based patch cycle until the application passes or the pass limit is reached.

It replaces brittle keyword heuristics and hardcoded interaction patterns with a structured plan emitted by an LLM, executed by a deterministic browser layer, and evaluated against explicit assertions.

## How it works

**Plan.** You provide a goal in plain language. An LLM reads your project context (dependency manifest, top-level file listing, and HTML pages) and emits a typed `TaskGraph`: an ordered sequence of nodes connected by explicit dependency edges. The HTML context lets the LLM use exact selectors (`#email`, `input[name="password"]`) rather than guessing. Structured output enforcement (via Gemini `response_json_schema` or OpenAI `response_format`) ensures the response is always a valid, schema-conforming TaskGraph with no parsing fallbacks. If planning fails, the error is surfaced immediately.

**Execute.** Nodes run in topological order. Stack detection scans for language and framework markers. Service startup launches your backend and frontend as subprocesses. Browser flow nodes drive a real Chrome instance using a validated Flow DSL; the LLM cannot run arbitrary scripts, only emit structured actions that are validated before execution. A JavaScript network interceptor captures `fetch`/`XHR` response statuses so `assert_http_status` works against real API calls, not just navigations. API check nodes hit endpoints directly with full request body and header support. Code patch nodes apply search-replace diffs to your source files. Screenshots are captured at every step.

**Evaluate.** A Reliability Evaluator checks every flow assertion, HTTP expectation, and accessibility requirement. It produces a machine-readable report with typed failure reasons and artifact pointers. Exit code is `0` on pass, `1` on failure.

**Patch.** If `--passes` is greater than 1 and the run failed, Symphony feeds the failure evidence back to the LLM, which proposes a targeted source patch. The graph reruns, repeating until the application passes or the pass limit is reached.

## Comparison

Symphony is a runtime reliability tool, not a coding assistant. The distinction matters.

Tools like GitHub Copilot and Cursor operate on source code. Their loop is: read code → suggest a change → you apply it. There is no running application involved, no browser, and no way to know whether the change actually worked without manually testing it.

Symphony operates on a running application. Its loop is: run the app → drive a real browser → observe what happens → patch if needed → retest automatically. Code edits are a side effect, not the primary action.

| | Coding assistants | Symphony |
| --- | --- | --- |
| Primary input | Source code | A goal and a running application |
| Evidence | Static analysis | Screenshots, HTTP traces, DOM snapshots |
| Loop | Edit → you test | Run → observe → patch → retest |
| Pass/fail signal | None | Typed assertions, machine-readable report |
| CI integration | No | Yes (exit code 0/1) |

The tools are complementary. A coding assistant writes the feature; Symphony verifies it works in a real browser before it ships.

## Requirements

- Python 3.12 or later
- Chrome and ChromeDriver (for browser execution)
- An OpenAI or Google Gemini API key

## Installation

```sh
# With OpenAI support
uv sync --extra openai

# With Gemini support
uv sync --extra gemini

# With token budget enforcement
uv sync --extra openai --extra budget
```

## Configuration

Set one of the following environment variables:

```sh
export OPENAI_API_KEY=sk-...
# or
export GEMINI_API_KEY=AI...
```

The provider is auto-detected from whichever key is present. If both are set, pass `--provider` explicitly.

## Usage

**Run a reliability check:**

```sh
symphony run \
  --project ./my-app \
  --goal "Submit the contact form and verify the success message" \
  --passes 3
```

**Inspect the plan without executing:**

```sh
symphony plan \
  --goal "Verify login with valid and invalid credentials" \
  --emit-taskgraph
```

**Replay a previous run:**

```sh
symphony replay --run-id run_20260419_150159
```

### Options

| Flag | Description |
|---|---|
| `--project` | Path to the project directory |
| `--goal` | Goal in plain language |
| `--profile` | Execution profile: `web` (default) or `api` |
| `--passes` | Number of fix-retest cycles (default: 1) |
| `--provider` | `openai` or `gemini` (auto-detected if unset) |
| `--model` | Model name (defaults to provider default) |
| `--token-budget` | Max prompt tokens (unlimited by default) |
| `--artifact-dir` | Override artifact output directory |
| `--edit-mode` | Patch behavior: `ask` (default), `suggest`, `auto` |
| `--write-scope` | Restrict editable paths (repeatable) |

## Provider defaults

| Provider | Default model |
|---|---|
| OpenAI | `gpt-5.4-mini` |
| Gemini | `gemini-3-flash-preview` |

Pass `--model` to override.

## Output

The terminal shows live progress as each node executes: plan generation, node-by-node status, and pass/fail inline, then prints the full report on completion.

Every run also writes artifacts to the **target project** directory:

```
<project>/artifacts/run_<timestamp>/
  taskgraph.json     TaskGraph the planner produced
  report.json        Machine-readable pass/fail report
  <node_id>/         Per-node screenshots and DOM snapshots
    step_001_navigate.png
    step_002_fill.png
    ...
```

Override the location with `--artifact-dir`.

`report.json` schema:

```json
{
  "status": "pass | fail",
  "failing_reasons": [
    { "id": "...", "severity": "critical | warning", "message": "...", "node_id": "..." }
  ],
  "assertion_results": [
    { "assertion_id": "...", "passed": true, "message": "...", "node_id": "..." }
  ],
  "artifacts": [
    { "type": "screenshot", "path": "...", "node_id": "..." }
  ],
  "patches": [
    {
      "proposal_id": "...",
      "mode": "ask | suggest | auto",
      "decision": "applied | rejected | suggested | skipped",
      "applied_files": ["..."],
      "blocked_reason": "manual_approval_required | user_rejected | ...",
      "proposal_path": "...",
      "diff_path": "...",
      "summary": { "files": 1, "added": 2, "removed": 1 }
    }
  ],
  "token_usage": { "total": 1240 },
  "planner_confidence": 0.9
}
```

Exit code is `0` on pass and `1` on failure, suitable for use in CI.

## Flow DSL

Symphony's browser actions are structured and validated before execution:

| Action | Required fields |
|---|---|
| `navigate` | `value` (URL) |
| `scroll` | `params.direction`, `params.pixels` |
| `click` | `selector` |
| `fill` | `selector`, `value` |
| `press` | `value` (key name) |
| `wait_for` | `selector` |
| `assert_text` | `selector`, `value` |
| `assert_http_status` | `value` (status code) |
| `assert_banner` | `value` (expected text) |
| `other` | `other_action_type` (description) |

`assert_http_status` works by injecting a JavaScript interceptor that captures every `fetch` and `XHR` response during the flow, not just page navigations. The `other` action type is logged and skipped; it exists so the LLM can annotate intent without causing execution errors.

## TaskGraph node types

| Type | Purpose |
|---|---|
| `stack_detect` | Identify project language and framework |
| `service_start` | Start backend or frontend servers |
| `ui_discovery` | Explore the UI before testing |
| `web_flow_test` | Execute a browser flow with assertions |
| `api_check` | Verify an API endpoint directly (supports request body and headers) |
| `code_patch` | Modify application source to fix a failure |
| `retest` | Re-run a prior test node after a patch |
| `finalize` | Cleanup and report generation |
| `other` | Intent annotation for tasks outside the standard types (logged, not executed) |

## Development

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[openai,budget,dev]"
python -m pytest
```

All tests run without a live browser or API key.

## License

MIT
