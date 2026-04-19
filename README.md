# Symphony

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Symphony is an intelligent orchestration tool that automatically detects your project stack, manages servers, and coordinates AI agents to refine your applications through visual audits and code improvements.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command-Line Options](#command-line-options)
  - [Vision Modes](#vision-modes)
  - [Example Workflows](#example-workflows)
- [Configuration](#configuration)
- [Testing](#testing)
- [Privacy and Safety](#privacy-and-safety)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Automatic Stack Detection**: Identifies frontend and backend technologies
- **Intelligent Server Management**: Starts and monitors development servers
- **AI-Powered Refinement**: Coordinates vision and code agents for UI/UX improvements
- **Real-Time Monitoring**: Live terminal UI with activity feeds and status updates
- **Safety First**: Confirms destructive operations and enforces iteration limits

## Prerequisites

- Python 3.10 or higher
- API keys for Symphony Brain and Vision services

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Symphony
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows (Git Bash)
   source venv/Scripts/activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

   > **Note**: If you encounter dependency conflicts:
   > ```bash
   > python -m pip cache purge
   > python -m pip install --no-cache-dir -r requirements.txt
   > ```

## Quick Start

1. Set up your API keys by creating a `.env` file in the project root:
   ```bash
   SYMPHONY_BRAIN_API_KEY=your_brain_api_key_here
   SYMPHONY_VISION_API_KEY=your_vision_api_key_here
   ```

2. Run Symphony on a project:
   ```bash
   python symphony.py --project ./projects/portfolio "Ensure contact form works"
   ```

## Usage

### Basic Command

Always run `symphony.py` from the repository root directory:

```bash
python symphony.py --project <project-path> "<goal-description>"
```

### Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--project <path>` | Path to the project directory | Required |
| `--open` / `--no-open` | Open detected URL on success | `--open` |
| `--vision-mode <mode>` | Vision analysis mode (see below) | `hybrid` |
| `--max-passes <int>` | Maximum refinement cycles | `3` |
| `--dry-run` | Preview routing without execution | `false` |
| `--detailed-log` / `--concise-log` | Toggle trace detail level | `--concise-log` |

### Vision Modes

- **`visual`**: Quick screenshot-based analysis
- **`hybrid`**: Combines screenshots with light DOM inspection
- **`qa`**: Full scripted quality assurance flows

### Example Workflows

**Refine UI/UX:**
```bash
python symphony.py --project ./projects/portfolio "Improve mobile responsiveness"
```

**Add new feature:**
```bash
python symphony.py --project ./projects/blog "Add dark mode toggle"
```

**Quality assurance:**
```bash
python symphony.py --project ./projects/ecommerce --vision-mode qa "Test checkout flow"
```

**Preview without executing:**
```bash
python symphony.py --project ./projects/dashboard --dry-run "Optimize load times"
```

## Configuration

### Environment Variables

Set these in your `.env` file or export them in your shell:

```bash
SYMPHONY_BRAIN_API_KEY=...    # Required: API key for code agent
SYMPHONY_VISION_API_KEY=...   # Required: API key for vision agent
```

### Shell Aliases (Optional)

For convenience, add an alias to your shell configuration:

**macOS/Linux (bash/zsh):**
```bash
alias symphony="python /path/to/Symphony/symphony.py"
```

**Windows Command Prompt:**
```cmd
doskey symphony=py C:\path\to\Symphony\symphony.py $*
```

**Windows PowerShell:**
```powershell
function symphony { py C:\path\to\Symphony\symphony.py $args }
```

## Testing

Run the complete test suite:

```bash
pytest
```

This includes:
- Unit tests
- Integration tests
- CLI functionality tests

## Privacy and Safety

Symphony prioritizes safe operation:

- ✓ Confirms before creating new directories or scaffolding projects
- ✓ Enforces maximum iteration limits to prevent runaway processes
- ✓ Prompts before shutting down services started during the run
- ✓ Provides dry-run mode for previewing actions

## Terminal UI

Symphony provides real-time feedback during execution:

```
┌──────────────────────── Symphony ────────────────────────┐
│ Project: …/projects/portfolio                            │
│ Goal: Ensure UI/UX is up to the mark                     │
│ Mode: refine (ui_ux)                                     │
│ Passes: 3                                                │
│ Run: run_20251026_104500                                 │
└──────────────────────────────────────────────────────────┘
┌──────────────────────── Status ──────────────────────────┐
│ Stack Detection: COMPLETE – frontend: start command ready│
│ Servers: READY – backend: http://localhost:5000,         │
│          frontend: http://localhost:8000/index.html      │
│ Expectations: READY – 4 capabilities                     │
│ Pass: 1/3 – running                                      │
│ Heartbeat: 42s                                           │
└──────────────────────────────────────────────────────────┘
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**Questions or Issues?** Please open an issue on GitHub.
