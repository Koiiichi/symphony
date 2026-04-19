"""Git-based checkpoint and rollback for safe Brain agent passes.

Before the Brain agent writes code, a checkpoint is created so that
broken changes can be reverted if the subsequent Vision pass detects
a regression.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def _run_git(project_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=30,
        check=check,
    )


def is_git_repo(project_root: Path) -> bool:
    """Return True if the project lives inside a git working tree."""
    try:
        result = _run_git(project_root, "rev-parse", "--is-inside-work-tree", check=False)
        return result.returncode == 0 and "true" in result.stdout.lower()
    except Exception:
        return False


def create_checkpoint(project_root: Path, label: str) -> Optional[str]:
    """Stage all changes and create a stash entry as a checkpoint.

    Returns the stash reference (e.g. ``stash@{0}``) on success, or
    ``None`` if there was nothing to stash or git is unavailable.
    """
    if not is_git_repo(project_root):
        return None
    try:
        # Stage everything so untracked files are captured
        _run_git(project_root, "add", "-A", check=False)
        result = _run_git(project_root, "stash", "push", "-m", label, check=False)
        if result.returncode == 0 and "No local changes" not in result.stdout:
            return "stash@{0}"
        return None
    except Exception:
        return None


def restore_checkpoint(project_root: Path, stash_ref: str) -> bool:
    """Pop the stash to revert the working tree to the checkpoint.

    Returns True on success.
    """
    if not is_git_repo(project_root):
        return False
    try:
        result = _run_git(project_root, "stash", "pop", check=False)
        return result.returncode == 0
    except Exception:
        return False


def discard_checkpoint(project_root: Path, stash_ref: str) -> bool:
    """Drop the checkpoint stash entry (changes were accepted).

    Returns True on success.
    """
    if not is_git_repo(project_root):
        return False
    try:
        result = _run_git(project_root, "stash", "drop", stash_ref, check=False)
        return result.returncode == 0
    except Exception:
        return False
