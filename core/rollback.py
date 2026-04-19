"""Git-based checkpoint and rollback for safe Brain agent passes.

Before the Brain agent writes code, a checkpoint is created so that
broken changes can be reverted if the subsequent Vision pass detects
a regression.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

_HEAD_PREFIX = "HEAD:"
_STASH_PREFIX = "STASH:"


def _run_git(project_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=30,
        check=check,
    )


def _repo_root(project_root: Path) -> Optional[Path]:
    result = _run_git(project_root, "rev-parse", "--show-toplevel", check=False)
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return Path(root) if root else None


def _project_pathspec(project_root: Path) -> str:
    """Return pathspec relative to repository root."""
    repo_root = _repo_root(project_root)
    if not repo_root:
        return "."
    try:
        rel = project_root.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return "."
    rel_str = str(rel)
    return rel_str if rel_str else "."


def _has_local_changes(project_root: Path) -> bool:
    """True when project subtree has uncommitted tracked or untracked files."""
    result = _run_git(
        project_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def is_git_repo(project_root: Path) -> bool:
    """Return True if the project lives inside a git working tree."""
    try:
        result = _run_git(project_root, "rev-parse", "--is-inside-work-tree", check=False)
        return result.returncode == 0 and "true" in result.stdout.lower()
    except Exception:
        return False


def create_checkpoint(project_root: Path, label: str) -> Optional[str]:
    """Create a rollback checkpoint.

    For safety, checkpoints are only created when the project subtree is clean.
    This avoids stashing and accidentally dropping pre-existing user edits.

    Returns a checkpoint token (``HEAD:<sha>``) on success.
    """
    _ = label  # Label reserved for future persistence and logging.
    if not is_git_repo(project_root):
        return None
    try:
        # Skip checkpointing in dirty trees to prevent data loss.
        if _has_local_changes(project_root):
            return None
        result = _run_git(project_root, "rev-parse", "HEAD", check=False)
        if result.returncode != 0:
            return None
        head_sha = result.stdout.strip()
        if not head_sha:
            return None
        return f"{_HEAD_PREFIX}{head_sha}"
    except Exception:
        return None


def restore_checkpoint(project_root: Path, stash_ref: str) -> bool:
    """Restore the working tree to the checkpoint token.

    Returns True on success.
    """
    if not is_git_repo(project_root):
        return False
    try:
        if stash_ref.startswith(_HEAD_PREFIX):
            head_sha = stash_ref[len(_HEAD_PREFIX) :].strip()
            if not head_sha:
                return False
            pathspec = _project_pathspec(project_root)
            restore_result = _run_git(
                project_root,
                "restore",
                "--source",
                head_sha,
                "--worktree",
                "--staged",
                "--",
                pathspec,
                check=False,
            )
            clean_result = _run_git(
                project_root,
                "clean",
                "-fd",
                "--",
                pathspec,
                check=False,
            )
            return restore_result.returncode == 0 and clean_result.returncode == 0

        ref = stash_ref[len(_STASH_PREFIX) :].strip() if stash_ref.startswith(_STASH_PREFIX) else stash_ref
        if not ref:
            return False
        result = _run_git(project_root, "stash", "pop", ref, check=False)
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
        # HEAD checkpoints are content-addressed snapshots and need no cleanup.
        if stash_ref.startswith(_HEAD_PREFIX):
            return True

        ref = stash_ref[len(_STASH_PREFIX) :].strip() if stash_ref.startswith(_STASH_PREFIX) else stash_ref
        if not ref:
            return False
        result = _run_git(project_root, "stash", "drop", ref, check=False)
        return result.returncode == 0
    except Exception:
        return False
