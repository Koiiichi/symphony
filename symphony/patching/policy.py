"""Safe patch proposal, validation, and apply policy."""

from __future__ import annotations

import difflib
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class EditMode(str, Enum):
    ASK = "ask"
    SUGGEST = "suggest"
    AUTO = "auto"


class PatchDecision(str, Enum):
    APPLIED = "applied"
    REJECTED = "rejected"
    SUGGESTED = "suggested"
    SKIPPED = "skipped"


@dataclass
class FilePatch:
    file: str
    absolute_path: Path
    original: str
    updated: str
    diff: str
    added: int
    removed: int
    preview: list[str]


@dataclass
class PatchProposal:
    proposal_id: str
    node_id: str
    mode: EditMode
    files: list[FilePatch]
    proposal_path: Path
    diff_path: Path

    @property
    def added(self) -> int:
        return sum(f.added for f in self.files)

    @property
    def removed(self) -> int:
        return sum(f.removed for f in self.files)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "node_id": self.node_id,
            "mode": self.mode.value,
            "proposal_path": str(self.proposal_path),
            "diff_path": str(self.diff_path),
            "summary": {
                "files": len(self.files),
                "added": self.added,
                "removed": self.removed,
            },
            "files": [
                {
                    "file": f.file,
                    "added": f.added,
                    "removed": f.removed,
                    "preview": f.preview,
                }
                for f in self.files
            ],
        }


@dataclass
class PatchOutcome:
    decision: PatchDecision
    applied_files: list[str]
    blocked_reason: str | None = None


class PatchPolicy:
    """Create and apply patch proposals with safety checks."""

    def __init__(
        self,
        *,
        project_root: Path,
        mode: EditMode,
        write_scopes: list[Path] | None = None,
        require_manual_approval: bool = False,
    ):
        self._project_root = project_root.resolve()
        self.mode = mode
        self.require_manual_approval = require_manual_approval
        scopes = write_scopes or [self._project_root]
        self._write_scopes = [scope.resolve() for scope in scopes]

    def build_proposal(
        self,
        *,
        node_id: str,
        patches: list[dict[str, Any]],
        artifact_dir: Path,
    ) -> PatchProposal:
        file_patches: list[FilePatch] = []
        for patch in patches:
            rel_path = str(patch.get("file", "")).strip()
            search = str(patch.get("search", ""))
            replace = str(patch.get("replace", ""))

            abs_path = self._validate_target_path(rel_path)
            original = abs_path.read_text(errors="replace")
            if search not in original:
                raise ValueError(f"Patch search string not found in {rel_path}")
            updated = original.replace(search, replace, 1)

            rel_display = str(abs_path.relative_to(self._project_root))
            diff = self._make_unified_diff(rel_display, original, updated)
            added, removed = self._count_changes(diff)
            preview = self._make_preview(diff)
            file_patches.append(
                FilePatch(
                    file=rel_display,
                    absolute_path=abs_path,
                    original=original,
                    updated=updated,
                    diff=diff,
                    added=added,
                    removed=removed,
                    preview=preview,
                )
            )

        proposal_id = f"{node_id}_{int(time.time() * 1000)}"
        patch_dir = artifact_dir / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        proposal_path = patch_dir / f"{proposal_id}.proposal.json"
        diff_path = patch_dir / f"{proposal_id}.diff"

        combined_diff = "".join(fp.diff for fp in file_patches)
        diff_path.write_text(combined_diff)
        proposal_path.write_text(
            json.dumps(
                {
                    "proposal_id": proposal_id,
                    "node_id": node_id,
                    "mode": self.mode.value,
                    "files": [
                        {
                            "file": fp.file,
                            "added": fp.added,
                            "removed": fp.removed,
                            "search": patches[idx].get("search", ""),
                            "replace": patches[idx].get("replace", ""),
                        }
                        for idx, fp in enumerate(file_patches)
                    ],
                    "summary": {
                        "files": len(file_patches),
                        "added": sum(fp.added for fp in file_patches),
                        "removed": sum(fp.removed for fp in file_patches),
                    },
                },
                indent=2,
            )
        )

        return PatchProposal(
            proposal_id=proposal_id,
            node_id=node_id,
            mode=self.mode,
            files=file_patches,
            proposal_path=proposal_path,
            diff_path=diff_path,
        )

    def decide_and_apply(
        self,
        proposal: PatchProposal,
        *,
        approval_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> PatchOutcome:
        if not proposal.files:
            return PatchOutcome(decision=PatchDecision.SKIPPED, applied_files=[])

        if self.mode == EditMode.SUGGEST:
            reason = "manual_approval_required" if self.require_manual_approval else "edit_mode_suggest"
            return PatchOutcome(
                decision=PatchDecision.SUGGESTED,
                applied_files=[],
                blocked_reason=reason,
            )

        if self.mode == EditMode.ASK:
            if approval_callback is None:
                return PatchOutcome(
                    decision=PatchDecision.SUGGESTED,
                    applied_files=[],
                    blocked_reason="manual_approval_required",
                )
            approved = approval_callback(proposal.to_event_payload())
            if not approved:
                return PatchOutcome(
                    decision=PatchDecision.REJECTED,
                    applied_files=[],
                    blocked_reason="user_rejected",
                )

        applied_files: list[str] = []
        for file_patch in proposal.files:
            file_patch.absolute_path.write_text(file_patch.updated)
            applied_files.append(file_patch.file)
        return PatchOutcome(decision=PatchDecision.APPLIED, applied_files=applied_files)

    def _validate_target_path(self, rel_path: str) -> Path:
        if not rel_path:
            raise ValueError("Patch file path is empty")

        candidate = (self._project_root / rel_path).resolve()
        if not self._is_within(candidate, self._project_root):
            raise ValueError(f"Patch file '{rel_path}' escapes project root")
        if not candidate.exists() or not candidate.is_file():
            raise ValueError(f"Patch target '{rel_path}' does not exist")
        if not any(self._is_within(candidate, scope) for scope in self._write_scopes):
            raise PermissionError(f"Patch target '{rel_path}' is outside write scope")
        return candidate

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _make_unified_diff(rel_path: str, original: str, updated: str) -> str:
        diff_lines = difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
        return "".join(diff_lines)

    @staticmethod
    def _count_changes(diff_text: str) -> tuple[int, int]:
        added = 0
        removed = 0
        for line in diff_text.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                removed += 1
        return added, removed

    @staticmethod
    def _make_preview(diff_text: str, max_lines: int = 12) -> list[str]:
        preview: list[str] = []
        for line in diff_text.splitlines():
            if line.startswith(("diff ", "index ")):
                continue
            preview.append(line)
            if len(preview) >= max_lines:
                break
        return preview
