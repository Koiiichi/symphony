"""Utilities for validating and normalising Vision agent output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_VALID_MODES = {"visual", "hybrid", "qa"}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            return float(value)
    except (TypeError, ValueError):
        return None
    return None


def _as_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return str(value)


@dataclass
class VisionInteraction:
    action: str
    selector: Optional[str] = None
    ok: bool = False
    notes: Optional[str] = None
    id: Optional[str] = None
    attempted: Optional[bool] = None
    http_status: Optional[int] = None
    success_banner: Optional[bool] = None
    error_banner: Optional[bool] = None

    @classmethod
    def from_payload(cls, data: Dict[str, Any]) -> "VisionInteraction":
        attempted = data.get("attempted")
        if attempted is not None:
            attempted_flag = bool(attempted)
        else:
            attempted_flag = bool(data.get("ok", False))
        return cls(
            action=_as_str(data.get("action")) or "unknown",
            selector=_as_str(data.get("selector")),
            ok=bool(data.get("ok", False)),
            notes=_as_str(data.get("notes")),
            id=_as_str(data.get("id")),
            attempted=attempted_flag,
            http_status=int(data["http_status"]) if isinstance(data.get("http_status"), (int, float)) else None,
            success_banner=bool(data.get("success_banner")) if data.get("success_banner") is not None else None,
            error_banner=bool(data.get("error_banner")) if data.get("error_banner") is not None else None,
        )


@dataclass
class VisionIssue:
    id: str
    status: str
    detail: Optional[str] = None

    @classmethod
    def from_payload(cls, data: Dict[str, Any]) -> "VisionIssue":
        return cls(
            id=_as_str(data.get("id")) or "unspecified_issue",
            status=_as_str(data.get("status")) or "unknown",
            detail=_as_str(data.get("detail")),
        )


@dataclass
class VisionSuggestion:
    area: Optional[str] = None
    change: Optional[str] = None

    @classmethod
    def from_payload(cls, data: Dict[str, Any]) -> "VisionSuggestion":
        return cls(area=_as_str(data.get("area")), change=_as_str(data.get("change")))


@dataclass
class VisionResult:
    version: str
    target_url: str
    mode: str
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    accessibility: Dict[str, Any] = field(default_factory=dict)
    interactions: List[VisionInteraction] = field(default_factory=list)
    issues: List[VisionIssue] = field(default_factory=list)
    suggestions: List[VisionSuggestion] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "target_url": self.target_url,
            "mode": self.mode,
            "scores": self.scores,
            "accessibility": self.accessibility,
            "interactions": [
                {
                    "id": item.id,
                    "action": item.action,
                    "selector": item.selector,
                    "ok": item.ok,
                    "notes": item.notes,
                    "attempted": item.attempted,
                    "http_status": item.http_status,
                    "success_banner": item.success_banner,
                    "error_banner": item.error_banner,
                }
                for item in self.interactions
            ],
            "issues": [
                {"id": issue.id, "status": issue.status, "detail": issue.detail}
                for issue in self.issues
            ],
            "suggestions": [
                {"area": s.area, "change": s.change} for s in self.suggestions
            ],
            "artifacts": self.artifacts,
        }

    def to_observations(self) -> Dict[str, Any]:
        """Convert to the structure expected by the gate engine."""

        score_source: Optional[str] = None
        raw_scores = self.raw.get("scores")
        if isinstance(raw_scores, dict):
            candidate = raw_scores.get("source")
            if isinstance(candidate, str) and candidate:
                score_source = candidate

        scores = {
            "alignment": self.scores.get("alignment"),
            "spacing": self.scores.get("spacing"),
            "contrast": self.scores.get("contrast"),
        }
        normalized_scores = {
            key: (value if value is not None else 0.0)
            for key, value in scores.items()
        }
        if score_source:
            normalized_scores["source"] = score_source  # type: ignore[assignment]

        interactions_map: Dict[str, Dict[str, Any]] = {}
        for entry in self.interactions:
            key = entry.id or f"{entry.action}:{entry.selector or 'unknown'}"
            interactions_map[key] = {
                "attempted": entry.attempted if entry.attempted is not None else entry.ok,
                "ok": entry.ok,
                "selector": entry.selector,
                "notes": entry.notes,
                "http_status": entry.http_status,
                "success_banner": entry.success_banner,
                "error_banner": entry.error_banner,
            }

        return {
            "target_url": self.target_url,
            "vision_scores": normalized_scores,
            "scores": scores,
            "interactions": interactions_map,
            "issues": [issue.__dict__ for issue in self.issues],
            "suggestions": [s.__dict__ for s in self.suggestions],
            "accessibility": self.accessibility,
        }


def _default_result(url: str, mode: str, error: Optional[str] = None) -> VisionResult:
    issues = []
    if error:
        issues.append(VisionIssue(id="vision_error", status="fail", detail=error))
    return VisionResult(
        version="1.0",
        target_url=url,
        mode=mode,
        scores={"alignment": None, "spacing": None, "contrast": None},
        accessibility={"violations": None, "target": "AA"},
        issues=issues,
    )


def _coerce_dict(payload: Any) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    if isinstance(payload, dict):
        return payload, warnings
    if hasattr(payload, "to_dict"):
        return getattr(payload, "to_dict")(), warnings
    if isinstance(payload, str):
        try:
            return json.loads(payload), warnings
        except json.JSONDecodeError as exc:
            warnings.append(f"JSON decode failed: {exc}")
            start = payload.find("{")
            end = payload.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(payload[start : end + 1]), warnings
                except json.JSONDecodeError:
                    warnings.append("Fallback JSON extraction failed")
    return {}, warnings


def parse_vision_payload(
    payload: Any,
    *,
    url: str,
    mode: str,
) -> Tuple[VisionResult, List[str]]:
    """Parse raw payload into :class:`VisionResult` with validation."""

    warnings: List[str] = []
    data, extra = _coerce_dict(payload)
    warnings.extend(extra)

    if not data:
        result = _default_result(url, mode, "Empty response from Vision agent")
        return result, warnings

    version = _as_str(data.get("version")) or "1.0"
    target_url = _as_str(data.get("target_url")) or url
    payload_mode = _as_str(data.get("mode")) or mode
    payload_mode = payload_mode.lower()
    if payload_mode not in _VALID_MODES:
        warnings.append(f"Unknown vision mode '{payload_mode}', defaulting to {mode}")
        payload_mode = mode

    scores_data = data.get("scores")
    if not isinstance(scores_data, dict):
        scores_data = {}
    scores = {
        "alignment": _safe_float(scores_data.get("alignment")),
        "spacing": _safe_float(scores_data.get("spacing")),
        "contrast": _safe_float(scores_data.get("contrast")),
    }

    accessibility = data.get("accessibility")
    if not isinstance(accessibility, dict):
        accessibility = {"violations": None, "target": "AA"}

    interactions_payload: Iterable[Any] = data.get("interactions") or []
    if not isinstance(interactions_payload, Iterable) or isinstance(
        interactions_payload, (str, bytes)
    ):
        interactions_payload = []
    interactions = [
        VisionInteraction.from_payload(item)
        for item in interactions_payload
        if isinstance(item, dict)
    ]

    issues_payload: Iterable[Any] = data.get("issues") or []
    if not isinstance(issues_payload, Iterable) or isinstance(issues_payload, (str, bytes)):
        issues_payload = []
    issues = [
        VisionIssue.from_payload(item)
        for item in issues_payload
        if isinstance(item, dict)
    ]

    suggestions_payload: Iterable[Any] = data.get("suggestions") or []
    if not isinstance(suggestions_payload, Iterable) or isinstance(
        suggestions_payload, (str, bytes)
    ):
        suggestions_payload = []
    suggestions = [
        VisionSuggestion.from_payload(item)
        for item in suggestions_payload
        if isinstance(item, dict)
    ]

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}

    result = VisionResult(
        version=version,
        target_url=target_url,
        mode=payload_mode,
        scores=scores,
        accessibility=accessibility,
        interactions=interactions,
        issues=issues,
        suggestions=suggestions,
        artifacts=artifacts,
        raw=data,
    )

    return result, warnings


def write_raw_payload(run_id: str, pass_index: int, payload: Any) -> Path:
    """Persist the raw payload for diagnostics."""

    artifacts_dir = Path("artifacts") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"vision_raw_pass_{pass_index}.json"
    try:
        if isinstance(payload, str):
            path.write_text(payload)
        else:
            path.write_text(json.dumps(payload, indent=2))
    except TypeError:
        path.write_text(json.dumps(str(payload)))
    return path
