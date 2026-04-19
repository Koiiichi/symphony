"""Sensory Agent Contract - Standardized JSON response format.

Defines the contract between Sensory and Brain agents to ensure
consistent feedback and targeted fixes.
"""

from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict, field
import json


@dataclass
class InteractionResult:
    """Results from a single interaction test (form submission, button click, etc)."""
    attempted: bool = False
    http_status: Optional[int] = None
    success_banner: bool = False
    error_banner: bool = False
    details: str = ""
    errors: List[str] = field(default_factory=list)
    
    # Legacy compatibility
    contact_submitted: bool = False
    
    def __post_init__(self):
        if self.contact_submitted:
            self.attempted = True
            self.success_banner = True


@dataclass
class AccessibilityResult:
    """Accessibility testing results."""
    violations: int = 0
    top_issues: List[str] = field(default_factory=list)
    wcag_level: str = "AA"


@dataclass
class PlaywrightResult:
    """Playwright test execution results."""
    passed: bool = True
    failed_tests: List[str] = field(default_factory=list)
    total_tests: int = 0

    @property
    def is_placeholder(self) -> bool:
        """True when no real tests were executed (total_tests == 0)."""
        return self.total_tests == 0


@dataclass
class Screenshot:
    """Screenshot metadata."""
    page: str
    path: str
    timestamp: Optional[str] = None
    viewport: Optional[str] = None


@dataclass
class SensoryReport:
    """Standardized sensory agent report format.
    
    This contract ensures the Brain agent receives consistent,
    actionable feedback for targeted fixes.
    """
    status: Literal["pass", "needs_fix"]
    alignment_score: float = 0.0
    spacing_score: float = 0.0
    contrast_score: float = 0.0
    visible_sections: List[str] = field(default_factory=list)
    interaction: InteractionResult = field(default_factory=InteractionResult)
    a11y: AccessibilityResult = field(default_factory=AccessibilityResult)
    playwright: PlaywrightResult = field(default_factory=PlaywrightResult)
    screens: List[Screenshot] = field(default_factory=list)
    
    # New generic fields
    elements: Dict[str, int] = field(default_factory=dict)
    interactions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    visited_urls: List[str] = field(default_factory=list)
    vision_scores: Dict[str, float] = field(default_factory=dict)
    model_ids: Dict[str, str] = field(default_factory=dict)
    expectations: Dict[str, Any] = field(default_factory=dict)
    failing_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    feature_verification: Dict[str, Any] = field(default_factory=dict)  # Feature verification results
    viewport_results: Dict[str, Any] = field(default_factory=dict)  # Per-viewport score breakdown

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "alignment_score": self.alignment_score,
            "spacing_score": self.spacing_score,
            "contrast_score": self.contrast_score,
            "visible_sections": self.visible_sections,
            "interaction": asdict(self.interaction),
            "a11y": asdict(self.a11y),
            "playwright": asdict(self.playwright),
            "screens": [asdict(s) for s in self.screens],
            "elements": self.elements,
            "interactions": self.interactions,
            "visited_urls": self.visited_urls,
            "vision_scores": self.vision_scores,
            "model_ids": self.model_ids,
            "expectations": self.expectations,
            "failing_reasons": self.failing_reasons,
            "warnings": self.warnings,
            "feature_verification": self.feature_verification,
            "viewport_results": self.viewport_results,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensoryReport":
        """Create report from dictionary."""
        # Convert nested dicts to dataclasses
        if "interaction" in data and isinstance(data["interaction"], dict):
            data["interaction"] = InteractionResult(**data["interaction"])
        
        if "a11y" in data and isinstance(data["a11y"], dict):
            data["a11y"] = AccessibilityResult(**data["a11y"])
        
        if "playwright" in data and isinstance(data["playwright"], dict):
            data["playwright"] = PlaywrightResult(**data["playwright"])
        
        if "screens" in data and isinstance(data["screens"], list):
            data["screens"] = [
                Screenshot(**s) if isinstance(s, dict) else s
                for s in data["screens"]
            ]

        warnings = data.get("warnings")
        if isinstance(warnings, list):
            data["warnings"] = [str(w) for w in warnings if w is not None]
        else:
            data["warnings"] = []

        return cls(**data)
    
    def get_failing_gates(self, thresholds: Optional[Dict[str, float]] = None) -> List[str]:
        """Get list of failing quality gates.
        
        Args:
            thresholds: Custom thresholds (uses defaults if None)
            
        Returns:
            List of failing gate names
        """
        if thresholds is None:
            thresholds = {
                "alignment_score": 0.90,
                "spacing_score": 0.90,
                "contrast_score": 0.75,
                "a11y_violations": 5
            }
        
        failing = []
        
        if self.alignment_score < thresholds["alignment_score"]:
            failing.append(f"alignment_score ({self.alignment_score:.2f} < {thresholds['alignment_score']})")
        
        if self.spacing_score < thresholds["spacing_score"]:
            failing.append(f"spacing_score ({self.spacing_score:.2f} < {thresholds['spacing_score']})")
        
        if self.contrast_score < thresholds["contrast_score"]:
            failing.append(f"contrast_score ({self.contrast_score:.2f} < {thresholds['contrast_score']})")
        
        if not self.interaction.contact_submitted and "contact" in self.visible_sections:
            failing.append("contact_form_not_working")
        
        if self.a11y.violations > thresholds["a11y_violations"]:
            failing.append(f"a11y_violations ({self.a11y.violations} > {thresholds['a11y_violations']})")
        
        if not self.playwright.passed and not self.playwright.is_placeholder:
            failing.append(f"playwright_tests_failed ({len(self.playwright.failed_tests)} tests)")
        
        return failing
    
    def passes_all_gates(self, thresholds: Optional[Dict[str, float]] = None) -> bool:
        """Check if all quality gates pass.
        
        Args:
            thresholds: Custom thresholds (uses defaults if None)
            
        Returns:
            True if all gates pass
        """
        return len(self.get_failing_gates(thresholds)) == 0
    
    def get_fix_instructions(self) -> str:
        """Generate targeted fix instructions for the Brain agent.
        
        Returns:
            Markdown-formatted fix instructions
        """
        failing = self.get_failing_gates()
        
        if not failing:
            return "All quality gates passed. No fixes needed."
        
        instructions = ["## Required Fixes\n"]
        
        for gate in failing:
            if "alignment_score" in gate:
                instructions.append(
                    "### Layout Alignment\n"
                    "- Improve CSS grid/flexbox alignment\n"
                    "- Ensure consistent element positioning\n"
                    "- Fix any visual misalignments\n"
                )
            
            elif "spacing_score" in gate:
                instructions.append(
                    "### Spacing and Typography\n"
                    "- Apply consistent spacing scale (8px, 16px, 24px, etc.)\n"
                    "- Improve margin/padding consistency\n"
                    "- Enhance typography hierarchy\n"
                )
            
            elif "contrast_score" in gate:
                instructions.append(
                    "### Contrast and Readability\n"
                    "- Increase text contrast ratios\n"
                    "- Use darker text on light backgrounds\n"
                    "- Ensure WCAG AA compliance\n"
                )
            
            elif "contact_form" in gate:
                instructions.append(
                    "### Contact Form Functionality\n"
                    "- Fix JavaScript form submission handler\n"
                    "- Ensure backend route processes POST requests\n"
                    "- Display success/error messages to user\n"
                    f"- Details: {self.interaction.details}\n"
                )
            
            elif "a11y_violations" in gate:
                instructions.append(
                    f"### Accessibility Issues ({self.a11y.violations} violations)\n"
                )
                for issue in self.a11y.top_issues[:3]:
                    instructions.append(f"- {issue}\n")
            
            elif "playwright_tests" in gate:
                instructions.append(
                    "### Test Failures\n"
                )
                for test in self.playwright.failed_tests[:3]:
                    instructions.append(f"- Fix: {test}\n")
        
        return "\n".join(instructions)


def create_sample_report() -> SensoryReport:
    """Create a sample report for testing/documentation."""
    return SensoryReport(
        status="needs_fix",
        alignment_score=0.85,
        spacing_score=0.92,
        contrast_score=0.78,
        visible_sections=["hero", "projects", "contact"],
        interaction=InteractionResult(
            contact_submitted=False,
            details="Form submit button did not trigger API call",
            errors=["TypeError: Cannot read property 'value' of null"]
        ),
        a11y=AccessibilityResult(
            violations=3,
            top_issues=[
                "Button missing accessible label",
                "Low contrast ratio (3.2:1) on muted text",
                "Form input missing associated label"
            ]
        ),
        playwright=PlaywrightResult(
            passed=False,
            failed_tests=["Contact form submission"],
            total_tests=4
        ),
        screens=[
            Screenshot(page="home", path="artifacts/step_1.png"),
            Screenshot(page="after_scroll", path="artifacts/step_2.png")
        ]
    )
