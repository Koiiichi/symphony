"""Test script to verify the new architecture components."""

import sys
import json
from pathlib import Path

# Add repository root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.goal_interpreter import build_expectations
from gates.engine import evaluate as evaluate_gates

def test_goal_interpreter():
    """Test goal interpreter with different goals."""
    print("Testing Goal Interpreter...")
    
    # Test 1: Dashboard goal
    expectations = build_expectations(
        "Analytics dashboard with 3 KPI tiles, a chart and a table",
        page_type_hint=None,
        stack=None,
        vision_mode="qa",
    )
    print("\n1. Dashboard Goal:")
    print(json.dumps(expectations, indent=2))
    assert expectations["capabilities"]["kpi_tiles"]["min"] >= 3
    assert expectations["capabilities"]["charts"]["min"] >= 1
    assert expectations["capabilities"]["tables"]["min"] >= 1
    
    # Test 2: Contact form goal
    expectations = build_expectations(
        "Contact page accepts messages",
        page_type_hint=None,
        stack=None,
        vision_mode="qa",
    )
    print("\n2. Contact Form Goal:")
    print(json.dumps(expectations, indent=2))
    assert len(expectations["interactions"]) >= 1
    assert any("contact" in i["id"] for i in expectations["interactions"])
    assert expectations["thresholds"]["alignment_score"] <= 0.70
    assert expectations["thresholds"]["spacing_score"] <= 0.70
    
    # Test 3: Landing page goal
    expectations = build_expectations(
        "Beautiful landing page with hero section",
        page_type_hint=None,
        stack=None,
        vision_mode="qa",
    )
    print("\n3. Landing Page Goal:")
    print(json.dumps(expectations, indent=2))
    assert expectations["capabilities"]["kpi_tiles"]["min"] == 0
    assert expectations["thresholds"]["alignment_score"] >= 0.90
    
    print("\nGoal Interpreter: PASSED")


def test_gate_engine():
    """Test gate engine evaluation."""
    print("\n\nTesting Gate Engine...")
    
    # Test 1: Passing all gates
    expectations = {
        "capabilities": {
            "kpi_tiles": {"min": 0},
            "charts": {"min": 0},
            "tables": {"min": 0},
            "filters": {"required": False}
        },
        "interactions": [],
        "thresholds": {
            "alignment_score": 0.90,
            "spacing_score": 0.90,
            "contrast_score": 0.75
        }
    }
    
    observations = {
        "elements": {"kpi_tiles": 0, "charts": 0, "tables": 0, "filters": 0},
        "interactions": {},
        "vision_scores": {"alignment": 0.95, "spacing": 0.92, "contrast": 0.80}
    }
    
    result = evaluate_gates(expectations, observations)
    print("\n1. All gates passing:")
    print(json.dumps(result, indent=2))
    assert result["passed"] == True
    assert len(result["failing_reasons"]) == 0
    
    # Test 2: Failing vision scores
    observations["vision_scores"]["alignment"] = 0.85
    result = evaluate_gates(expectations, observations)
    print("\n2. Alignment score failing:")
    print(json.dumps(result, indent=2))
    assert result["passed"] == False
    assert any("alignment_score" in r for r in result["failing_reasons"])
    
    # Test 3: Failing form interaction
    expectations["interactions"] = [{
        "id": "contact_submit",
        "type": "form_submit",
        "selector": "#contact",
        "expect_http_2xx": True,
        "expect_success_banner": True
    }]
    
    observations["interactions"] = {
        "contact_submit": {
            "attempted": True,
            "http_status": 501,
            "success_banner": False,
            "error_banner": True
        }
    }
    
    result = evaluate_gates(expectations, observations)
    print("\n3. Form interaction failing:")
    print(json.dumps(result, indent=2))
    assert result["passed"] == False
    assert any("contact_submit" in r for r in result["failing_reasons"])
    
    print("\nGate Engine: PASSED")


def test_gate_engine_handles_legacy_capability_shapes():
    """Ensure gate predicates accept shorthand capability declarations."""

    expectations = {
        "capabilities": {
            "kpi_tiles": 2,  # numeric shorthand should map to min requirement
            "charts": {"min": 0},
            "tables": 0,
            "filters": True,  # boolean shorthand should map to required flag
        },
        "interactions": [],
    }

    # Not enough KPI tiles and missing filter => fails
    failing_observations = {
        "elements": {"kpi_tiles": 1, "charts": 0, "tables": 0, "filters": 0},
        "vision_scores": {"alignment": 1.0, "spacing": 1.0, "contrast": 1.0},
    }

    result = evaluate_gates(expectations, failing_observations)
    assert result["passed"] is False
    assert any("kpi_tiles" in reason for reason in result["failing_reasons"])
    assert any("filters" in reason for reason in result["failing_reasons"])

    # Meets shorthand requirements => passes
    passing_observations = {
        "elements": {"kpi_tiles": 3, "charts": 0, "tables": 2, "filters": 1},
        "vision_scores": {"alignment": 1.0, "spacing": 1.0, "contrast": 1.0},
    }

    result = evaluate_gates(expectations, passing_observations)
    assert result["passed"] is True
    assert result["failing_reasons"] == []


def test_gate_engine_uses_ui_success_when_http_status_missing():
    expectations = {
        "capabilities": {
            "kpi_tiles": {"min": 0},
            "charts": {"min": 0},
            "tables": {"min": 0},
            "filters": {"required": False},
        },
        "interactions": [
            {
                "id": "contact_submit",
                "type": "form_submit",
                "selector": "#contact",
                "expect_http_2xx": True,
                "expect_success_banner": True,
            }
        ],
        "thresholds": {
            "alignment_score": 0.70,
            "spacing_score": 0.70,
            "contrast_score": 0.70,
        },
    }

    observations = {
        "elements": {"kpi_tiles": 0, "charts": 0, "tables": 0, "filters": 0},
        "interactions": {
            "contact_submit": {
                "attempted": True,
                "http_status": None,
                "success_banner": True,
                "error_banner": False,
                "ok": True,
            }
        },
        "vision_scores": {"alignment": 0.80, "spacing": 0.80, "contrast": 0.80},
    }

    result = evaluate_gates(expectations, observations)
    assert result["passed"] is True
    assert result["failing_reasons"] == []


def test_integration():
    """Test integration of goal interpreter and gate engine."""
    print("\n\nTesting Integration...")
    
    # Build expectations for a contact page
    expectations = build_expectations("Contact page accepts messages", vision_mode="qa")
    
    # Simulate broken contact form (like test-broken project)
    observations = {
        "elements": {"kpi_tiles": 0, "charts": 0, "tables": 0, "filters": 0},
        "interactions": {
            "contact_submit": {
                "attempted": True,
                "http_status": 501,
                "success_banner": False,
                "error_banner": True
            }
        },
        "vision_scores": {"alignment": 0.95, "spacing": 0.92, "contrast": 0.80},
        "visited_urls": ["http://localhost:3000"]
    }
    
    result = evaluate_gates(expectations, observations)
    
    print("\nBroken contact form:")
    print(json.dumps(result, indent=2))
    
    assert result["passed"] == False
    assert len(result["failing_reasons"]) > 0
    
    # Now simulate fixed contact form
    observations["interactions"]["contact_submit"] = {
        "attempted": True,
        "http_status": 200,
        "success_banner": True,
        "error_banner": False
    }
    
    result = evaluate_gates(expectations, observations)
    
    print("\nFixed contact form:")
    print(json.dumps(result, indent=2))
    
    assert result["passed"] == True
    assert len(result["failing_reasons"]) == 0
    
    print("\nIntegration: PASSED")


if __name__ == "__main__":
    try:
        test_goal_interpreter()
        test_gate_engine()
        test_integration()
        print("\n\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        sys.exit(0)
    except Exception as e:
        print(f"\n\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
