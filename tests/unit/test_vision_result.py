from core.vision_result import parse_vision_payload


def test_parse_vision_payload_valid() -> None:
    payload = {
        "version": "1.0",
        "target_url": "http://localhost:3000",
        "mode": "qa",
        "scores": {"alignment": 0.9, "spacing": 0.91, "contrast": 0.92},
        "accessibility": {"violations": 0, "target": "AA"},
        "interactions": [
            {
                "id": "contact_submit",
                "action": "form_submit",
                "ok": True,
                "selector": "#contact",
                "attempted": True,
                "http_status": 200,
                "success_banner": True,
                "error_banner": False,
            }
        ],
        "issues": [],
        "suggestions": [],
        "artifacts": {},
    }

    result, warnings = parse_vision_payload(payload, url="http://localhost:3000", mode="qa")

    assert not warnings
    observations = result.to_observations()
    assert observations["vision_scores"]["alignment"] == 0.9
    assert "contact_submit" in observations["interactions"]
    assert observations["interactions"]["contact_submit"]["http_status"] == 200
    assert observations["interactions"]["contact_submit"]["success_banner"] is True


def test_parse_vision_payload_with_noise() -> None:
    raw = "IGNORE {\"invalid\": \"json\"}"

    result, warnings = parse_vision_payload(raw, url="http://localhost:4000", mode="visual")

    assert warnings  # JSON extraction failed but returned default
    assert result.mode == "visual"
    assert result.scores["alignment"] is None
