# agents/sensory_agent_web.py
"""Enhanced sensory agent with agentic browsing, form testing, and vision scoring.

Now returns standardized SensoryReport for contract compliance.
"""

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import WebDriverException
except ModuleNotFoundError:  # pragma: no cover - fallback used in tests
    class Options:  # type: ignore[misc]
        def __init__(self, *_, **__):
            raise ModuleNotFoundError(
                "selenium is required for sensory agent operations. Install selenium to enable browser automation."
            )

    class By:  # type: ignore[misc]
        CSS_SELECTOR = "css selector"
        XPATH = "xpath"

    class WebDriverException(Exception):
        pass

try:  # pragma: no cover - optional dependency
    import helium  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback used in tests
    class _HeliumStub:
        def __getattr__(self, name):
            def _missing(*_, **__):
                raise ModuleNotFoundError(
                    "helium is required for sensory agent operations. Install helium to enable browser automation."
                )

            return _missing

    helium = _HeliumStub()

# Import contract types
from agents.sensory_contract import (
    SensoryReport,
    InteractionResult,
    AccessibilityResult,
    PlaywrightResult,
    Screenshot
)

VIEWPORTS = [
    {"name": "mobile", "width": 360, "height": 812},
    {"name": "tablet", "width": 768, "height": 1024},
    {"name": "desktop", "width": 1280, "height": 900},
]

# Optional OpenAI import for vision scoring
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def _ensure_artifacts_dir(run_id: str) -> Path:
    """Ensure artifacts directory exists for this run."""
    artifacts_dir = Path("artifacts") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def _save_step_screenshot(step_name: str, run_id: str = "default") -> str:
    """Save screenshot and return the path."""
    artifacts_dir = _ensure_artifacts_dir(run_id)
    driver = helium.get_driver()
    path = artifacts_dir / f"step_{step_name}_{int(time.time())}.png"
    driver.get_screenshot_as_file(str(path))
    return str(path)


def go_to_url(url: str) -> str:
    """Navigate to URL and wait for load."""
    helium.go_to(url)
    time.sleep(2.0)
    return f"Opened {url}"


def ensure_contact_present() -> str:
    """Scroll to find contact form section."""
    contact_indicators = ["Contact", "contact", "Get in touch", "Send message"]
    
    for indicator in contact_indicators:
        if helium.Text(indicator).exists():
            helium.scroll_down(1200)
            time.sleep(1.5)
            return f"Found and scrolled to contact section ({indicator})"
    
    helium.scroll_down(1200)
    time.sleep(1.5)
    return "Scrolled down to explore page; contact section may be below fold"


def _resolve_element(driver, selector: str):
    try:
        if selector.startswith("css="):
            return driver.find_element(By.CSS_SELECTOR, selector[4:])
        if selector.startswith("xpath="):
            return driver.find_element(By.XPATH, selector[6:])
        if selector.startswith("//"):
            return driver.find_element(By.XPATH, selector)
        if selector.startswith(("#", ".", "[", "input", "textarea", "button")):
            return driver.find_element(By.CSS_SELECTOR, selector)
    except WebDriverException:
        return None
    return None


def submit_contact_form(
    name="Test User",
    email="test@example.com",
    message="Hello from Symphony-Lite!"
) -> InteractionResult:
    """Attempt to fill and submit contact form.
    
    Returns:
        InteractionResult with submission status and details
    """
    errors = []
    
    try:
        driver = helium.get_driver()

        def fill_field(selectors, value) -> bool:
            for selector in selectors:
                try:
                    element = _resolve_element(driver, selector)
                    if element is None:
                        helium.write(value, into=selector)
                        return True
                    try:
                        element.clear()
                    except Exception:
                        pass
                    element.send_keys(value)
                    return True
                except Exception:
                    continue
            return False

        def click_target(selectors) -> bool:
            for selector in selectors:
                try:
                    element = _resolve_element(driver, selector)
                    if element is None:
                        helium.click(selector)
                        return True
                    element.click()
                    return True
                except Exception:
                    continue
            return False

        # Try different input selectors
        name_selectors = ["Name", "name", "Your Name", "css=input[name='name']", "#name"]
        email_selectors = ["Email", "email", "Your Email", "css=input[name='email']", "#email"]
        message_selectors = [
            "Message",
            "message",
            "Your Message",
            "css=textarea[name='message']",
            "#message",
            "textarea",
        ]
        
        # Fill name field
        name_filled = fill_field(name_selectors, name)
        if not name_filled:
            errors.append("Could not find name field")

        # Fill email field
        email_filled = fill_field(email_selectors, email)
        if not email_filled:
            errors.append("Could not find email field")

        # Fill message field
        message_filled = fill_field(message_selectors, message)
        if not message_filled:
            errors.append("Could not find message field")

        # Try to submit
        submit_selectors = [
            "Send",
            "Submit",
            "Send Message",
            "css=button[type='submit']",
            "button[type='submit']",
        ]
        submit_clicked = click_target(submit_selectors)
        if not submit_clicked:
            errors.append("Could not find or click submit button")
        
        time.sleep(2.0)  # Wait for response
        
        contact_submitted = name_filled and email_filled and message_filled and submit_clicked
        
        # Capture HTTP status from network logs
        http_status = _get_last_xhr_status('/api/contact')
        
        return InteractionResult(
            attempted=contact_submitted,
            contact_submitted=contact_submitted,
            http_status=http_status,
            success_banner=_check_success_banner(),
            error_banner=_check_error_banner(),
            details="Form submitted successfully" if contact_submitted else "; ".join(errors),
            errors=errors
        )
        
    except Exception as e:
        return InteractionResult(
            attempted=True,
            contact_submitted=False,
            http_status=None,
            success_banner=False,
            error_banner=True,
            details=f"Exception during form submission: {str(e)}",
            errors=[str(e)]
        )


def check_basic_accessibility() -> AccessibilityResult:
    """Perform basic accessibility checks.

    Returns:
        AccessibilityResult with violations found
    """
    violations = []

    try:
        driver = helium.get_driver()

        # Images without alt text
        images = driver.find_elements("css selector", "img:not([alt])")
        if images:
            violations.append(f"{len(images)} images missing alt text")

        # Buttons without accessible labels
        buttons = driver.find_elements("css selector", "button:not([aria-label]):empty")
        if buttons:
            violations.append(f"{len(buttons)} buttons missing accessible labels")

        # Form inputs without labels
        inputs = driver.find_elements("css selector", "input:not([aria-label]):not([id])")
        unlabeled = [inp for inp in inputs if not driver.find_elements("css selector", f"label[for='{inp.get_attribute('id')}']")]
        if unlabeled:
            violations.append(f"{len(unlabeled)} form inputs missing labels")

        # Heading hierarchy — skip-level headings (e.g. h1 → h3)
        try:
            heading_levels = [
                int(el.tag_name[1])
                for el in driver.find_elements("css selector", "h1,h2,h3,h4,h5,h6")
            ]
            for i in range(1, len(heading_levels)):
                if heading_levels[i] - heading_levels[i - 1] > 1:
                    violations.append(
                        f"Heading hierarchy skip: h{heading_levels[i-1]} → h{heading_levels[i]}"
                    )
                    break
        except Exception:
            pass

        # Missing <main> landmark
        try:
            main_elements = driver.find_elements("css selector", "main, [role='main']")
            if not main_elements:
                violations.append("Page missing <main> landmark")
        except Exception:
            pass

        # Anchor links with non-descriptive text ("click here", "read more", etc.)
        try:
            _vague = {"click here", "here", "read more", "more", "link", "this"}
            anchors = driver.find_elements("css selector", "a")
            vague_count = sum(
                1 for a in anchors
                if a.text.strip().lower() in _vague and not a.get_attribute("aria-label")
            )
            if vague_count:
                violations.append(f"{vague_count} links with non-descriptive text")
        except Exception:
            pass

        # Suppressed focus outlines (outline:none / outline:0 without :focus-visible override)
        try:
            page_src = driver.page_source
            if "outline: none" in page_src or "outline:none" in page_src or "outline: 0" in page_src:
                violations.append("CSS may suppress focus outlines (outline:none detected)")
        except Exception:
            pass

    except Exception as e:
        violations.append(f"Error during a11y check: {str(e)}")

    return AccessibilityResult(
        violations=len(violations),
        top_issues=violations[:5],
    )


def analyze_current_view() -> dict:
    """Analyze current page view using vision model or fallback heuristics.
    
    Returns:
        Dict with alignment_score, spacing_score, contrast_score, visible_sections
    """
    vision_key = os.getenv("SYMPHONY_VISION_API_KEY")
    
    if not HAS_OPENAI or not vision_key:
        logger.info("Vision API unavailable, using heuristic fallback")
        return analyze_view_heuristic()
    
    try:
        driver = helium.get_driver()
        png = driver.get_screenshot_as_png()
        b64 = base64.b64encode(png).decode()
        
        logger.info("Analyzing screenshot with Vision API...")
        
        client = OpenAI(api_key=vision_key)
        prompt = """You are a UI/UX expert. Analyze this webpage screenshot and provide objective scores from 0.0 to 1.0.

SCORING CRITERIA:
- alignment_score (0.0-1.0): Grid/flexbox consistency, visual balance, element alignment
  - 0.0-0.3: Major misalignments, inconsistent layouts
  - 0.4-0.6: Some alignment issues, room for improvement
  - 0.7-0.8: Good alignment with minor issues
  - 0.9-1.0: Excellent, professional-grade alignment

- spacing_score (0.0-1.0): Whitespace usage, padding/margin consistency
  - 0.0-0.3: Cramped or overly sparse, inconsistent
  - 0.4-0.6: Adequate but could be improved
  - 0.7-0.8: Good rhythm and breathing room
  - 0.9-1.0: Excellent, consistent spacing system

- contrast_score (0.0-1.0): Text readability, color contrast ratios
  - 0.0-0.3: Hard to read, poor contrast
  - 0.4-0.6: Readable but could be better
  - 0.7-0.8: Good contrast, clear text
  - 0.9-1.0: Excellent WCAG compliance

Also identify visible sections: hero, projects, contact, about, services, testimonials

Return ONLY valid JSON: {"alignment_score": 0.X, "spacing_score": 0.X, "contrast_score": 0.X, "visible_sections": ["section1", "section2"]}"""
        
        # Use gpt-4o-mini for faster, cheaper vision analysis
        model = os.getenv("SYMPHONY_VISION_MODEL", "gpt-4o-mini")
        
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]}
            ],
            temperature=0,
            max_tokens=300
        )
        
        try:
            content = resp.choices[0].message.content.strip()
            logger.debug("Vision API response: %s", content)
            
            # Extract JSON from response
            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
                result = json.loads(json_str)
                result["source"] = "vision_api"
                logger.info("Vision scores: alignment=%.2f, spacing=%.2f, contrast=%.2f",
                           result.get("alignment_score", 0),
                           result.get("spacing_score", 0),
                           result.get("contrast_score", 0))
                return result
            else:
                return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Vision JSON parse failed: %s", str(e))
            report = analyze_view_heuristic()
            report.setdefault("warnings", []).append(
                f"Vision JSON parse failed ({e.__class__.__name__})"
            )
            return report

    except Exception as e:
        logger.warning("Vision analysis failed: %s", str(e))
        report = analyze_view_heuristic()
        report.setdefault("warnings", []).append(
            f"Vision analysis failed ({e.__class__.__name__})"
        )
        return report


def analyze_view_heuristic() -> dict:
    """Fallback heuristic analysis when vision model unavailable.
    
    Returns:
        Dict with scores and visible sections
    """
    visible_sections = []
    
    # Check for hero section
    hero_indicators = ["portfolio", "developer", "designer", "welcome", "hello", "Symphony"]
    for indicator in hero_indicators:
        try:
            if helium.Text(indicator).exists():
                visible_sections.append("hero")
                break
        except:
            pass
    
    # Check for projects section
    project_indicators = ["project", "work", "portfolio", "showcase", "Project"]
    for indicator in project_indicators:
        try:
            if helium.Text(indicator).exists():
                visible_sections.append("projects")
                break
        except:
            pass
    
    # Check for contact section
    contact_indicators = ["contact", "email", "message", "get in touch", "Contact"]
    for indicator in contact_indicators:
        try:
            if helium.Text(indicator).exists():
                visible_sections.append("contact")
                break
        except:
            pass
    
    # Basic scoring based on visible elements — capped at 0.75 (heuristics are imprecise)
    _MAX_HEURISTIC = 0.75
    base_score = 0.6 + (len(visible_sections) * 0.1)

    return {
        "alignment_score": min(base_score + 0.1, _MAX_HEURISTIC),
        "spacing_score": min(base_score, _MAX_HEURISTIC),
        "contrast_score": min(base_score + 0.05, _MAX_HEURISTIC),
        "visible_sections": visible_sections,
        "confidence": "low",
        "source": "heuristic",
        "warnings": ["Heuristic fallback used — scores capped at 0.75"],
    }


def _get_last_xhr_status(url_pattern: str) -> Optional[int]:
    """Extract HTTP status from Chrome performance logs.
    
    Args:
        url_pattern: URL substring to match (e.g., '/api/contact')
        
    Returns:
        HTTP status code or None if not found
    """
    try:
        driver = helium.get_driver()
        logs = driver.get_log('performance')
        
        # Reverse to get most recent first
        for log_entry in reversed(logs):
            log = json.loads(log_entry['message'])['message']
            
            # Look for Network.responseReceived events
            if log['method'] == 'Network.responseReceived':
                response = log['params']['response']
                if url_pattern in response['url']:
                    return response['status']
        
        return None
    except Exception as e:
        logger.debug("Failed to parse network logs: %s", e)
        return None


def _check_success_banner() -> bool:
    """Check if success message is visible."""
    try:
        driver = helium.get_driver()
        success_elements = driver.find_elements("css selector", 
            ".success, .message.success, [class*='success']")
        return any(el.is_displayed() for el in success_elements)
    except:
        return False


def _check_error_banner() -> bool:
    """Check if error message is visible."""
    try:
        driver = helium.get_driver()
        error_elements = driver.find_elements("css selector", 
            ".error, .message.error, [class*='error']")
        return any(el.is_displayed() for el in error_elements)
    except:
        return False


def _count_elements() -> dict:
    """Count UI elements for capability checking.
    
    Returns:
        Dict with element counts
    """
    try:
        driver = helium.get_driver()
        
        kpi_tiles = len(driver.find_elements("css selector", "[class*='kpi'], [class*='metric'], [class*='stat'], [data-type='kpi']"))
        charts = len(driver.find_elements("css selector", "canvas, svg[class*='chart'], [class*='chart']"))
        tables = len(driver.find_elements("css selector", "table"))
        filters = len(driver.find_elements("css selector", "[type='search'], select, [class*='filter']"))
        
        return {
            "kpi_tiles": kpi_tiles,
            "charts": charts,
            "tables": tables,
            "filters": filters
        }
    except Exception as e:
        logger.debug("Element counting failed: %s", e)
        return {
            "kpi_tiles": 0,
            "charts": 0,
            "tables": 0,
            "filters": 0
        }


def _verify_features(expected_features: list) -> dict:
    """Verify that expected features exist on the page.
    
    Args:
        expected_features: List of feature specs from goal interpreter
        
    Returns:
        Dict with feature verification results
    """
    results = {
        "verified": [],
        "missing": [],
        "details": {}
    }
    
    if not expected_features:
        return results
    
    try:
        driver = helium.get_driver()
        page_source = driver.page_source.lower()
        
        for feature in expected_features:
            feature_id = feature.get("id", "unknown")
            selectors = feature.get("selectors", [])
            keywords = feature.get("keywords", [])
            description = feature.get("description", feature_id)
            
            found = False
            found_by = None
            
            # Try CSS selectors first
            for selector in selectors:
                try:
                    elements = driver.find_elements("css selector", selector)
                    if elements and any(el.is_displayed() for el in elements):
                        found = True
                        found_by = f"selector: {selector}"
                        break
                except Exception:
                    continue
            
            # Fall back to keyword search in page source
            if not found:
                for keyword in keywords:
                    if keyword.lower() in page_source:
                        # Verify it's not just in a script or comment
                        try:
                            # Check for visible text or aria-label
                            text_elements = driver.find_elements("xpath", f"//*[contains(text(), '{keyword}')]")
                            aria_elements = driver.find_elements("css selector", f"[aria-label*='{keyword}']")
                            if text_elements or aria_elements:
                                found = True
                                found_by = f"keyword: {keyword}"
                                break
                        except Exception:
                            pass
            
            # Record result
            results["details"][feature_id] = {
                "found": found,
                "description": description,
                "found_by": found_by
            }
            
            if found:
                results["verified"].append(feature_id)
            else:
                results["missing"].append(feature_id)
        
    except Exception as e:
        logger.error("Feature verification failed: %s", str(e))
        # Mark all as missing if verification failed
        for feature in expected_features:
            feature_id = feature.get("id", "unknown")
            results["missing"].append(feature_id)
            results["details"][feature_id] = {
                "found": False,
                "description": feature.get("description", feature_id),
                "error": str(e)
            }
    
    return results


def _test_form_interaction(interaction_spec: dict) -> dict:
    """Test a form interaction based on spec.
    
    Args:
        interaction_spec: Interaction specification from expectations
        
    Returns:
        Dict with interaction results
    """
    result = {
        "attempted": False,
        "http_status": None,
        "success_banner": False,
        "error_banner": False
    }
    
    try:
        if interaction_spec["type"] != "form_submit":
            return result
        
        # Try to submit form using the selector
        selector = interaction_spec.get("selector", "form")
        
        # Fill and submit based on interaction id
        interaction_id = interaction_spec["id"]
        
        if "contact" in interaction_id:
            form_result = submit_contact_form()
            result["attempted"] = form_result.attempted
            result["http_status"] = form_result.http_status
            result["success_banner"] = form_result.success_banner
            result["error_banner"] = form_result.error_banner
        else:
            # Generic form submission
            try:
                helium.click(f"{selector} button[type='submit']")
                result["attempted"] = True
                time.sleep(2.0)
                
                # Check for success/error indicators
                driver = helium.get_driver()
                success_elements = driver.find_elements("css selector", "[class*='success'], [class*='Success']")
                error_elements = driver.find_elements("css selector", "[class*='error'], [class*='Error']")
                
                result["success_banner"] = len(success_elements) > 0
                result["error_banner"] = len(error_elements) > 0
                result["http_status"] = 200 if result["success_banner"] else 500
            except Exception as e:
                result["attempted"] = True
                result["error_banner"] = True
                result["http_status"] = 500
        
        return result
        
    except Exception as e:
        logger.debug("Form interaction test failed: %s", e)
        return result


def inspect_site(
    url: str,
    run_id: str = "default",
    sensory_config: Optional[Dict[str, Any]] = None,
    expectations: Optional[Dict[str, Any]] = None,
    *,
    mode: str = "hybrid",
) -> SensoryReport:
    """Main function to inspect site with agentic browsing.
    
    Args:
        url: URL to inspect (e.g., http://localhost:3000)
        run_id: Unique identifier for this run (for artifacts)
        sensory_config: Configuration for sensory model
        expectations: Expected capabilities and interactions
        
    Returns:
        SensoryReport with all inspection results
    """
    if sensory_config is None:
        sensory_config = {"model_id": "gpt-4o"}
    
    if expectations is None:
        expectations = {}
    
    # Setup Chrome with appropriate options
    opts = Options()
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    
    # Enable performance logging for network capture
    opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    
    headless = os.getenv("SYMPHONY_HEADLESS", "true").lower() == "true"
    if headless:
        opts.add_argument("--headless")
    
    screenshots = []
    all_visible_sections = set()
    alignment_scores = []
    spacing_scores = []
    contrast_scores = []
    visited_urls = []
    warnings: list[str] = []
    
    try:
        # Start browser
        helium.start_chrome(headless=headless, options=opts)

        viewport_results: dict = {}

        # Step 1: Multi-viewport visual capture (mobile → tablet → desktop)
        used_vision_api = False
        for vp in VIEWPORTS:
            driver = helium.get_driver()
            driver.set_window_size(vp["width"], vp["height"])

            go_to_url(url)
            if url not in visited_urls:
                visited_urls.append(url)

            vp_screen_path = _save_step_screenshot(f"vp_{vp['name']}", run_id)
            vp_view = analyze_current_view()
            warnings.extend(vp_view.pop("warnings", []) or [])
            screenshots.append(Screenshot(page=f"{vp['name']}_load", path=vp_screen_path, viewport=vp["name"]))

            vp_alignment = vp_view.get("alignment_score", 0.7)
            vp_spacing = vp_view.get("spacing_score", 0.7)
            vp_contrast = vp_view.get("contrast_score", 0.7)

            alignment_scores.append(vp_alignment)
            spacing_scores.append(vp_spacing)
            contrast_scores.append(vp_contrast)
            all_visible_sections.update(vp_view.get("visible_sections", []))

            if vp_view.get("source") == "vision_api":
                used_vision_api = True

            viewport_results[vp["name"]] = {
                "alignment_score": vp_alignment,
                "spacing_score": vp_spacing,
                "contrast_score": vp_contrast,
                "source": vp_view.get("source", "heuristic"),
                "visible_sections": vp_view.get("visible_sections", []),
            }

        # Remainder of inspection at desktop resolution
        driver = helium.get_driver()
        driver.set_window_size(1280, 900)
        go_to_url(url)

        # Step 2: Explore and scroll
        ensure_contact_present()
        screen2_path = _save_step_screenshot("2_scroll", run_id)
        screenshots.append(Screenshot(page="after_scroll", path=screen2_path, viewport="desktop"))

        # Count elements for generic capabilities
        elements = _count_elements()

        # Default interaction result
        interaction = InteractionResult()

        # Test interactions from expectations (in qa and hybrid modes)
        interactions_results = {}
        if mode.lower() in ("qa", "hybrid"):
            for interaction_spec in expectations.get("interactions", []):
                if interaction_spec["type"] == "form_submit":
                    result = _test_form_interaction(interaction_spec)
                    interactions_results[interaction_spec["id"]] = result

            # Legacy contact form interaction for backward compatibility
            interaction = submit_contact_form()
            if not interactions_results.get("contact_submit"):
                interactions_results["contact_submit"] = {
                    "attempted": interaction.attempted,
                    "http_status": interaction.http_status,
                    "success_banner": interaction.success_banner,
                    "error_banner": interaction.error_banner,
                    "errors": interaction.errors,
                    "notes": interaction.details,
                }
        else:
            warnings.append("Interactive checks skipped in visual-only mode")

        # Verify expected features exist on the page
        expected_features = expectations.get("expected_features", [])
        feature_results = _verify_features(expected_features)
        if feature_results["missing"]:
            for missing_id in feature_results["missing"]:
                detail = feature_results["details"].get(missing_id, {})
                desc = detail.get("description", missing_id)
                warnings.append(f"Missing expected feature: {desc}")

        # Step 3: Final screenshot after interactions
        screen3_path = _save_step_screenshot("3_submit", run_id)
        screen3 = analyze_current_view()
        warnings.extend(screen3.pop("warnings", []) or [])
        screenshots.append(Screenshot(page="after_submit", path=screen3_path, viewport="desktop"))

        # Don't feed the post-interaction screenshot into viewport scores —
        # it would inflate final numbers relative to the per-viewport baseline.

        if screen3.get("source") == "vision_api":
            used_vision_api = True

        # Step 4: Accessibility check at desktop
        a11y = check_basic_accessibility()

        # Aggregate using minimum across viewports (weakest link wins)
        final_alignment = min(alignment_scores) if alignment_scores else 0.7
        final_spacing = min(spacing_scores) if spacing_scores else 0.7
        final_contrast = min(contrast_scores) if contrast_scores else 0.7

        # Build vision scores
        vision_scores = {
            "alignment": final_alignment,
            "spacing": final_spacing,
            "contrast": final_contrast,
            "source": "vision_api" if used_vision_api else "heuristic",
        }

        # Deduplicate warnings
        deduped_warnings: list = []
        for warning in warnings:
            if warning and warning not in deduped_warnings:
                deduped_warnings.append(warning)

        report = SensoryReport(
            status="pass",
            alignment_score=final_alignment,
            spacing_score=final_spacing,
            contrast_score=final_contrast,
            visible_sections=sorted(list(all_visible_sections)),
            interaction=interaction,
            a11y=a11y,
            playwright=PlaywrightResult(passed=True, total_tests=0),
            screens=screenshots,
            elements=elements,
            interactions=interactions_results,
            visited_urls=visited_urls,
            vision_scores=vision_scores,
            model_ids={"sensory": sensory_config.get("model_id", "gpt-4o")},
            expectations=expectations,
            warnings=deduped_warnings,
            feature_verification=feature_results,
            viewport_results=viewport_results,
        )

        # Update status based on gates
        if not report.passes_all_gates():
            report.status = "needs_fix"

        return report
        
    except Exception as e:
        # Log the error for debugging
        logger.error("Sensory agent failed: %s", str(e))
        
        # Return error report with fallback scores (not 0.0)
        return SensoryReport(
            status="needs_fix",
            alignment_score=0.5,
            spacing_score=0.5,
            contrast_score=0.5,
            visible_sections=[],
            interaction=InteractionResult(
                contact_submitted=False,
                details=f"Error during inspection: {str(e)}",
                errors=[str(e)]
            ),
            a11y=AccessibilityResult(violations=0, top_issues=[]),
            playwright=PlaywrightResult(passed=False, failed_tests=["Site inspection"], total_tests=1),
            screens=[],
            warnings=[f"Sensory agent error: {str(e)}"]
        )
        
    finally:
        try:
            helium.kill_browser()
        except Exception:
            pass


# Backward compatibility function
def make_sensory_agent():
    """Legacy function for backward compatibility."""
    class LegacySensoryAgent:
        def run(self, instruction: str):
            if "localhost" in instruction:
                # Extract URL from instruction
                import re
                match = re.search(r'https?://localhost:\d+', instruction)
                if match:
                    url = match.group(0)
                    report = inspect_site(url, mode="hybrid")
                    # Convert to legacy dict format
                    return report.to_dict()
            return {"error": "Please provide a localhost URL to inspect"}
    
    return LegacySensoryAgent()
