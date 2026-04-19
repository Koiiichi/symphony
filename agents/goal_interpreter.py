"""Goal Interpreter - Converts natural language goals into structured expectations.

Uses gpt-4o-mini to derive capability-based expectations from user goals.
This removes the need for hard-coded page types.
"""

import json
import logging
import os
import time
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# Model-specific temperature constraints
MODEL_TEMPERATURE_SUPPORT = {
    "gpt-4o": 0.0,
    "gpt-4o-mini": 0.0,
    "gpt-4-turbo": 0.0,
    "claude-3-opus": 0.0,
    "claude-3-sonnet": 0.0,
}

# Simple in-memory cache keyed on (goal, hint, stack_repr)
_expectations_cache: Dict[str, Dict[str, Any]] = {}


def get_model_temperature(model_id: str, desired_temp: float = 0.0) -> float:
    """Get appropriate temperature for model.
    
    Args:
        model_id: Model identifier
        desired_temp: Desired temperature (default 0.0 for deterministic)
        
    Returns:
        Actual temperature to use for this model
    """
    if model_id in MODEL_TEMPERATURE_SUPPORT:
        required_temp = MODEL_TEMPERATURE_SUPPORT[model_id]
        if required_temp != desired_temp:
            print(f"[Goal Interpreter] {model_id} requires temperature={required_temp}")
        return required_temp
    return desired_temp


def build_expectations(
    goal: str,
    page_type_hint: Optional[str] = None,
    stack: Optional[Dict[str, Any]] = None,
    expectations_file: Optional[str] = None,
    vision_mode: str = "hybrid",
) -> Dict[str, Any]:
    """Build expectations from goal using LLM or file override.
    
    Args:
        goal: Natural language goal from user
        page_type_hint: Optional hint about page type
        stack: Detected project stack information
        expectations_file: Optional path to JSON file with expectations (bypasses LLM)
        
    Returns:
        Dictionary with capabilities and interactions expectations
    """
    
    if expectations_file:
        expectations = _load_expectations_from_file(expectations_file)
        return _apply_mode_filters(expectations, vision_mode)

    if not HAS_OPENAI or not os.getenv("SYMPHONY_BRAIN_API_KEY"):
        return _build_expectations_heuristic(goal, page_type_hint, vision_mode=vision_mode)

    try:
        expectations = _build_expectations_llm(goal, page_type_hint, stack)
        return _apply_mode_filters(expectations, vision_mode)
    except Exception as e:
        print(f"Goal interpreter LLM call failed: {e}, falling back to heuristic")
        return _build_expectations_heuristic(goal, page_type_hint, vision_mode=vision_mode)


def _load_expectations_from_file(filepath: str) -> Dict[str, Any]:
    """Load expectations from JSON file for deterministic CI."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def _build_expectations_llm(
    goal: str,
    page_type_hint: Optional[str],
    stack: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Use gpt-4o-mini to derive expectations from goal.

    Retries up to 3 times with exponential backoff.  Results are cached
    in-memory so identical goals in multi-pass runs avoid redundant calls.
    """

    # Cache lookup
    stack_key = json.dumps(stack, sort_keys=True) if stack else ""
    cache_key = f"{goal}|{page_type_hint}|{stack_key}"
    if cache_key in _expectations_cache:
        logger.debug("Goal interpreter cache hit")
        return _expectations_cache[cache_key]

    client = OpenAI(api_key=os.getenv("SYMPHONY_BRAIN_API_KEY"))

    stack_info = ""
    if stack:
        stack_info = f"\nDetected stack: Frontend: {stack.get('frontend', 'unknown')}, Backend: {stack.get('backend', 'unknown')}"

    hint_info = ""
    if page_type_hint:
        hint_info = f"\nPage type hint: {page_type_hint}"

    prompt = f"""You are a requirements analyzer. Given a user goal for a web application, extract structured expectations.

User Goal: {goal}{hint_info}{stack_info}

Return ONLY a JSON object with this structure:
{{
  "capabilities": {{
    "kpi_tiles": {{"min": <number or 0 if not applicable>}},
    "charts": {{"min": <number or 0 if not applicable>}},
    "tables": {{"min": <number or 0 if not applicable>}},
    "filters": {{"required": <true/false>}}
  }},
  "interactions": [
    {{
      "id": "<unique_id>",
      "type": "form_submit",
      "selector": "<css_selector>",
      "expect_http_2xx": true,
      "expect_success_banner": true
    }}
  ]
}}

Guidelines:
- If goal mentions "dashboard" or "analytics", set kpi_tiles, charts, tables accordingly
- If goal mentions "contact form", "newsletter signup", "login", add a form_submit interaction
- If goal mentions "landing page" or "portfolio", set all capabilities to 0
- Be conservative: only require what's explicitly mentioned or strongly implied
- Use meaningful interaction IDs like "contact_submit", "newsletter_signup", "login_form"
"""

    temperature = get_model_temperature("gpt-4o-mini", desired_temp=0.0)

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a requirements extraction specialist. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_completion_tokens=500,
            )

            content = response.choices[0].message.content.strip()

            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
                expectations = json.loads(json_str)
            else:
                expectations = json.loads(content)

            _expectations_cache[cache_key] = expectations
            return expectations

        except Exception as exc:
            last_error = exc
            if attempt < 2:
                delay = 2 ** attempt  # 1s, 2s
                logger.warning("Goal interpreter attempt %d failed: %s – retrying in %ds", attempt + 1, exc, delay)
                time.sleep(delay)

    raise last_error  # type: ignore[misc]


def _build_expectations_heuristic(
    goal: str,
    page_type_hint: Optional[str],
    *,
    vision_mode: str = "hybrid",
) -> Dict[str, Any]:
    """Fallback heuristic when LLM unavailable."""
    
    goal_lower = goal.lower()
    hint_lower = (page_type_hint or "").lower()
    
    expectations = {
        "capabilities": {
            "kpi_tiles": {"min": 0},
            "charts": {"min": 0},
            "tables": {"min": 0},
            "filters": {"required": False}
        },
        "interactions": []
    }
    
    if "dashboard" in goal_lower or "dashboard" in hint_lower or "analytics" in goal_lower:
        expectations["capabilities"]["kpi_tiles"]["min"] = 3
        expectations["capabilities"]["charts"]["min"] = 1
        expectations["capabilities"]["tables"]["min"] = 1
        # Only require filters if explicitly mentioned
        if "filter" in goal_lower:
            expectations["capabilities"]["filters"]["required"] = True
    
    if any(keyword in goal_lower for keyword in ["contact", "contact form", "get in touch"]):
        expectations["interactions"].append({
            "id": "contact_submit",
            "type": "form_submit",
            "selector": "#contact",
            "expect_http_2xx": True,
            "expect_success_banner": True
        })
    
    if "newsletter" in goal_lower or "signup" in goal_lower or "subscribe" in goal_lower:
        expectations["interactions"].append({
            "id": "newsletter_signup",
            "type": "form_submit",
            "selector": "form",
            "expect_http_2xx": True,
            "expect_success_banner": True
        })
    
    if "login" in goal_lower or "sign in" in goal_lower:
        expectations["interactions"].append({
            "id": "login_form",
            "type": "form_submit",
            "selector": "#login",
            "expect_http_2xx": True,
            "expect_success_banner": True
        })
    
    # Extract expected features from the goal
    expected_features = []
    
    # Toggle/switch features
    if any(word in goal_lower for word in ["toggle", "switch", "dark mode", "light mode", "theme"]):
        expected_features.append({
            "id": "theme_toggle",
            "type": "button",
            "keywords": ["toggle", "switch", "theme", "mode", "dark", "light"],
            "selectors": ["[class*='toggle']", "[class*='switch']", "[class*='theme']", "button[aria-label*='theme']", "button[aria-label*='mode']"],
            "description": "Dark/light mode toggle button"
        })
    
    # Search functionality
    if any(word in goal_lower for word in ["search", "search bar", "search box"]):
        expected_features.append({
            "id": "search_feature",
            "type": "input",
            "keywords": ["search"],
            "selectors": ["input[type='search']", "[class*='search']", "#search"],
            "description": "Search input field"
        })
    
    # Modal/popup features
    if any(word in goal_lower for word in ["modal", "popup", "dialog"]):
        expected_features.append({
            "id": "modal_feature",
            "type": "element",
            "keywords": ["modal", "popup", "dialog"],
            "selectors": ["[class*='modal']", "[class*='popup']", "[role='dialog']"],
            "description": "Modal or popup element"
        })
    
    # Navigation menu
    if any(word in goal_lower for word in ["menu", "hamburger", "nav", "navigation"]):
        expected_features.append({
            "id": "navigation_menu",
            "type": "element",
            "keywords": ["menu", "nav", "navigation"],
            "selectors": ["nav", "[class*='menu']", "[class*='nav']", "[role='navigation']"],
            "description": "Navigation menu"
        })
    
    # Dropdown
    if "dropdown" in goal_lower or "select" in goal_lower:
        expected_features.append({
            "id": "dropdown_feature",
            "type": "element",
            "keywords": ["dropdown", "select"],
            "selectors": ["select", "[class*='dropdown']", "[class*='select']"],
            "description": "Dropdown or select element"
        })
    
    # Carousel/slider
    if any(word in goal_lower for word in ["carousel", "slider", "slideshow"]):
        expected_features.append({
            "id": "carousel_feature",
            "type": "element",
            "keywords": ["carousel", "slider", "slide"],
            "selectors": ["[class*='carousel']", "[class*='slider']", "[class*='swiper']"],
            "description": "Carousel or slider component"
        })
    
    # Tabs
    if "tab" in goal_lower:
        expected_features.append({
            "id": "tabs_feature",
            "type": "element",
            "keywords": ["tab"],
            "selectors": ["[role='tablist']", "[class*='tab']"],
            "description": "Tab navigation component"
        })
    
    # Add to button
    if any(word in goal_lower for word in ["add button", "create button", "new button"]):
        expected_features.append({
            "id": "action_button",
            "type": "button",
            "keywords": ["add", "create", "new"],
            "selectors": ["button"],
            "description": "Action button"
        })
    
    if expected_features:
        expectations["expected_features"] = expected_features

    return _apply_mode_filters(expectations, vision_mode)


def _apply_mode_filters(expectations: Dict[str, Any], vision_mode: str) -> Dict[str, Any]:
    mode = (vision_mode or "hybrid").lower()
    # Only strip interactions in visual-only mode; hybrid and qa both test forms
    if mode == "visual":
        expectations = dict(expectations)
        expectations["interactions"] = []
    capabilities = expectations.get("capabilities")
    if isinstance(capabilities, dict):
        for key in ("kpi_tiles", "charts", "tables"):
            value = capabilities.get(key)
            if isinstance(value, (int, float)):
                capabilities[key] = {"min": int(value)}
        filters = capabilities.get("filters")
        if isinstance(filters, bool):
            capabilities["filters"] = {"required": filters}
    return expectations


def save_expectations(expectations: Dict[str, Any], output_path: str):
    """Save expectations to JSON file for reuse."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(expectations, indent=2, fp=f)
