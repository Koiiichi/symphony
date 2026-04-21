"""Deterministic Flow DSL executor — drives browser via Selenium/Helium."""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException

from symphony.flow.dsl import ActionType, FlowAction, FlowScript

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Evidence collection
# ------------------------------------------------------------------

@dataclass
class Evidence:
    screenshots: list[Path] = field(default_factory=list)
    network_traces: list[dict[str, Any]] = field(default_factory=list)
    dom_snapshots: list[str] = field(default_factory=list)
    focused_elements: list[str] = field(default_factory=list)
    console_logs: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Action result
# ------------------------------------------------------------------

@dataclass
class ActionResult:
    action: FlowAction
    success: bool
    message: str = ""
    elapsed_ms: float = 0.0
    evidence: Evidence = field(default_factory=Evidence)


# ------------------------------------------------------------------
# Flow execution result
# ------------------------------------------------------------------

@dataclass
class FlowResult:
    script_name: str
    passed: bool
    results: list[ActionResult] = field(default_factory=list)
    evidence: Evidence = field(default_factory=Evidence)

    @property
    def failures(self) -> list[ActionResult]:
        return [r for r in self.results if not r.success]


# ------------------------------------------------------------------
# Key mapping
# ------------------------------------------------------------------

_KEY_MAP: dict[str, str] = {
    "enter": Keys.ENTER,
    "return": Keys.RETURN,
    "tab": Keys.TAB,
    "escape": Keys.ESCAPE,
    "space": Keys.SPACE,
    "backspace": Keys.BACKSPACE,
    "delete": Keys.DELETE,
    "arrowup": Keys.ARROW_UP,
    "arrowdown": Keys.ARROW_DOWN,
    "arrowleft": Keys.ARROW_LEFT,
    "arrowright": Keys.ARROW_RIGHT,
}


# ------------------------------------------------------------------
# Executor
# ------------------------------------------------------------------

class FlowExecutor:
    """Execute a FlowScript against a live browser session."""

    def __init__(
        self,
        driver: WebDriver,
        artifact_dir: Path,
        *,
        collect_network: bool = False,
        collect_dom: bool = False,
        collect_focus: bool = False,
    ):
        self._driver = driver
        self._artifact_dir = artifact_dir
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._collect_network = collect_network
        self._collect_dom = collect_dom
        self._collect_focus = collect_focus
        self._step = 0
        self._captured_responses: list[dict[str, Any]] = []

    # ---- public API ------------------------------------------------

    def execute(self, script: FlowScript) -> FlowResult:
        """Run all actions in *script* sequentially, collecting evidence."""
        self._inject_network_interceptor()
        results: list[ActionResult] = []
        combined_evidence = Evidence()
        all_passed = True

        for action in script.actions:
            result = self._run_action(action)
            results.append(result)

            # aggregate evidence
            combined_evidence.screenshots.extend(result.evidence.screenshots)
            combined_evidence.network_traces.extend(result.evidence.network_traces)
            combined_evidence.dom_snapshots.extend(result.evidence.dom_snapshots)
            combined_evidence.focused_elements.extend(
                result.evidence.focused_elements
            )

            if not result.success:
                all_passed = False
                # stop on first non-assertion failure
                if action.action not in {
                    ActionType.ASSERT_TEXT,
                    ActionType.ASSERT_HTTP_STATUS,
                    ActionType.ASSERT_BANNER,
                }:
                    break

        if self._collect_network:
            harvested = self._harvest_responses()
            if harvested:
                self._captured_responses.extend(harvested)
            network_path = self._artifact_dir / "network_trace.json"
            try:
                network_path.write_text(
                    json.dumps(self._captured_responses, indent=2, ensure_ascii=True)
                )
                combined_evidence.network_traces.append(
                    {"path": str(network_path), "entries": len(self._captured_responses)}
                )
            except OSError as exc:
                logger.warning("Could not save network trace at %s: %s", network_path, exc)

        return FlowResult(
            script_name=script.name,
            passed=all_passed,
            results=results,
            evidence=combined_evidence,
        )

    # ---- private dispatch ------------------------------------------

    def _run_action(self, action: FlowAction) -> ActionResult:
        self._step += 1
        t0 = time.monotonic()
        try:
            handler = self._dispatch(action.action)
            msg = handler(action)
            elapsed = (time.monotonic() - t0) * 1000
            evidence = self._collect_evidence(action)
            return ActionResult(
                action=action,
                success=True,
                message=msg,
                elapsed_ms=elapsed,
                evidence=evidence,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            evidence = self._collect_evidence(action)
            return ActionResult(
                action=action,
                success=False,
                message=self._format_error(exc, action),
                elapsed_ms=elapsed,
                evidence=evidence,
            )

    @staticmethod
    def _format_error(exc: Exception, action: FlowAction) -> str:
        """Extract a human-readable message from Selenium exceptions."""
        raw = str(exc)
        # Selenium exceptions embed "Message: ...\nStacktrace:\n..." —
        # strip the native chromedriver stacktrace, keep only the message.
        if "Stacktrace:" in raw:
            raw = raw.split("Stacktrace:")[0].strip()
        # TimeoutException from WebDriverWait often has an empty Message.
        msg = raw.removeprefix("Message:").strip()
        if not msg:
            sel = action.selector or action.value or "unknown"
            exc_name = type(exc).__name__
            if "Timeout" in exc_name:
                msg = f"Timed out after {action.timeout_ms}ms waiting for '{sel}'"
            else:
                msg = f"{exc_name}: element '{sel}' not found or not interactable"
        return msg

    def _dispatch(self, action_type: ActionType):
        return {
            ActionType.NAVIGATE: self._do_navigate,
            ActionType.SCROLL: self._do_scroll,
            ActionType.CLICK: self._do_click,
            ActionType.FILL: self._do_fill,
            ActionType.PRESS: self._do_press,
            ActionType.WAIT_FOR: self._do_wait_for,
            ActionType.ASSERT_TEXT: self._do_assert_text,
            ActionType.ASSERT_HTTP_STATUS: self._do_assert_http_status,
            ActionType.ASSERT_BANNER: self._do_assert_banner,
            ActionType.OTHER: self._do_other,
        }[action_type]

    # ---- action handlers -------------------------------------------

    def _do_navigate(self, a: FlowAction) -> str:
        if self._collect_network:
            harvested = self._harvest_responses()
            if harvested:
                self._captured_responses.extend(harvested)
        self._driver.get(a.value)
        # Re-inject interceptor — navigation clears page JS state
        self._inject_network_interceptor()
        return f"Navigated to {a.value}"

    def _do_scroll(self, a: FlowAction) -> str:
        direction = a.params.get("direction", "down")
        pixels = a.params.get("pixels", 300)
        if a.selector:
            el = self._find(a.selector, a.timeout_ms)
            self._driver.execute_script(
                f"arguments[0].scrollBy(0, {pixels if direction == 'down' else -pixels})",
                el,
            )
        else:
            sign = 1 if direction == "down" else -1
            self._driver.execute_script(f"window.scrollBy(0, {sign * pixels})")
        return f"Scrolled {direction} {pixels}px"

    def _do_click(self, a: FlowAction) -> str:
        assert a.selector is not None
        el = self._find(a.selector, a.timeout_ms)
        el.click()
        return f"Clicked {a.selector}"

    def _do_fill(self, a: FlowAction) -> str:
        assert a.selector is not None and a.value is not None
        el = self._find(a.selector, a.timeout_ms)
        el.clear()
        el.send_keys(a.value)
        return f"Filled {a.selector}"

    def _do_press(self, a: FlowAction) -> str:
        assert a.value is not None
        key = _KEY_MAP.get(a.value.lower(), a.value)
        if a.selector:
            el = self._find(a.selector, a.timeout_ms)
            el.send_keys(key)
        else:
            from selenium.webdriver.common.action_chains import ActionChains
            ActionChains(self._driver).send_keys(key).perform()
        return f"Pressed {a.value}"

    def _do_wait_for(self, a: FlowAction) -> str:
        assert a.selector is not None
        self._find(a.selector, a.timeout_ms)
        return f"Element {a.selector} appeared"

    def _do_assert_text(self, a: FlowAction) -> str:
        assert a.selector is not None and a.value is not None
        el = self._find(a.selector, a.timeout_ms)
        text = el.text
        if a.value not in text:
            raise AssertionError(
                f"Expected text '{a.value}' in '{a.selector}', "
                f"got: '{text[:200]}'"
            )
        return f"Text assertion passed for {a.selector}"

    def _do_assert_http_status(self, a: FlowAction) -> str:
        assert a.value is not None
        expected = int(a.value)
        url_pattern = a.params.get("url_pattern", "")

        # Harvest responses captured by our JS interceptor
        harvested = self._harvest_responses()
        self._captured_responses.extend(harvested)

        # Check all captured responses (manually recorded + intercepted)
        all_statuses = []
        for resp in self._captured_responses:
            url = resp.get("url", "")
            status = resp.get("status", 0)
            all_statuses.append(f"{url} -> {status}")
            if url_pattern and url_pattern not in url:
                continue
            if status == expected:
                return f"HTTP {expected} assertion passed (url: {url})"

        # Fallback: JS performance entries
        try:
            entries = self._driver.execute_script(
                "return performance.getEntriesByType('resource')"
                ".concat(performance.getEntriesByType('navigation'))"
                ".map(e => ({name: e.name, status: e.responseStatus || 0}))"
            )
            for entry in entries:
                name = entry.get("name", "")
                status = entry.get("status", 0)
                all_statuses.append(f"{name} -> {status}")
                if url_pattern and url_pattern not in name:
                    continue
                if status == expected:
                    return f"HTTP {expected} assertion passed (url: {name})"
        except WebDriverException:
            pass

        detail = "; ".join(all_statuses[-10:]) if all_statuses else "none captured"
        raise AssertionError(
            f"Expected HTTP {expected} but not found. "
            f"Captured responses: [{detail}]"
            + (f" (filter: '{url_pattern}')" if url_pattern else "")
        )

    def _do_assert_banner(self, a: FlowAction) -> str:
        assert a.value is not None
        # Look for visible banner/alert/toast containing expected text
        selectors = [
            "[role='alert']",
            "[role='status']",
            ".alert", ".banner", ".toast", ".notification",
            ".flash", ".message", ".success", ".error",
        ]
        deadline = time.monotonic() + a.timeout_ms / 1000
        while time.monotonic() < deadline:
            for sel in selectors:
                try:
                    elements = self._driver.find_elements(By.CSS_SELECTOR, sel)
                    for el in elements:
                        if el.is_displayed() and a.value.lower() in el.text.lower():
                            return f"Banner assertion passed: found '{a.value}'"
                except WebDriverException:
                    continue
            # Also check page body as fallback
            body = self._driver.find_element(By.TAG_NAME, "body").text
            if a.value.lower() in body.lower():
                return f"Banner assertion passed: found '{a.value}' in page"
            time.sleep(0.3)
        raise AssertionError(f"Banner containing '{a.value}' not found")

    def _do_other(self, a: FlowAction) -> str:
        # OTHER actions are logged but not executed as raw scripts
        logger.info(
            "Other action '%s': %s (params=%s)",
            a.other_action_type, a.value, a.params,
        )
        return f"Other action '{a.other_action_type}' noted (no-op execution)"

    # ---- network interception ---------------------------------------

    _INTERCEPTOR_JS = """
    if (!window.__symphonyResponses) {
        window.__symphonyResponses = [];
        // Intercept fetch()
        const origFetch = window.fetch;
        window.fetch = async function(...args) {
            const resp = await origFetch.apply(this, args);
            const url = (typeof args[0] === 'string') ? args[0]
                      : (args[0] && args[0].url) ? args[0].url : '';
            window.__symphonyResponses.push({url: url, status: resp.status});
            return resp;
        };
        // Intercept XMLHttpRequest
        const origOpen = XMLHttpRequest.prototype.open;
        const origSend = XMLHttpRequest.prototype.send;
        XMLHttpRequest.prototype.open = function(method, url, ...rest) {
            this.__symUrl = url;
            return origOpen.apply(this, [method, url, ...rest]);
        };
        XMLHttpRequest.prototype.send = function(...args) {
            this.addEventListener('load', function() {
                window.__symphonyResponses.push({
                    url: this.__symUrl || '', status: this.status
                });
            });
            return origSend.apply(this, args);
        };
    }
    """

    def _inject_network_interceptor(self) -> None:
        """Inject JS that records fetch/XHR response status codes."""
        try:
            self._driver.execute_script(self._INTERCEPTOR_JS)
        except WebDriverException:
            pass

    def _harvest_responses(self) -> list[dict[str, Any]]:
        """Pull captured responses from the injected interceptor."""
        try:
            responses = self._driver.execute_script(
                "var r = window.__symphonyResponses || [];"
                "window.__symphonyResponses = [];"
                "return r;"
            )
            return responses if isinstance(responses, list) else []
        except WebDriverException:
            return []

    # ---- helpers ---------------------------------------------------

    def _find(self, selector: str, timeout_ms: int):
        """Find element by CSS selector with timeout."""
        wait = WebDriverWait(self._driver, timeout_ms / 1000)
        return wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))

    def _collect_evidence(self, action: FlowAction) -> Evidence:
        ev = Evidence()
        try:
            ss_path = self._artifact_dir / f"step_{self._step:03d}_{action.action.value}.png"
            self._driver.save_screenshot(str(ss_path))
            ev.screenshots.append(ss_path)
        except WebDriverException:
            pass

        if self._collect_dom:
            try:
                html = self._driver.page_source
                ev.dom_snapshots.append(html[:50_000])
            except WebDriverException:
                pass

        if self._collect_focus:
            try:
                tag = self._driver.execute_script(
                    "var el = document.activeElement;"
                    "return el ? el.tagName + '#' + el.id + '.' + el.className : 'none'"
                )
                ev.focused_elements.append(tag)
            except WebDriverException:
                pass

        return ev

    def record_response(self, url: str, status: int):
        """Allow external callers to feed captured HTTP responses."""
        self._captured_responses.append({"url": url, "status": status})
