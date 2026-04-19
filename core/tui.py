from __future__ import annotations

from contextlib import contextmanager
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner


from .spinners import ensure_bw_spinners


ensure_bw_spinners()

SPINNER_MAP = {
    "pulsing_star": "pulsing_star_bw",
    "orbit": "orbit_bw",
    "dots": "dots",
    "bouncingBall": "bounce_bw",
}


@dataclass
class StatusLine:
    label: str
    status: str
    detail: Optional[str] = None


class SymphonyTUI:
    """Dynamic terminal UI for Symphony runs."""

    def __init__(self, console: Optional[Console] = None, detailed: bool = False) -> None:
        self.console = console or Console()
        self.detailed = detailed
        self.header: Dict[str, str] = {}
        self.status_lines: Dict[str, StatusLine] = {}
        self.voice_lines: List[str] = []
        self.todo_lines: List[str] = []
        self.footer: Optional[str] = None
        self._live: Optional[Live] = None
        self._active_activity: Optional[str] = None
        self._spinner_style: str = "dots"
        self._activity_progress: List[str] = []
        self._heartbeat_interval = 3.0
        self._heartbeat_text = "…"
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop: Optional[threading.Event] = None
        self._start_time = time.monotonic()

    @contextmanager
    def live(self):
        with Live(self._render(), console=self.console, refresh_per_second=8, transient=False) as live:
            self._live = live
            self._start_time = time.monotonic()
            self._start_heartbeat()
            try:
                yield self
            finally:
                self._stop_heartbeat()
                self._live = None

    def set_header(self, **fields: str) -> None:
        self.header.update({k: str(v) for k, v in fields.items() if v is not None})
        self._refresh()

    def update_status(self, label: str, status: str, detail: Optional[str] = None) -> None:
        self.status_lines[label] = StatusLine(label=label, status=status, detail=detail)
        self._refresh()

    def add_voice(self, text: str, *, icon: str = "•") -> None:
        line = f"{icon} {text.strip()}"
        self.voice_lines.append(line)
        self._refresh()

    def add_sub_info(self, text: str) -> None:
        self.voice_lines.append(f"  ⎿ {text.strip()}")
        self._refresh()

    def start_activity(self, text: str, *, spinner: str = "dots") -> None:
        """Show a spinner while a long-running step is in progress."""

        self._active_activity = text.strip()
        self._spinner_style = SPINNER_MAP.get(spinner, spinner)
        self._activity_progress = []
        self._refresh()

    def stop_activity(self, completion: Optional[str] = None, *, icon: str = "•") -> None:
        self._active_activity = None
        self._activity_progress = []
        if completion:
            self.add_voice(completion, icon=icon)
        else:
            self._refresh()

    def update_activity_progress(self, detail: str) -> None:
        if not detail:
            return
        self._activity_progress.append(detail)
        self._refresh()

    def mark_timeout(self, label: str) -> None:
        self.add_voice(f"{label} – TIMED OUT", icon="[warn]")

    def set_footer(self, text: str) -> None:
        self.footer = text
        self._refresh()

    def set_todos(self, todos: Dict[str, bool]) -> None:
        self.todo_lines = []
        if todos:
            self.todo_lines.append("Update Todos")
            for label, done in todos.items():
                checkbox = "☒" if done else "☐"
                self.todo_lines.append(f"{checkbox} {label}")
        self._refresh()

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    def _render(self):
        header_table = Table.grid(expand=True)
        for key, value in self.header.items():
            header_table.add_row(f"[bold]{key}[/]: {value}")

        status_table = Table.grid(padding=0)
        for status in self.status_lines.values():
            detail = f" – {status.detail}" if status.detail else ""
            status_table.add_row(f"{status.label}: {status.status}{detail}")
        status_table.add_row(f"Heartbeat: {self._heartbeat_text}")

        voice = Table.grid(padding=0)
        for line in self.voice_lines[-15:]:
            voice.add_row(line)
        if self._active_activity:
            spinner_text = self._active_activity
            if self._activity_progress:
                spinner_text += " – " + " | ".join(self._activity_progress[-3:])
            voice.add_row(Spinner(self._spinner_style, text=spinner_text))

        items: List = [Panel(header_table, title="Symphony"), Panel(status_table, title="Status")]

        if self.todo_lines:
            todo_table = Table.grid(padding=0)
            for line in self.todo_lines:
                todo_table.add_row(line)
            items.append(Panel(todo_table, title="Todos"))

        items.append(Panel(voice, title="Activity"))

        if self.footer:
            items.append(Panel(Text(self.footer), title="Summary"))

        return Group(*items)

    def print_detailed(self, text: str) -> None:
        if self.detailed:
            self.console.print(Text(text, style="dim"))

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop = threading.Event()

        def _pulse() -> None:
            while self._heartbeat_stop and not self._heartbeat_stop.wait(self._heartbeat_interval):
                elapsed = int(time.monotonic() - self._start_time)
                self._heartbeat_text = f"{elapsed}s"
                self._refresh()

        self._heartbeat_thread = threading.Thread(target=_pulse, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_stop:
            self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=0.2)
        self._heartbeat_thread = None
        self._heartbeat_stop = None
