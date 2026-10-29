"""Shared paths for timestamped Small Brain diagnostic sessions."""

from __future__ import annotations

from datetime import datetime
import asyncio
from collections import Counter
import copy
import json
import os
from pathlib import Path
import threading
import time


_SMALL_BRAIN_ROOT = Path(__file__).resolve().parents[1]
_SESSION_STAMP = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def default_output_directory() -> Path:
    configured = os.environ.get("DJ_SMALLBRAIN_LOG_DIR")
    if configured:
        return Path(configured).expanduser()
    return _SMALL_BRAIN_ROOT.parent / "diagnostics" / "smallbrain_sessions"


def session_log_path(stream_name: str) -> Path:
    """Return a log path sharing one timestamp across the current process."""
    return default_output_directory() / f"{_SESSION_STAMP}-{stream_name}.jsonl"


class BestEffortSessionLogger:
    """Small rate-limited JSONL writer that never raises into robot logic."""

    def __init__(self, stream_name, *, interval_seconds=1.0,
                 max_bytes=5 * 1024 * 1024):
        self.path = session_log_path(stream_name)
        self.interval_seconds = float(interval_seconds)
        self.max_bytes = int(max_bytes)
        self._last_write_at = None
        self._disabled = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._disabled = True

    def write(self, event, *, now=None):
        now = time.monotonic() if now is None else float(now)
        if (self._disabled or
                (self._last_write_at is not None and
                 now - self._last_write_at < self.interval_seconds)):
            return False
        try:
            if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                self.path.replace(self.path.with_name(self.path.name + ".1"))
            payload = {
                "timestamp": datetime.now().astimezone().isoformat(),
                **event,
            }
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(
                    payload,
                    separators=(",", ":"),
                    default=lambda value: value.item()
                    if hasattr(value, "item") else str(value),
                    allow_nan=False,
                ) + "\n")
            self._last_write_at = now
            return True
        except (OSError, TypeError, ValueError):
            self._disabled = True
            return False


class SmallBrainDiagnostics:
    """Collect component state and publish one correlated snapshot per second."""

    def __init__(self):
        self.path = session_log_path("smallbrain")
        self._writer = BestEffortSessionLogger(
            "smallbrain", interval_seconds=0.0)
        self._lock = threading.Lock()
        self._components = {}
        self._windows = {}
        self._sequence = 0
        self._started_at = time.monotonic()

    def update(self, component, state, *, tally_group=None, tally_value=None):
        """Replace one component snapshot; failures remain diagnostics-only."""
        try:
            with self._lock:
                self._components[str(component)] = copy.deepcopy(dict(state))
                window = self._windows.setdefault(
                    str(component), {
                        "updates": 0, "tallies": {}, "events": []
                    })
                window["updates"] += 1
                if tally_group is not None and tally_value is not None:
                    counter = window["tallies"].setdefault(
                        str(tally_group), Counter())
                    counter[str(tally_value)] += 1
        except Exception:
            return False
        return True

    def record_event(self, component, event, **data):
        """Preserve a fast lifecycle event until the next 1 Hz snapshot."""
        try:
            with self._lock:
                self._components.setdefault(str(component), {})
                window = self._windows.setdefault(
                    str(component), {
                        "updates": 0, "tallies": {}, "events": []
                    })
                window["events"].append({
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "event": str(event),
                    **copy.deepcopy(data),
                })
                del window["events"][:-50]
        except Exception:
            return False
        return True

    def snapshot(self):
        with self._lock:
            components = copy.deepcopy(self._components)
            windows = self._windows
            self._windows = {}
            self._sequence += 1
            sequence = self._sequence
        for component, window in windows.items():
            if component not in components:
                continue
            components[component]["window"] = {
                "updates": window["updates"],
                "events": window["events"],
                **{
                    group: dict(counts)
                    for group, counts in window["tallies"].items()
                },
            }
        return {
            "event": "smallbrain_snapshot",
            "sequence": sequence,
            "uptime_seconds": round(
                time.monotonic() - self._started_at, 3),
            "components": components,
        }

    async def run(self):
        """Write combined state on a stable one-second cadence."""
        loop = asyncio.get_running_loop()
        while True:
            started_at = loop.time()
            self._writer.write(self.snapshot(), now=started_at)
            await asyncio.sleep(max(0.0, 1.0 - (loop.time() - started_at)))


smallbrain_diagnostics = SmallBrainDiagnostics()
