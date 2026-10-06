#!/usr/bin/env python3
"""Fullscreen local control surface for the DJ robot."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402


TARGET = "dj-robot.target"
ROS_SERVICE = "dj-ros.service"
BRAIN_SERVICE = "dj-small-brain.service"

CSS = b"""
window, .page { background: #080b10; color: #f4f7fb; }
.header { margin: 28px 42px 0 42px; }
.brand { color: #f4f7fb; font: bold 18px Ubuntu; }
.clock { color: #8893a4; font: 16px Ubuntu; }
.card { background: #111722; border-radius: 24px; padding: 30px 42px; }
.indicator {
    color: #080b10; border-radius: 54px; font: bold 34px Ubuntu;
    min-width: 108px; min-height: 108px;
}
.indicator-stopped { background: #8893a4; }
.indicator-starting, .indicator-stopping { background: #ffc857; }
.indicator-running { background: #45e58b; }
.indicator-degraded, .indicator-failed { background: #ff5865; }
.title { color: #f4f7fb; font: bold 32px Ubuntu; margin-top: 12px; }
.detail { color: #8893a4; font: 16px Ubuntu; margin-top: 4px; }
.hint { color: #8893a4; font: 11px Ubuntu; margin-top: 12px; }
button.action {
    border: 0; border-radius: 13px; font: bold 18px Ubuntu;
    min-width: 330px; min-height: 74px; margin-top: 28px; box-shadow: none;
}
button.start { background: #173d2b; color: #8bf2b8; }
button.start:hover, button.start:active { background: #245b40; }
button.stop { background: #471d26; color: #ff9ba3; }
button.stop:hover, button.stop:active { background: #6a2935; }
button.wait { background: #292f39; color: #8893a4; }
.footer { color: #596273; font: 11px Ubuntu; margin-bottom: 16px; }
.compact { background: #111722; border: 2px solid #45e58b; border-radius: 16px; padding: 10px; }
.compact-status { color: #8bf2b8; font: bold 15px Ubuntu; }
button.compact-stop {
    background: #6a2935; color: #ffd5d8; border: 0; border-radius: 11px;
    font: bold 17px Ubuntu; min-width: 215px; min-height: 58px;
}
"""


def unit_property(unit: str, prop: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, f"--property={prop}", "--value"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def robot_vision_visible() -> bool:
    """Return true only after Small Brain has created its debug window."""
    try:
        result = subprocess.run(
            ["xwininfo", "-root", "-tree"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and '"Robot Vision"' in result.stdout


class RobotFace:
    def __init__(self, *, windowed: bool = False) -> None:
        provider = Gtk.CssProvider()
        provider.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self.window = Gtk.Window(title="DJ Robot")
        self.window.set_name("dj-robot-face")
        self.window.set_default_size(1024, 600)
        self.window.set_decorated(windowed)
        self.window.set_keep_above(not windowed)
        self.window.connect("destroy", Gtk.main_quit)
        self.window.connect("key-press-event", self._on_key)
        self.fullscreen = not windowed
        if self.fullscreen:
            self.window.fullscreen()

        self.action: str | None = None
        self.error = ""
        self.debug_overlay = False
        self.results: queue.Queue[tuple[bool, str]] = queue.Queue()
        self.last_state = ""
        self._build()
        # Capture native touch before child widgets process it. This works
        # around panels that emit TouchBegin/TouchEnd without mouse buttons.
        self.window_touch = Gtk.GestureMultiPress.new(self.window)
        self.window_touch.set_button(0)
        self.window_touch.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        self.window_touch.connect("pressed", self._window_touch_pressed)
        self._tick_clock()
        GLib.timeout_add(750, self._poll)
        self.window.show_all()

    @staticmethod
    def _add_class(widget: Gtk.Widget, name: str) -> None:
        widget.get_style_context().add_class(name)

    def _build(self) -> None:
        self.stack = Gtk.Stack()
        self.stack.set_hhomogeneous(False)
        self.stack.set_vhomogeneous(False)
        self.window.add(self.stack)

        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._add_class(page, "page")
        self.stack.add_named(page, "full")

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self._add_class(header, "header")
        page.pack_start(header, False, False, 0)

        brand = Gtk.Label(label="DJ  /  ROBOT")
        brand.set_halign(Gtk.Align.START)
        self._add_class(brand, "brand")
        header.pack_start(brand, False, False, 0)

        self.clock = Gtk.Label()
        self.clock.set_halign(Gtk.Align.END)
        self.clock.set_hexpand(True)
        self._add_class(self.clock, "clock")
        header.pack_end(self.clock, False, False, 0)

        center = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        center.set_valign(Gtk.Align.CENTER)
        center.set_halign(Gtk.Align.CENTER)
        center.set_vexpand(True)
        page.pack_start(center, True, True, 12)

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        card.set_size_request(720, 430)
        card.set_halign(Gtk.Align.CENTER)
        self._add_class(card, "card")
        center.pack_start(card, False, False, 0)

        self.indicator = Gtk.Label(label="●")
        self.indicator.set_halign(Gtk.Align.CENTER)
        self.indicator.set_valign(Gtk.Align.CENTER)
        self._add_class(self.indicator, "indicator")
        card.pack_start(self.indicator, False, False, 0)

        self.title = Gtk.Label(label="Robot is stopped")
        self.title.set_halign(Gtk.Align.CENTER)
        self._add_class(self.title, "title")
        card.pack_start(self.title, False, False, 0)

        self.detail = Gtk.Label(label="Ready to start on this NUC")
        self.detail.set_halign(Gtk.Align.CENTER)
        self._add_class(self.detail, "detail")
        card.pack_start(self.detail, False, False, 0)

        self.button = Gtk.Button(label="START ROBOT")
        self.button.set_halign(Gtk.Align.CENTER)
        self.button.set_can_focus(False)
        self.button.connect("clicked", self._button_clicked)
        self._add_class(self.button, "action")
        card.pack_start(self.button, False, False, 0)

        # This consumes the panel's native XInput2 TouchBegin event and fires
        # on contact; the WaveShare does not synthesize a mouse button press.
        self.touch_gesture = Gtk.GestureMultiPress.new(self.button)
        self.touch_gesture.set_button(0)
        self.touch_gesture.connect("pressed", self._touch_pressed)

        self.hint = Gtk.Label()
        self.hint.set_halign(Gtk.Align.CENTER)
        self.hint.set_line_wrap(True)
        self.hint.set_max_width_chars(72)
        self._add_class(self.hint, "hint")
        card.pack_start(self.hint, False, False, 0)

        footer = Gtk.Label(label="Local control  •  No client computer required")
        footer.set_halign(Gtk.Align.CENTER)
        self._add_class(footer, "footer")
        page.pack_end(footer, False, False, 0)

        compact = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self._add_class(compact, "compact")
        compact_status = Gtk.Label(label="●  ROBOT RUNNING")
        self._add_class(compact_status, "compact-status")
        compact.pack_start(compact_status, False, False, 4)
        self.compact_button = Gtk.Button(label="STOP ROBOT")
        self.compact_button.set_can_focus(False)
        self.compact_button.connect("clicked", self._button_clicked)
        self._add_class(self.compact_button, "compact-stop")
        compact.pack_end(self.compact_button, False, False, 0)
        self.stack.add_named(compact, "debug")

        self.compact_touch_gesture = Gtk.GestureMultiPress.new(self.compact_button)
        self.compact_touch_gesture.set_button(0)
        self.compact_touch_gesture.connect("pressed", self._touch_pressed)

        self._render("stopped")

    def _on_key(self, _window: Gtk.Window, event: Gdk.EventKey) -> bool:
        if event.keyval == Gdk.KEY_F11:
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                self.window.fullscreen()
                self.window.set_keep_above(True)
            else:
                self.window.unfullscreen()
                self.window.set_keep_above(False)
            return True
        modifiers = event.state & Gtk.accelerator_get_default_mod_mask()
        if event.keyval in (Gdk.KEY_q, Gdk.KEY_Q) and modifiers == (
            Gdk.ModifierType.CONTROL_MASK | Gdk.ModifierType.SHIFT_MASK
        ):
            self.window.destroy()
            return True
        return False

    def _tick_clock(self) -> bool:
        self.clock.set_text(dt.datetime.now().strftime("%H:%M  ·  %a %d %b"))
        GLib.timeout_add_seconds(1, self._tick_clock)
        return False

    def _touch_pressed(
        self, gesture: Gtk.GestureMultiPress, _count: int, _x: float, _y: float
    ) -> None:
        button = gesture.get_widget()
        if button.get_sensitive() and not self.action:
            button.emit("clicked")

    def _window_touch_pressed(
        self, _gesture: Gtk.GestureMultiPress, _count: int, x: float, y: float
    ) -> None:
        active_button = self.compact_button if self.debug_overlay else self.button
        origin = active_button.translate_coordinates(self.window, 0, 0)
        if origin is None:
            return
        button_x, button_y = origin
        allocation = active_button.get_allocation()
        padding = 35
        inside = (
            button_x - padding <= x <= button_x + allocation.width + padding
            and button_y - padding <= y <= button_y + allocation.height + padding
        )
        print(
            f"touch x={x:.0f} y={y:.0f} button="
            f"{button_x},{button_y},{allocation.width},{allocation.height} inside={inside}",
            flush=True,
        )
        if inside and active_button.get_sensitive() and not self.action:
            active_button.emit("clicked")

    def _set_debug_overlay(self, enabled: bool) -> None:
        if enabled == self.debug_overlay:
            return
        self.debug_overlay = enabled
        if enabled:
            self.stack.set_visible_child_name("debug")
            self.window.unfullscreen()
            self.window.set_keep_above(True)
            self._place_debug_overlay()
            GLib.timeout_add(150, self._place_debug_overlay)
        else:
            self.stack.set_visible_child_name("full")
            self.window.set_keep_above(True)
            self.window.fullscreen()

    def _place_debug_overlay(self) -> bool:
        self.window.resize(390, 92)
        screen = self.window.get_screen()
        monitor = screen.get_monitor_geometry(screen.get_primary_monitor())
        width, height = self.window.get_size()
        self.window.move(
            monitor.x + monitor.width - width - 20,
            monitor.y + monitor.height - height - 20,
        )
        return False

    def _button_clicked(self, _button: Gtk.Button) -> None:
        if self.action:
            return
        if self.last_state in {"starting", "running", "degraded"}:
            self._stop()
        else:
            self._start()

    def _run_action(self, action: str) -> None:
        try:
            if action == "start":
                names = [
                    name
                    for name in (
                        "DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY",
                        "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR", "OPENAI_API_KEY",
                    )
                    if os.environ.get(name)
                ]
                if names:
                    subprocess.run(
                        ["systemctl", "--user", "import-environment", *names],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                command = ["systemctl", "--user", "start", TARGET]
            else:
                command = ["systemctl", "--user", "stop", TARGET]
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=90,
                check=False,
            )
            self.results.put((result.returncode == 0, result.stdout.strip()))
        except (OSError, subprocess.TimeoutExpired) as error:
            self.results.put((False, str(error)))

    def _start(self) -> None:
        self.error = ""
        self.action = "start"
        self._render("starting")
        threading.Thread(target=self._run_action, args=("start",), daemon=True).start()

    def _stop(self) -> None:
        self.error = ""
        self.action = "stop"
        self._render("stopping")
        threading.Thread(target=self._run_action, args=("stop",), daemon=True).start()

    def _service_state(self) -> str:
        target = unit_property(TARGET, "ActiveState")
        ros = unit_property(ROS_SERVICE, "ActiveState")
        brain = unit_property(BRAIN_SERVICE, "ActiveState")
        if target == "active" and ros == "active" and brain == "active":
            return "running" if robot_vision_visible() else "starting"
        if any(value == "activating" for value in (target, ros, brain)):
            return "starting"
        if any(value == "deactivating" for value in (target, ros, brain)):
            return "stopping"
        if any(value == "active" for value in (target, ros, brain)):
            return "degraded"
        if any(value == "failed" for value in (ros, brain)):
            return "failed"
        return "stopped"

    def _poll(self) -> bool:
        try:
            ok, message = self.results.get_nowait()
            completed_action = self.action
            self.action = None
            if not ok:
                self.error = message or f"Could not {completed_action} robot"
        except queue.Empty:
            pass

        if not self.action:
            state = self._service_state()
            if self.error and state == "stopped":
                state = "failed"
            if state != self.last_state:
                self._render(state, self.error)
        return True

    def _render(self, state: str, error: str = "") -> None:
        self.last_state = state
        options = {
            "stopped": ("●", "Robot is stopped", "Ready to start on this NUC"),
            "starting": ("…", "Starting robot", "Connecting hardware and cognition…"),
            "running": ("✓", "Robot is running", "ROS and Small Brain are active"),
            "degraded": ("!", "Robot is partially running", "A service failed; stop before trying again"),
            "stopping": ("…", "Stopping safely", "Sending zero velocity, then closing services…"),
            "failed": ("!", "Robot needs attention", "Startup did not complete"),
        }
        mark, title, detail = options[state]
        self.indicator.set_text(mark)
        self.title.set_text(title)
        self.detail.set_text(detail)

        indicator_style = self.indicator.get_style_context()
        for name in options:
            indicator_style.remove_class(f"indicator-{name}")
        indicator_style.add_class(f"indicator-{state}")

        button_style = self.button.get_style_context()
        for name in ("start", "stop", "wait"):
            button_style.remove_class(name)
        if state in {"running", "degraded"}:
            self.button.set_label("STOP ROBOT")
            self.button.set_sensitive(True)
            button_style.add_class("stop")
            self.hint.set_text(
                "Stop performs a controlled software shutdown."
                if state == "running"
                else "Part of the robot is still active. Use Stop to shut it down safely."
            )
        elif state == "starting":
            self.button.set_label("STOP STARTUP")
            self.button.set_sensitive(True)
            button_style.add_class("stop")
            self.hint.set_text("Models are loading. You can safely cancel startup.")
        elif state == "stopping":
            self.button.set_label("PLEASE WAIT")
            self.button.set_sensitive(False)
            button_style.add_class("wait")
            self.hint.set_text("")
        else:
            self.button.set_label("START ROBOT" if state == "stopped" else "TRY AGAIN")
            self.button.set_sensitive(True)
            button_style.add_class("start")
            if state == "failed":
                concise = " ".join(error.split())[-180:] if error else "Check the service logs."
                self.hint.set_text(concise)
            else:
                self.hint.set_text("")
        self.compact_button.set_sensitive(state == "running" and not self.action)
        self._set_debug_overlay(state == "running")


def acquire_single_instance() -> object:
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
    lock = open(runtime / f"dj-face-{os.getuid()}.lock", "w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(0)
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windowed", action="store_true", help="Do not use fullscreen")
    args = parser.parse_args()
    lock = acquire_single_instance()
    app = RobotFace(windowed=args.windowed)
    Gtk.main()
    del app, lock
    return 0


if __name__ == "__main__":
    sys.exit(main())
