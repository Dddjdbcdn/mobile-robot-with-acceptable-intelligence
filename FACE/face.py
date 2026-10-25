#!/usr/bin/env python3
"""Fullscreen local control surface for the DJ robot."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

try:
    import zmq
except ImportError:  # FACE still needs to offer Stop if the optional map client is absent.
    zmq = None

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, GLib, Gtk  # noqa: E402


TARGET = "dj-robot.target"
ROS_SERVICE = "dj-ros.service"
BRAIN_SERVICE = "dj-small-brain.service"
MAP_ENDPOINT = os.environ.get("MAP_STREAM_ENDPOINT", "tcp://127.0.0.1:5559")
RESET_ENDPOINT = os.environ.get("RESET_ENDPOINT", "tcp://127.0.0.1:5561")
MAP_REFRESH_SECONDS = 2.0
MAP_LOCAL_DEFAULT_METERS = 3.0
MAP_LOCAL_MIN_METERS = 3.0
MAP_LOCAL_MAX_METERS = 10.0

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
.wifi-status { color: #ffc857; font: bold 15px Ubuntu; }
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
button.wifi {
    background: #292f39; color: #ffd978; border: 0; border-radius: 10px;
    font: bold 14px Ubuntu; min-width: 180px; min-height: 44px;
}
button.wifi:hover, button.wifi:active { background: #3a424f; }
.footer { color: #596273; font: 11px Ubuntu; margin-bottom: 16px; }
.compact { background: #111722; border: 2px solid #45e58b; border-radius: 13px; padding: 7px; }
button.compact-stop {
    background: #6a2935; color: #ffd5d8; border: 0; border-radius: 9px;
    font: bold 14px Ubuntu; min-width: 135px; min-height: 44px;
}
button.compact-map {
    background: #173d4d; color: #9cdef5; border: 0; border-radius: 9px;
    font: bold 14px Ubuntu; min-width: 80px; min-height: 44px;
}
button.compact-step {
    background: #173d4d; color: #9cdef5; border: 0; border-radius: 9px;
    font: bold 18px Ubuntu; min-width: 44px; min-height: 44px;
}
.map-size {
    color: #d8f3fc; font: bold 14px Ubuntu; min-width: 58px;
}
button.compact-reset {
    background: #493718; color: #ffda83; border: 0; border-radius: 9px;
    font: bold 14px Ubuntu; min-width: 82px; min-height: 44px;
}
button.compact-reset:hover, button.compact-reset:active {
    background: #654d20;
}
.map-page { background: #080b10; }
.map-status {
    background: rgba(8, 11, 16, 0.82); color: #aab4c4;
    border-radius: 8px; padding: 6px 10px; font: 13px Ubuntu;
}
.reset-overlay {
    background: rgba(8, 11, 16, 0.94); color: #ffc857;
    border: 4px solid #ffc857; border-radius: 16px;
    padding: 26px 42px; font: bold 25px Ubuntu;
}
.reset-danger { color: #ff9ba3; border-color: #ff5865; }
.reset-ready { color: #8bf2b8; border-color: #45e58b; }
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


def active_wifi_name() -> str | None:
    """Return NetworkManager's active Wi-Fi connection name."""
    try:
        result = subprocess.run(
            [
                "nmcli", "-t", "-f", "TYPE,STATE,CONNECTION",
                "device", "status",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return next((
        fields[2].replace(r"\:", ":")
        for line in result.stdout.splitlines()
        if len(fields := line.split(":", 2)) == 3
        and fields[0] == "wifi"
        and fields[1] == "connected"
        and fields[2]
    ), None)


def duration_milliseconds(value: object) -> float:
    """Return a safe, non-negative duration for display."""
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, duration)


class CpuUsage:
    """Measure whole-system CPU utilization from Linux ``/proc/stat``."""

    def __init__(self, sample_interval_s: float = 0.5) -> None:
        self.sample_interval_s = float(sample_interval_s)
        self.last_sample_at: float | None = None
        self.last_total: int | None = None
        self.last_idle: int | None = None
        self.usage = 0.0

    @staticmethod
    def _read_times() -> tuple[int, int]:
        with open("/proc/stat", encoding="ascii") as stat_file:
            fields = stat_file.readline().split()
        if not fields or fields[0] != "cpu" or len(fields) < 5:
            raise ValueError("unexpected /proc/stat CPU data")
        times = [int(value) for value in fields[1:9]]
        idle = times[3] + (times[4] if len(times) > 4 else 0)
        return sum(times), idle

    def sample(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else float(now)
        if (
            self.last_sample_at is not None
            and now - self.last_sample_at < self.sample_interval_s
        ):
            return self.usage
        try:
            total, idle = self._read_times()
        except (OSError, ValueError):
            return self.usage
        if self.last_total is not None and self.last_idle is not None:
            total_delta = total - self.last_total
            idle_delta = idle - self.last_idle
            if total_delta > 0:
                self.usage = max(
                    0.0,
                    min(
                        100.0,
                        100.0 * (total_delta - idle_delta) / total_delta,
                    ),
                )
        self.last_total = total
        self.last_idle = idle
        self.last_sample_at = now
        return self.usage


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
        self.window.connect("window-state-event", self._window_state_changed)
        self.fullscreen = not windowed
        if self.fullscreen:
            self.window.fullscreen()

        self.action: str | None = None
        self.error = ""
        self.debug_overlay = False
        self.map_visible = False
        self.map_layer = "combined"
        self.map_crop_size_m: float | None = None
        self.map_local_size_m = MAP_LOCAL_DEFAULT_METERS
        self.map_request_pending = False
        self.reset_pending = False
        self.map_reset_state = None
        self.map_ready_generation = 0
        self.map_pixbuf = None
        self.map_results: queue.Queue[
            tuple[bool, object, object, object, float]
        ] = queue.Queue()
        self.map_cpu_usage = CpuUsage()
        self.map_cpu_usage.sample()
        self.reset_results: queue.Queue[tuple[bool, str]] = queue.Queue()
        self.wifi_action = False
        self.wifi_name: str | None = None
        self.wifi_checked_at = -float("inf")
        self.results: queue.Queue[tuple[bool, str]] = queue.Queue()
        self.last_state = ""
        self.last_touch_button = None
        self.last_touch_at = -float("inf")
        self.last_map_view_change_at = -float("inf")
        self.last_map_size_change_at = -float("inf")
        self._build()
        # Capture native touch before child widgets process it. This works
        # around panels that emit TouchBegin/TouchEnd without mouse buttons.
        self.window_touch = Gtk.GestureMultiPress.new(self.window)
        self.window_touch.set_button(0)
        self.window_touch.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        self.window_touch.connect("pressed", self._window_touch_pressed)
        self._tick_clock()
        GLib.timeout_add(750, self._poll)
        GLib.timeout_add(250, self._poll_map)
        GLib.timeout_add(100, self._poll_reset)
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

        wifi_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        wifi_box.set_halign(Gtk.Align.CENTER)
        wifi_box.set_margin_top(20)
        self.wifi_status = Gtk.Label(label="Wi-Fi: checking…")
        self.wifi_status.set_halign(Gtk.Align.CENTER)
        self._add_class(self.wifi_status, "wifi-status")
        wifi_box.pack_start(self.wifi_status, False, False, 0)
        self.wifi_button = Gtk.Button(label="CHANGE WI-FI")
        self.wifi_button.set_can_focus(False)
        self.wifi_button.connect("clicked", self._wifi_button_clicked)
        self._add_class(self.wifi_button, "wifi")
        wifi_box.pack_start(self.wifi_button, False, False, 0)
        card.pack_start(wifi_box, False, False, 0)

        self.wifi_touch_gesture = Gtk.GestureMultiPress.new(self.wifi_button)
        self.wifi_touch_gesture.set_button(0)
        self.wifi_touch_gesture.connect("pressed", self._touch_pressed)

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

        footer = Gtk.Label(label="DJ 2026 • Made in Vietnam")
        footer.set_halign(Gtk.Align.CENTER)
        self._add_class(footer, "footer")
        page.pack_end(footer, False, False, 0)

        compact = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=7)
        self._add_class(compact, "compact")
        self.compact_button = Gtk.Button(label="STOP ROBOT")
        self.compact_button.set_can_focus(False)
        self.compact_button.connect("clicked", self._button_clicked)
        self._add_class(self.compact_button, "compact-stop")
        compact.pack_end(self.compact_button, False, False, 0)
        self.compact_reset_button = Gtk.Button(label="RESET")
        self.compact_reset_button.set_can_focus(False)
        self.compact_reset_button.connect("clicked", self._reset_button_clicked)
        self._add_class(self.compact_reset_button, "compact-reset")
        compact.pack_end(self.compact_reset_button, False, False, 0)
        self.compact_map_button = Gtk.Button(label="MAP")
        self.compact_map_button.set_can_focus(False)
        self.compact_map_button.connect("clicked", self._map_button_clicked)
        self._add_class(self.compact_map_button, "compact-map")
        compact.pack_end(self.compact_map_button, False, False, 0)
        self.stack.add_named(compact, "debug")

        self.compact_touch_gesture = Gtk.GestureMultiPress.new(self.compact_button)
        self.compact_touch_gesture.set_button(0)
        self.compact_touch_gesture.connect("pressed", self._touch_pressed)

        self.compact_map_touch_gesture = Gtk.GestureMultiPress.new(
            self.compact_map_button
        )
        self.compact_map_touch_gesture.set_button(0)
        self.compact_map_touch_gesture.connect("pressed", self._touch_pressed)

        self.compact_reset_touch_gesture = Gtk.GestureMultiPress.new(
            self.compact_reset_button
        )
        self.compact_reset_touch_gesture.set_button(0)
        self.compact_reset_touch_gesture.connect("pressed", self._touch_pressed)

        self._build_map_page()

        self._render("stopped")

    def _build_map_page(self) -> None:
        page = Gtk.Overlay()
        self._add_class(page, "map-page")
        self.stack.add_named(page, "map")

        self.map_status = Gtk.Label(label="Waiting for map service…")
        self.map_status.set_halign(Gtk.Align.END)
        self.map_status.set_valign(Gtk.Align.START)
        self.map_status.set_margin_top(20)
        self.map_status.set_margin_end(20)
        self.map_status.set_xalign(0.0)
        self.map_status.set_justify(Gtk.Justification.LEFT)
        self._add_class(self.map_status, "map-status")

        # Keep the bitmap's natural dimensions out of page layout. The image
        # is scaled against the entire screen. Every map aspect ratio remains
        # visible without distortion; controls are overlaid instead of taking
        # space away from the map.
        self.map_viewport = Gtk.ScrolledWindow()
        self.map_viewport.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.NEVER
        )
        self.map_viewport.set_shadow_type(Gtk.ShadowType.NONE)
        self.map_viewport.set_min_content_width(1)
        self.map_viewport.set_min_content_height(1)
        self.map_viewport.set_propagate_natural_width(False)
        self.map_viewport.set_propagate_natural_height(False)
        self.map_viewport.set_hexpand(True)
        self.map_viewport.set_vexpand(True)
        self.map_viewport.connect("size-allocate", self._map_image_resized)

        self.map_image = Gtk.Image()
        self.map_image.set_size_request(1, 1)
        self.map_image.set_halign(Gtk.Align.CENTER)
        self.map_image.set_valign(Gtk.Align.CENTER)
        self.map_viewport.add_with_viewport(self.map_image)
        page.add(self.map_viewport)
        page.add_overlay(self.map_status)

        self.map_reset_overlay = Gtk.Label()
        self.map_reset_overlay.set_halign(Gtk.Align.CENTER)
        self.map_reset_overlay.set_valign(Gtk.Align.CENTER)
        self.map_reset_overlay.set_justify(Gtk.Justification.CENTER)
        self.map_reset_overlay.set_no_show_all(True)
        self._add_class(self.map_reset_overlay, "reset-overlay")
        page.add_overlay(self.map_reset_overlay)

        controls = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=7)
        controls.set_halign(Gtk.Align.END)
        controls.set_valign(Gtk.Align.END)
        controls.set_margin_end(20)
        controls.set_margin_bottom(20)
        self._add_class(controls, "compact")

        self.map_back_button = Gtk.Button(label="CAMERA")
        self.map_back_button.set_can_focus(False)
        self.map_back_button.connect("clicked", self._map_back_clicked)
        self._add_class(self.map_back_button, "compact-map")

        self.map_view_button = Gtk.Button(label="LOCAL")
        self.map_view_button.set_can_focus(False)
        self.map_view_button.connect("clicked", self._map_view_clicked)
        self._add_class(self.map_view_button, "compact-map")

        self.map_size_down_button = Gtk.Button(label="−")
        self.map_size_down_button.set_can_focus(False)
        self.map_size_down_button.connect(
            "clicked", self._map_size_down_clicked
        )
        self._add_class(self.map_size_down_button, "compact-step")

        self.map_size_label = Gtk.Label(label="3×3")
        self.map_size_label.set_valign(Gtk.Align.CENTER)
        self._add_class(self.map_size_label, "map-size")

        self.map_size_up_button = Gtk.Button(label="+")
        self.map_size_up_button.set_can_focus(False)
        self.map_size_up_button.connect("clicked", self._map_size_up_clicked)
        self._add_class(self.map_size_up_button, "compact-step")

        self.map_stop_button = Gtk.Button(label="STOP ROBOT")
        self.map_stop_button.set_can_focus(False)
        self.map_stop_button.connect("clicked", self._button_clicked)
        self._add_class(self.map_stop_button, "compact-stop")

        self.map_reset_button = Gtk.Button(label="RESET")
        self.map_reset_button.set_can_focus(False)
        self.map_reset_button.connect("clicked", self._reset_button_clicked)
        self._add_class(self.map_reset_button, "compact-reset")

        self.map_size_controls = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=4
        )
        self.map_size_controls.set_no_show_all(True)
        self.map_size_controls.pack_start(
            self.map_size_down_button, False, False, 0
        )
        self.map_size_controls.pack_start(
            self.map_size_label, False, False, 0
        )
        self.map_size_controls.pack_start(
            self.map_size_up_button, False, False, 0
        )

        controls.pack_start(self.map_size_controls, False, False, 0)
        controls.pack_start(self.map_view_button, False, False, 0)
        controls.pack_start(self.map_back_button, False, False, 0)
        controls.pack_start(self.map_reset_button, False, False, 0)
        controls.pack_start(self.map_stop_button, False, False, 0)
        page.add_overlay(controls)

        self.map_touch_gestures = []
        for button in (
            self.map_view_button,
            self.map_size_down_button,
            self.map_size_up_button,
            self.map_back_button,
            self.map_reset_button,
            self.map_stop_button,
        ):
            gesture = Gtk.GestureMultiPress.new(button)
            gesture.set_button(0)
            gesture.connect("pressed", self._touch_pressed)
            self.map_touch_gestures.append(gesture)

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

    def _window_state_changed(
        self, _window: Gtk.Window, event: Gdk.EventWindowState
    ) -> bool:
        if (
            self.debug_overlay
            and not self.map_visible
            and not (
                event.new_window_state & Gdk.WindowState.FULLSCREEN
            )
        ):
            GLib.idle_add(self._place_debug_overlay)
        return False

    def _tick_clock(self) -> bool:
        self.clock.set_text(dt.datetime.now().strftime("%H:%M  ·  %a %d %b"))
        GLib.timeout_add_seconds(1, self._tick_clock)
        return False

    def _touch_pressed(
        self, gesture: Gtk.GestureMultiPress, _count: int, _x: float, _y: float
    ) -> None:
        self._emit_touch_click(gesture.get_widget())

    def _emit_touch_click(self, button: Gtk.Button) -> None:
        """Emit once when both window and child touch gestures see a press."""
        now = time.monotonic()
        if button is self.last_touch_button and now - self.last_touch_at < 0.20:
            return
        self.last_touch_button = button
        self.last_touch_at = now
        if button.get_sensitive() and not self.action:
            button.emit("clicked")

    def _window_touch_pressed(
        self, _gesture: Gtk.GestureMultiPress, _count: int, x: float, y: float
    ) -> None:
        active_buttons = (
            [
                self.map_view_button,
                self.map_size_down_button,
                self.map_size_up_button,
                self.map_back_button,
                self.map_reset_button,
                self.map_stop_button,
            ]
            if self.map_visible else
            [
                self.compact_map_button,
                self.compact_reset_button,
                self.compact_button,
            ]
            if self.debug_overlay else [self.wifi_button, self.button]
        )
        for active_button in active_buttons:
            origin = active_button.translate_coordinates(self.window, 0, 0)
            if origin is None:
                continue
            button_x, button_y = origin
            allocation = active_button.get_allocation()
            padding = 3 if self.debug_overlay else 20
            inside = (
                button_x - padding <= x <= button_x + allocation.width + padding
                and button_y - padding <= y <= button_y + allocation.height + padding
            )
            if inside and active_button.get_sensitive() and not self.action:
                self._emit_touch_click(active_button)
                return

    def _set_debug_overlay(self, enabled: bool) -> None:
        if enabled == self.debug_overlay:
            return
        self.debug_overlay = enabled
        if enabled:
            self.map_visible = False
            self.stack.set_visible_child_name("debug")
            self.window.unfullscreen()
            self.window.set_keep_above(True)
            self._place_debug_overlay()
            GLib.timeout_add(150, self._place_debug_overlay)
        else:
            self.map_visible = False
            self.stack.set_visible_child_name("full")
            self.window.set_keep_above(True)
            self.window.fullscreen()

    def _place_debug_overlay(self) -> bool:
        if self.map_visible:
            return False
        self.window.resize(165, 174)
        screen = self.window.get_screen()
        monitor = screen.get_monitor_geometry(screen.get_primary_monitor())
        width, height = self.window.get_size()
        self.window.move(
            monitor.x + monitor.width - width - 20,
            monitor.y + monitor.height - height - 20,
        )
        return False

    def _map_button_clicked(self, _button: Gtk.Button) -> None:
        if self.action or self.last_state != "running":
            return
        self.map_visible = True
        self.stack.set_visible_child_name("map")
        self.window.fullscreen()
        self.window.set_keep_above(True)
        self.map_status.set_text("Loading map…")
        self._request_map_if_due(force=True)

    def _map_back_clicked(self, _button: Gtk.Button) -> None:
        if not self.map_visible:
            return
        self.map_visible = False
        self.stack.set_visible_child_name("debug")
        self.window.unfullscreen()
        self._place_debug_overlay()
        GLib.timeout_add(150, self._place_debug_overlay)

    def _map_view_clicked(self, _button: Gtk.Button) -> None:
        if self.action or not self.map_visible:
            return
        now = time.monotonic()
        if now - self.last_map_view_change_at < 0.25:
            return
        self.last_map_view_change_at = now
        self.map_crop_size_m = (
            None
            if self.map_crop_size_m is not None
            else self.map_local_size_m
        )
        self._update_map_view_controls()
        self.map_status.set_text(
            "Loading full overview…"
            if self.map_crop_size_m is None else
            f"Loading {self.map_local_size_m:g}×"
            f"{self.map_local_size_m:g} m local map…"
        )
        self._request_map_if_due(force=True)

    def _map_size_down_clicked(self, _button: Gtk.Button) -> None:
        self._change_map_local_size(-1.0)

    def _map_size_up_clicked(self, _button: Gtk.Button) -> None:
        self._change_map_local_size(1.0)

    def _change_map_local_size(self, delta: float) -> None:
        if (
            self.action
            or not self.map_visible
            or self.map_crop_size_m is None
        ):
            return
        now = time.monotonic()
        if now - self.last_map_size_change_at < 0.25:
            return
        self.last_map_size_change_at = now
        new_size = min(
            MAP_LOCAL_MAX_METERS,
            max(MAP_LOCAL_MIN_METERS, self.map_local_size_m + delta),
        )
        if new_size == self.map_local_size_m:
            return
        self.map_local_size_m = new_size
        self.map_crop_size_m = new_size
        self._update_map_view_controls()
        self.map_status.set_text(
            f"Loading {new_size:g}×{new_size:g} m local map…"
        )
        self._request_map_if_due(force=True)

    def _update_map_view_controls(self) -> None:
        self.map_view_button.set_label(
            "LOCAL" if self.map_crop_size_m is None else "FULL"
        )
        self.map_size_label.set_text(
            f"{self.map_local_size_m:g}×{self.map_local_size_m:g}"
        )
        if self.map_crop_size_m is None:
            self.map_size_controls.hide()
        else:
            # This subtree is excluded from the window's startup show_all(),
            # so reveal both the container and its children on LOCAL entry.
            self.map_size_down_button.show()
            self.map_size_label.show()
            self.map_size_up_button.show()
            self.map_size_controls.show()
        available = self.last_state == "running" and not self.action
        self.map_size_down_button.set_sensitive(
            available and self.map_local_size_m > MAP_LOCAL_MIN_METERS
        )
        self.map_size_up_button.set_sensitive(
            available and self.map_local_size_m < MAP_LOCAL_MAX_METERS
        )

    def _reset_button_clicked(self, _button: Gtk.Button) -> None:
        if (
            self.action
            or self.reset_pending
            or not self.debug_overlay
            or self.last_state != "running"
        ):
            return
        self.reset_pending = True
        self._set_map_reset_overlay("resetting_manual")
        self.map_reset_button.set_label("RESETTING…")
        self.map_reset_button.set_sensitive(False)
        self.compact_reset_button.set_label("RESETTING…")
        self.compact_reset_button.set_sensitive(False)
        if self.map_visible:
            self.map_status.set_text("Requesting robot reset…")
        threading.Thread(target=self._request_reset, daemon=True).start()

    def _request_reset(self) -> None:
        if zmq is None:
            self.reset_results.put((False, "python3-zmq is not installed"))
            return
        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.connect(RESET_ENDPOINT)
            request_id = uuid.uuid4().hex
            socket.send_json({
                "schema_version": 1,
                "operation": "reset",
                "request_id": request_id,
                "source": "face_display",
            })
            if not socket.poll(15000, zmq.POLLIN):
                raise TimeoutError("Reset service did not reply within 15 seconds")
            response = socket.recv_json()
            if not isinstance(response, dict) or not response.get("ok"):
                error = (
                    response.get("error")
                    if isinstance(response, dict) else None
                )
                raise RuntimeError(error or "Reset service rejected the request")
            if response.get("request_id") != request_id:
                raise RuntimeError("Reset service returned a mismatched reply")
            self.reset_results.put((True, "Reset requested"))
        except Exception as error:
            self.reset_results.put((False, str(error)))
        finally:
            socket.close(linger=0)

    def _poll_reset(self) -> bool:
        try:
            ok, message = self.reset_results.get_nowait()
        except queue.Empty:
            return True
        self.reset_pending = False
        self.map_reset_button.set_label("RESET")
        self.compact_reset_button.set_label("RESET")
        reset_available = self.last_state == "running" and not self.action
        self.map_reset_button.set_sensitive(reset_available)
        self.compact_reset_button.set_sensitive(reset_available)
        if self.map_visible:
            self.map_status.set_text(
                "Robot reset requested; refreshing map…"
                if ok else f"Reset failed: {message}"
            )
        if ok:
            self.map_requested_at = -float("inf")
            self._set_map_reset_overlay("ready", show_initial_ready=True)
        elif self.map_reset_state == "resetting_manual":
            self._set_map_reset_overlay(None)
        return True

    def _set_map_reset_overlay(
        self, state, *, show_initial_ready=False
    ) -> None:
        state = str(state or "").lower() or None
        previous = self.map_reset_state
        self.map_reset_state = state
        messages = {
            "initializing": "INITIALIZING\nKEEP ROBOT STILL",
            "lifted": "ROBOT LIFTED\nPLACE ROBOT DOWN",
            "resetting_lift": "RESETTING MAP\nPLEASE WAIT",
            "resetting_manual": "RESETTING ROBOT\nPLEASE WAIT",
            "ready": "READY\nROBOT CAN MOVE",
        }
        context = self.map_reset_overlay.get_style_context()
        context.remove_class("reset-danger")
        context.remove_class("reset-ready")
        if state == "ready" and previous == "ready":
            return
        if state == "ready" and not (
            show_initial_ready
            or previous in {
                "initializing", "lifted", "resetting_lift", "resetting_manual",
            }
        ):
            self.map_reset_overlay.hide()
            return
        if state not in messages:
            self.map_reset_overlay.hide()
            return
        if state == "lifted":
            context.add_class("reset-danger")
        elif state == "ready":
            context.add_class("reset-ready")
        self.map_reset_overlay.set_text(messages[state])
        self.map_reset_overlay.show()
        self.map_ready_generation += 1
        generation = self.map_ready_generation
        if state == "ready":
            GLib.timeout_add(
                3000, self._hide_ready_overlay, generation
            )

    def _hide_ready_overlay(self, generation: int) -> bool:
        if (
            generation == self.map_ready_generation
            and self.map_reset_state == "ready"
        ):
            self.map_reset_overlay.hide()
        return False

    def _request_map_if_due(self, *, force: bool = False) -> None:
        if not self.map_visible or self.map_request_pending:
            return
        now = time.monotonic()
        last_request = getattr(self, "map_requested_at", -float("inf"))
        if not force and now - last_request < MAP_REFRESH_SECONDS:
            return
        self.map_requested_at = now
        self.map_request_pending = True
        crop_size_m = self.map_crop_size_m
        threading.Thread(
            target=self._request_map,
            args=(self.map_layer, crop_size_m),
            daemon=True,
        ).start()

    def _request_map(self, layer: str, crop_size_m: float | None) -> None:
        started_at = time.monotonic()
        if zmq is None:
            self.map_results.put((
                False,
                layer,
                crop_size_m,
                "python3-zmq is not installed",
                0.0,
            ))
            return
        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.connect(MAP_ENDPOINT)
            request = {
                "schema_version": 1,
                "operation": "snapshot",
                "request_id": uuid.uuid4().hex,
                "render_frontiers": False,
                "rotate_180": True,
            }
            if layer != "combined":
                request["display_layer"] = layer
            if crop_size_m is not None:
                request["crop_size_m"] = crop_size_m
            socket.send_json(request)
            if not socket.poll(3000, zmq.POLLIN):
                raise TimeoutError("Map service did not reply within 3 seconds")
            parts = socket.recv_multipart()
            if len(parts) == 1:
                response = json.loads(parts[0].decode("utf-8"))
                raise RuntimeError(response.get("error") or "Map is not ready")
            if len(parts) != 2 or not parts[1].startswith(bytes.fromhex("ffd8")):
                raise RuntimeError("Map service returned an invalid snapshot")
            metadata = json.loads(parts[0].decode("utf-8"))
            self.map_results.put((
                True,
                layer,
                crop_size_m,
                (bytes(parts[1]), metadata),
                (time.monotonic() - started_at) * 1000.0,
            ))
        except Exception as error:
            self.map_results.put((
                False,
                layer,
                crop_size_m,
                str(error),
                (time.monotonic() - started_at) * 1000.0,
            ))
        finally:
            socket.close(linger=0)

    def _poll_map(self) -> bool:
        try:
            ok, layer, crop_size_m, payload, request_ms = (
                self.map_results.get_nowait()
            )
        except queue.Empty:
            self._request_map_if_due()
            return True
        self.map_request_pending = False
        if (
            layer != self.map_layer
            or crop_size_m != self.map_crop_size_m
        ):
            self._request_map_if_due(force=True)
            return True
        if not ok:
            self.map_status.set_text(str(payload))
            return True
        jpeg, metadata = payload
        try:
            loader = GdkPixbuf.PixbufLoader.new_with_type("jpeg")
            loader.write(jpeg)
            loader.close()
            self.map_pixbuf = loader.get_pixbuf()
            self._scale_map_image()
            if not self.reset_pending:
                self._set_map_reset_overlay(metadata.get("reset_state"))
            age = float(metadata.get("source_age_ms", 0.0)) / 1000.0
            crop_size_m = metadata.get("map_crop_size_m")
            view_name = (
                f"{float(crop_size_m):g}×{float(crop_size_m):g} M LOCAL MAP"
                if crop_size_m is not None else "FULL OVERVIEW"
            )
            compute_ms = duration_milliseconds(metadata.get("analysis_ms"))
            render_ms = duration_milliseconds(metadata.get("render_ms"))
            total_map_ms = duration_milliseconds(request_ms)
            cpu_usage = self.map_cpu_usage.sample()
            detail = (
                f"{view_name}  •  AGE {age:.1f}s\n"
                f"COMPUTE {compute_ms:.0f} ms  •  "
                f"RENDER {render_ms:.0f} ms\n"
                f"TOTAL MAP {total_map_ms:.0f} ms  •  CPU {cpu_usage:.0f}%"
            )
            self.map_status.set_text(detail)
        except Exception as error:
            self.map_status.set_text(f"Cannot display map: {error}")
        self._request_map_if_due()
        return True

    def _map_image_resized(self, _image, _allocation) -> None:
        self._scale_map_image()

    def _scale_map_image(self) -> None:
        if self.map_pixbuf is None:
            return
        allocation = self.map_viewport.get_allocation()
        source_width = self.map_pixbuf.get_width()
        source_height = self.map_pixbuf.get_height()
        if allocation.width <= 1 or allocation.height <= 1:
            return
        scale = min(
            allocation.width / source_width,
            allocation.height / source_height,
        )
        width = max(1, int(source_width * scale))
        height = max(1, int(source_height * scale))
        scaled = self.map_pixbuf.scale_simple(
            width, height, GdkPixbuf.InterpType.BILINEAR
        )
        self.map_image.set_from_pixbuf(scaled)

    def _button_clicked(self, _button: Gtk.Button) -> None:
        if self.action:
            return
        if self.last_state in {"starting", "running", "degraded"}:
            self._stop()
        else:
            self._start()

    def _wifi_button_clicked(self, _button: Gtk.Button) -> None:
        if self.action or self.wifi_action:
            return
        self.wifi_action = True
        self.wifi_button.set_sensitive(False)
        self.window.set_keep_above(False)
        self.window.hide()
        threading.Thread(target=self._run_wifi_settings, daemon=True).start()

    def _run_wifi_settings(self) -> None:
        try:
            subprocess.run(
                ["gnome-control-center", "wifi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        finally:
            GLib.idle_add(self._wifi_settings_closed)

    def _wifi_settings_closed(self) -> bool:
        self.wifi_action = False
        self.wifi_checked_at = -float("inf")
        self._refresh_wifi()
        self.window.show_all()
        self.window.set_keep_above(True)
        if self.fullscreen:
            self.window.fullscreen()
        self.window.present()
        self.wifi_button.set_sensitive(
            self.last_state in {"stopped", "failed"}
        )
        return False

    def _refresh_wifi(self) -> None:
        now = time.monotonic()
        if now - self.wifi_checked_at < 2.0:
            return
        self.wifi_checked_at = now
        self.wifi_name = active_wifi_name()
        self.wifi_status.set_text(
            f"Wi-Fi: {self.wifi_name}"
            if self.wifi_name else "Wi-Fi: disconnected"
        )

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
        self._refresh_wifi()
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
            "stopped": ("●", "Robot is stopped", "Ready to get rolling, bro?"),
            "starting": ("…", "Starting robot", "Things gon be just fireeee!"),
            "running": ("✓", "Robot is running", "Woohoo I am alive!"),
            "degraded": ("!", "Robot is partially running", "I am not feeling well, brother..."),
            "stopping": ("…", "Stopping robot", "Ok fine."),
            "failed": ("!", "Robot needs attention", "Something went wrong, HELP!"),
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
        self.compact_map_button.set_sensitive(state == "running" and not self.action)
        self.compact_reset_button.set_sensitive(
            state == "running" and not self.action and not self.reset_pending
        )
        self.map_stop_button.set_sensitive(
            state in {"starting", "running", "degraded"} and not self.action
        )
        self.map_view_button.set_sensitive(state == "running" and not self.action)
        self._update_map_view_controls()
        self.map_reset_button.set_sensitive(
            state == "running" and not self.action and not self.reset_pending
        )
        self.wifi_button.set_sensitive(
            state in {"stopped", "failed"}
            and not self.action
            and not self.wifi_action
        )
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
