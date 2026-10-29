"""Write a compact, event-oriented micro-ROS session log.

The recorder deliberately does not serialize sensor payloads.  It records the
STM32's own health strings, stream liveness transitions, ROS endpoint presence,
the ST-Link serial-device state, and relevant kernel USB events.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import threading
import time
from typing import Any

import rclpy
from geometry_msgs.msg import Point32
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Int16MultiArray, String


SERIAL_LINK = (
    "/dev/serial/by-id/"
    "usb-STMicroelectronics_STM32_STLink_066BFF485270535067113035-if02"
)
STREAM_SPECS = {
    "/stm32/imu_accel": (Point32, 2.0),
    "/stm32/imu_gyro": (Point32, 2.0),
    "/stm32/range_msg": (Point32, 2.0),
    "/stm32/tof_raw_data": (Int16MultiArray, 2.0),
    "/stm32/wheel_states": (JointState, 2.0),
    "/stm32/debug_msg": (String, 4.5),
}
KERNEL_PATTERN = re.compile(
    r"(?:st-?link|ttyacm|cdc_acm|"
    r"usb .*?(?:disconnect|reset|error|fail|descriptor|new .*speed usb))",
    re.IGNORECASE,
)


def stream_state(
    publisher_count: int,
    age: float | None,
    stale_after: float,
    in_startup_grace: bool,
) -> str:
    """Classify one stream without logging its payload."""
    if publisher_count == 0:
        return "starting" if in_startup_grace else "no_publisher"
    if age is None:
        return "starting" if in_startup_grace else "no_messages"
    if age > stale_after:
        return "stale"
    return "live"


def kernel_line_is_relevant(line: str) -> bool:
    """Keep USB transport failures while excluding unrelated kernel noise."""
    return KERNEL_PATTERN.search(line) is not None


def default_output_directory() -> Path:
    """Prefer the workspace diagnostics directory when launched from it."""
    configured = os.environ.get("DJ_MICROROS_LOG_DIR")
    if configured:
        return Path(configured).expanduser()

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "MICRO_ROS").is_dir() and (candidate / "BIG_BRAIN").is_dir():
            return candidate / "diagnostics" / "microros-sessions"

    state_home = Path(
        os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
    )
    return state_home / "dj-robot" / "microros-sessions"


def matching_process_ids(required_fragments: tuple[bytes, ...]) -> list[int]:
    """Return PIDs whose command lines contain every requested fragment."""
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command_line = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if all(fragment in command_line for fragment in required_fragments):
            matches.append(int(entry.name))
    return sorted(matches)


class MicroRosRecorder(Node):
    """Record low-rate evidence needed to diagnose micro-ROS outages."""

    def __init__(self) -> None:
        super().__init__("microros_recorder")
        self.declare_parameter("output_directory", str(default_output_directory()))
        self.declare_parameter("serial_link", SERIAL_LINK)
        self.declare_parameter("startup_grace", 10.0)
        self.declare_parameter("heartbeat_period", 60.0)

        output_directory = Path(
            str(self.get_parameter("output_directory").value)
        ).expanduser()
        output_directory.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now().astimezone()
        filename = started_at.strftime("%Y%m%d-%H%M%S-microros.jsonl")
        self.log_path = output_directory / filename
        self._log = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._log_lock = threading.Lock()
        self._started_monotonic = time.monotonic()
        self._last_heartbeat = self._started_monotonic
        self._last_received: dict[str, float] = {}
        self._message_counts: dict[str, int] = {}
        self._states: dict[str, str] = {}
        self._stream_subscriptions = []
        self._journal_process: subprocess.Popen[str] | None = None
        self._journal_thread: threading.Thread | None = None
        self._closing = False

        for topic, (message_type, _stale_after) in STREAM_SPECS.items():
            callback = self._debug_callback if topic == "/stm32/debug_msg" else (
                lambda _message, stream=topic: self._record_message(stream)
            )
            self._stream_subscriptions.append(
                self.create_subscription(
                    message_type,
                    topic,
                    callback,
                    qos_profile_sensor_data,
                )
            )

        self._write_event(
            "session_start",
            pid=os.getpid(),
            kernel=platform.release(),
            serial_link=str(self.get_parameter("serial_link").value),
        )
        self._start_kernel_monitor()
        self.create_timer(1.0, self._sample_state)
        self.get_logger().info(f"micro-ROS recording to {self.log_path}")

    def _write_event(self, event: str, **fields: Any) -> None:
        if self._log.closed:
            return
        record = {
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "up_s": round(time.monotonic() - self._started_monotonic, 3),
            "event": event,
            **fields,
        }
        encoded = json.dumps(record, separators=(",", ":"), sort_keys=True)
        with self._log_lock:
            self._log.write(encoded + "\n")

    def _record_message(self, topic: str) -> None:
        self._last_received[topic] = time.monotonic()
        self._message_counts[topic] = self._message_counts.get(topic, 0) + 1

    def _debug_callback(self, message: String) -> None:
        self._record_message("/stm32/debug_msg")
        self._write_event("stm32_debug", message=message.data)

    def _start_kernel_monitor(self) -> None:
        command = [
            "journalctl",
            "-k",
            "--follow",
            "--no-pager",
            "--output=short-iso-precise",
            "--since=now",
        ]
        try:
            self._journal_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            self._write_event("kernel_monitor_error", error=str(error))
            return

        self._journal_thread = threading.Thread(
            target=self._read_kernel_log,
            name="microros-kernel-monitor",
            daemon=True,
        )
        self._journal_thread.start()

    def _read_kernel_log(self) -> None:
        process = self._journal_process
        if process is None or process.stdout is None:
            return
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if kernel_line_is_relevant(line):
                self._write_event("kernel", message=line)
            elif "journalctl:" in line.lower():
                self._write_event("kernel_monitor_error", error=line)
            if self._closing:
                break
        if not self._closing:
            self._write_event(
                "kernel_monitor_exit",
                return_code=process.poll(),
            )

    def _transition(self, key: str, value: str, **fields: Any) -> None:
        previous = self._states.get(key)
        if previous == value:
            return
        self._states[key] = value
        self._write_event(
            "state",
            key=key,
            previous=previous,
            value=value,
            **fields,
        )

    def _sample_state(self) -> None:
        now = time.monotonic()
        startup_grace = float(self.get_parameter("startup_grace").value)
        in_startup_grace = now - self._started_monotonic < startup_grace

        for topic, (_message_type, stale_after) in STREAM_SPECS.items():
            received = self._last_received.get(topic)
            age = None if received is None else max(0.0, now - received)
            publishers = self.count_publishers(topic)
            state = stream_state(publishers, age, stale_after, in_startup_grace)
            self._transition(
                f"stream:{topic}",
                state,
                age_s=None if age is None else round(age, 3),
                publishers=publishers,
            )

        serial_link = Path(str(self.get_parameter("serial_link").value))
        serial_present = serial_link.exists()
        serial_target = str(serial_link.resolve()) if serial_present else None
        self._transition(
            "device:stlink_serial",
            "present" if serial_present else "missing",
            target=serial_target,
        )

        available_nodes = {
            f"{namespace.rstrip('/')}/{name}" if namespace.rstrip("/") else f"/{name}"
            for name, namespace in self.get_node_names_and_namespaces()
        }
        for node_name in ("/cubemx_node",):
            self._transition(
                f"node:{node_name}",
                "present" if node_name in available_nodes else "missing",
            )

        agent_pids = matching_process_ids((b"micro_ros_agent", b"serial"))
        agent_state = "missing" if not agent_pids else (
            "present:" + ",".join(str(pid) for pid in agent_pids)
        )
        self._transition("process:micro_ros_agent", agent_state)

        heartbeat_period = float(self.get_parameter("heartbeat_period").value)
        if now - self._last_heartbeat >= heartbeat_period:
            self._last_heartbeat = now
            self._write_event(
                "heartbeat",
                states=self._states,
                message_counts=self._message_counts,
            )

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._write_event(
            "session_end",
            states=self._states,
            message_counts=self._message_counts,
        )
        process = self._journal_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        if self._journal_thread is not None:
            self._journal_thread.join(timeout=2.0)
        with self._log_lock:
            self._log.close()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MicroRosRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
