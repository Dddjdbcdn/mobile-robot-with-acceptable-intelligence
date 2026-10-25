"""Pan/tilt camera servo control and an interactive hardware test."""

import argparse
import math
import select
import sys
import termios
import threading
import time
import tty

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32


class CameraServo:
    def __init__(self, pan_pub, tilt_pub, logger=None):
        self.servo_pan_pub = pan_pub
        self.servo_tilt_pub = tilt_pub
        self.logger = logger
        self.reset_pan_angle = 95.0
        self.reset_tilt_angle = 90.0
        self.pan_angle = self.reset_pan_angle
        self.tilt_angle = self.reset_tilt_angle
        self.min_pan_angle = 30.0
        self.max_pan_angle = 160.0
        self.min_tilt_angle = 30.0
        self.max_tilt_angle = 120.0

        self.delta_pan_angle = 0.0
        self.delta_tilt_angle = 0.0
        self.tracking_sequence = None
        self.tracking_timing = None
        self.tracking_received_at = None
        self.tracking_received_unix_ns = None
        self.last_visual_pan_error = 0.0
        self.last_visual_at = None
        self.last_applied_tracking_sequence = None
        self.command_lock = threading.Lock()
        self._latency_window_started = time.monotonic()
        self._latency_samples = []
        self.Kp = 0.30
        self.deadband_degrees = 0.5
        self.max_step_degrees = 4.0

    def set_error(self, pan, tilt, tracking_sequence=None, tracking_timing=None):
        with self.command_lock:
            self.delta_pan_angle = float(pan)
            self.delta_tilt_angle = float(tilt)
            self.tracking_sequence = tracking_sequence
            self.tracking_timing = tracking_timing
            self.tracking_received_at = time.monotonic()
            self.tracking_received_unix_ns = time.time_ns()
            if tracking_sequence is not None:
                self.last_visual_pan_error = float(pan)
                self.last_visual_at = self.tracking_received_at

    def set_angles(self, pan_angle, tilt_angle):
        """Publish one absolute, limit-clamped pan/tilt command."""
        self.pan_angle = max(
            self.min_pan_angle, min(self.max_pan_angle, float(pan_angle))
        )
        self.tilt_angle = max(
            self.min_tilt_angle, min(self.max_tilt_angle, float(tilt_angle))
        )
        self._publish_angles()

    def handle_teleop_key(self, key, step_degrees=5.0):
        """Apply an I/J/K/L/comma teleop key and report whether it was handled."""
        key = key.lower()
        if key == "i":
            self.set_angles(self.pan_angle, self.tilt_angle - step_degrees)
        elif key == "j":
            self.set_angles(self.pan_angle - step_degrees, self.tilt_angle)
        elif key == "k":
            self.set_angles(self.reset_pan_angle, self.reset_tilt_angle)
        elif key == "l":
            self.set_angles(self.pan_angle + step_degrees, self.tilt_angle)
        elif key == ",":
            self.set_angles(self.pan_angle, self.tilt_angle + step_degrees)
        else:
            return False
        return True

    def tracking_snapshot(self):
        with self.command_lock:
            return {
                "pan_angle": self.pan_angle,
                "pan_error": self.last_visual_pan_error,
                "observed_at": self.last_visual_at,
            }

    def clear_tracking_observation(self):
        """Require a new visual observation before tracking can resume."""
        with self.command_lock:
            self.delta_pan_angle = 0.0
            self.delta_tilt_angle = 0.0
            self.tracking_sequence = None
            self.tracking_timing = None
            self.tracking_received_at = None
            self.tracking_received_unix_ns = None
            self.last_visual_pan_error = 0.0
            self.last_visual_at = None
            self.last_applied_tracking_sequence = None

    @staticmethod
    def _latency_percentile(values, fraction):
        ordered = sorted(values)
        index = max(0, math.ceil(len(ordered) * fraction) - 1)
        return ordered[index]

    def _record_tracking_latency(self, timing, received_at, received_unix_ns):
        if not isinstance(timing, dict) or received_at is None:
            return

        capture_to_send_ms = timing.get("capture_to_send_ms")
        sent_at_unix_ns = timing.get("sent_at_unix_ns")
        if not isinstance(capture_to_send_ms, (int, float)):
            return

        published_at = time.monotonic()
        queue_ms = max(0.0, (published_at - received_at) * 1000.0)
        transport_ms = 0.0
        if isinstance(sent_at_unix_ns, int) and received_unix_ns is not None:
            transport_ms = max(
                0.0, (received_unix_ns - sent_at_unix_ns) / 1_000_000.0
            )

        self._latency_samples.append({
            "total": float(capture_to_send_ms) + transport_ms + queue_ms,
            "capture_wait": float(timing.get("capture_wait_ms") or 0.0),
            "inference": float(timing.get("inference_ms") or 0.0),
            "post_inference": float(timing.get("post_inference_ms") or 0.0),
            "transport": transport_ms,
            "queue": queue_ms,
        })

        now = time.monotonic()
        if now - self._latency_window_started < 1.0:
            return

        samples = self._latency_samples
        self._latency_samples = []
        self._latency_window_started = now
        if not samples or self.logger is None:
            return

        def average(name):
            return sum(sample[name] for sample in samples) / len(samples)

        totals = [sample["total"] for sample in samples]
        self.logger.info(
            "[TRACK LATENCY] capture->ROS-servo-publish "
            f"avg={average('total'):.1f}ms "
            f"p95={self._latency_percentile(totals, 0.95):.1f}ms "
            f"max={max(totals):.1f}ms; stage averages: "
            f"camera-wait={average('capture_wait'):.1f}ms, "
            f"YOLO={average('inference'):.1f}ms, "
            f"tracking={average('post_inference'):.1f}ms, "
            f"ZMQ={average('transport'):.1f}ms, "
            f"timer-queue={average('queue'):.1f}ms (n={len(samples)})"
        )

    def _publish_angles(self):
        pan_msg = Float32()
        tilt_msg = Float32()
        pan_msg.data = self.pan_angle
        tilt_msg.data = self.tilt_angle
        self.servo_tilt_pub.publish(tilt_msg)
        self.servo_pan_pub.publish(pan_msg)

    def publish_servo_command(self, tracking=False):
        remaining_pan_angle = 0.0

        with self.command_lock:
            if (
                tracking
                and self.tracking_sequence is not None
                and self.tracking_sequence == self.last_applied_tracking_sequence
            ):
                self.delta_pan_angle = 0.0
                self.delta_tilt_angle = 0.0
                return remaining_pan_angle

            delta_pan_angle = self.delta_pan_angle
            delta_tilt_angle = self.delta_tilt_angle
            if tracking and self.tracking_sequence is not None:
                self.last_applied_tracking_sequence = self.tracking_sequence
            self.delta_pan_angle = 0.0
            self.delta_tilt_angle = 0.0
            self.tracking_timing = None
            self.tracking_received_at = None
            self.tracking_received_unix_ns = None

        if abs(delta_pan_angle) < self.deadband_degrees:
            delta_pan_angle = 0.0
        if abs(delta_tilt_angle) < self.deadband_degrees:
            delta_tilt_angle = 0.0

        step_pan = delta_pan_angle
        step_tilt = delta_tilt_angle
        if tracking:
            step_pan = max(
                min(delta_pan_angle * self.Kp, self.max_step_degrees),
                -self.max_step_degrees,
            )
            step_tilt = max(
                min(delta_tilt_angle * self.Kp, self.max_step_degrees),
                -self.max_step_degrees,
            )

        requested_pan = self.pan_angle + step_pan
        self.pan_angle = max(
            self.min_pan_angle, min(self.max_pan_angle, requested_pan)
        )
        remaining_pan_angle = requested_pan - self.pan_angle
        self.tilt_angle = max(
            self.min_tilt_angle,
            min(self.max_tilt_angle, self.tilt_angle + step_tilt),
        )
        self._publish_angles()
        return remaining_pan_angle


def test_servo(pan_angle=95.0, tilt_angle=90.0):
    """Publish a one-shot camera pose for a quick hardware check."""
    initialized_here = not rclpy.ok()
    if initialized_here:
        rclpy.init(args=[])
    node = Node("camera_servo_test")
    try:
        pan_pub = node.create_publisher(Float32, "/stm32/servo_pan", 10)
        tilt_pub = node.create_publisher(Float32, "/stm32/servo_tilt", 10)
        servo = CameraServo(pan_pub, tilt_pub, node.get_logger())
        rclpy.spin_once(node, timeout_sec=0.5)
        servo.set_angles(pan_angle, tilt_angle)
        rclpy.spin_once(node, timeout_sec=0.2)
        return servo.pan_angle, servo.tilt_angle
    finally:
        node.destroy_node()
        if initialized_here:
            rclpy.shutdown()


def teleop_servo(pan_angle=95.0, tilt_angle=90.0, step_degrees=5.0):
    """Control the camera servos from a terminal until Q or Ctrl-C."""
    if not sys.stdin.isatty():
        raise RuntimeError("servo teleop requires an interactive terminal")
    if step_degrees <= 0.0:
        raise ValueError("step_degrees must be greater than zero")

    initialized_here = not rclpy.ok()
    if initialized_here:
        rclpy.init(args=[])
    node = Node("camera_servo_teleop")
    original_terminal = termios.tcgetattr(sys.stdin)
    try:
        pan_pub = node.create_publisher(Float32, "/stm32/servo_pan", 10)
        tilt_pub = node.create_publisher(Float32, "/stm32/servo_tilt", 10)
        servo = CameraServo(pan_pub, tilt_pub, node.get_logger())
        rclpy.spin_once(node, timeout_sec=0.5)
        servo.set_angles(pan_angle, tilt_angle)

        print(
            "Camera servo teleop: I=tilt up, J=pan left, "
            "K=reset, L=pan right, ,=tilt down, Q=quit"
        )
        print(f"Step: {step_degrees:g} degrees")
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            key = sys.stdin.read(1)
            if key.lower() == "q" or key == "\x03":
                break
            if servo.handle_teleop_key(key, step_degrees):
                print(
                    f"\rpan={servo.pan_angle:5.1f}  "
                    f"tilt={servo.tilt_angle:5.1f}   ",
                    end="",
                    flush=True,
                )
        print()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal)
        node.destroy_node()
        if initialized_here:
            rclpy.shutdown()


def main(args=None):
    parser = argparse.ArgumentParser(description="Teleoperate the camera servos")
    parser.add_argument("--pan", type=float, default=95.0)
    parser.add_argument("--tilt", type=float, default=90.0)
    parser.add_argument(
        "--step",
        type=float,
        default=5.0,
        help="degrees moved by each key press (default: 5)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="publish --pan/--tilt once instead of starting keyboard teleop",
    )
    parsed, _ = parser.parse_known_args(args)
    if parsed.once:
        pan, tilt = test_servo(parsed.pan, parsed.tilt)
        print(f"Published camera servo pose: pan={pan:.1f}, tilt={tilt:.1f}")
        return
    try:
        teleop_servo(parsed.pan, parsed.tilt, parsed.step)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
