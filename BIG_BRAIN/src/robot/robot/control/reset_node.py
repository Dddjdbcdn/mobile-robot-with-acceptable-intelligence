"""Stop a lifted robot and start a fresh spatial session after placement."""

from __future__ import annotations

import math
import os
import queue
import threading
import time

from geometry_msgs.msg import TwistStamped
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid
import rclpy
import zmq
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2
from slam_toolbox.srv import Reset
from std_msgs.msg import Empty, String


BASELINE_SAMPLES = 25
LIFT_RATE_RAD_S = 0.7
LIFT_ROTATION_RAD = 0.18
MINIMUM_LIFT_SECONDS = 1.0
STABLE_SECONDS = 1.5
STABLE_GYRO_RAD_S = 0.12
GRAVITY_M_S2 = 9.80665
GRAVITY_TOLERANCE_M_S2 = 1.5
PLACEMENT_COSINE = math.cos(math.radians(10.0))
CLIFF_FRESH_SECONDS = 0.5
RESET_ENDPOINT = os.environ.get("RESET_ENDPOINT", "tcp://127.0.0.1:5561")
MANUAL_RESET_TIMEOUT_SECONDS = 12.0


class LiftDetector:
    """Detect a side lift followed by stable upright placement."""

    def __init__(self) -> None:
        self.state = "calibrating"
        self.baseline_sum = [0.0, 0.0, 0.0]
        self.baseline_count = 0
        self.gravity_direction: tuple[float, float, float] | None = None
        self.last_time: float | None = None
        self.lift_rotation = 0.0
        self.lifted_at: float | None = None
        self.stable_since: float | None = None

    @staticmethod
    def norm(values: tuple[float, float, float]) -> float:
        return math.sqrt(sum(value * value for value in values))

    def observe(
        self,
        now: float,
        acceleration: tuple[float, float, float],
        angular_velocity: tuple[float, float, float],
        placement_clear: bool = True,
    ) -> str | None:
        values = (*acceleration, *angular_velocity, now)
        if not all(math.isfinite(value) for value in values):
            return None

        dt = 0.0 if self.last_time is None else max(
            0.0, min(0.1, now - self.last_time)
        )
        self.last_time = now
        acceleration_norm = self.norm(acceleration)
        gyro_norm = self.norm(angular_velocity)

        if self.state == "calibrating":
            stable = (
                gyro_norm <= STABLE_GYRO_RAD_S
                and acceleration_norm > 1e-6
                and abs(acceleration_norm - GRAVITY_M_S2)
                <= GRAVITY_TOLERANCE_M_S2
            )
            if not stable:
                self.baseline_sum = [0.0, 0.0, 0.0]
                self.baseline_count = 0
                return None
            for index, value in enumerate(acceleration):
                self.baseline_sum[index] += value / acceleration_norm
            self.baseline_count += 1
            if self.baseline_count >= BASELINE_SAMPLES:
                baseline_norm = self.norm(tuple(self.baseline_sum))
                self.gravity_direction = tuple(
                    value / baseline_norm for value in self.baseline_sum
                )
                self.state = "normal"
            return None

        if self.state == "normal":
            roll_rate = abs(angular_velocity[0])
            if roll_rate >= LIFT_RATE_RAD_S:
                self.lift_rotation += roll_rate * dt
            else:
                self.lift_rotation *= 0.5
            if self.lift_rotation >= LIFT_ROTATION_RAD:
                self.state = "lifted"
                self.lifted_at = now
                self.stable_since = None
                return "lifted"
            return None

        if self.state != "lifted" or self.gravity_direction is None:
            return None
        if self.lifted_at is None or now - self.lifted_at < MINIMUM_LIFT_SECONDS:
            return None

        direction = (
            tuple(value / acceleration_norm for value in acceleration)
            if acceleration_norm > 1e-6
            else (0.0, 0.0, 0.0)
        )
        alignment = sum(
            value * baseline
            for value, baseline in zip(
                direction, self.gravity_direction, strict=True
            )
        )
        stable = (
            gyro_norm <= STABLE_GYRO_RAD_S
            and abs(acceleration_norm - GRAVITY_M_S2)
            <= GRAVITY_TOLERANCE_M_S2
            and alignment >= PLACEMENT_COSINE
            and placement_clear
        )
        if not stable:
            self.stable_since = None
            return None
        if self.stable_since is None:
            self.stable_since = now
            return None
        if now - self.stable_since >= STABLE_SECONDS:
            self.state = "resetting"
            return "placed"
        return None

    def reset_complete(self) -> None:
        self.state = "normal"
        self.lift_rotation = 0.0
        self.lifted_at = None
        self.stable_since = None


class ResetNode(Node):
    """Coordinate stopping and a fresh spatial session after placement."""

    def __init__(self) -> None:
        super().__init__("reset_node")
        self.detector = LiftDetector()
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, "/imu", self._imu_callback, qos)
        self.create_subscription(
            PointCloud2, "/cliff_layer", self._cliff_callback, qos
        )
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            OccupancyGrid, "/map", self._map_callback, map_qos
        )
        self.stop_publisher = self.create_publisher(
            TwistStamped, "/diff_drive_controller/cmd_vel", 10
        )
        self.reset_publisher = self.create_publisher(
            Empty, "/big_brain/reset", 10
        )
        self.reset_reason_publisher = self.create_publisher(
            String, "/big_brain/reset_reason", 10
        )
        self.state_publisher = self.create_publisher(
            String,
            "/lift_detector/state",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.reset_state_publisher = self.create_publisher(
            String,
            "/reset/state",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.slam = self.create_client(Reset, "/slam_toolbox/reset")
        self.local_costmap = self.create_client(
            ClearEntireCostmap,
            "/local_costmap/clear_entirely_local_costmap",
        )
        self.global_costmap = self.create_client(
            ClearEntireCostmap,
            "/global_costmap/clear_entirely_global_costmap",
        )
        self._reset_futures = None
        self._reset_mode: str | None = None
        self._manual_reset = None
        self._manual_requests: queue.Queue = queue.Queue()
        self._retry_after = 0.0
        self._last_state = None
        self._last_reset_state = None
        self._last_cliff_at: float | None = None
        self._cliff_point_count = 0
        self._map_revision = 0
        self._map_wait_revision: int | None = None

        self.create_timer(0.1, self._control_tick)
        self._zmq_stop = threading.Event()
        self._zmq_context = zmq.Context()
        self._zmq_thread = threading.Thread(
            target=self._serve_reset_requests,
            name="lift-reset-zmq",
            daemon=True,
        )
        self._zmq_thread.start()
        self._publish_state()
        self._publish_reset_state()
        self.get_logger().info(
            f"Listening for manual reset requests on {RESET_ENDPOINT}"
        )

    def _imu_callback(self, message: Imu) -> None:
        now = time.monotonic()
        event = self.detector.observe(
            now,
            (
                float(message.linear_acceleration.x),
                float(message.linear_acceleration.y),
                float(message.linear_acceleration.z),
            ),
            (
                float(message.angular_velocity.x),
                float(message.angular_velocity.y),
                float(message.angular_velocity.z),
            ),
            placement_clear=self._placement_clear(now),
        )
        if event == "lifted":
            self.get_logger().warning("Lift detected: canceling motion")
            self._publish_state()
            self._publish_reset_state()
            self.stop_movement("ROBOT_LIFTED")
        elif event == "placed":
            self.get_logger().info(
                "Stable upright placement detected; requesting spatial reset"
            )
            self._publish_state()

    def _cliff_callback(self, message: PointCloud2) -> None:
        self._last_cliff_at = time.monotonic()
        self._cliff_point_count = message.width * message.height
        if self._cliff_point_count > 0:
            self.detector.stable_since = None

    def _map_callback(self, message: OccupancyGrid) -> None:
        cell_count = int(message.info.width) * int(message.info.height)
        if cell_count <= 0 or len(message.data) != cell_count:
            return
        self._map_revision += 1

    def _cliff_fresh(self, now: float) -> bool:
        return (
            self._last_cliff_at is not None
            and now - self._last_cliff_at <= CLIFF_FRESH_SECONDS
        )

    def _placement_clear(self, now: float) -> bool:
        return (
            not self._cliff_fresh(now)
            or self._cliff_point_count == 0
        )

    def _publish_stop(self) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_footprint"
        self.stop_publisher.publish(message)

    def stop_movement(self, reason: str) -> None:
        """Stop all motion and clear transient behavior for either reset path."""
        self._publish_stop()
        self.reset_publisher.publish(Empty())
        self.reset_reason_publisher.publish(String(data=reason))

    def reset_map(self) -> bool:
        """Start one SLAM and costmap reset; return false until services exist."""
        clients = (self.slam, self.local_costmap, self.global_costmap)
        if not all(client.service_is_ready() for client in clients):
            return False
        request = Reset.Request()
        request.pause_new_measurements = False
        self._reset_futures = (
            self.slam.call_async(request),
            self.local_costmap.call_async(ClearEntireCostmap.Request()),
            self.global_costmap.call_async(ClearEntireCostmap.Request()),
        )
        return True

    def _serve_reset_requests(self) -> None:
        socket = self._zmq_context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(RESET_ENDPOINT)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            while not self._zmq_stop.is_set():
                if socket not in dict(poller.poll(timeout=100)):
                    continue
                request = socket.recv_json()
                request_id = (
                    str(request.get("request_id") or "")
                    if isinstance(request, dict) else ""
                )
                if (
                    not isinstance(request, dict)
                    or request.get("operation") != "reset"
                ):
                    socket.send_json({
                        "ok": False,
                        "error": "Unsupported reset request",
                        "request_id": request_id,
                    })
                    continue
                completion: queue.Queue = queue.Queue(maxsize=1)
                self._manual_requests.put((request_id, completion))
                response_deadline = (
                    time.monotonic() + MANUAL_RESET_TIMEOUT_SECONDS + 1.0
                )
                while not self._zmq_stop.is_set():
                    try:
                        ok, error = completion.get(timeout=0.1)
                        break
                    except queue.Empty:
                        if time.monotonic() >= response_deadline:
                            ok, error = False, "Manual reset timed out"
                            break
                else:
                    break
                response = {
                    "ok": ok,
                    "operation": "reset",
                    "request_id": request_id,
                }
                if error:
                    response["error"] = error
                socket.send_json(response)
        except Exception as error:
            if not self._zmq_stop.is_set():
                self.get_logger().error(f"Reset ZMQ service failed: {error}")
        finally:
            socket.close(linger=0)

    def _reset_succeeded(self) -> bool | None:
        if self._reset_futures is None:
            return None
        if not all(future.done() for future in self._reset_futures):
            return None
        try:
            slam_response = self._reset_futures[0].result()
            self._reset_futures[1].result()
            self._reset_futures[2].result()
            return int(slam_response.result) == int(
                Reset.Response.RESULT_SUCCESS
            )
        except Exception as error:
            self.get_logger().error(f"Spatial reset failed: {error}")
            return False

    def _control_tick(self) -> None:
        self._accept_manual_reset()
        state = self.detector.state
        if state in {"lifted", "resetting"} or self._reset_mode is not None:
            self._publish_stop()
        if state == "resetting" and self._reset_mode is None:
            self._reset_mode = "lift"
        if self._reset_mode is not None:
            now = time.monotonic()
            if self._map_wait_revision is not None:
                if self._map_revision > self._map_wait_revision:
                    if self._reset_mode == "lift":
                        self.detector.reset_complete()
                    self._finish_reset(True, None)
                    self.get_logger().info(
                        "Fresh SLAM map received; movement may resume"
                    )
            else:
                if self._reset_futures is None and now >= self._retry_after:
                    self.reset_map()
                result = self._reset_succeeded()
                if result is True:
                    self._map_wait_revision = self._map_revision
                    self.get_logger().info(
                        "SLAM and costmaps reset; waiting for a fresh map"
                    )
                elif result is False:
                    self._reset_futures = None
                    self._retry_after = now + 1.0
        self._publish_state()
        self._publish_reset_state()

    def _accept_manual_reset(self) -> None:
        try:
            request_id, completion = self._manual_requests.get_nowait()
        except queue.Empty:
            return
        if self._reset_mode is not None:
            completion.put((False, "A spatial reset is already running"))
            return
        if self.detector.state in {"lifted", "resetting"}:
            completion.put((False, "Place the robot down before resetting"))
            return
        self._manual_reset = (
            request_id,
            completion,
        )
        self._reset_mode = "manual"
        self._retry_after = 0.0
        self.stop_movement("MANUAL_RESET")
        self.get_logger().info("Manual reset: motion stopped; resetting map")

    def _finish_reset(self, ok: bool, error: str | None) -> None:
        if self._reset_mode == "manual" and self._manual_reset is not None:
            self._manual_reset[1].put((ok, error))
        self._manual_reset = None
        self._reset_mode = None
        self._reset_futures = None
        self._map_wait_revision = None
        self._retry_after = 0.0

    def _publish_state(self) -> None:
        state = self.detector.state
        if state == self._last_state:
            return
        self._last_state = state
        self.state_publisher.publish(String(data=state))
        self.get_logger().info(f"Lift state: {state}")

    def _publish_reset_state(self) -> None:
        if self._reset_mode == "manual":
            state = "resetting_manual"
        elif self.detector.state == "lifted":
            state = "lifted"
        elif self.detector.state == "resetting" or self._reset_mode == "lift":
            state = "resetting_lift"
        elif self.detector.state == "calibrating":
            state = "initializing"
        else:
            state = "ready"
        if state == self._last_reset_state:
            return
        self._last_reset_state = state
        self.reset_state_publisher.publish(String(data=state))
        self.get_logger().info(f"Reset state: {state}")

    def destroy_node(self) -> None:
        self._zmq_stop.set()
        self._zmq_thread.join(timeout=1.0)
        self._zmq_context.term()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ResetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
