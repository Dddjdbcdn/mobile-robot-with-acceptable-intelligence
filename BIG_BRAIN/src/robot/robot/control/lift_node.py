"""Stop a lifted robot and start a fresh spatial session after placement."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import time

from geometry_msgs.msg import TwistStamped
from nav2_msgs.srv import ClearEntireCostmap
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
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
DIAGNOSTIC_PERIOD_SECONDS = 0.1
DIAGNOSTIC_LOG_MAX_BYTES = 5 * 1024 * 1024
CLIFF_FRESH_SECONDS = 0.5
BIG_BRAIN_ROOT = Path(__file__).resolve().parents[4]


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
        self.last_acceleration = (math.nan, math.nan, math.nan)
        self.last_angular_velocity = (math.nan, math.nan, math.nan)
        self.last_acceleration_norm = math.nan
        self.last_gyro_norm = math.nan
        self.last_gx = math.nan
        self.last_alignment = math.nan
        self.sample_valid = False

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
            self.sample_valid = False
            return None
        self.sample_valid = True

        dt = 0.0 if self.last_time is None else max(
            0.0, min(0.1, now - self.last_time)
        )
        self.last_time = now
        acceleration_norm = self.norm(acceleration)
        gyro_norm = self.norm(angular_velocity)
        self.last_acceleration = acceleration
        self.last_angular_velocity = angular_velocity
        self.last_acceleration_norm = acceleration_norm
        self.last_gyro_norm = gyro_norm
        self.last_gx = angular_velocity[0]

        if self.gravity_direction is not None and acceleration_norm > 1e-6:
            self.last_alignment = sum(
                value * baseline / acceleration_norm
                for value, baseline in zip(
                    acceleration, self.gravity_direction, strict=True
                )
            )

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
        self.last_alignment = alignment
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


class LiftNode(Node):
    """Coordinate stopping and a fresh spatial session after placement."""

    def __init__(self) -> None:
        super().__init__("lift_detector")
        self.detector = LiftDetector()
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, "/imu", self._imu_callback, qos)
        self.create_subscription(
            PointCloud2, "/cliff_layer", self._cliff_callback, qos
        )
        self.stop_publisher = self.create_publisher(
            TwistStamped, "/diff_drive_controller/cmd_vel", 10
        )
        self.reset_publisher = self.create_publisher(
            Empty, "/big_brain/reset", 10
        )
        self.state_publisher = self.create_publisher(
            String, "/lift_detector/state", 10
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
        self._retry_after = 0.0
        self._last_state = None
        self._last_diagnostic_at = -math.inf
        self._diagnostic_sequence = 0
        self._last_cliff_at: float | None = None
        self._cliff_point_count = 0
        self._diagnostic_log_path: Path | None = (
            BIG_BRAIN_ROOT / "results" / "lift_detection.log"
        )
        try:
            self._diagnostic_log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.get_logger().error(
                f"Lift diagnostics disabled: cannot create log directory: {error}"
            )
            self._diagnostic_log_path = None

        self.create_timer(0.1, self._control_tick)
        self._publish_state()
        self._write_session_start()
        if self._diagnostic_log_path is not None:
            self.get_logger().info(
                f"Lift diagnostics: {self._diagnostic_log_path}"
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
            self._placement_clear(now),
        )
        if event == "lifted":
            self.get_logger().warning(
                "Lift detected: canceling motion"
            )
            self.reset_publisher.publish(Empty())
            self._publish_stop()
            self._publish_state()
        elif event == "placed":
            self.get_logger().info(
                "Stable upright placement detected; requesting spatial reset"
            )
            self._publish_state()
        self._write_diagnostic(event or "", force=event is not None)

    def _cliff_callback(self, message: PointCloud2) -> None:
        self._last_cliff_at = time.monotonic()
        self._cliff_point_count = message.width * message.height
        if self._cliff_point_count > 0:
            self.detector.stable_since = None

    def _cliff_fresh(self, now: float) -> bool:
        return (
            self._last_cliff_at is not None
            and now - self._last_cliff_at <= CLIFF_FRESH_SECONDS
        )

    def _placement_clear(self, now: float) -> bool:
        return not self._cliff_fresh(now) or self._cliff_point_count == 0

    def _publish_stop(self) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_footprint"
        self.stop_publisher.publish(message)

    def _request_spatial_reset(self) -> None:
        clients = (self.slam, self.local_costmap, self.global_costmap)
        if not all(client.service_is_ready() for client in clients):
            return
        request = Reset.Request()
        # SLAM remains active while carried; this reset starts a clean map.
        request.pause_new_measurements = False
        self._reset_futures = (
            self.slam.call_async(request),
            self.local_costmap.call_async(ClearEntireCostmap.Request()),
            self.global_costmap.call_async(ClearEntireCostmap.Request()),
        )
        self._write_diagnostic("spatial_reset_requested", force=True)

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
        state = self.detector.state
        if state in {"lifted", "resetting"}:
            self._publish_stop()
        if state == "resetting":
            now = time.monotonic()
            if self._reset_futures is None and now >= self._retry_after:
                self._request_spatial_reset()
            result = self._reset_succeeded()
            if result is True:
                self.detector.reset_complete()
                self._reset_futures = None
                self.get_logger().info(
                    "SLAM and costmaps reset; movement may resume"
                )
                self._write_diagnostic("spatial_reset_complete", force=True)
            elif result is False:
                self._reset_futures = None
                self._retry_after = now + 1.0
                self._write_diagnostic("spatial_reset_failed", force=True)
        self._publish_state()

    @staticmethod
    def _number(value: float, digits: int = 3) -> str:
        return f"{value:.{digits}f}" if math.isfinite(value) else "--"

    def _write_log_line(self, line: str) -> None:
        path = self._diagnostic_log_path
        if path is None:
            return
        try:
            if path.exists() and path.stat().st_size >= DIAGNOSTIC_LOG_MAX_BYTES:
                path.replace(path.with_suffix(".log.1"))
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except OSError as error:
            self.get_logger().error(f"Lift diagnostics disabled: {error}")
            self._diagnostic_log_path = None

    def _write_session_start(self) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self._write_log_line(
            f"[LIFT] session=start wall_time={timestamp} "
            f"rate_threshold={LIFT_RATE_RAD_S:.3f}rad/s "
            f"rotation_threshold={LIFT_ROTATION_RAD:.3f}rad "
            f"minimum_lift={MINIMUM_LIFT_SECONDS:.2f}s "
            f"stable_gyro={STABLE_GYRO_RAD_S:.3f}rad/s "
            f"gravity={GRAVITY_M_S2:.3f}+/-{GRAVITY_TOLERANCE_M_S2:.3f}m/s2 "
            f"upright=10.0deg stable_time={STABLE_SECONDS:.2f}s "
            f"cliff_fresh={CLIFF_FRESH_SECONDS:.2f}s"
        )

    def _write_diagnostic(self, event: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_diagnostic_at < DIAGNOSTIC_PERIOD_SECONDS:
            return
        self._last_diagnostic_at = now
        self._diagnostic_sequence += 1
        detector = self.detector
        alignment = detector.last_alignment
        tilt = (
            math.degrees(math.acos(max(-1.0, min(1.0, alignment))))
            if math.isfinite(alignment)
            else math.nan
        )
        stable_for = (
            max(0.0, now - detector.stable_since)
            if detector.stable_since is not None
            else 0.0
        )
        cliff_fresh = self._cliff_fresh(now)
        placement_clear = not cliff_fresh or self._cliff_point_count == 0
        acceleration = detector.last_acceleration
        gyro = detector.last_angular_velocity
        criteria = (
            abs(detector.last_gx) >= LIFT_RATE_RAD_S,
            detector.lift_rotation >= LIFT_ROTATION_RAD,
            detector.lifted_at is not None
            and now - detector.lifted_at >= MINIMUM_LIFT_SECONDS,
            detector.last_gyro_norm <= STABLE_GYRO_RAD_S,
            abs(detector.last_acceleration_norm - GRAVITY_M_S2)
            <= GRAVITY_TOLERANCE_M_S2,
            math.isfinite(alignment) and alignment >= PLACEMENT_COSINE,
            placement_clear,
            stable_for >= STABLE_SECONDS,
        )
        values = (
            self._number(acceleration[0]), self._number(acceleration[1]),
            self._number(acceleration[2]), self._number(detector.last_acceleration_norm),
            self._number(gyro[0]), self._number(gyro[1]), self._number(gyro[2]),
            self._number(detector.last_gyro_norm), self._number(detector.lift_rotation),
            self._number(alignment), self._number(tilt),
        )
        self._write_log_line(
            f"[LIFT] seq={self._diagnostic_sequence} state={detector.state} "
            f"event={event or '--'} valid={int(detector.sample_valid)} "
            f"accel=({values[0]},{values[1]},{values[2]})m/s2 "
            f"accel_norm={values[3]}m/s2 "
            f"gyro=({values[4]},{values[5]},{values[6]})rad/s "
            f"gyro_norm={values[7]}rad/s lift_rotation={values[8]}rad "
            f"alignment={values[9]} tilt={values[10]}deg "
            f"criteria=rate:{int(criteria[0])},rotation:{int(criteria[1])},"
            f"min_time:{int(criteria[2])},gyro:{int(criteria[3])},"
            f"gravity:{int(criteria[4])},upright:{int(criteria[5])},"
            f"cliff_clear:{int(criteria[6])},stable:{int(criteria[7])} "
            f"cliff_fresh:{int(cliff_fresh)},"
            f"cliff_points:{self._cliff_point_count} "
            f"stable_for={stable_for:.2f}s"
        )

    def _publish_state(self) -> None:
        state = self.detector.state
        if state == self._last_state:
            return
        self._last_state = state
        self.state_publisher.publish(String(data=state))
        self.get_logger().info(f"Lift state: {state}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LiftNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
