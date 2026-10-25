"""Connect cognition commands and robot state to ROS interfaces."""

import json
import math
import threading
import time

from geometry_msgs.msg import PoseStamped, TwistStamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Range
from std_msgs.msg import Float32, String
import tf2_geometry_msgs  # noqa: F401
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import zmq

from robot.control.camera_servo import CameraServo
from robot.map.map_stream import MapImageStream
from robot.bridge.navigation_controller import (
    NAVIGATION_COMMANDS,
    NavigationController,
)

def _yaw_from_quaternion(rotation):
    return math.atan2(
        2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1.0 - 2.0 * (rotation.y ** 2 + rotation.z ** 2),
    )
def _planar_pose_from_transform(transform, frame_id):
    translation = transform.transform.translation
    return {
        "x": float(translation.x),
        "y": float(translation.y),
        "yaw": _yaw_from_quaternion(transform.transform.rotation),
        "frame_id": str(frame_id),
    }

class LLMRosBridge(Node):
    """Orchestrate transport, sensing, tracking, and navigation helpers."""

    def __init__(self):
        super().__init__('llm_ros_bridge')
        self.get_logger().info("Starting DJ-ROS Bridge (Async Mode)...")
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._stop_event = threading.Event()

        # ZMQ transport to and from SMALL_BRAIN.
        self.zmq_context = zmq.Context()
        self.rep_socket = self.zmq_context.socket(zmq.REP)
        self.rep_socket.setsockopt(zmq.LINGER, 0)
        self.rep_socket.setsockopt(zmq.RCVTIMEO, 200)
        self.rep_socket.bind("tcp://*:5555")
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.LINGER, 0)
        self.pub_socket.bind("tcp://*:5556")
        self.sub_socket = self.zmq_context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.LINGER, 0)
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 200)
        self.sub_socket.bind("tcp://*:5557")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.pub_lock = threading.Lock()

        # Base and camera control.
        self.cmd_pub = self.create_publisher(
            TwistStamped, "/diff_drive_controller/cmd_vel", 10
        )
        self.servo_pan_pub = self.create_publisher(Float32, "/stm32/servo_pan", 10)
        self.servo_tilt_pub = self.create_publisher(Float32, "/stm32/servo_tilt", 10)
        self.servo = CameraServo(
            self.servo_pan_pub,
            self.servo_tilt_pub,
            self.get_logger(),
        )

        # Shared robot state and domain helpers.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_image_stream = MapImageStream(self, self.tf_buffer, self.zmq_context)
        self.camera_tof_range = 0.0
        self.robot_pose = None
        self.robot_pose_at = None
        self.latest_person_pose_map = None
        self.latest_person_pose_at = None
        self.person_tracker_status = {"state": None}
        self.lift_state = None
        self.reset_state = None
        self.stm32_health = None
        self.stm32_health_at = None
        self.navigation = NavigationController(
            node=self,
            tf_buffer=self.tf_buffer,
            map_stream=self.map_image_stream,
            send_event=self.send_event,
            stop_robot=lambda: self.publish_cmd(0.0, 0.0),
            get_robot_pose=self._robot_state,
            get_person_state=self._person_state,
        )

        # Camera/body tracking state.
        self.tracking_enabled = False
        self.tracking_body_active = False
        self.track_body_pan_margin_deg = 15.0
        self.track_body_pan_hysteresis_deg = 5.0
        self.track_body_kp = 0.05
        self.track_body_max_angular_vel = 1.0
        self.track_body_visual_timeout_s = 1.0

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Range, '/camera_tof', self.camera_tof_callback, qos)
        self.create_subscription(PoseStamped, "/person_pose", self.person_pose_callback, 10)
        self.create_subscription(
            String, "/person_tracker/status", self.person_tracker_status_callback, 10
        )
        self.create_subscription(
            String,
            "/lift_detector/state",
            self.lift_state_callback,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.create_subscription(
            String,
            "/reset/state",
            self.reset_state_callback,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.create_subscription(
            String, "/stm32/debug_msg", self.stm32_debug_callback, qos
        )
        self.create_subscription(
            String, "/big_brain/reset_reason", self.reset_callback, 10
        )
        self.create_timer(0.05, self.state_pub_loop)
        self.create_timer(0.1, self.update_robot_pose)
        self.track_action_timer = self.create_timer(0.01, self.track_action_loop)
        self.track_action_timer.cancel()

        # Blocking ZMQ loops run outside the ROS executor.
        self.llm_thread = threading.Thread(
            target=self.listen_for_llm,
            name="llm-command-server",
            daemon=True,
        )
        self.tracking_thread = threading.Thread(
            target=self.listen_for_tracking_updates,
            name="camera-tracking-listener",
            daemon=True,
        )
        self.llm_thread.start()
        self.tracking_thread.start()

    # ROS state ---------------------------------------------------------

    def publish_cmd(self, linear_x, angular_z):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def reset_callback(self, message: String) -> None:
        self._set_tracking_enabled(False)
        self.tracking_body_active = False
        self.stop_all_motion()
        self.servo.clear_tracking_observation()
        self.servo.set_angles(
            self.servo.reset_pan_angle,
            self.servo.reset_tilt_angle,
        )
        self.latest_person_pose_map = None
        self.latest_person_pose_at = None
        reason = str(message.data or "MANUAL_RESET")
        self.send_event("robot_reset", "requested", reason_code=reason)
        self.get_logger().info(
            "Robot stopped, camera recentered, and SMALL_BRAIN reset requested"
        )

    def state_pub_loop(self):
        with self.pub_lock:
            self.pub_socket.send_json({
                "type": "state",
                "camera_tof_range": self.camera_tof_range,
                "servo_pan_angle": self.servo.pan_angle,
                "servo_tilt_angle": self.servo.tilt_angle,
                "robot_pose": self.robot_pose,
                "person_pose": self._person_state(),
                "person_tracker_state": self.person_tracker_status.get("state"),
                "person_tracker_status": self.person_tracker_status,
                "lift_state": self.lift_state,
                "reset_state": self.reset_state,
                "navigation_status": self.navigation.status_snapshot(),
                "stm32_health": (
                    {
                        "message": self.stm32_health,
                        "age_seconds": max(
                            0.0, time.monotonic() - self.stm32_health_at
                        ),
                    }
                    if self.stm32_health_at is not None else None
                ),
            })

    def update_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02),
            )
        except Exception:
            return
        self.robot_pose = _planar_pose_from_transform(transform, "map")
        self.robot_pose_at = time.monotonic()

    def _robot_state(self):
        if self.robot_pose is None or self.robot_pose_at is None:
            return None
        state = dict(self.robot_pose)
        state["age_seconds"] = max(0.0, time.monotonic() - self.robot_pose_at)
        return state

    def _person_state(self):
        if self.latest_person_pose_map is None or self.latest_person_pose_at is None:
            return None
        position = self.latest_person_pose_map.pose.position
        return {
            "x": float(position.x),
            "y": float(position.y),
            "frame_id": "map",
            "age_seconds": max(0.0, time.monotonic() - self.latest_person_pose_at),
        }

    def person_pose_callback(self, msg):
        if msg.header.frame_id != "map":
            self.get_logger().warning(
                f"Ignoring /person_pose in unexpected frame {msg.header.frame_id!r}",
                throttle_duration_sec=2.0,
            )
            return
        self.latest_person_pose_map = msg
        self.latest_person_pose_at = time.monotonic()

    def person_tracker_status_callback(self, msg):
        try:
            status = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        state = str(status.get("state") or "")
        previous = self.person_tracker_status.get("state")
        self.person_tracker_status = status
        if state == "disabled":
            self.latest_person_pose_map = None
            self.latest_person_pose_at = None
        self.navigation.note_person_tracker_state(state, previous, status)

    def lift_state_callback(self, msg):
        self.lift_state = str(msg.data or "").strip() or None

    def reset_state_callback(self, msg):
        self.reset_state = str(msg.data or "").strip() or None

    def stm32_debug_callback(self, msg):
        message = str(msg.data or "").strip()
        if not message.startswith("[HEALTH]"):
            return
        self.stm32_health = message.removeprefix("[HEALTH]").strip()
        self.stm32_health_at = time.monotonic()

    def camera_tof_callback(self, msg):
        self.camera_tof_range = msg.range

    # SMALL_BRAIN events and command orchestration ---------------------

    def send_event(self, event_name, status_msg, action_id=None, **fields):
        with self.pub_lock:
            payload = {"type": "event", "event": event_name, "status": status_msg}
            if action_id is not None:
                payload["action_id"] = action_id
            payload.update(fields)
            self.pub_socket.send_json(payload)
            self.get_logger().info(f"Broadcasted to LLM: {payload}")

    def resolve_request(self, request):
        try:
            command = request.get("command")
            if (
                self.lift_state in {"lifted", "resetting"}
                and command not in {
                    "stop_follow_action", "stop_moving", "stop_tracking"
                }
            ):
                return {
                    "status": "error",
                    "message": f"Robot is {self.lift_state}",
                }
            if command == "move_camera":
                self.servo.set_error(
                    request.get("delta_pan_angle", 0.0),
                    request.get("delta_tilt_angle", 0.0),
                )
                self.servo.publish_servo_command(tracking=False)
                return {"status": "accepted"}
            if command == "track_action":
                self.servo.clear_tracking_observation()
                self._set_tracking_enabled(True)
                return {"status": "accepted"}
            if command == "stop_tracking":
                return self._stop_tracking(request)
            if command in {"stop_follow_action", "stop_moving"}:
                self.stop_all_motion()
                return {"status": "accepted"}
            if command in NAVIGATION_COMMANDS:
                return self.navigation.handle_request(request)
            return {"status": "error", "message": f"Unknown command: {command}"}
        except Exception as error:
            self.get_logger().error(f"Command resolution failed: {error}")
            return {"status": "error", "message": f"Internal exception: {error}"}

    # Camera/body tracking ---------------------------------------------

    def _set_tracking_enabled(self, enabled):
        self.tracking_enabled = bool(enabled)
        if self.tracking_enabled:
            self.track_action_timer.reset()
        else:
            self.track_action_timer.cancel()

    def _stop_tracking(self, request):
        self._set_tracking_enabled(False)
        self.tracking_body_active = False
        self.publish_cmd(0.0, 0.0)
        if request.get("reset_camera", True) is not False:
            self.servo.set_error(
                self.servo.reset_pan_angle - self.servo.pan_angle,
                self.servo.reset_tilt_angle - self.servo.tilt_angle,
            )
            self.servo.publish_servo_command(tracking=False)
        return {"status": "accepted"}

    def stop_all_motion(self):
        self.navigation.stop()

    def _stop_tracking_body(self):
        if self.tracking_body_active:
            self.publish_cmd(0.0, 0.0)
        self.tracking_body_active = False

    def track_action_loop(self):
        if not self.tracking_enabled:
            self._stop_tracking_body()
            return
        self.servo.publish_servo_command(tracking=True)
        if self.navigation.is_active:
            self._stop_tracking_body()
            return

        observed_at = self.servo.tracking_snapshot()["observed_at"]
        observation_is_stale = (
            observed_at is None
            or time.monotonic() - observed_at > self.track_body_visual_timeout_s
        )
        if observation_is_stale:
            self._stop_tracking_body()
            return

        start_margin = max(0.0, self.track_body_pan_margin_deg)
        release_margin = start_margin + max(0.0, self.track_body_pan_hysteresis_deg)
        start_low = self.servo.min_pan_angle + start_margin
        start_high = self.servo.max_pan_angle - start_margin
        release_low = self.servo.min_pan_angle + release_margin
        release_high = self.servo.max_pan_angle - release_margin
        pan_angle = self.servo.pan_angle
        if not self.tracking_body_active and start_low <= pan_angle <= start_high:
            return
        if pan_angle < release_low:
            body_pan_error = pan_angle - release_low
        elif pan_angle > release_high:
            body_pan_error = pan_angle - release_high
        else:
            self.publish_cmd(0.0, 0.0)
            self.tracking_body_active = False
            return
        max_ang_vel = max(0.0, self.track_body_max_angular_vel)
        ang_vel = max(
            min(self.track_body_kp * body_pan_error, max_ang_vel),
            -max_ang_vel,
        )
        self.publish_cmd(0.0, ang_vel)
        self.tracking_body_active = True

    # Background transport and lifecycle -------------------------------

    def listen_for_llm(self):
        while not self._stop_event.is_set():
            try:
                request = self.rep_socket.recv_json()
                self.rep_socket.send_json(self.resolve_request(request))
            except zmq.Again:
                continue
            except zmq.ZMQError as error:
                if self._stop_event.is_set():
                    return
                if rclpy.ok(context=self.context):
                    self.get_logger().error(f"LLM transport failed: {error}")
                return
            except Exception as error:
                if (
                    not self._stop_event.is_set()
                    and rclpy.ok(context=self.context)
                ):
                    self.get_logger().error(f"LLM request loop failed: {error}")

    def listen_for_tracking_updates(self):
        while not self._stop_event.is_set():
            try:
                message = self.sub_socket.recv_json()
                if (
                    message.get("tracking_valid") is not True
                    or message.get("tracking_sequence") is None
                ):
                    continue
                self.servo.set_error(
                    message.get("delta_pan_angle", 0.0),
                    message.get("delta_tilt_angle", 0.0),
                    message.get("tracking_sequence"),
                    message.get("tracking_timing"),
                )
            except zmq.Again:
                continue
            except zmq.ZMQError as error:
                if self._stop_event.is_set():
                    return
                if rclpy.ok(context=self.context):
                    self.get_logger().error(
                        f"Camera tracking transport failed: {error}"
                    )
                return
            except Exception as error:
                if (
                    not self._stop_event.is_set()
                    and rclpy.ok(context=self.context)
                ):
                    self.get_logger().error(
                        f"Camera tracking listener failed: {error}"
                    )

    def shutdown(self):
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True

        context_is_valid = rclpy.ok(context=self.context)
        try:
            if context_is_valid:
                self.get_logger().info("Shutting down LLM ROS Bridge...")
                self._set_tracking_enabled(False)
                self.navigation.shutdown()
        except Exception as error:
            if rclpy.ok(context=self.context):
                self.get_logger().warning(
                    f"Motion cleanup during shutdown failed: {error}"
                )
        finally:
            self._stop_event.set()
            self.map_image_stream.close()
            for thread in (self.llm_thread, self.tracking_thread):
                if thread is not threading.current_thread():
                    thread.join(timeout=1.0)
            self.rep_socket.close(linger=0)
            self.pub_socket.close(linger=0)
            self.sub_socket.close(linger=0)
            self.zmq_context.term()


def main(args=None):
    rclpy.init(args=args)
    node = LLMRosBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
