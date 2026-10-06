"""Connect cognition commands and robot state to ROS interfaces."""

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Range
from std_msgs.msg import Float32, String
import tf2_geometry_msgs  # noqa: F401
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import zmq

from robot.map.map_stream import MapImageStream
from robot.person_pose.follow_goal_generator import FollowGoalGenerator
from robot.control.camera_servo import CameraServo


@dataclass
class ActiveNavigation:
    mode: str
    action_id: str | None
    destination: dict | None = None
    goal_handle: object | None = None


SEQUENCE_COMMANDS = {
    "open_space_middle", "exit_room", "go_to_another_room",
}
ROOM_SEQUENCE_COMMANDS = {"exit_room", "go_to_another_room"}


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


def _make_pose_stamped(x, y, yaw, frame_id, stamp):
    pose = PoseStamped()
    pose.header.frame_id = str(frame_id)
    pose.header.stamp = stamp
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = 0.0
    pose.pose.orientation.z = math.sin(float(yaw) * 0.5)
    pose.pose.orientation.w = math.cos(float(yaw) * 0.5)
    return pose


class LLMRosBridge(Node):
    def __init__(self):
        super().__init__('llm_ros_bridge')
        self.get_logger().info("Starting DJ-ROS Bridge (Async Mode)...")

        self.zmq_context = zmq.Context()

        self.rep_socket = self.zmq_context.socket(zmq.REP)
        self.rep_socket.bind("tcp://*:5555")

        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:5556")

        self.sub_socket = self.zmq_context.socket(zmq.SUB)
        self.sub_socket.bind("tcp://*:5557")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.pub_lock = threading.Lock()

        self.cmd_pub = self.create_publisher(
            TwistStamped, "/diff_drive_controller/cmd_vel", 10
        )
        self.servo_pan_pub = self.create_publisher(
            Float32, "/stm32/servo_pan", 10
        )
        self.servo_tilt_pub = self.create_publisher(
            Float32, "/stm32/servo_tilt", 10
        )
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.goal_update_pub = self.create_publisher(
            PoseStamped, "/goal_update", 10
        )
        config_dir = Path(get_package_share_directory("robot")) / "config"
        self.dj_bt = str(config_dir / "behavior_dj.xml")
        self.follow_bt = str(config_dir / "behavior_follow.xml")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.servo = CameraServo(
            self.servo_pan_pub,
            self.servo_tilt_pub,
            self.get_logger(),
        )
        self.follow_standoff_m = 0.90
        self.follow_heading_comfort_deg = 20.0
        self.follow_heading_critical_deg = 45.0
        self.follow_goal_generator = FollowGoalGenerator(
            standoff_m=self.follow_standoff_m,
            camera_center_deg=self.servo.reset_pan_angle,
            heading_comfort_deg=self.follow_heading_comfort_deg,
            heading_critical_deg=self.follow_heading_critical_deg,
        )
        self.track_body_pan_margin_deg = 15.0
        self.track_body_pan_hysteresis_deg = 5.0
        self.track_body_kp = 0.08
        self.track_body_max_angular_vel = 1.0
        self.sequence_map_update_timeout_s = 5.0
        self.map_image_stream = MapImageStream(self, self.tf_buffer, self.zmq_context)

        self.camera_tof_range = 0.0
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Range, '/camera_tof', self.camera_tof_callback, qos)
        self.robot_pose = None
        self.robot_pose_at = None

        self.create_timer(0.05, self.state_pub_loop)
        self.create_timer(0.1, self.update_robot_pose)

        self.navigation = None
        self.track_action_timer = None
        self.follow_enabled = False
        self.follow_action_id = None
        self.person_tracker_status = {"state": None}
        self.follow_had_person_track = False
        self.latest_person_pose_map = None
        self.latest_person_pose_at = None
        self.pending_person_pose = None
        self.last_follow_goal = None
        self.last_follow_goal_at = None
        self.previous_navigation_pose = None
        self.sequence = None
        self.sequence_map_wait_timer = None
        self.tracking_body_active = False
        self.create_subscription(
            PoseStamped, "/person_pose", self.person_pose_callback, 10
        )
        self.create_subscription(
            String,
            "/person_tracker/status",
            self.person_tracker_status_callback,
            10,
        )
        self.create_timer(0.1, self.process_pending_person_pose)
        self.create_timer(0.4, self.update_follow_goal)

        threading.Thread(target=self.listen_for_llm, daemon=True).start()
        threading.Thread(target=self.background_listener, daemon=True).start()

    def publish_cmd(self, linear_x, angular_z):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

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
            })

    def update_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02),
            )
        except Exception:
            return None

        self.robot_pose = _planar_pose_from_transform(transform, "map")
        self.robot_pose_at = time.monotonic()

    def _person_state(self):
        if self.latest_person_pose_map is None or self.latest_person_pose_at is None:
            return None
        position = self.latest_person_pose_map.pose.position
        return {
            "x": float(position.x), "y": float(position.y), "frame_id": "map",
            "age_seconds": max(0.0, time.monotonic() - self.latest_person_pose_at),
        }

    def person_pose_callback(self, msg):
        """Keep the newest lidar pose until its map transform is available."""
        self.pending_person_pose = msg
        self.process_pending_person_pose()

    def person_tracker_status_callback(self, msg):
        try:
            status = json.loads(msg.data)
        except (TypeError, ValueError):
            return

        state = str(status.get("state") or "")
        previous = self.person_tracker_status.get("state")
        self.person_tracker_status = status
        if not self.follow_enabled:
            return

        if state in {"tracking", "coasting", "recovering"}:
            self.follow_had_person_track = True
        if (
            state == "lost"
            and previous != "lost"
            and self.follow_had_person_track
        ):
            self.send_event(
                "person_tracker",
                "lost",
                self.follow_action_id,
            )

    def process_pending_person_pose(self):
        msg = self.pending_person_pose
        if msg is None:
            return

        try:
            if msg.header.frame_id == "map":
                pose_map = msg
            else:
                # The pose is expressed in odom coordinates, but its lidar
                # timestamp is normally newer than SLAM's latest map->odom
                # update. An exact-time lookup therefore waits and repeatedly
                # fails with future extrapolation while newer poses replace it.
                # Use the latest frame transform without blocking instead.
                transform = self.tf_buffer.lookup_transform(
                    "map",
                    msg.header.frame_id,
                    rclpy.time.Time(),
                )
                pose_map = tf2_geometry_msgs.do_transform_pose_stamped(
                    msg, transform
                )
        except Exception as error:
            self.get_logger().warning(
                f"Waiting to transform /person_pose to map: {error}",
                throttle_duration_sec=2.0,
            )
            return

        # A newer callback may have replaced this pose while TF was queried.
        if self.pending_person_pose is not msg:
            return
        self.pending_person_pose = None
        pose_map.header.stamp = self.get_clock().now().to_msg()
        self.latest_person_pose_map = pose_map
        self.latest_person_pose_at = time.monotonic()

    def update_follow_goal(self):
        """Publish a stable standoff goal with adaptive camera-cone yaw."""
        if not self.follow_enabled:
            return

        now = time.monotonic()
        if (
            self.latest_person_pose_map is None
            or self.latest_person_pose_at is None
            or now - self.latest_person_pose_at > 0.75
            or self.robot_pose is None
            or self.robot_pose_at is None
            or now - self.robot_pose_at > 0.50
        ):
            return

        camera = self.servo.tracking_snapshot()
        observed_at = camera["observed_at"]
        camera_fresh = (
            observed_at is not None and now - observed_at <= 0.30
        )
        # Include the newest image error so the goal can react before the
        # physical camera has completed its next servo step.
        camera_pan = camera["pan_angle"] + camera["pan_error"]
        person = self.latest_person_pose_map.pose.position
        generated = self.follow_goal_generator.generate(
            person_x=float(person.x),
            person_y=float(person.y),
            robot_x=float(self.robot_pose["x"]),
            robot_y=float(self.robot_pose["y"]),
            robot_yaw=float(self.robot_pose["yaw"]),
            camera_pan_deg=camera_pan,
            camera_fresh=camera_fresh,
        )
        if generated is None:
            return

        if not camera_fresh:
            desired_pan = max(
                self.servo.min_pan_angle,
                min(
                    self.servo.max_pan_angle,
                    generated.camera_pan_target_deg,
                ),
            )
            self.servo.set_error(
                desired_pan - float(camera["pan_angle"]),
                0.0,
            )
            self.servo.publish_servo_command(tracking=True)

        goal_pose = _make_pose_stamped(
            generated.x,
            generated.y,
            generated.yaw,
            "map",
            self.get_clock().now().to_msg(),
        )

        elapsed_since_publish = (
            None
            if self.last_follow_goal_at is None
            else now - self.last_follow_goal_at
        )
        should_publish = self.follow_goal_generator.should_publish(
            generated,
            self.last_follow_goal,
            elapsed_since_publish,
        )

        if self.navigation is None:
            self.start_follow_navigation(goal_pose)
            should_publish = True
        elif self.navigation.mode == "follow" and should_publish:
            self.goal_update_pub.publish(goal_pose)

        if should_publish:
            self.last_follow_goal = (
                generated.x,
                generated.y,
                generated.yaw,
            )
            self.last_follow_goal_at = now

    def camera_tof_callback(self, msg):
        self.camera_tof_range = msg.range

    def send_event(self, event_name, status_msg, action_id=None, **fields):
        with self.pub_lock:
            payload = {
                "type": "event",
                "event": event_name,
                "status": status_msg,
            }
            if action_id is not None:
                payload["action_id"] = action_id
            payload.update(fields)
            self.pub_socket.send_json(payload)
            self.get_logger().info(f"Broadcasted to LLM: {payload}")

    def nav_goal_response_cb(self, future, navigation):
        goal_handle = future.result()
        if self.navigation is not navigation:
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
            return
        if not goal_handle.accepted:
            self.navigation = None
            if navigation.mode == "sequence":
                self._finish_sequence(
                    "failed", "navigation_rejected", "NAVIGATION_REJECTED"
                )
            elif navigation.mode in {"local", "approach"}:
                self.send_event(
                    "navigation",
                    "failed",
                    navigation.action_id,
                    mode=navigation.mode,
                    outcome="navigation_rejected",
                    reason_code="NAVIGATION_REJECTED",
                    data={
                        "destination": navigation.destination,
                        "robot_status": "Goal Rejected by Nav2",
                    },
                )
            else:
                if navigation.mode == "follow":
                    self.follow_enabled = False
                    self.follow_action_id = None
                self.send_event(
                    "navigation", "Goal Rejected by Nav2", navigation.action_id
                )
            return

        navigation.goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda completed: self.nav_result_cb(completed, navigation)
        )

    def nav_result_cb(self, future, navigation):
        if self.navigation is not navigation:
            return

        self.navigation = None
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            if navigation.mode == "sequence":
                if self.sequence is None:
                    return
                if self.sequence["complete_after_goal"]:
                    self._finish_sequence("succeeded", "navigation_reached")
                    return
                self._wait_for_sequence_map_update()
            elif navigation.mode in {"local", "approach"}:
                self.send_event(
                    "navigation",
                    "succeeded",
                    navigation.action_id,
                    mode=navigation.mode,
                    outcome=(
                        "navigation_reached"
                        if navigation.mode == "approach"
                        else "explicit_navigation_completed"
                    ),
                    data={"destination": navigation.destination},
                )
            else:
                if navigation.mode == "follow":
                    self.follow_enabled = False
                    self.follow_action_id = None
                self.send_event(
                    "navigation", "Goal Reached", navigation.action_id
                )
        elif navigation.mode == "sequence":
            self._finish_sequence(
                "failed", "navigation_failed", "NAVIGATION_FAILED"
            )
        elif navigation.mode in {"local", "approach"}:
            self.send_event(
                "navigation",
                "failed",
                navigation.action_id,
                mode=navigation.mode,
                outcome="navigation_failed",
                reason_code="NAVIGATION_FAILED",
                data={
                    "destination": navigation.destination,
                    "robot_status": f"Nav2 status {status}",
                },
            )
        else:
            if navigation.mode == "follow":
                self.follow_enabled = False
                self.follow_action_id = None
            self.send_event(
                "navigation", f"Failed or Canceled (Status {status})",
                navigation.action_id,
            )

    def stop_all_motion(self):
        self._cancel_sequence_map_wait()
        navigation = self.navigation
        self.navigation = None
        if navigation is not None and navigation.goal_handle is not None:
            navigation.goal_handle.cancel_goal_async()
        self.follow_enabled = False
        self.follow_action_id = None
        self.follow_had_person_track = False
        self.pending_person_pose = None
        self.last_follow_goal = None
        self.last_follow_goal_at = None
        self.sequence = None
        self.publish_cmd(0.0, 0.0)

    def _cancel_sequence_map_wait(self):
        timer = self.sequence_map_wait_timer
        self.sequence_map_wait_timer = None
        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)

    def _wait_for_sequence_map_update(self):
        """Resume a sequence only after /map updates following arrival."""
        sequence = self.sequence
        if sequence is None:
            return
        self._cancel_sequence_map_wait()
        sequence["arrival_map_revision"] = self.map_image_stream.map_revision()
        sequence["map_wait_started"] = time.monotonic()
        self.sequence_map_wait_timer = self.create_timer(
            0.05, self._resume_sequence_after_map_update
        )

    def _resume_sequence_after_map_update(self):
        sequence = self.sequence
        if sequence is None:
            self._cancel_sequence_map_wait()
            return

        arrival_revision = int(sequence["arrival_map_revision"])
        if self.map_image_stream.map_revision() <= arrival_revision:
            elapsed = time.monotonic() - float(sequence["map_wait_started"])
            if elapsed < self.sequence_map_update_timeout_s:
                return
            self._cancel_sequence_map_wait()
            self._finish_sequence(
                "failed", "map_update_timeout", "MAP_UPDATE_TIMEOUT"
            )
            return

        self._cancel_sequence_map_wait()
        try:
            prepared, pose = self.map_image_stream.prepare_map()
            self._advance_sequence(prepared, pose)
        except (KeyError, TypeError, ValueError) as error:
            self.get_logger().warning(
                f"Navigation sequence resolution failed: {error}"
            )
            self._finish_sequence(
                "failed",
                "navigation_resolution_failed",
                "NAVIGATION_REJECTED",
            )

    def start_follow_navigation(self, pose_map):
        """Start one long-running Nav2 goal; GoalUpdater handles later poses."""
        if not self.follow_enabled or self.navigation is not None:
            return

        navigation = ActiveNavigation(
            mode="follow",
            action_id=self.follow_action_id,
        )
        self.navigation = navigation
        goal = NavigateToPose.Goal()
        goal.pose = pose_map
        goal.behavior_tree = self.follow_bt
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(
            lambda completed: self.nav_goal_response_cb(
                completed, navigation
            )
        )

    def track_action_loop(self):
        self.servo.publish_servo_command(tracking=True)

        if self.navigation is not None:
            self.tracking_body_active = False
            return

        # Do not wait for a requested servo step to overflow the hard limit.
        # That overflow exists for only one timer tick, which makes the base
        # repeatedly start and stop at the camera frame rate.  Instead, turn
        # continuously whenever the actual pan leaves a soft comfort band.
        start_margin = max(0.0, self.track_body_pan_margin_deg)
        release_margin = start_margin + max(
            0.0, self.track_body_pan_hysteresis_deg
        )
        start_low = self.servo.min_pan_angle + start_margin
        start_high = self.servo.max_pan_angle - start_margin
        release_low = self.servo.min_pan_angle + release_margin
        release_high = self.servo.max_pan_angle - release_margin
        pan_angle = self.servo.pan_angle

        should_turn = self.tracking_body_active or (
            pan_angle < start_low or pan_angle > start_high
        )
        if not should_turn:
            return

        if pan_angle < release_low:
            body_pan_error = pan_angle - release_low
        elif pan_angle > release_high:
            body_pan_error = pan_angle - release_high
        else:
            self.publish_cmd(0.0, 0.0)
            self.tracking_body_active = False
            return

        ang_vel = self.track_body_kp * body_pan_error
        max_ang_vel = max(0.0, self.track_body_max_angular_vel)
        ang_vel = max(min(ang_vel, max_ang_vel), -max_ang_vel)

        self.publish_cmd(0.0, ang_vel)
        self.tracking_body_active = True

    def dispatch_navigation_goal(
        self, destination, action_id, navigation_mode
    ):
        """Transform and dispatch one destination through the shared Nav2 path."""
        pose_in = _make_pose_stamped(
            destination["x"], destination["y"], destination["angle"],
            destination.get("frame_id", "base_footprint"),
            rclpy.time.Time().to_msg(),
        )
        if pose_in.header.frame_id == "map":
            pose_map = pose_in
        else:
            try:
                pose_map = self.tf_buffer.transform(
                    pose_in,
                    "map",
                    timeout=rclpy.duration.Duration(seconds=0.1),
                )
            except Exception as error:
                return f"TF Transform failed: {error}"

        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            return "Nav2 not available"

        pose_map.header.stamp = self.get_clock().now().to_msg()
        navigation = ActiveNavigation(
            mode=navigation_mode,
            action_id=action_id,
            destination=dict(destination),
        )
        self.navigation = navigation

        goal = NavigateToPose.Goal()
        goal.pose = pose_map
        goal.behavior_tree = self.dj_bt
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(
            lambda completed: self.nav_goal_response_cb(completed, navigation)
        )
        return None

    def _start_sequence(self, request, prepared, pose):
        command = str(request.get("sequence_command") or "")
        if command not in SEQUENCE_COMMANDS:
            raise ValueError(f"Unknown navigation sequence: {command}")
        self.sequence = {
            "action_id": request.get("action_id"),
            "command": command,
            "face_person": bool(request.get("face_person")),
            "state": {},
            "attempts": 0,
            "complete_after_goal": False,
        }
        self._advance_sequence(prepared, pose)

    def _advance_sequence(self, prepared, pose):
        sequence = self.sequence
        if sequence is None:
            raise ValueError("No navigation sequence is active")

        command = sequence["command"]
        state = sequence["state"]
        if command in ROOM_SEQUENCE_COMMANDS:
            if command == "go_to_another_room" and "origin_room_id" not in state:
                room = (
                    self.map_image_stream.room_registry.room_at(prepared, pose)
                    or self.map_image_stream.room_registry.ensure_startup_room(
                        prepared, pose
                    )
                )
                if room is not None:
                    state["origin_room_id"] = room["room_id"]
            step = self.map_image_stream.logic.resolve_exit_room_step(
                prepared, pose, state
            )
        else:
            step = self.map_image_stream.logic.resolve_open_space_middle_step(
                prepared,
                pose,
                state,
                person_pose=(
                    self._person_state() if sequence["face_person"] else None
                ),
                face_person=sequence["face_person"],
            )

        if step["complete"]:
            if command == "go_to_another_room":
                self.map_image_stream.room_registry.observe_room(
                    prepared,
                    pose,
                    connected_from=step["state"].get("origin_room_id"),
                )
            self._finish_sequence("succeeded", "navigation_reached")
            return
        if command == "exit_room" and step["phase"] == "cross_room_boundary":
            self._finish_sequence("succeeded", "navigation_reached")
            return

        limit = 12 if command in ROOM_SEQUENCE_COMMANDS else 4
        if sequence["attempts"] >= limit:
            raise ValueError(f"Navigation sequence exceeded {limit} goals")

        sequence["state"] = step["state"]
        sequence["attempts"] += 1
        sequence["complete_after_goal"] = (
            command == "open_space_middle"
            and step["phase"] == "move_to_room_core"
        )
        error = self.dispatch_navigation_goal(
            step["destination"], sequence["action_id"], "sequence"
        )
        if error:
            raise ValueError(error)

    def _finish_sequence(self, status, outcome, reason_code=None):
        self._cancel_sequence_map_wait()
        sequence = self.sequence
        self.sequence = None
        if sequence is not None:
            self.send_event(
                "navigation",
                status,
                sequence["action_id"],
                mode="sequence",
                outcome=outcome,
                reason_code=reason_code,
            )

    def background_listener(self):
        while rclpy.ok():
            try:
                message = self.sub_socket.recv_json()
                self.servo.set_error(
                    message.get("delta_pan_angle", 0.0),
                    message.get("delta_tilt_angle", 0.0),
                    message.get("tracking_sequence"),
                    message.get("tracking_timing"),
                )
            except Exception as e:
                self.get_logger().error(f"Internal Loop Error: {e}")

    def shutdown(self):
        self.get_logger().info("Shutting down LLM ROS Bridge...")

        self._cancel_sequence_map_wait()

        if self.track_action_timer is not None:
            self.track_action_timer.cancel()

        self.publish_cmd(0.0, 0.0)

        self.rep_socket.close(linger=0)
        self.pub_socket.close(linger=0)
        self.map_image_stream.close()
        self.sub_socket.close(linger=0)

        self.zmq_context.term()

    def listen_for_llm(self):
        while rclpy.ok():
            try:
                request = self.rep_socket.recv_json()
                cmd = request.get("command")

                if cmd == "move_camera":
                    self.servo.set_error(
                        request.get("delta_pan_angle", 0.0),
                        request.get("delta_tilt_angle", 0.0),
                    )
                    self.servo.publish_servo_command(tracking=False)
                    response = {"status": "accepted"}

                elif cmd == "track_action":
                    if self.track_action_timer is None:
                        self.track_action_timer = self.create_timer(
                            0.01, self.track_action_loop
                        )
                    response = {"status": "accepted"}

                elif cmd == "stop_tracking":
                    if self.track_action_timer is not None:
                        self.track_action_timer.cancel()
                        self.track_action_timer = None
                    self.publish_cmd(0.0, 0.0)
                    reset_camera = request.get("reset_camera", True) is not False
                    if reset_camera:
                        self.servo.set_error(
                            self.servo.reset_pan_angle - self.servo.pan_angle,
                            self.servo.reset_tilt_angle - self.servo.tilt_angle,
                        )
                        self.servo.publish_servo_command(tracking=False)
                    response = {"status": "accepted"}

                elif cmd == "follow_action":
                    self.stop_all_motion()
                    if not self.nav_client.wait_for_server(timeout_sec=1.0):
                        response = {
                            "status": "error", "message": "Nav2 not available"
                        }
                    else:
                        self.follow_enabled = True
                        self.follow_action_id = request.get("action_id")
                        self.follow_had_person_track = False
                        self.last_follow_goal = None
                        self.last_follow_goal_at = None
                        response = {"status": "accepted"}

                elif cmd == "stop_follow_action":
                    self.stop_all_motion()
                    response = {"status": "accepted"}

                elif cmd == "navigate_sequence":
                    try:
                        prepared, pose = self.map_image_stream.prepare_map()
                        start_pose = dict(pose)
                        self.stop_all_motion()
                        self._start_sequence(request, prepared, pose)
                        self.previous_navigation_pose = start_pose
                        response = {"status": "accepted"}
                    except (KeyError, TypeError, ValueError) as error:
                        self.sequence = None
                        response = {
                            "status": "error", "message": str(error),
                        }

                elif cmd == "navigate_local":
                    try:
                        prepared, pose = self.map_image_stream.prepare_map()
                        start_pose = dict(pose)
                        self.stop_all_motion()
                        local_command = str(request.get("local_command") or "")
                        face_person = bool(request.get("face_person"))
                        person_pose = self._person_state()
                        if local_command in {
                            "go_to_room", "return_to_initial_place"
                        }:
                            room_id = request.get("room_id")
                            if local_command == "return_to_initial_place":
                                room_id = 1
                            destination = self.map_image_stream.room_registry.destination(
                                room_id
                            )
                        else:
                            destination = self.map_image_stream.logic.resolve_local_destination(
                                prepared,
                                pose,
                                local_command,
                                previous_pose=self.previous_navigation_pose,
                                person_pose=person_pose,
                                face_person=face_person,
                            )
                        error = self.dispatch_navigation_goal(
                            destination,
                            request.get("action_id"),
                            "local",
                        )
                        if error:
                            raise ValueError(error)
                        self.previous_navigation_pose = start_pose
                        response = {"status": "accepted"}
                    except (KeyError, TypeError, ValueError) as error:
                        response = {
                            "status": "error", "message": str(error),
                        }

                elif cmd in {"navigate_to_pose", "navigate_to_approach"}:
                    try:
                        destination = {
                            "x": float(request.get("x", 0.0)),
                            "y": float(request.get("y", 0.0)),
                            "angle": float(request.get("angle", 0.0)),
                            "frame_id": str(
                                request.get("frame_id", "base_footprint")
                            ),
                        }
                        if cmd == "navigate_to_approach":
                            prepared, pose = self.map_image_stream.prepare_map()
                            destination = self.map_image_stream.logic.resolve_approach_destination(
                                prepared,
                                pose,
                                destination["x"],
                                destination["y"],
                                standoff_m=request.get("standoff_m"),
                            )

                        start_pose = (
                            dict(self.robot_pose)
                            if isinstance(self.robot_pose, dict) else None
                        )
                        self.stop_all_motion()
                        navigation_mode = (
                            "approach"
                            if cmd == "navigate_to_approach" else "navigate"
                        )
                        error = self.dispatch_navigation_goal(
                            destination,
                            request.get("action_id"),
                            navigation_mode,
                        )
                        if error:
                            response = {
                                "status": "error", "message": error,
                            }
                        else:
                            if start_pose is not None:
                                self.previous_navigation_pose = start_pose
                            response = {"status": "accepted"}
                    except (KeyError, TypeError, ValueError) as error:
                        response = {
                            "status": "error", "message": str(error),
                        }

                elif cmd == "stop_moving":
                    self.stop_all_motion()
                    response = {"status": "accepted"}

                else:
                    response = {
                        "status": "error",
                        "message": f"Unknown command: {cmd}",
                    }

                self.rep_socket.send_json(response)

            except Exception as error:
                self.get_logger().error(f"Internal Loop Error: {error}")
                try:
                    self.rep_socket.send_json({
                        "status": "error",
                        "message": f"Internal exception: {error}",
                    })
                except Exception as socket_error:
                    self.get_logger().error(
                        f"Could not recover ZMQ state: {socket_error}"
                    )

def main(args=None):
    rclpy.init(args=args)
    node = LLMRosBridge()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Ctrl-C received")

    finally:
        node.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
