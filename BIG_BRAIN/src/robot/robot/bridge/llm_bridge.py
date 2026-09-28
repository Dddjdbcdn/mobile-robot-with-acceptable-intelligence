import math
import threading
import json
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Range
from nav2_msgs.action import NavigateToPose
import zmq
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
# Registers geometry message conversions with tf2's Python transform system.
import tf2_geometry_msgs  # noqa: F401
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy
from robot.map.map_stream import MapImageStream
from robot.utilities.camera_servo import CameraServo
from robot.person_pose.follow_goal_generator import FollowGoalGenerator


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

        self.cmd_pub = self.create_publisher(TwistStamped, '/diff_drive_controller/cmd_vel', 10)
        self.servo_pan_pub = self.create_publisher(Float32, '/stm32/servo_pan', 10)
        self.servo_tilt_pub = self.create_publisher(Float32, '/stm32/servo_tilt', 10)
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
        self.declare_parameter("follow_standoff_m", 0.90)
        self.declare_parameter("follow_heading_comfort_deg", 20.0)
        self.declare_parameter("follow_heading_critical_deg", 45.0)
        self.follow_goal_generator = FollowGoalGenerator(
            standoff_m=float(self.get_parameter("follow_standoff_m").value),
            camera_center_deg=self.servo.reset_pan_angle,
            heading_comfort_deg=float(
                self.get_parameter("follow_heading_comfort_deg").value
            ),
            heading_critical_deg=float(
                self.get_parameter("follow_heading_critical_deg").value
            ),
        )
        self.map_image_stream = MapImageStream(self, self.tf_buffer, self.zmq_context)

        self.camera_tof_range = 0.0
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Range, '/camera_tof', self.camera_tof_callback, qos)
        self.robot_pose = None
        self.robot_pose_at = None

        self.state_timer = self.create_timer(0.05, self.state_pub_loop)
        self.robot_pose_timer = self.create_timer(0.1, self.update_robot_pose)

        self.current_nav_goal_handle = None
        self.current_nav_action_id = None
        self.track_action_timer = None
        self.navigation_active = False
        self.navigation_mode = None
        self.nav_goal_generation = 0
        self.follow_enabled = False
        self.follow_action_id = None
        self.person_tracker_state = None
        self.follow_had_person_track = False
        self.latest_person_pose_map = None
        self.latest_person_pose_at = None
        self.pending_person_pose = None
        self.last_follow_goal = None
        self.last_follow_goal_at = None
        self.previous_navigation_pose = None
        self.tracking_body_active = False
        self.max_tracking_time = 10.0
        self.create_subscription(
            PoseStamped, "/person_pose", self.person_pose_callback, 10
        )
        self.create_subscription(
            String,
            "/person_tracker/status",
            self.person_tracker_status_callback,
            10,
        )
        self.person_pose_timer = self.create_timer(
            0.1, self.process_pending_person_pose
        )
        self.follow_goal_timer = self.create_timer(
            0.4, self.update_follow_goal
        )

        self.zmq_thread = threading.Thread(target=self.listen_for_llm, daemon=True)
        self.zmq_thread.start()

        self.background_listener_thread = threading.Thread(target=self.background_listener, daemon=True)
        self.background_listener_thread.start()

    def publish_cmd(self, linear_x, angular_z):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def state_pub_loop(self):
        with self.pub_lock:
            self.pub_socket.send_json(
                {
                "type": "state",
                "camera_tof_range": self.camera_tof_range,
                "servo_pan_angle": self.servo.pan_angle,
                "servo_tilt_angle": self.servo.tilt_angle,
                "robot_pose": self.robot_pose,
                "person_pose": self._person_state(),
                "person_tracker_state": self.person_tracker_state,
                }
            )

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
        previous = self.person_tracker_state
        self.person_tracker_state = state
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
            pose_map = (
                msg
                if msg.header.frame_id == "map"
                else self.tf_buffer.transform(
                    msg,
                    "map",
                    timeout=rclpy.duration.Duration(seconds=0.05),
                )
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

        if not self.navigation_active:
            self.start_follow_navigation(goal_pose)
            should_publish = True
        elif self.navigation_mode == "follow" and should_publish:
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

    def send_event(self, event_name, status_msg, action_id=None):
        with self.pub_lock:
            payload = {
                "type": "event",
                "event": event_name,
                "status": status_msg,
            }
            if action_id is not None:
                payload["action_id"] = action_id
            self.pub_socket.send_json(payload)
            self.get_logger().info(f"Broadcasted to LLM: {payload}")

    def nav_goal_response_cb(
        self, future, action_id, navigation_mode, generation
    ):
        goal_handle = future.result()
        if generation != self.nav_goal_generation:
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
            return
        if not goal_handle.accepted:
            self.navigation_active = False
            self.navigation_mode = None
            self.current_nav_action_id = None
            if navigation_mode == "follow":
                self.follow_enabled = False
                self.follow_action_id = None
            self.send_event(
                "navigation",
                "Goal Rejected by Nav2",
                action_id,
            )
            return

        self.current_nav_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda completed: self.nav_result_cb(
                completed, goal_handle, action_id, navigation_mode, generation
            )
        )

    def nav_result_cb(
        self, future, goal_handle, action_id, navigation_mode, generation
    ):
        status = future.result().status
        is_current = (
            generation == self.nav_goal_generation
            and self.current_nav_goal_handle is goal_handle
        )
        if not is_current:
            return

        self.navigation_active = False
        self.navigation_mode = None
        self.current_nav_goal_handle = None
        self.current_nav_action_id = None
        if navigation_mode == "follow":
            self.follow_enabled = False
            self.follow_action_id = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.send_event("navigation", "Goal Reached", action_id)
        else:
            self.send_event(
                "navigation",
                f"Failed or Canceled (Status {status})",
                action_id,
            )

    def stop_all_motion(self):
        self.nav_goal_generation += 1
        if self.current_nav_goal_handle:
            self.current_nav_goal_handle.cancel_goal_async()
        self.current_nav_goal_handle = None
        self.current_nav_action_id = None
        self.navigation_active = False
        self.navigation_mode = None
        self.follow_enabled = False
        self.follow_action_id = None
        self.follow_had_person_track = False
        self.pending_person_pose = None
        self.last_follow_goal = None
        self.last_follow_goal_at = None

        self.publish_cmd(0.0, 0.0)

    def start_follow_navigation(self, pose_map):
        """Start one long-running Nav2 goal; GoalUpdater handles later poses."""
        if not self.follow_enabled or self.navigation_active:
            return

        action_id = self.follow_action_id
        self.navigation_active = True
        self.navigation_mode = "follow"
        self.current_nav_action_id = action_id
        self.nav_goal_generation += 1
        generation = self.nav_goal_generation

        goal = NavigateToPose.Goal()
        goal.pose = pose_map
        goal.behavior_tree = self.follow_bt
        send_goal_future = self.nav_client.send_goal_async(goal)
        send_goal_future.add_done_callback(
            lambda completed: self.nav_goal_response_cb(
                completed, action_id, "follow", generation
            )
        )

    def track_action_loop(self):
        remaining_pan_angle = self.servo.publish_servo_command(
            tracking=True
        )

        if self.navigation_active:
            self.tracking_body_active = False
            return

        remaining_rad = math.radians(remaining_pan_angle)

        if abs(remaining_rad) <= math.radians(1.0):
            if self.tracking_body_active:
                self.publish_cmd(0.0, 0.0)
                self.tracking_body_active = False
            return

        body_kp = 100.0
        ang_vel = body_kp * remaining_rad

        max_ang_vel = 1.0
        ang_vel = max(min(ang_vel, max_ang_vel),-max_ang_vel)

        self.publish_cmd(0.0, ang_vel)
        self.tracking_body_active = True

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

                if cmd == 'move_camera':
                    self.servo.set_error(
                        request.get("delta_pan_angle", 0.0),
                        request.get("delta_tilt_angle", 0.0),
                    )

                    self.servo.publish_servo_command(tracking=False)
                    self.rep_socket.send_json(
                        {
                            "status": "accepted",
                            "pan_angle": self.servo.pan_angle,
                            "tilt_angle": self.servo.tilt_angle,
                        }
                    )
                
                elif cmd == "track_action":
                    if self.track_action_timer is None:
                        self.track_action_timer = self.create_timer(
                            0.01, self.track_action_loop
                        )

                    self.rep_socket.send_json({"status": "accepted", "message": "Object is being tracked"})

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

                    self.rep_socket.send_json({
                        "status": "accepted",
                        "message": "Object tracking is stopped",
                        "camera_reset": reset_camera,
                    })

                elif cmd == "follow_action":
                    self.stop_all_motion()
                    if not self.nav_client.wait_for_server(timeout_sec=1.0):
                        self.rep_socket.send_json({
                            "status": "error", "message": "Nav2 not available"
                        })
                        continue

                    self.follow_enabled = True
                    self.follow_action_id = request.get("action_id")
                    self.follow_had_person_track = False
                    self.last_follow_goal = None
                    self.last_follow_goal_at = None
                    self.rep_socket.send_json({
                        "status": "accepted",
                        "message": "Waiting for fresh lidar person pose",
                    })

                elif cmd == "stop_follow_action":
                    self.stop_all_motion()
                    self.rep_socket.send_json({
                        "status": "accepted",
                        "message": "Following stopped",
                    })

                elif cmd in {
                    "navigate_to_pose", "navigate_to_approach",
                    "navigate_local",
                }:
                    x = float(request.get("x", 0.0))
                    y = float(request.get("y", 0.0))
                    angle = float(request.get("angle", 0.0))
                    frame_id = str(request.get("frame_id", "base_footprint"))
                    destination = None

                    if cmd in {"navigate_to_approach", "navigate_local"}:
                        with self.map_image_stream.state_lock:
                            prepared = self.map_image_stream.prepared
                        if prepared is None:
                            self.rep_socket.send_json({
                                "status": "error", "message": "Map is not ready"
                            })
                            continue
                        after_revision = request.get("after_map_revision")
                        after_map_received_at = request.get(
                            "after_map_received_at_unix_ns"
                        )
                        map_revision = int(prepared.get(
                            "map_revision",
                            prepared.get("prepared_at_unix_ns", 0),
                        ))
                        map_received_at = int(prepared.get(
                            "map_received_at_unix_ns", 0
                        ))
                        room_step_commands = {
                            "exit_room", "go_to_another_room"
                        }
                        if (
                            cmd == "navigate_local"
                            and request.get("local_command") in room_step_commands
                            and (
                                (
                                    after_revision is not None
                                    and map_revision <= int(after_revision)
                                )
                                or (
                                    after_map_received_at is not None
                                    and map_received_at
                                    <= int(after_map_received_at)
                                )
                            )
                        ):
                            self.rep_socket.send_json({"status": "map_updating"})
                            continue
                        map_frame = str(prepared.get("frame_id") or "map")
                        pose = self.map_image_stream._pose(map_frame)
                        if pose is None:
                            self.rep_socket.send_json({
                                "status": "error",
                                "message": f"TF unavailable: {map_frame} -> base_footprint",
                            })
                            continue
                        try:
                            if cmd == "navigate_to_approach":
                                destination = self.map_image_stream.logic.resolve_approach_destination(
                                    prepared, pose, x, y,
                                    standoff_m=request.get("standoff_m"),
                                )
                            elif cmd == "navigate_local":
                                local_command = str(
                                    request.get("local_command") or ""
                                )
                                if local_command in room_step_commands:
                                    exit_state = dict(
                                        request.get("exit_state") or {}
                                    )
                                    if exit_state.get("origin_room_id") is None:
                                        current_room = (
                                            self.map_image_stream.room_registry.room_at(
                                                prepared, pose
                                            )
                                            or self.map_image_stream.room_registry.ensure_startup_room(
                                                prepared, pose
                                            )
                                        )
                                        if current_room is not None:
                                            exit_state["origin_room_id"] = current_room[
                                                "room_id"
                                            ]
                                    exit_step = self.map_image_stream.logic.resolve_exit_room_step(
                                        prepared, pose, exit_state
                                    )
                                    if exit_step["complete"]:
                                        room = None
                                        if local_command == "go_to_another_room":
                                            room = self.map_image_stream.room_registry.observe_room(
                                                prepared,
                                                pose,
                                                connected_from=exit_step["state"].get(
                                                    "origin_room_id"
                                                ),
                                            )
                                        self.rep_socket.send_json({
                                            "status": "complete",
                                            "exit_phase": exit_step["phase"],
                                            "exit_state": exit_step["state"],
                                            "map_revision": map_revision,
                                            "map_received_at_unix_ns": map_received_at,
                                            "room": room,
                                            "room_graph": self.map_image_stream.room_registry.snapshot(),
                                        })
                                        continue
                                    if (
                                        local_command == "exit_room"
                                        and exit_step["phase"] == "cross_room_boundary"
                                    ):
                                        self.rep_socket.send_json({
                                            "status": "complete",
                                            "exit_phase": "crossing_ready",
                                            "exit_state": exit_step["state"],
                                            "crossing_destination": exit_step["destination"],
                                            "map_revision": map_revision,
                                            "map_received_at_unix_ns": map_received_at,
                                            "room_graph": self.map_image_stream.room_registry.snapshot(),
                                        })
                                        continue
                                    destination = exit_step["destination"]
                                elif local_command in {
                                    "go_to_room", "return_to_initial_place"
                                }:
                                    room_id = (
                                        1 if local_command == "return_to_initial_place"
                                        else request.get("room_id")
                                    )
                                    destination = self.map_image_stream.room_registry.destination(
                                        room_id
                                    )
                                else:
                                    destination = self.map_image_stream.logic.resolve_local_destination(
                                        prepared, pose, local_command,
                                        previous_pose=self.previous_navigation_pose,
                                        person_pose=self._person_state(),
                                    )
                        except (KeyError, TypeError, ValueError) as error:
                            self.rep_socket.send_json({
                                "status": "error", "message": str(error)
                            })
                            continue
                        x = destination["x"]
                        y = destination["y"]
                        angle = destination["angle"]
                        frame_id = destination["frame_id"]

                    # A zero stamp requests the latest complete TF chain.
                    pose_in = _make_pose_stamped(
                        x, y, angle, frame_id, rclpy.time.Time().to_msg()
                    )

                    if pose_in.header.frame_id == "map":
                        pose_map = pose_in
                    else:
                        try:
                            timeout = rclpy.duration.Duration(seconds=0.1)
                            pose_map = self.tf_buffer.transform(
                                pose_in, "map", timeout=timeout
                            )
                        except Exception as error:
                            self.rep_socket.send_json({
                                "status": "error",
                                "message": f"TF Transform failed: {error}",
                            })
                            continue

                    # GoalUpdater rejects zero-stamped messages and uses the stamp
                    # to decide whether an update is newer than the seed goal.
                    pose_map.header.stamp = self.get_clock().now().to_msg()

                    start_pose = dict(self.robot_pose) if isinstance(self.robot_pose, dict) else None
                    self.stop_all_motion()
                    if not self.nav_client.wait_for_server(timeout_sec=1.0):
                        self.rep_socket.send_json({
                            "status": "error", "message": "Nav2 not available"
                        })
                        continue

                    action_id = request.get("action_id")
                    navigation_mode = "navigate"
                    self.navigation_active = True
                    self.navigation_mode = navigation_mode
                    self.current_nav_action_id = action_id
                    self.nav_goal_generation += 1
                    generation = self.nav_goal_generation

                    goal = NavigateToPose.Goal()
                    goal.pose = pose_map
                    goal.behavior_tree = self.dj_bt
                    send_goal_future = self.nav_client.send_goal_async(goal)
                    send_goal_future.add_done_callback(
                        lambda completed: self.nav_goal_response_cb(
                            completed, action_id, navigation_mode, generation
                        )
                    )
                    if start_pose is not None:
                        self.previous_navigation_pose = start_pose

                    response = {
                        "status": "accepted",
                        "message": "Nav2 Goal Dispatched",
                    }
                    if destination is not None:
                        response["destination"] = destination
                    if (
                        cmd == "navigate_local"
                        and request.get("local_command") in room_step_commands
                    ):
                        response.update({
                            "exit_phase": exit_step["phase"],
                            "exit_state": exit_step["state"],
                            "map_revision": map_revision,
                            "map_received_at_unix_ns": map_received_at,
                        })
                        if exit_step.get("recovery_reason") is not None:
                            response["recovery_reason"] = exit_step[
                                "recovery_reason"
                            ]
                    self.rep_socket.send_json(response)

                elif cmd == "stop_moving":
                    self.stop_all_motion()
                    self.rep_socket.send_json({"status": "accepted", "message": "All motion stopped"})

                else:
                    self.rep_socket.send_json({"status": "error", "message": "Unknown command"})

            except Exception as e:
                self.get_logger().error(f"Internal Loop Error: {e}")
                try:
                    self.rep_socket.send_json({"status": "error", "message": f"Internal exception: {str(e)}"})
                except Exception as zmq_e:
                    self.get_logger().error(f"Could not recover ZMQ state: {zmq_e}")

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
