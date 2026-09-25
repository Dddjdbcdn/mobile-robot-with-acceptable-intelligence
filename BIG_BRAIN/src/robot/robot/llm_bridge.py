import math
import threading
import json
import struct
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, TwistStamped
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import CameraInfo, Image, Imu, Range, PointCloud2
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
import zmq
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
import time
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from robot.follow_goal_generator import FollowGoalGenerator, FollowGoalSettings
from robot.room_geometry import RoomGeometryEstimator
from robot.map_stream import MapImageStream

class CameraServo():
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
        self.Kp = 0.15

        self.deadband_degrees = 0.5
        self.max_step_degrees = 4

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

    def tracking_snapshot(self):
        with self.command_lock:
            return {
                "pan_angle": self.pan_angle,
                "pan_error": self.last_visual_pan_error,
                "observed_at": self.last_visual_at,
            }

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

    def publish_servo_command(self,tracking=False):
        remaining_pan_angle = 0.0

        with self.command_lock:
            if (
                tracking
                and self.tracking_sequence is not None
                and self.tracking_sequence
                == self.last_applied_tracking_sequence
            ):
                self.delta_pan_angle = 0.0
                self.delta_tilt_angle = 0.0
                return remaining_pan_angle

            delta_pan_angle = self.delta_pan_angle
            delta_tilt_angle = self.delta_tilt_angle
            tracking_timing = self.tracking_timing
            tracking_received_at = self.tracking_received_at
            tracking_received_unix_ns = self.tracking_received_unix_ns
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
            step_pan = delta_pan_angle * self.Kp
            step_tilt = delta_tilt_angle * self.Kp

            step_pan = max(min(step_pan, self.max_step_degrees), -self.max_step_degrees)
            step_tilt = max(min(step_tilt, self.max_step_degrees), -self.max_step_degrees)

        self.pan_angle = self.pan_angle + step_pan
        self.tilt_angle = self.tilt_angle + step_tilt

        if self.pan_angle < self.min_pan_angle: 
            remaining_pan_angle = self.pan_angle - self.min_pan_angle
            self.pan_angle = self.min_pan_angle

        if self.pan_angle > self.max_pan_angle: 
            remaining_pan_angle = self.pan_angle - self.max_pan_angle
            self.pan_angle = self.max_pan_angle

        if self.tilt_angle < self.min_tilt_angle: 
            self.tilt_angle = self.min_tilt_angle

        if self.tilt_angle > self.max_tilt_angle: 
            self.tilt_angle = self.max_tilt_angle

        pan_msg = Float32()
        tilt_msg = Float32()
        reset_tilt_msg = Float32()

        pan_msg.data = self.pan_angle
        tilt_msg.data = self.tilt_angle
        reset_tilt_msg.data = self.reset_tilt_angle

        self.servo_tilt_pub.publish(tilt_msg)
        self.servo_pan_pub.publish(pan_msg)
        # if tracking:
        #     self._record_tracking_latency(
        #         tracking_timing,
        #         tracking_received_at,
        #         tracking_received_unix_ns,
        #     )

        return remaining_pan_angle
        

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

        self.camera_pub_socket = self.zmq_context.socket(zmq.PUB)
        # Keep live camera frames on this machine only.
        self.camera_pub_socket.setsockopt(zmq.SNDHWM, 2)
        self.camera_pub_socket.bind("tcp://127.0.0.1:5558")
        
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
            FollowGoalSettings(
                standoff_m=float(
                    self.get_parameter("follow_standoff_m").value
                ),
                camera_center_deg=self.servo.reset_pan_angle,
                heading_comfort_deg=float(
                    self.get_parameter("follow_heading_comfort_deg").value
                ),
                heading_critical_deg=float(
                    self.get_parameter("follow_heading_critical_deg").value
                ),
            )
        )
        self.map_image_stream = MapImageStream(self, self.tf_buffer, self.zmq_context)

        self.camera_tof_range = 0.0
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Range, '/camera_tof', self.camera_tof_callback, qos)
        self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.astra_color_callback,
            qos,
        )
        self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.astra_depth_callback,
            qos,
        )
        self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.astra_camera_info_callback,
            qos,
        )

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_message = None
        self.robot_pose = None
        self.robot_pose_at = None
        self.room_geometry = None
        self.room_geometry_estimator = RoomGeometryEstimator()
        self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos
        )

        self.state_timer = self.create_timer(0.05, self.state_pub_loop)
        self.robot_pose_timer = self.create_timer(0.1, self.update_robot_pose)
        self.room_geometry_timer = self.create_timer(
            1.0, self.update_room_geometry
        )

        self.current_nav_goal_handle = None
        self.current_nav_action_id = None
        self.move_action_timer = None
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
        self.moving_active = False
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

        self.reset_semantic_memory()

    def reset_semantic_memory(self):
        semantic_memory_path = next(
            (
                parent / "SMALL_BRAIN" / "database" / "semantic_memory.json"
                for parent in Path(__file__).resolve().parents
                if (parent / "SMALL_BRAIN" / "database").is_dir()
            ),
            None,
        )
        if semantic_memory_path is None:
            raise FileNotFoundError("Could not locate semantic_memory.json")

        with semantic_memory_path.open("w", encoding="utf-8") as file:
            json.dump({"objects": {}, "version": 1}, file, indent=2)
            file.write("\n")

        self.get_logger().info("Semantic memory cleared for the new map")

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
                "room_geometry": self.room_geometry,
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

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y ** 2 + rotation.z ** 2),
        )
        pose = {
            "x": translation.x,
            "y": translation.y,
            "yaw": yaw,
            "frame_id": "map",
        }

        if pose is not None:
            self.robot_pose = pose
            self.robot_pose_at = time.monotonic()

    def map_callback(self, msg):
        self.map_message = msg

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

    @staticmethod
    def _angle_distance(first, second):
        return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)

    @staticmethod
    def _angle_distance_signed(first, second):
        return (first - second + math.pi) % (2.0 * math.pi) - math.pi

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
        if not camera_fresh:
            lidar_deviation = self._angle_distance_signed(
                math.atan2(
                    float(person.y) - float(self.robot_pose["y"]),
                    float(person.x) - float(self.robot_pose["x"]),
                ),
                float(self.robot_pose["yaw"]),
            )
            desired_pan = self.servo.reset_pan_angle + math.degrees(
                lidar_deviation
            )
            desired_pan = max(
                self.servo.min_pan_angle,
                min(self.servo.max_pan_angle, desired_pan),
            )
            self.servo.set_error(
                desired_pan - float(camera["pan_angle"]),
                0.0,
            )
            self.servo.publish_servo_command(tracking=True)
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

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = generated.x
        goal_pose.pose.position.y = generated.y
        goal_pose.pose.orientation.z = math.sin(generated.yaw * 0.5)
        goal_pose.pose.orientation.w = math.cos(generated.yaw * 0.5)

        should_publish = self.last_follow_goal is None
        if self.last_follow_goal is not None:
            previous_x, previous_y, previous_yaw = self.last_follow_goal
            position_change = math.hypot(
                generated.x - previous_x,
                generated.y - previous_y,
            )
            yaw_change = self._angle_distance(generated.yaw, previous_yaw)
            refresh_due = (
                self.last_follow_goal_at is None
                or now - self.last_follow_goal_at >= 0.8
            )
            should_publish = (
                position_change >= 0.10
                or yaw_change >= math.radians(5.0)
                or refresh_due
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

    def update_room_geometry(self):
        pose = self.robot_pose
        if self.map_message is None or pose is None:
            return
        try:
            self.room_geometry = self.room_geometry_estimator.estimate(
                self.map_message, pose["x"], pose["y"], pose["yaw"]
            )
        except Exception as error:
            self.get_logger().warning(
                f"Could not estimate room geometry: {error}"
            )

    def camera_tof_callback(self, msg):
        self.camera_tof_range = msg.range

    @staticmethod
    def _serialize_ros_image(msg):
        metadata = json.dumps(
            {
                "width": msg.width,
                "height": msg.height,
                "step": msg.step,
                "encoding": msg.encoding,
                "is_bigendian": msg.is_bigendian,
                "frame_id": msg.header.frame_id,
                "stamp_sec": msg.header.stamp.sec,
                "stamp_nanosec": msg.header.stamp.nanosec,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return struct.pack("!I", len(metadata)) + metadata + bytes(msg.data)

    def _publish_astra_stream(self, topic, payload):
        """Publish a typed camera payload without blocking ROS callbacks."""
        try:
            self.camera_pub_socket.send_multipart(
                [topic, payload], flags=zmq.NOBLOCK
            )
        except zmq.Again:
            pass

    def astra_color_callback(self, msg):
        self._publish_astra_stream(
            b"astra/color", self._serialize_ros_image(msg)
        )

    def astra_depth_callback(self, msg):
        self._publish_astra_stream(
            b"astra/depth", self._serialize_ros_image(msg)
        )

    def astra_camera_info_callback(self, msg):
        payload = json.dumps(
            {
                "width": msg.width,
                "height": msg.height,
                "frame_id": msg.header.frame_id,
                "k": list(msg.k),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self._publish_astra_stream(b"astra/camera_info", payload)

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

    def move_action_loop(
        self,
        lin_vel,
        ang_vel,
        fwd_dur,
        rot_dur,
        start_time,
        action_id,
        send_event=False,
    ):
        elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
        
        if elapsed >= max(fwd_dur,rot_dur):
            self.publish_cmd(0.0, 0.0)
            self.move_action_timer.cancel()
            self.move_action_timer = None
            if send_event:
                self.send_event("move_action", "completed", action_id)
            self.moving_active = False
            return

        lin = lin_vel if elapsed < fwd_dur else 0.0
        ang = ang_vel if elapsed < rot_dur else 0.0
        self.publish_cmd(lin, ang)

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
        if self.move_action_timer:
            self.move_action_timer.cancel()
            self.move_action_timer = None
            self.moving_active = False
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

    def track_action_loop(self, start_time):
        remaining_pan_angle = self.servo.publish_servo_command(
            tracking=True
        )

        if self.navigation_active or self.moving_active:
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

        if self.move_action_timer is not None:
            self.move_action_timer.cancel()

        if self.track_action_timer is not None:
            self.track_action_timer.cancel()

        self.publish_cmd(0.0, 0.0)

        self.rep_socket.close(linger=0)
        self.pub_socket.close(linger=0)
        self.camera_pub_socket.close(linger=0)
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
                
                elif cmd == "move_action":
                    self.stop_all_motion()
                    action_id = request.get("action_id")
                    
                    lin_vel = float(request.get("linear_velocity", 0.0))
                    dist = float(request.get("distance", 0.0))
                    ang_vel = float(request.get("angular_velocity", 0.0))
                    angle = float(request.get("angle", 0.0))
                    fwd_dur = abs(dist / lin_vel) if lin_vel != 0 else 0.0
                    rot_dur = abs(angle / ang_vel) if ang_vel != 0 else 0.0
                    start_time = self.get_clock().now()
        
                    self.move_action_timer = self.create_timer(
                        0.05,
                        lambda: self.move_action_loop(
                            lin_vel,
                            ang_vel,
                            fwd_dur,
                            rot_dur,
                            start_time,
                            action_id,
                            True,
                        ),
                    )
                    self.moving_active = True
                    self.rep_socket.send_json({"status": "accepted", "message": "Blind move started"})

                elif cmd == "track_action":
                    start_time = self.get_clock().now()
                    if self.track_action_timer is None:
                        self.track_action_timer = self.create_timer(0.01, lambda: self.track_action_loop(start_time))

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
                    # A new follow session must wait for a pose produced after
                    # this command, never reuse the previous person's last pose.
                    self.latest_person_pose_map = None
                    self.latest_person_pose_at = None
                    self.pending_person_pose = None
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

                elif cmd in {"navigate_to_pose", "navigate_to_approach"}:
                    x = float(request.get("x", 0.0))
                    y = float(request.get("y", 0.0))
                    angle = float(request.get("angle", 0.0))
                    frame_id = str(request.get("frame_id", "base_footprint"))
                    destination = None

                    if cmd == "navigate_to_approach":
                        with self.map_image_stream.state_lock:
                            prepared = self.map_image_stream.prepared
                        if prepared is None:
                            self.rep_socket.send_json({
                                "status": "error", "message": "Map is not ready"
                            })
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
                            destination = self.map_image_stream.logic.resolve_approach_destination(
                                prepared, pose, x, y,
                                standoff_m=request.get("standoff_m"),
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

                    pose_in = PoseStamped()
                    pose_in.header.frame_id = frame_id
                    # A zero stamp requests the latest complete TF chain.
                    pose_in.header.stamp = rclpy.time.Time().to_msg()
                    pose_in.pose.position.x = x
                    pose_in.pose.position.y = y
                    pose_in.pose.position.z = 0.0
                    pose_in.pose.orientation.z = math.sin(angle / 2.0)
                    pose_in.pose.orientation.w = math.cos(angle / 2.0)

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

                    response = {
                        "status": "accepted",
                        "message": "Nav2 Goal Dispatched",
                    }
                    if destination is not None:
                        response["destination"] = destination
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
