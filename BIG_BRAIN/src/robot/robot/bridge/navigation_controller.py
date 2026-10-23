"""Navigation behaviors used by the LLM bridge.

The controller exposes the navigation vocabulary in one place:

* ``follow_action`` continuously follows the latest person or hand target.
* ``navigate_sequence`` resolves map-aware, multi-goal room behaviors.
* ``navigate_local`` resolves a named destination from the current map.
* ``navigate_to_pose`` sends an explicit pose to Nav2.
* ``navigate_to_approach`` finds a reachable pose near a requested target.

It is deliberately a helper owned by the bridge, rather than another ROS node.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose, Spin
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from robot.person_pose.follow_goal_generator import FollowGoalGenerator


NAVIGATION_COMMANDS = {
    "follow_action",
    "set_follow_target_override",
    "clear_follow_target_override",
    "navigate_sequence",
    "navigate_local",
    "navigate_to_pose",
    "navigate_to_approach",
}
SEQUENCE_COMMANDS = {
    "open_space_middle", "exit_room", "go_to_another_room",
}
ROOM_SEQUENCE_COMMANDS = {"exit_room", "go_to_another_room"}


@dataclass
class ActiveNavigation:
    mode: str
    action_id: str | None
    destination: dict | None = None
    goal_handle: object | None = None


@dataclass
class ActivePersonRecoverySpin:
    """A Nav2 spin requested by one parent follow action."""

    action_id: str
    parent_action_id: str
    target_yaw: float
    goal_handle: object | None = None
    angular_distance_traveled: float = 0.0


def _make_pose(x, y, yaw, frame_id, stamp):
    pose = PoseStamped()
    pose.header.frame_id = str(frame_id)
    pose.header.stamp = stamp
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.orientation.z = math.sin(float(yaw) * 0.5)
    pose.pose.orientation.w = math.cos(float(yaw) * 0.5)
    return pose


class NavigationController:
    """Own Nav2 actions and the robot's available navigation behaviors."""

    FOLLOW_RECOVERY_WINDOW_SECONDS = 3.0
    FOLLOW_PERSON_RECOVERY_ARRIVAL_M = 0.10
    PERSON_LIDAR_RECOVERY_SWEEP_RAD = 2.0 * math.pi
    PERSON_VISUAL_RECOVERY_SWEEP_RAD = math.pi
    PERSON_VISUAL_RECOVERY_TIMEOUT_SECONDS = 8.0
    FATAL_FOLLOW_ERROR_CODES = {101, 103, 201}
    MAP_UPDATE_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        node,
        tf_buffer,
        map_stream,
        send_event,
        stop_robot,
        get_robot_pose,
        get_person_state,
    ):
        self.node = node
        self.logger = node.get_logger()
        self.tf_buffer = tf_buffer
        self.map_stream = map_stream
        self.send_event = send_event
        self.stop_robot = stop_robot
        self.get_robot_pose = get_robot_pose
        self.get_person_state = get_person_state

        self.nav_client = ActionClient(node, NavigateToPose, "navigate_to_pose")
        self.spin_client = ActionClient(node, Spin, "spin")
        self.goal_update_pub = node.create_publisher(PoseStamped, "/goal_update", 10)
        selector_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.goal_checker_pub = node.create_publisher(
            String, "/goal_checker_selector", selector_qos
        )
        config_dir = Path(get_package_share_directory("robot")) / "config"
        self.navigate_bt = str(config_dir / "behavior_navigate.xml")
        self.follow_bt = str(config_dir / "behavior_follow.xml")

        self.active = None
        self.previous_pose = None
        self.sequence = None
        self.sequence_map_timer = None

        self.follow_enabled = False
        self.follow_action_id = None
        self.follow_had_person_track = False
        self.person_tracker_state = None
        self.person_tracker_status = {}
        self.follow_person_recovery_arrived = False
        self.follow_target_override = None
        self.follow_target_override_at = None
        self.last_follow_goal = None
        self.follow_recovery_deadline = None
        self.follow_recovery_failure = None
        self.follow_goal_generator = FollowGoalGenerator()
        self.follow_timer = node.create_timer(0.5, self.update_follow)
        self.follow_timer.cancel()

        # Shared LiDAR/visual person recovery -------------------------
        self.person_recovery_spin = None
        self.person_visual_recovery_active = False
        self.person_recovery_remaining_yaw = 0.0

    @property
    def is_active(self):
        return (
            self.active is not None
            or self.person_recovery_spin is not None
            or self.follow_enabled
        )

    def handle_request(self, request):
        """Dispatch one navigation command from SMALL_BRAIN."""
        command = request.get("command")
        if command == "follow_action":
            return self.start_follow(request)
        if command == "set_follow_target_override":
            return self.set_follow_target_override(request)
        if command == "clear_follow_target_override":
            return self.clear_follow_target_override(request)
        if command == "navigate_sequence":
            return self.navigate_sequence(request)
        if command == "navigate_local":
            return self.navigate_local(request)
        if command in {"navigate_to_pose", "navigate_to_approach"}:
            return self.navigate_to_pose(request, approach=command.endswith("approach"))
        return {"status": "error", "message": f"Unknown navigation command: {command}"}

    def stop(self):
        """Cancel every navigation behavior and clear its transient state."""
        if self.sequence_map_timer is not None:
            self.sequence_map_timer.cancel()
            self.node.destroy_timer(self.sequence_map_timer)
            self.sequence_map_timer = None
        active = self.active
        self.active = None
        if active is not None and active.goal_handle is not None:
            active.goal_handle.cancel_goal_async()
        recovery_spin = getattr(self, "person_recovery_spin", None)
        self.person_recovery_spin = None
        if recovery_spin is not None and recovery_spin.goal_handle is not None:
            recovery_spin.goal_handle.cancel_goal_async()
        self.person_visual_recovery_active = False
        self.person_recovery_remaining_yaw = 0.0
        self.follow_enabled = False
        self.follow_timer.cancel()
        self.follow_action_id = None
        self.follow_had_person_track = False
        self.person_tracker_state = None
        self.person_tracker_status = {}
        self.follow_person_recovery_arrived = False
        self.follow_target_override = None
        self.follow_target_override_at = None
        self.last_follow_goal = None
        self.follow_recovery_deadline = None
        self.follow_recovery_failure = None
        self.follow_goal_generator.reset()
        self.sequence = None
        self.stop_robot()

    def shutdown(self):
        self.stop()

    def note_person_tracker_state(self, state, previous_state, status=None):
        """Remember whether follow navigation is chasing a held person pose."""
        self.person_tracker_state = state
        status = dict(status or {})
        self.person_tracker_status = status
        if state == "tracking":
            self.follow_person_recovery_arrived = False
        elif state == "recovering" and previous_state != "recovering":
            self.follow_person_recovery_arrived = False
        if state == "visual_recovery" and previous_state != "visual_recovery":
            if not self.follow_enabled:
                return
            self._start_or_resume_visual_recovery(status)
            return
        if not self.follow_enabled:
            return
        if getattr(self, "person_visual_recovery_active", False):
            if state == "validating" and state != previous_state:
                self.person_recovery_remaining_yaw = (
                    self._stop_person_recovery_spin()
                )
                return
            if state in {"reacquiring", "awaiting_seed"}:
                return
            if state in {"seeded", "tracking"}:
                self._complete_visual_recovery()
                return
            if state == "lost":
                self._fail_visual_recovery(
                    "LIDAR_TRACK_LOST",
                    {"tracker_state": "lost"},
                )
                return
        if state in {"tracking", "recovering"}:
            self.follow_had_person_track = True
        if (
            state == "lost"
            and previous_state != "lost"
            and self.follow_had_person_track
        ):
            self.send_event("person_tracker", "lost", self.follow_action_id)

    # Shared LiDAR/visual person recovery -----------------------------

    def _pause_follow_for_visual_recovery(self):
        """Stop follow motion before the shared local person-search spin."""
        self.follow_timer.cancel()
        active = self.active
        if active is not None and active.mode == "follow":
            self.active = None
            if active.goal_handle is not None:
                active.goal_handle.cancel_goal_async()
        self.stop_robot()

    def _start_or_resume_visual_recovery(self, status, *, relocated=False):
        """Start the shared local search for LiDAR or visual track loss."""
        if self.person_recovery_spin is not None:
            return
        if not self.person_visual_recovery_active:
            self._pause_follow_for_visual_recovery()
            self.person_visual_recovery_active = True
            if relocated:
                # The saved visual bearing was measured before navigation and
                # is no longer relative to the robot's arrival pose. Search a
                # full circle instead of assigning meaning to its stale sign.
                target_yaw = self.PERSON_LIDAR_RECOVERY_SWEEP_RAD
            else:
                direction = int(status.get("visual_search_direction") or 0)
                last_bearing = status.get("last_visual_base_bearing_rad")
                try:
                    last_bearing = float(last_bearing)
                except (TypeError, ValueError):
                    last_bearing = 0.0
                if not math.isfinite(last_bearing):
                    last_bearing = 0.0
                if direction == 0:
                    direction = 1 if last_bearing >= 0.0 else -1
                direction = 1 if direction > 0 else -1
                target_yaw = direction * max(
                    self.PERSON_VISUAL_RECOVERY_SWEEP_RAD,
                    abs(last_bearing),
                )
        else:
            target_yaw = self.person_recovery_remaining_yaw
            if abs(target_yaw) < 1e-3:
                direction = int(status.get("visual_search_direction") or 1)
                direction = 1 if direction > 0 else -1
                target_yaw = (
                    direction * self.PERSON_VISUAL_RECOVERY_SWEEP_RAD
                )
        self.person_recovery_remaining_yaw = target_yaw
        self._start_person_recovery_spin(target_yaw)

    def _start_person_recovery_spin(self, target_yaw):
        parent_action_id = self.follow_action_id
        if not parent_action_id:
            return False
        if not self.spin_client.wait_for_server(timeout_sec=1.0):
            self._fail_visual_recovery(
                "PERSON_RECOVERY_SPIN_UNAVAILABLE",
                {"message": "Nav2 spin behavior is unavailable"},
            )
            return False

        recovery = ActivePersonRecoverySpin(
            f"{parent_action_id}:visual-recovery",
            parent_action_id,
            target_yaw,
        )
        self.person_recovery_spin = recovery

        goal = Spin.Goal()
        goal.target_yaw = target_yaw
        timeout = self.PERSON_VISUAL_RECOVERY_TIMEOUT_SECONDS
        seconds = int(timeout)
        goal.time_allowance = DurationMsg(
            sec=seconds,
            nanosec=int((timeout - seconds) * 1_000_000_000),
        )
        future = self.spin_client.send_goal_async(
            goal,
            feedback_callback=lambda feedback: (
                self._person_recovery_spin_feedback(feedback, recovery)
            ),
        )
        future.add_done_callback(
            lambda completed: self._person_recovery_spin_response(
                completed, recovery
            )
        )
        return True

    def _stop_person_recovery_spin(self):
        """Cancel the local spin and return its signed untravelled yaw."""
        recovery = getattr(self, "person_recovery_spin", None)
        if recovery is None:
            self.stop_robot()
            return 0.0
        self.person_recovery_spin = None
        if recovery.goal_handle is not None:
            recovery.goal_handle.cancel_goal_async()
        self.stop_robot()
        remaining = max(
            0.0,
            abs(recovery.target_yaw)
            - abs(recovery.angular_distance_traveled),
        )
        return math.copysign(remaining, recovery.target_yaw)

    def _complete_visual_recovery(self):
        """Resume follow after either LiDAR or visual reacquisition."""
        self._stop_person_recovery_spin()
        self.person_visual_recovery_active = False
        self.person_recovery_remaining_yaw = 0.0
        self.follow_had_person_track = True
        self.last_follow_goal = None
        self.follow_recovery_deadline = None
        self.follow_recovery_failure = None
        self.follow_timer.reset()

    def _fail_visual_recovery(self, reason_code, data=None):
        """End follow and report the terminal shared-search failure."""
        action_id = self.follow_action_id
        recovery = self.person_recovery_spin
        self.person_recovery_spin = None
        if recovery is not None and recovery.goal_handle is not None:
            recovery.goal_handle.cancel_goal_async()
        self.person_visual_recovery_active = False
        self.person_recovery_remaining_yaw = 0.0
        self.follow_enabled = False
        self.follow_timer.cancel()
        self.follow_action_id = None
        self.follow_had_person_track = False
        self.follow_person_recovery_arrived = False
        self.follow_target_override = None
        self.follow_target_override_at = None
        self.last_follow_goal = None
        self.follow_recovery_deadline = None
        self.follow_recovery_failure = None
        self.follow_goal_generator.reset()
        self.stop_robot()
        self.send_event(
            "navigation",
            "failed",
            action_id,
            mode="person_recovery_spin",
            outcome=(
                "person_recovery_sweep_exhausted"
                if reason_code == "PERSON_VISUAL_RECOVERY_EXHAUSTED"
                else "person_recovery_spin_failed"
            ),
            reason_code=reason_code,
            data=dict(data or {}),
        )

    def _person_recovery_spin_feedback(self, feedback, recovery):
        if self.person_recovery_spin is not recovery:
            return
        recovery.angular_distance_traveled = float(
            feedback.feedback.angular_distance_traveled
        )

    def _person_recovery_spin_response(self, future, recovery):
        try:
            goal_handle = future.result()
        except Exception as error:
            self._finish_person_recovery_spin(
                recovery,
                "failed",
                "PERSON_RECOVERY_SPIN_RESPONSE_FAILED",
                {"error": f"{type(error).__name__}: {error}"},
            )
            return
        if self.person_recovery_spin is not recovery:
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
            return
        if not goal_handle.accepted:
            self._finish_person_recovery_spin(
                recovery,
                "failed",
                "PERSON_RECOVERY_SPIN_REJECTED",
            )
            return
        recovery.goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(
            lambda completed: self._person_recovery_spin_result(
                completed, recovery
            )
        )

    def _person_recovery_spin_result(self, future, recovery):
        if self.person_recovery_spin is not recovery:
            return
        try:
            response = future.result()
            status = response.status
            result = response.result
            succeeded = status == GoalStatus.STATUS_SUCCEEDED
            data = {
                "action_status": int(status),
                "nav2_error_code": int(getattr(result, "error_code", 0)),
                "nav2_error_msg": str(
                    getattr(result, "error_msg", "") or ""
                ),
                "target_yaw": recovery.target_yaw,
            }
        except Exception as error:
            self._finish_person_recovery_spin(
                recovery,
                "failed",
                "PERSON_RECOVERY_SPIN_RESULT_FAILED",
                {"error": f"{type(error).__name__}: {error}"},
            )
            return
        self._finish_person_recovery_spin(
            recovery,
            "succeeded" if succeeded else "failed",
            None if succeeded else "PERSON_RECOVERY_SPIN_FAILED",
            data,
        )

    def _finish_person_recovery_spin(
        self,
        recovery,
        status,
        reason_code,
        data=None,
    ):
        if self.person_recovery_spin is not recovery:
            return
        terminal_reason = (
            "PERSON_VISUAL_RECOVERY_EXHAUSTED"
            if status == "succeeded" else reason_code
        )
        self._fail_visual_recovery(terminal_reason, data)

    # Follow navigation -------------------------------------------------

    def start_follow(self, request):
        self.stop()
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            return {"status": "error", "message": "Nav2 not available"}
        self.follow_action_id = request.get("action_id")
        self.follow_enabled = True
        self.follow_person_recovery_arrived = False
        self.follow_timer.reset()
        return {"status": "accepted"}

    def set_follow_target_override(self, request):
        try:
            if (
                not self.follow_enabled
                or request.get("action_id") != self.follow_action_id
                or request.get("target") != "hand"
                or request.get("frame_id") != "map"
            ):
                raise ValueError("follow action is not active")
            target_x = float(request["x"])
            target_y = float(request["y"])
            override_type = str(request.get("override_type") or "")
            if override_type not in {"welcome", "push"}:
                raise ValueError("unknown hand follow override")
            if not math.isfinite(target_x) or not math.isfinite(target_y):
                raise ValueError("hand target must be finite")
            self.follow_target_override = {
                "x": target_x,
                "y": target_y,
                "override_type": override_type,
            }
            self.follow_target_override_at = time.monotonic()
            return {"status": "accepted"}
        except (KeyError, TypeError, ValueError) as error:
            return {"status": "error", "message": str(error)}

    def clear_follow_target_override(self, request):
        if request.get("action_id") == self.follow_action_id:
            self.follow_target_override = None
            self.follow_target_override_at = None
        return {"status": "accepted"}

    def update_follow(self):
        """Generate or update the Nav2 goal for the latest follow target."""
        if not self.follow_enabled or not self._handle_follow_recovery():
            return

        now = time.monotonic()
        robot_pose = self.get_robot_pose()
        if (
            not isinstance(robot_pose, dict)
            or robot_pose.get("age_seconds", 1.0) > 0.5
        ):
            return

        override_fresh = (
            isinstance(self.follow_target_override, dict)
            and self.follow_target_override_at is not None
            and now - self.follow_target_override_at <= 0.75
        )
        if not override_fresh and self.follow_target_override is not None:
            self.follow_target_override = None
            self.follow_target_override_at = None
        person = self.get_person_state()
        if override_fresh:
            target_x = float(self.follow_target_override["x"])
            target_y = float(self.follow_target_override["y"])
            target_mode = self.follow_target_override["override_type"]
        elif isinstance(person, dict) and person.get("age_seconds", 1.0) <= 0.75:
            target_x = float(person["x"])
            target_y = float(person["y"])
            target_mode = (
                "recovery"
                if getattr(self, "person_tracker_state", None) == "recovering"
                else "person"
            )
        else:
            return

        if (
            target_mode == "recovery"
            and self.follow_person_recovery_arrived
        ):
            return
        if (
            target_mode == "recovery"
            and math.hypot(
                target_x - float(robot_pose["x"]),
                target_y - float(robot_pose["y"]),
            ) <= getattr(
                self,
                "FOLLOW_PERSON_RECOVERY_ARRIVAL_M",
                0.10,
            )
        ):
            NavigationController._notify_follow_recovery_arrived(
                self, self.follow_action_id
            )
            return

        generated = self.follow_goal_generator.generate(
            person_x=target_x,
            person_y=target_y,
            robot_x=float(robot_pose["x"]),
            robot_y=float(robot_pose["y"]),
            target_mode=target_mode,
        )
        if generated is None:
            return

        goal_pose = _make_pose(
            generated.x,
            generated.y,
            generated.yaw,
            "map",
            self.node.get_clock().now().to_msg(),
        )
        should_publish = self.follow_goal_generator.should_publish(
            generated, self.last_follow_goal
        )
        if self.active is None:
            navigation = ActiveNavigation("follow", self.follow_action_id)
            self.active = navigation
            goal = NavigateToPose.Goal()
            goal.pose = goal_pose
            goal.behavior_tree = self.follow_bt
            future = self.nav_client.send_goal_async(goal)
            future.add_done_callback(
                lambda completed: self._goal_response(completed, navigation)
            )
            should_publish = True
        elif self.active.mode == "follow" and should_publish:
            self.goal_update_pub.publish(goal_pose)
        if should_publish:
            self.last_follow_goal = (generated.x, generated.y, generated.yaw)

    def _notify_follow_recovery_arrived(self, action_id):
        """Replace the old idle grace period with the shared local search."""
        if self.follow_person_recovery_arrived:
            return
        active = getattr(self, "active", None)
        if active is not None and active.mode == "follow":
            self.active = None
            if active.goal_handle is not None:
                active.goal_handle.cancel_goal_async()
            stop_robot = getattr(self, "stop_robot", None)
            if callable(stop_robot):
                stop_robot()
        self.follow_person_recovery_arrived = True
        self.follow_recovery_deadline = None
        self.follow_recovery_failure = None
        self._start_or_resume_visual_recovery(
            getattr(self, "person_tracker_status", {}),
            relocated=True,
        )

    def _handle_follow_recovery(self, navigation=None, status=None, result_data=None):
        """Start, monitor, complete, or fail the single follow retry window."""
        now = time.monotonic()
        if navigation is not None:
            if not self.follow_enabled or navigation.action_id != self.follow_action_id:
                return False
            result_data = dict(result_data or {})
            error_code = result_data.get("nav2_error_code")
            if error_code in self.FATAL_FOLLOW_ERROR_CODES:
                outcome = "navigation_failed"
                reason_code = "NAVIGATION_FAILED"
            else:
                if self.follow_recovery_deadline is None:
                    self.follow_recovery_deadline = now + self.FOLLOW_RECOVERY_WINDOW_SECONDS
                self.follow_recovery_failure = (navigation, status, result_data)
                remaining = self.follow_recovery_deadline - now
                if remaining > 0.0:
                    self.logger.warning(
                        "Follow navigation ended unexpectedly; retrying with fresh "
                        f"goals for another {remaining:.2f}s: status={status!r} "
                        f"error_code={error_code!r}"
                    )
                    return True
                outcome = "follow_recovery_failed"
                reason_code = "FOLLOW_RECOVERY_FAILED"
        elif self.follow_recovery_deadline is None:
            return True
        elif now < self.follow_recovery_deadline:
            return True
        elif (
            self.active is not None
            and self.active.mode == "follow"
            and self.active.goal_handle is not None
        ):
            self.follow_recovery_deadline = None
            self.follow_recovery_failure = None
            self.logger.info("Follow navigation recovered within the retry window")
            return True
        else:
            navigation, status, result_data = self.follow_recovery_failure or (
                ActiveNavigation("follow", self.follow_action_id),
                "Follow recovery timed out",
                {},
            )
            result_data = dict(result_data)
            result_data["recovery_timeout_seconds"] = self.FOLLOW_RECOVERY_WINDOW_SECONDS
            outcome = "follow_recovery_failed"
            reason_code = "FOLLOW_RECOVERY_FAILED"

        active = self.active
        self.active = None
        if active is not None and active.goal_handle is not None:
            active.goal_handle.cancel_goal_async()
        self.follow_enabled = False
        self.follow_timer.cancel()
        self.follow_action_id = None
        self.follow_recovery_deadline = None
        self.follow_recovery_failure = None
        self.send_event(
            "navigation",
            status,
            navigation.action_id,
            mode="follow",
            outcome=outcome,
            reason_code=reason_code,
            data=result_data,
        )
        return False

    # Explicit, local, and approach navigation -------------------------

    def navigate_to_pose(self, request, approach=False):
        try:
            destination = {
                "x": float(request.get("x", 0.0)),
                "y": float(request.get("y", 0.0)),
                "angle": float(request.get("angle", 0.0)),
                "frame_id": str(request.get("frame_id", "base_footprint")),
            }
            if approach:
                prepared, pose = self.map_stream.prepare_map()
                destination = self.map_stream.logic.resolve_approach_destination(
                    prepared,
                    pose,
                    destination["x"],
                    destination["y"],
                    standoff_m=request.get("standoff_m"),
                )
            robot_pose = self.get_robot_pose()
            start_pose = dict(robot_pose) if isinstance(robot_pose, dict) else None
            self.stop()
            error = self._send_goal(
                destination,
                request.get("action_id"),
                "approach" if approach else "navigate",
            )
            if error:
                return {"status": "error", "message": error}
            if start_pose is not None:
                self.previous_pose = start_pose
            return {"status": "accepted"}
        except (KeyError, TypeError, ValueError) as error:
            return {"status": "error", "message": str(error)}

    def navigate_local(self, request):
        try:
            prepared, pose = self.map_stream.prepare_map()
            start_pose = dict(pose)
            self.stop()
            local_command = str(request.get("local_command") or "")
            if local_command in {"go_to_room", "return_to_initial_place"}:
                room_id = (
                    1
                    if local_command == "return_to_initial_place"
                    else request.get("room_id")
                )
                destination = self.map_stream.room_registry.destination(room_id)
            else:
                destination = self.map_stream.logic.resolve_local_destination(
                    prepared,
                    pose,
                    local_command,
                    previous_pose=self.previous_pose,
                    person_pose=self.get_person_state(),
                    face_person=bool(request.get("face_person")),
                )
            error = self._send_goal(destination, request.get("action_id"), "local")
            if error:
                raise ValueError(error)
            self.previous_pose = start_pose
            return {"status": "accepted"}
        except (KeyError, TypeError, ValueError) as error:
            return {"status": "error", "message": str(error)}

    # Map-aware navigation sequences -----------------------------------

    def navigate_sequence(self, request):
        try:
            command = str(request.get("sequence_command") or "")
            if command not in SEQUENCE_COMMANDS:
                raise ValueError(f"Unknown navigation sequence: {command}")
            prepared, pose = self.map_stream.prepare_map()
            start_pose = dict(pose)
            self.stop()
            self.sequence = {
                "action_id": request.get("action_id"),
                "command": command,
                "face_person": bool(request.get("face_person")),
                "state": {},
                "attempts": 0,
                "complete_after_goal": False,
            }
            self._advance_sequence(prepared, pose)
            self.previous_pose = start_pose
            return {"status": "accepted"}
        except (KeyError, TypeError, ValueError) as error:
            self.sequence = None
            return {"status": "error", "message": str(error)}

    def _advance_sequence(self, prepared, pose):
        sequence = self.sequence
        if sequence is None:
            raise ValueError("No navigation sequence is active")
        command = sequence["command"]
        state = sequence["state"]
        if command in ROOM_SEQUENCE_COMMANDS:
            if command == "go_to_another_room" and "origin_room_id" not in state:
                room = (
                    self.map_stream.room_registry.room_at(prepared, pose)
                    or self.map_stream.room_registry.ensure_startup_room(prepared, pose)
                )
                if room is not None:
                    state["origin_room_id"] = room["room_id"]
            step = self.map_stream.logic.resolve_exit_room_step(prepared, pose, state)
        else:
            step = self.map_stream.logic.resolve_open_space_middle_step(
                prepared,
                pose,
                state,
                person_pose=self.get_person_state() if sequence["face_person"] else None,
                face_person=sequence["face_person"],
            )

        if step["complete"] or (
            command == "exit_room" and step["phase"] == "cross_room_boundary"
        ):
            if command == "go_to_another_room" and step["complete"]:
                self.map_stream.room_registry.observe_room(
                    prepared,
                    pose,
                    connected_from=step["state"].get("origin_room_id"),
                )
            self._finish_sequence("succeeded", "navigation_reached")
            return

        limit = 12 if command in ROOM_SEQUENCE_COMMANDS else 4
        if sequence["attempts"] >= limit:
            raise ValueError(f"Navigation sequence exceeded {limit} goals")
        sequence["state"] = step["state"]
        sequence["attempts"] += 1
        sequence["complete_after_goal"] = (
            command == "open_space_middle" and step["phase"] == "move_to_room_core"
        )
        error = self._send_goal(step["destination"], sequence["action_id"], "sequence")
        if error:
            raise ValueError(error)

    def _wait_for_fresh_map(self):
        sequence = self.sequence
        if sequence is None:
            return
        sequence["arrival_map_revision"] = self.map_stream.map_revision()
        sequence["map_wait_started"] = time.monotonic()

        def resume():
            if self.sequence is None:
                self.sequence_map_timer.cancel()
                self.node.destroy_timer(self.sequence_map_timer)
                self.sequence_map_timer = None
                return
            if self.map_stream.map_revision() <= sequence["arrival_map_revision"]:
                map_wait_age = time.monotonic() - sequence["map_wait_started"]
                if map_wait_age < self.MAP_UPDATE_TIMEOUT_SECONDS:
                    return
                self._finish_sequence("failed", "map_update_timeout", "MAP_UPDATE_TIMEOUT")
                return
            self.sequence_map_timer.cancel()
            self.node.destroy_timer(self.sequence_map_timer)
            self.sequence_map_timer = None
            try:
                prepared, pose = self.map_stream.prepare_map()
                self._advance_sequence(prepared, pose)
            except (KeyError, TypeError, ValueError) as error:
                self.logger.warning(f"Navigation sequence resolution failed: {error}")
                self._finish_sequence(
                    "failed", "navigation_resolution_failed", "NAVIGATION_REJECTED"
                )

        self.sequence_map_timer = self.node.create_timer(0.05, resume)

    def _finish_sequence(self, status, outcome, reason_code=None):
        if self.sequence_map_timer is not None:
            self.sequence_map_timer.cancel()
            self.node.destroy_timer(self.sequence_map_timer)
            self.sequence_map_timer = None
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

    # Shared Nav2 action lifecycle -------------------------------------

    def _send_goal(self, destination, action_id, mode):
        pose = _make_pose(
            destination["x"],
            destination["y"],
            destination["angle"],
            destination.get("frame_id", "base_footprint"),
            rclpy.time.Time().to_msg(),
        )
        if pose.header.frame_id != "map":
            try:
                pose = self.tf_buffer.transform(
                    pose,
                    "map",
                    timeout=rclpy.duration.Duration(seconds=0.3),
                )
            except Exception as error:
                return f"TF Transform failed: {error}"
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            return "Nav2 not available"

        pose.header.stamp = self.node.get_clock().now().to_msg()
        navigation = ActiveNavigation(mode, action_id, dict(destination))
        self.active = navigation
        goal_checker = "approach_goal_checker" if mode == "approach" else "general_goal_checker"
        self.goal_checker_pub.publish(String(data=goal_checker))
        goal = NavigateToPose.Goal()
        goal.pose = pose
        goal.behavior_tree = self.navigate_bt
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(lambda completed: self._goal_response(completed, navigation))
        return None

    def _goal_response(self, future, navigation):
        try:
            goal_handle = future.result()
        except Exception as error:
            if self.active is not navigation:
                return
            self.active = None
            if navigation.mode == "follow":
                self._handle_follow_recovery(
                    navigation,
                    "Navigation goal response failed",
                    {"error": f"{type(error).__name__}: {error}"},
                )
            else:
                self.logger.error(
                    f"Navigation goal response failed: {type(error).__name__}: {error}"
                )
            return
        if self.active is not navigation:
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
            return
        if not goal_handle.accepted:
            self.active = None
            if navigation.mode == "follow":
                self._handle_follow_recovery(navigation, "Goal Rejected by Nav2", {})
            elif navigation.mode == "sequence":
                self._finish_sequence("failed", "navigation_rejected", "NAVIGATION_REJECTED")
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
                self.send_event("navigation", "Goal Rejected by Nav2", navigation.action_id)
            return
        navigation.goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(
            lambda completed: self._goal_result(completed, navigation)
        )

    def _goal_result(self, future, navigation):
        if self.active is not navigation:
            return
        self.active = None
        try:
            response = future.result()
        except Exception as error:
            if navigation.mode == "follow":
                self._handle_follow_recovery(
                    navigation,
                    "Navigation result failed",
                    {"error": f"{type(error).__name__}: {error}"},
                )
            else:
                self.logger.error(f"Navigation result failed: {type(error).__name__}: {error}")
            return

        status = response.status
        result = response.result
        error_code = getattr(result, "error_code", None)
        result_data = {
            "action_status": int(status),
            "nav2_error_code": int(error_code) if error_code is not None else None,
            "nav2_error_msg": str(getattr(result, "error_msg", "") or ""),
        }
        if navigation.mode == "follow":
            if (
                status == GoalStatus.STATUS_SUCCEEDED
                and getattr(self, "person_tracker_state", None) == "recovering"
            ):
                NavigationController._notify_follow_recovery_arrived(
                    self, navigation.action_id
                )
                return
            follow_status = (
                "Goal Reached"
                if status == GoalStatus.STATUS_SUCCEEDED
                else f"Failed or Canceled (Status {status})"
            )
            self._handle_follow_recovery(navigation, follow_status, result_data)
            return
        if status == GoalStatus.STATUS_SUCCEEDED:
            if navigation.mode == "sequence":
                if self.sequence is None:
                    return
                if self.sequence["complete_after_goal"]:
                    self._finish_sequence("succeeded", "navigation_reached")
                else:
                    self._wait_for_fresh_map()
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
                self.send_event(
                    "navigation", "Goal Reached", navigation.action_id,
                    mode=navigation.mode, data=result_data,
                )
        elif navigation.mode == "sequence":
            self._finish_sequence("failed", "navigation_failed", "NAVIGATION_FAILED")
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
            self.send_event(
                "navigation",
                f"Failed or Canceled (Status {status})",
                navigation.action_id,
                mode=navigation.mode,
                outcome="navigation_failed",
                reason_code="NAVIGATION_FAILED",
                data=result_data,
            )
