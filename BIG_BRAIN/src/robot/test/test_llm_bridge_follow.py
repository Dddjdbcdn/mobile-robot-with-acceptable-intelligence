import math
import time
from unittest.mock import Mock

import pytest
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String

from robot.bridge.llm_bridge import (
    LLMRosBridge,
    _make_pose_stamped,
    _planar_pose_from_transform,
)
from robot.person_pose.follow_goal_generator import FollowGoalGenerator


def test_bridge_pose_helpers_convert_planar_ros_geometry():
    transform = TransformStamped()
    transform.transform.translation.x = 1.5
    transform.transform.translation.y = -0.5
    transform.transform.rotation.z = math.sin(math.pi / 4.0)
    transform.transform.rotation.w = math.cos(math.pi / 4.0)

    planar = _planar_pose_from_transform(transform, "map")
    stamped = _make_pose_stamped(
        planar["x"], planar["y"], planar["yaw"], "map", transform.header.stamp
    )

    assert planar["x"] == pytest.approx(1.5)
    assert planar["y"] == pytest.approx(-0.5)
    assert planar["yaw"] == pytest.approx(math.pi / 2.0)
    assert planar["frame_id"] == "map"
    assert stamped.header.frame_id == "map"
    assert stamped.pose.position.x == pytest.approx(1.5)
    assert stamped.pose.orientation.z == pytest.approx(math.sin(math.pi / 4.0))


class FakeClock:
    class Now:
        @staticmethod
        def to_msg():
            return Time(sec=12, nanosec=34)

    def now(self):
        return self.Now()


def bridge_for_pose_callback():
    bridge = LLMRosBridge.__new__(LLMRosBridge)
    bridge.get_clock = lambda: FakeClock()
    bridge.goal_update_pub = Mock()
    bridge.latest_person_pose_map = None
    bridge.latest_person_pose_at = None
    bridge.pending_person_pose = None
    bridge.robot_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    bridge.robot_pose_at = time.monotonic()
    bridge.follow_goal_generator = FollowGoalGenerator(standoff_m=0.9)
    bridge.last_follow_goal = None
    bridge.last_follow_goal_at = None
    bridge.servo = Mock()
    bridge.servo.reset_pan_angle = 95.0
    bridge.servo.min_pan_angle = 30.0
    bridge.servo.max_pan_angle = 160.0
    bridge.servo.tracking_snapshot.return_value = {
        "pan_angle": 95.0,
        "pan_error": 0.0,
        "observed_at": time.monotonic(),
    }
    bridge.follow_enabled = True
    bridge.navigation_active = True
    bridge.navigation_mode = "follow"
    bridge.start_follow_navigation = Mock()
    bridge.get_logger = Mock(return_value=Mock())
    return bridge


def bridge_for_tracking_loop(pan_angle=95.0):
    bridge = LLMRosBridge.__new__(LLMRosBridge)
    bridge.servo = Mock()
    bridge.servo.pan_angle = pan_angle
    bridge.servo.min_pan_angle = 30.0
    bridge.servo.max_pan_angle = 160.0
    bridge.navigation_active = False
    bridge.tracking_body_active = False
    bridge.track_body_pan_margin_deg = 15.0
    bridge.track_body_pan_hysteresis_deg = 5.0
    bridge.track_body_kp = 0.08
    bridge.track_body_max_angular_vel = 1.0
    bridge.publish_cmd = Mock()
    return bridge


def test_tracking_turns_body_before_servo_hard_limit_and_keeps_turning():
    bridge = bridge_for_tracking_loop(pan_angle=150.0)
    bridge.servo.publish_servo_command.return_value = 0.0

    bridge.track_action_loop()
    bridge.track_action_loop()

    assert bridge.tracking_body_active is True
    assert bridge.publish_cmd.call_count == 2
    bridge.publish_cmd.assert_called_with(0.0, pytest.approx(0.8))


def test_tracking_stops_after_pan_reenters_hysteresis_band():
    bridge = bridge_for_tracking_loop(pan_angle=150.0)
    bridge.track_action_loop()
    bridge.servo.pan_angle = 139.0

    bridge.track_action_loop()

    assert bridge.tracking_body_active is False
    bridge.publish_cmd.assert_called_with(0.0, 0.0)


def test_active_follow_generates_standoff_goal_update():
    bridge = bridge_for_pose_callback()
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x = 2.0

    bridge.person_pose_callback(pose)
    bridge.update_follow_goal()

    published = bridge.goal_update_pub.publish.call_args.args[0]
    assert published.pose.position.x == 1.1
    assert bridge.latest_person_pose_map is pose
    assert pose.header.stamp == Time(sec=12, nanosec=34)
    bridge.start_follow_navigation.assert_not_called()


def test_first_person_pose_starts_a_waiting_follow_action():
    bridge = bridge_for_pose_callback()
    bridge.navigation_active = False
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x = 2.0

    bridge.person_pose_callback(pose)
    bridge.update_follow_goal()

    started_goal = bridge.start_follow_navigation.call_args.args[0]
    assert started_goal.pose.position.x == 1.1
    bridge.goal_update_pub.publish.assert_not_called()


def test_person_pose_is_retried_until_tf_catches_up():
    bridge = bridge_for_pose_callback()
    pose_odom = PoseStamped()
    pose_odom.header.frame_id = "odom"
    pose_map = PoseStamped()
    pose_map.header.frame_id = "map"
    pose_map.pose.position.x = 2.0
    bridge.tf_buffer = Mock()
    bridge.tf_buffer.transform.side_effect = [
        Exception("future extrapolation"),
        pose_map,
    ]

    bridge.person_pose_callback(pose_odom)

    assert bridge.pending_person_pose is pose_odom
    bridge.goal_update_pub.publish.assert_not_called()

    bridge.process_pending_person_pose()
    bridge.update_follow_goal()

    assert bridge.pending_person_pose is None
    assert bridge.latest_person_pose_map is pose_map
    published = bridge.goal_update_pub.publish.call_args.args[0]
    assert published.pose.position.x == 1.1


def test_lost_event_is_sent_only_after_lidar_track_existed():
    bridge = LLMRosBridge.__new__(LLMRosBridge)
    bridge.follow_enabled = True
    bridge.follow_action_id = "follow-1"
    bridge.person_tracker_state = None
    bridge.follow_had_person_track = False
    bridge.send_event = Mock()

    initial_lost = String()
    initial_lost.data = '{"state":"lost"}'
    bridge.person_tracker_status_callback(initial_lost)
    bridge.send_event.assert_not_called()

    tracking = String()
    tracking.data = '{"state":"tracking"}'
    bridge.person_tracker_status_callback(tracking)

    lost = String()
    lost.data = '{"state":"lost"}'
    bridge.person_tracker_status_callback(lost)

    bridge.send_event.assert_called_once_with(
        "person_tracker", "lost", "follow-1"
    )


def test_stale_vision_uses_lidar_bearing_to_guide_camera():
    bridge = bridge_for_pose_callback()
    bridge.servo.tracking_snapshot.return_value = {
        "pan_angle": 95.0,
        "pan_error": 0.0,
        "observed_at": None,
    }
    bearing = 0.5
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x = 2.0 * math.cos(bearing)
    pose.pose.position.y = 2.0 * math.sin(bearing)

    bridge.person_pose_callback(pose)
    bridge.update_follow_goal()

    bridge.servo.set_error.assert_called_once()
    pan_error, tilt_error = bridge.servo.set_error.call_args.args
    assert pan_error == pytest.approx(math.degrees(bearing))
    assert tilt_error == 0.0
    bridge.servo.publish_servo_command.assert_called_once_with(tracking=True)
