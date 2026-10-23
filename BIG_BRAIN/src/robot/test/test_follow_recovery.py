import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time

from robot.bridge.navigation_controller import (
    ActiveNavigation,
    ActivePersonRecoverySpin,
    NavigationController,
)


class FollowGoalTimestampTests(unittest.TestCase):

    def test_updated_follow_goal_uses_current_ros_timestamp(self):
        generated = SimpleNamespace(x=1.0, y=2.0, yaw=0.25)
        clock = Mock()
        clock.now.return_value.to_msg.return_value = Time(
            sec=1, nanosec=500_000_002
        )
        controller = SimpleNamespace(
            node=SimpleNamespace(get_clock=Mock(return_value=clock)),
            follow_enabled=True,
            follow_recovery_deadline=None,
            get_robot_pose=Mock(return_value={
                "x": 0.0,
                "y": 0.0,
                "age_seconds": 0.0,
            }),
            get_person_state=Mock(return_value={
                "x": 1.5,
                "y": 2.5,
                "age_seconds": 0.0,
            }),
            follow_target_override=None,
            follow_target_override_at=None,
            follow_goal_generator=SimpleNamespace(
                generate=Mock(return_value=generated),
                should_publish=Mock(return_value=True),
            ),
            last_follow_goal=None,
            active=ActiveNavigation("follow", "follow-1"),
            goal_update_pub=Mock(),
            _handle_follow_recovery=Mock(return_value=True),
        )

        NavigationController.update_follow(controller)

        goal = controller.goal_update_pub.publish.call_args.args[0]
        self.assertEqual(goal.header.stamp.sec, 1)
        self.assertEqual(goal.header.stamp.nanosec, 500_000_002)

    def test_recovering_person_requests_zero_standoff_mode(self):
        generated = SimpleNamespace(x=1.5, y=0.0, yaw=0.0)
        generator = SimpleNamespace(
            generate=Mock(return_value=generated),
            should_publish=Mock(return_value=False),
        )
        clock = Mock()
        clock.now.return_value.to_msg.return_value = Time()
        controller = SimpleNamespace(
            node=SimpleNamespace(get_clock=Mock(return_value=clock)),
            follow_enabled=True,
            follow_recovery_deadline=None,
            person_tracker_state="recovering",
            follow_person_recovery_arrived=False,
            get_robot_pose=Mock(return_value={
                "x": 0.0,
                "y": 0.0,
                "age_seconds": 0.0,
            }),
            get_person_state=Mock(return_value={
                "x": 1.5,
                "y": 0.0,
                "age_seconds": 0.0,
            }),
            follow_target_override=None,
            follow_target_override_at=None,
            follow_goal_generator=generator,
            last_follow_goal=(1.5, 0.0, 0.0),
            active=ActiveNavigation("follow", "follow-1"),
            goal_update_pub=Mock(),
            _handle_follow_recovery=Mock(return_value=True),
        )

        NavigationController.update_follow(controller)

        self.assertEqual(
            generator.generate.call_args.kwargs["target_mode"],
            "recovery",
        )

    def test_already_at_recovery_pose_starts_shared_local_search(self):
        generator = Mock()
        tracker_status = {
            "state": "recovering",
            "visual_search_direction": -1,
            "last_visual_base_bearing_rad": -0.2,
        }
        controller = SimpleNamespace(
            follow_enabled=True,
            follow_recovery_deadline=None,
            follow_recovery_failure=None,
            follow_action_id="follow-1",
            person_tracker_state="recovering",
            person_tracker_status=tracker_status,
            follow_person_recovery_arrived=False,
            get_robot_pose=Mock(return_value={
                "x": 1.0,
                "y": 2.0,
                "age_seconds": 0.0,
            }),
            get_person_state=Mock(return_value={
                "x": 1.0,
                "y": 2.0,
                "age_seconds": 0.0,
            }),
            follow_target_override=None,
            follow_target_override_at=None,
            follow_goal_generator=generator,
            _handle_follow_recovery=Mock(return_value=True),
            _start_or_resume_visual_recovery=Mock(),
        )

        NavigationController.update_follow(controller)

        self.assertTrue(controller.follow_person_recovery_arrived)
        controller._start_or_resume_visual_recovery.assert_called_once_with(
            tracker_status,
            relocated=True,
        )
        generator.generate.assert_not_called()


class FollowRecoveryTests(unittest.TestCase):

    def make_controller(self):
        controller = SimpleNamespace(
            FOLLOW_RECOVERY_WINDOW_SECONDS=3.0,
            FATAL_FOLLOW_ERROR_CODES={101, 103, 201},
            follow_enabled=True,
            follow_action_id="follow-1",
            follow_recovery_deadline=None,
            follow_recovery_failure=None,
            active=None,
            follow_timer=Mock(),
            node=SimpleNamespace(
                get_clock=Mock(return_value=SimpleNamespace(
                    now=Mock(return_value=SimpleNamespace(
                        to_msg=Mock(return_value=Time())
                    ))
                ))
            ),
            logger=Mock(),
            send_event=Mock(),
        )
        return controller

    def test_tf_failure_opens_fixed_recovery_window(self):
        controller = self.make_controller()
        navigation = ActiveNavigation("follow", "follow-1")

        with patch(
            "robot.bridge.navigation_controller.time.monotonic",
            return_value=10.0,
        ):
            NavigationController._handle_follow_recovery(
                controller,
                navigation,
                "Failed or Canceled (Status 6)",
                {
                    "action_status": GoalStatus.STATUS_ABORTED,
                    "nav2_error_code": 102,
                },
            )

        self.assertEqual(controller.follow_recovery_deadline, 13.0)
        self.assertTrue(controller.follow_enabled)
        controller.follow_timer.cancel.assert_not_called()
        controller.send_event.assert_not_called()

    def test_repeated_failure_does_not_extend_recovery_window(self):
        controller = self.make_controller()
        navigation = ActiveNavigation("follow", "follow-1")

        with patch(
            "robot.bridge.navigation_controller.time.monotonic",
            side_effect=[10.0, 11.0],
        ):
            NavigationController._handle_follow_recovery(
                controller, navigation, "first", {"nav2_error_code": 102}
            )
            NavigationController._handle_follow_recovery(
                controller, navigation, "second", {"nav2_error_code": 102}
            )

        self.assertEqual(controller.follow_recovery_deadline, 13.0)
        self.assertEqual(controller.follow_recovery_failure[1], "second")

    def test_invalid_controller_fails_without_retry(self):
        controller = self.make_controller()
        navigation = ActiveNavigation("follow", "follow-1")

        NavigationController._handle_follow_recovery(
            controller,
            navigation,
            "Failed or Canceled (Status 6)",
            {"nav2_error_code": 101},
        )

        self.assertFalse(controller.follow_enabled)
        controller.follow_timer.cancel.assert_called_once()
        controller.send_event.assert_called_once()
        self.assertIsNone(controller.follow_action_id)

    def test_running_retry_is_accepted_at_deadline(self):
        controller = self.make_controller()
        controller.follow_recovery_deadline = time.monotonic() - 0.1
        controller.follow_recovery_failure = (
            ActiveNavigation("follow", "follow-1"),
            "failed",
            {"nav2_error_code": 102},
        )
        controller.active = ActiveNavigation(
            "follow", "follow-1", goal_handle=Mock()
        )

        recovered = NavigationController._handle_follow_recovery(controller)

        self.assertTrue(recovered)
        self.assertIsNone(controller.follow_recovery_deadline)
        controller.send_event.assert_not_called()

    def test_missing_retry_goal_fails_at_deadline(self):
        controller = self.make_controller()
        controller.follow_recovery_deadline = time.monotonic() - 0.1
        controller.follow_recovery_failure = (
            ActiveNavigation("follow", "follow-1"),
            "Failed or Canceled (Status 6)",
            {"nav2_error_code": 102},
        )

        recovered = NavigationController._handle_follow_recovery(controller)

        self.assertFalse(recovered)
        self.assertFalse(controller.follow_enabled)
        event = controller.send_event.call_args
        self.assertEqual(event.kwargs["reason_code"], "FOLLOW_RECOVERY_FAILED")
        self.assertEqual(event.kwargs["data"]["nav2_error_code"], 102)
        self.assertEqual(event.kwargs["data"]["recovery_timeout_seconds"], 3.0)

    def test_recovery_goal_arrival_starts_shared_local_search(self):
        navigation = ActiveNavigation("follow", "follow-1")
        response = SimpleNamespace(
            status=GoalStatus.STATUS_SUCCEEDED,
            result=SimpleNamespace(error_code=0, error_msg=""),
        )
        controller = SimpleNamespace(
            active=navigation,
            person_tracker_state="recovering",
            person_tracker_status={"visual_search_direction": 1},
            follow_person_recovery_arrived=False,
            follow_recovery_deadline=12.0,
            follow_recovery_failure=(navigation, "old", {}),
            _handle_follow_recovery=Mock(),
            _start_or_resume_visual_recovery=Mock(),
        )
        future = Mock()
        future.result.return_value = response

        NavigationController._goal_result(controller, future, navigation)

        self.assertTrue(controller.follow_person_recovery_arrived)
        controller._start_or_resume_visual_recovery.assert_called_once_with(
            {"visual_search_direction": 1},
            relocated=True,
        )
        self.assertIsNone(controller.follow_recovery_deadline)
        controller._handle_follow_recovery.assert_not_called()

    def test_lost_tracker_state_is_forwarded_to_active_follow(self):
        controller = SimpleNamespace(
            person_tracker_state="recovering",
            follow_person_recovery_arrived=True,
            follow_enabled=True,
            follow_had_person_track=True,
            follow_action_id="follow-1",
            send_event=Mock(),
        )

        NavigationController.note_person_tracker_state(
            controller, "lost", "recovering"
        )

        controller.send_event.assert_called_once_with(
            "person_tracker", "lost", "follow-1"
        )

    def test_visual_recovery_starts_locally_without_progress_event(self):
        controller = SimpleNamespace(
            person_tracker_state="tracking",
            follow_person_recovery_arrived=False,
            follow_enabled=True,
            follow_had_person_track=True,
            follow_action_id="follow-1",
            _start_or_resume_visual_recovery=Mock(),
            send_event=Mock(),
        )
        status = {
            "state": "visual_recovery",
            "visual_search_direction": -1,
        }

        NavigationController.note_person_tracker_state(
            controller,
            "visual_recovery",
            "tracking",
            status,
        )

        controller._start_or_resume_visual_recovery.assert_called_once_with(
            status
        )
        controller.send_event.assert_not_called()

    def test_local_visual_recovery_uses_tracker_search_direction(self):
        controller = SimpleNamespace(
            PERSON_VISUAL_RECOVERY_SWEEP_RAD=math.pi / 2.0,
            person_visual_recovery_active=False,
            person_recovery_spin=None,
            person_recovery_remaining_yaw=0.0,
            _pause_follow_for_visual_recovery=Mock(),
            _start_person_recovery_spin=Mock(return_value=True),
        )

        NavigationController._start_or_resume_visual_recovery(
            controller,
            {
                "visual_search_direction": -1,
                "last_visual_base_bearing_rad": -0.3,
            },
        )

        self.assertTrue(controller.person_visual_recovery_active)
        self.assertAlmostEqual(
            controller.person_recovery_remaining_yaw,
            -math.pi / 2.0,
        )
        controller._pause_follow_for_visual_recovery.assert_called_once_with()
        controller._start_person_recovery_spin.assert_called_once_with(
            -math.pi / 2.0
        )

    def test_relocated_lidar_recovery_ignores_stale_visual_bearing(self):
        controller = SimpleNamespace(
            PERSON_LIDAR_RECOVERY_SWEEP_RAD=2.0 * math.pi,
            person_visual_recovery_active=False,
            person_recovery_spin=None,
            person_recovery_remaining_yaw=0.0,
            _pause_follow_for_visual_recovery=Mock(),
            _start_person_recovery_spin=Mock(return_value=True),
        )

        NavigationController._start_or_resume_visual_recovery(
            controller,
            {
                "visual_search_direction": -1,
                "last_visual_base_bearing_rad": -0.3,
            },
            relocated=True,
        )

        self.assertAlmostEqual(
            controller.person_recovery_remaining_yaw,
            2.0 * math.pi,
        )
        controller._start_person_recovery_spin.assert_called_once_with(
            2.0 * math.pi
        )

    def test_validating_stops_local_spin_and_remembers_remaining_yaw(self):
        controller = SimpleNamespace(
            person_tracker_state="reacquiring",
            follow_person_recovery_arrived=False,
            follow_enabled=True,
            follow_had_person_track=True,
            follow_action_id="follow-1",
            person_visual_recovery_active=True,
            person_recovery_remaining_yaw=-1.0,
            _stop_person_recovery_spin=Mock(return_value=-0.7),
            send_event=Mock(),
        )

        NavigationController.note_person_tracker_state(
            controller,
            "validating",
            "reacquiring",
            {"state": "validating"},
        )

        self.assertEqual(controller.person_recovery_remaining_yaw, -0.7)
        controller.send_event.assert_not_called()

    def test_tracking_after_visual_recovery_resumes_follow_locally(self):
        controller = SimpleNamespace(
            person_tracker_state="validating",
            follow_person_recovery_arrived=False,
            follow_enabled=True,
            follow_had_person_track=False,
            follow_action_id="follow-1",
            person_visual_recovery_active=True,
            _complete_visual_recovery=Mock(),
            send_event=Mock(),
        )

        NavigationController.note_person_tracker_state(
            controller,
            "tracking",
            "awaiting_seed",
            {"state": "tracking"},
        )

        controller._complete_visual_recovery.assert_called_once_with()
        controller.send_event.assert_not_called()

    def test_local_person_recovery_spin_sends_bounded_nav2_goal(self):
        future = Mock()
        spin_client = Mock()
        spin_client.wait_for_server.return_value = True
        spin_client.send_goal_async.return_value = future
        controller = SimpleNamespace(
            spin_client=spin_client,
            follow_action_id="follow-1",
            PERSON_VISUAL_RECOVERY_TIMEOUT_SECONDS=8.5,
            person_recovery_spin=None,
        )

        started = NavigationController._start_person_recovery_spin(
            controller,
            -math.pi / 2.0,
        )

        self.assertTrue(started)
        goal = spin_client.send_goal_async.call_args.args[0]
        self.assertAlmostEqual(goal.target_yaw, -math.pi / 2.0)
        self.assertEqual(goal.time_allowance.sec, 8)
        self.assertEqual(goal.time_allowance.nanosec, 500_000_000)
        future.add_done_callback.assert_called_once()
        self.assertEqual(controller.person_recovery_spin.parent_action_id, "follow-1")

    def test_stopping_recovery_spin_returns_untravelled_yaw(self):
        goal_handle = Mock()
        recovery = ActivePersonRecoverySpin(
            "follow-1:visual-recovery",
            "follow-1",
            -math.pi / 2.0,
            goal_handle=goal_handle,
            angular_distance_traveled=0.5,
        )
        controller = SimpleNamespace(
            person_recovery_spin=recovery,
            stop_robot=Mock(),
        )

        remaining_yaw = NavigationController._stop_person_recovery_spin(
            controller
        )

        self.assertAlmostEqual(
            remaining_yaw,
            -(math.pi / 2.0 - 0.5),
        )
        goal_handle.cancel_goal_async.assert_called_once()

    def test_completed_local_spin_reports_terminal_follow_failure(self):
        recovery = ActivePersonRecoverySpin(
            "follow-1:visual-recovery",
            "follow-1",
            math.pi / 2.0,
        )
        controller = SimpleNamespace(
            person_recovery_spin=recovery,
            _fail_visual_recovery=Mock(),
        )

        NavigationController._finish_person_recovery_spin(
            controller,
            recovery,
            "succeeded",
            None,
            {"target_yaw": math.pi / 2.0},
        )

        controller._fail_visual_recovery.assert_called_once_with(
            "PERSON_VISUAL_RECOVERY_EXHAUSTED",
            {"target_yaw": math.pi / 2.0},
        )


if __name__ == "__main__":
    unittest.main()
