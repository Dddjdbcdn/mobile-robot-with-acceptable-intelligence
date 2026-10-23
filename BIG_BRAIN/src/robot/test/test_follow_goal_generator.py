import unittest
import math

from robot.person_pose.follow_goal_generator import FollowGoalGenerator


class FollowGoalGeneratorTests(unittest.TestCase):

    def setUp(self):
        self.generator = FollowGoalGenerator()

    def generate(self, target_x, **options):
        return self.generator.generate(
            person_x=target_x,
            person_y=0.0,
            robot_x=0.0,
            robot_y=0.0,
            **options,
        )

    def test_normal_follow_holds_position_inside_standoff(self):
        goal = self.generate(0.3)

        self.assertAlmostEqual(goal.x, 0.0)
        self.assertAlmostEqual(goal.y, 0.0)

    def test_normal_follow_advances_to_standoff_above_start_distance(self):
        goal = self.generate(2.0)

        self.assertAlmostEqual(
            goal.x,
            2.0 - self.generator.settings.standoff_m,
        )

    def test_normal_follow_holds_at_move_start_distance(self):
        goal = self.generate(0.5)

        self.assertAlmostEqual(goal.x, 0.0)

    def test_normal_follow_uses_distance_hysteresis(self):
        standoff = self.generator.settings.standoff_m
        start = self.generator.settings.move_start_distance_m
        moving = self.generate(start + 0.1)
        still_moving = self.generate(standoff + 0.05)
        arrived = self.generate(standoff)
        held = self.generate(standoff + 0.05)

        self.assertAlmostEqual(moving.x, start + 0.1 - standoff)
        self.assertAlmostEqual(still_moving.x, 0.05)
        self.assertAlmostEqual(arrived.x, 0.0)
        self.assertAlmostEqual(held.x, 0.0)

    def test_goal_yaw_faces_person_from_standoff(self):
        goal = self.generator.generate(
            person_x=1.0,
            person_y=1.0,
            robot_x=0.0,
            robot_y=0.0,
        )

        expected = math.atan2(1.0 - goal.y, 1.0 - goal.x)
        self.assertAlmostEqual(goal.yaw, expected)

    def test_recovery_goal_has_zero_standoff(self):
        goal = self.generate(2.0, target_mode="recovery")

        self.assertAlmostEqual(goal.x, 2.0)
        self.assertAlmostEqual(goal.y, 0.0)

    def test_recovery_goal_keeps_approach_bearing(self):
        goal = self.generator.generate(
            person_x=1.0,
            person_y=1.0,
            robot_x=0.0,
            robot_y=0.0,
            target_mode="recovery",
        )

        self.assertAlmostEqual(goal.yaw, math.pi / 4.0)

    def test_welcome_hand_can_use_close_standoff(self):
        goal = self.generate(0.3, target_mode="welcome")

        self.assertAlmostEqual(
            goal.x,
            0.3 - self.generator.settings.hand_standoff_m,
        )

    def test_push_hand_retreats_to_regular_standoff(self):
        goal = self.generate(
            0.3,
            target_mode="push",
        )

        self.assertAlmostEqual(
            goal.x,
            0.3 - self.generator.settings.standoff_m,
        )


if __name__ == "__main__":
    unittest.main()
