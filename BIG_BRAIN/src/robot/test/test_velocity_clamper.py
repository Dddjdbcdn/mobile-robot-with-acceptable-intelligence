import math
import unittest

from robot.control.velocity_clamper import (
    WHEEL_SEPARATION,
    condition_diff_drive_velocity,
)


def condition(linear, angular):
    return condition_diff_drive_velocity(linear, angular)


def wheels(linear, angular):
    return (
        linear - angular * WHEEL_SEPARATION * 0.5,
        linear + angular * WHEEL_SEPARATION * 0.5,
    )


class VelocityClamperTests(unittest.TestCase):
    def test_clamps_straight_command_to_minimum_wheel_velocity(self):
        self.assertEqual(condition(0.02, 0.0), (0.05, 0.0))

    def test_clamps_each_wheel_independently(self):
        linear, angular = condition(0.06, 0.2)
        conditioned = wheels(linear, angular)

        self.assertAlmostEqual(conditioned[0], 0.05)
        self.assertAlmostEqual(conditioned[1], 0.0871121)

    def test_keeps_stationary_pivot_wheel_at_zero(self):
        angular = 0.1
        linear = angular * WHEEL_SEPARATION * 0.5
        conditioned = wheels(*condition(linear, angular))

        self.assertAlmostEqual(conditioned[0], 0.0)
        self.assertAlmostEqual(conditioned[1], 0.05)

    def test_snaps_near_zero_wheel_to_zero(self):
        left = 0.0005
        right = 0.03
        linear = (left + right) * 0.5
        angular = (right - left) / WHEEL_SEPARATION
        conditioned = wheels(*condition(linear, angular))

        self.assertAlmostEqual(conditioned[0], 0.0)
        self.assertAlmostEqual(conditioned[1], 0.05)

    def test_restores_two_deadband_wheels_with_original_signs(self):
        left = -0.0005
        right = 0.0008
        linear = (left + right) * 0.5
        angular = (right - left) / WHEEL_SEPARATION
        conditioned = wheels(*condition(linear, angular))

        self.assertAlmostEqual(conditioned[0], -0.05)
        self.assertAlmostEqual(conditioned[1], 0.05)

    def test_stops_when_both_wheels_are_in_absolute_deadband(self):
        self.assertEqual(condition(0.0005, 0.0), (0.0, 0.0))

    def test_restores_command_above_absolute_deadband(self):
        self.assertEqual(condition(0.0006, 0.0), (0.05, 0.0))

    def test_keeps_exact_stop_stopped(self):
        self.assertEqual(condition(0.0, 0.0), (0.0, 0.0))

    def test_rejects_nonfinite_command(self):
        self.assertEqual(condition(math.nan, 0.1), (0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
