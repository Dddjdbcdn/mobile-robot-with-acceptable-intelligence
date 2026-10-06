"""Condition chassis commands using the drivetrain's usable wheel speed."""

import math

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node


# Effective separation: 0.29 m controller value * 0.9349 multiplier.
WHEEL_SEPARATION = 0.271121
MINIMUM_WHEEL_VELOCITY = 0.05
ABSOLUTE_STOP_THRESHOLD = 0.0001
WHEEL_STOP_THRESHOLD = 0.001


def condition_diff_drive_velocity(
    linear: float,
    angular: float,
    wheel_separation: float = WHEEL_SEPARATION,
    absolute_stop_threshold: float = ABSOLUTE_STOP_THRESHOLD,
    wheel_stop_threshold: float = WHEEL_STOP_THRESHOLD,
    minimum_wheel_velocity: float = MINIMUM_WHEEL_VELOCITY,
):

    if not math.isfinite(linear) or not math.isfinite(angular):
        return 0.0, 0.0

    half_separation = wheel_separation * 0.5
    requested_left = linear - angular * half_separation
    requested_right = linear + angular * half_separation

    if (
        abs(requested_left) <= absolute_stop_threshold
        and abs(requested_right) <= absolute_stop_threshold
    ):
        return 0.0, 0.0

    def condition_wheel(value: float) -> float:
        if abs(value) <= wheel_stop_threshold:
            return 0.0
        if abs(value) < minimum_wheel_velocity:
            return math.copysign(minimum_wheel_velocity, value)
        return value

    left = condition_wheel(requested_left)
    right = condition_wheel(requested_right)

    if left == 0.0 and right == 0.0:
        if requested_left != 0.0:
            left = math.copysign(minimum_wheel_velocity, requested_left)
        if requested_right != 0.0:
            right = math.copysign(minimum_wheel_velocity, requested_right)

    conditioned_linear = (left + right) * 0.5
    conditioned_angular = (right - left) / wheel_separation
    return conditioned_linear, conditioned_angular


class VelocityClamper(Node):
    def __init__(self) -> None:
        super().__init__('velocity_clamper')

        self.publisher = self.create_publisher(TwistStamped, 'cmd_vel_conditioned', 10)
        self.subscription = self.create_subscription(TwistStamped, 'cmd_vel_smoothed', self.command_callback, 10)

    def command_callback(self, message: TwistStamped) -> None:
        output = TwistStamped()
        output.header = message.header
        output.twist = message.twist
        linear, angular = condition_diff_drive_velocity(
            message.twist.linear.x,
            message.twist.angular.z,
        )
        output.twist.linear.x = linear
        output.twist.angular.z = angular
        self.publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VelocityClamper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
