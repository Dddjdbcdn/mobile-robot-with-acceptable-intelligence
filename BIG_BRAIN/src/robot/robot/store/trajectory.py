#!/usr/bin/env python3

import math
from enum import Enum

import rclpy
from geometry_msgs.msg import TwistStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter


class SquareState(Enum):
    FORWARD = 0
    ROTATE = 1
    IDLE = 2


class TrajectoryNode(Node):
    MAX_LINEAR_SPEED = 0.8
    MAX_ANGULAR_SPEED = 5.0
    VALID_TRAJECTORIES = {'idle', 'square', 'circle'}

    def __init__(self):
        super().__init__('trajectory_node')

        self.declare_parameter('trajectory_type', 'idle')
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('square_length', 1.0)
        self.declare_parameter('square_linear_speed', 0.2)
        self.declare_parameter('square_angular_speed', 0.5)
        self.declare_parameter('circle_radius', 0.5)
        self.declare_parameter('circle_linear_speed', 0.2)
        self.declare_parameter('clockwise', False)

        self.trajectory_type = 'idle'
        self.publish_rate_hz = 20.0
        self.square_length = 1.0
        self.square_linear_speed = 0.2
        self.square_angular_speed = 0.5
        self.circle_radius = 0.5
        self.circle_linear_speed = 0.2
        self.clockwise = False

        self.square_state = SquareState.IDLE
        self.square_edge_count = 0
        self.segment_start_time = self.get_clock().now()

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/diff_drive_controller/cmd_vel',
            10,
        )

        self.param_callback_handle = self.add_on_set_parameters_callback(
            self.on_params_changed
        )

        self.load_params()
        self.reset_trajectory()
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.on_timer)

        self.get_logger().info(
            'Trajectory node ready. Set trajectory_type to square/circle/idle.'
        )

    def load_params(self):
        self.trajectory_type = self.get_parameter('trajectory_type').value
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.square_length = float(self.get_parameter('square_length').value)
        self.square_linear_speed = float(self.get_parameter('square_linear_speed').value)
        self.square_angular_speed = float(self.get_parameter('square_angular_speed').value)
        self.circle_radius = float(self.get_parameter('circle_radius').value)
        self.circle_linear_speed = float(self.get_parameter('circle_linear_speed').value)
        self.clockwise = bool(self.get_parameter('clockwise').value)

    def on_params_changed(self, params):
        updates = {param.name: param.value for param in params}
        ok, reason = self.validate_params(updates)
        if not ok:
            return SetParametersResult(successful=False, reason=reason)

        reset_needed = any(name in updates for name in {
            'trajectory_type',
            'square_length',
            'square_linear_speed',
            'square_angular_speed',
            'circle_radius',
            'circle_linear_speed',
            'clockwise',
        })

        old_trajectory = self.trajectory_type

        for name, value in updates.items():
            setattr(self, name, value)

        if 'publish_rate_hz' in updates:
            self.timer.cancel()
            self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.on_timer)

        if reset_needed:
            if old_trajectory != 'idle' and self.trajectory_type == 'idle':
                self.publish_cmd(0.0, 0.0)
            self.reset_trajectory()

        return SetParametersResult(successful=True)

    def validate_params(self, updates):
        trajectory_type = updates.get('trajectory_type', self.trajectory_type)
        publish_rate_hz = float(updates.get('publish_rate_hz', self.publish_rate_hz))
        square_length = float(updates.get('square_length', self.square_length))
        square_linear_speed = float(updates.get('square_linear_speed', self.square_linear_speed))
        square_angular_speed = float(updates.get('square_angular_speed', self.square_angular_speed))
        circle_radius = float(updates.get('circle_radius', self.circle_radius))
        circle_linear_speed = float(updates.get('circle_linear_speed', self.circle_linear_speed))

        if trajectory_type not in self.VALID_TRAJECTORIES:
            return False, 'trajectory_type must be idle, square, or circle'
        if publish_rate_hz <= 0.0:
            return False, 'publish_rate_hz must be positive'
        if square_length <= 0.0:
            return False, 'square_length must be positive'
        if square_linear_speed <= 0.0 or square_linear_speed > self.MAX_LINEAR_SPEED:
            return False, f'square_linear_speed must be in (0, {self.MAX_LINEAR_SPEED}]'
        if square_angular_speed <= 0.0 or square_angular_speed > self.MAX_ANGULAR_SPEED:
            return False, f'square_angular_speed must be in (0, {self.MAX_ANGULAR_SPEED}]'
        if circle_radius <= 0.0:
            return False, 'circle_radius must be positive'
        if circle_linear_speed <= 0.0 or circle_linear_speed > self.MAX_LINEAR_SPEED:
            return False, f'circle_linear_speed must be in (0, {self.MAX_LINEAR_SPEED}]'

        circle_angular_speed = abs(circle_linear_speed / circle_radius)
        if circle_angular_speed > self.MAX_ANGULAR_SPEED:
            return False, (
                f'circle_linear_speed / circle_radius = {circle_angular_speed:.3f} rad/s, '
                f'max {self.MAX_ANGULAR_SPEED}'
            )

        return True, ''

    def reset_trajectory(self):
        self.segment_start_time = self.get_clock().now()
        self.square_edge_count = 0

        if self.trajectory_type == 'square':
            self.square_state = SquareState.FORWARD
            self.get_logger().info(
                f'Square: length={self.square_length:.3f} m, '
                f'linear={self.square_linear_speed:.3f} m/s, '
                f'angular={self.turn_direction() * self.square_angular_speed:.3f} rad/s'
            )
        elif self.trajectory_type == 'circle':
            self.square_state = SquareState.IDLE
            angular = self.turn_direction() * self.circle_linear_speed / self.circle_radius
            self.get_logger().info(
                f'Circle: radius={self.circle_radius:.3f} m, '
                f'linear={self.circle_linear_speed:.3f} m/s, '
                f'angular={angular:.3f} rad/s'
            )
        else:
            self.square_state = SquareState.IDLE
            self.get_logger().info('Trajectory idle')

    def turn_direction(self):
        return -1.0 if self.clockwise else 1.0

    def elapsed_in_segment(self):
        return (self.get_clock().now() - self.segment_start_time).nanoseconds / 1e9

    def start_next_segment(self):
        self.segment_start_time = self.get_clock().now()

    def on_timer(self):
        if self.trajectory_type == 'square':
            linear_x, angular_z = self.square_command()
        elif self.trajectory_type == 'circle':
            linear_x = self.circle_linear_speed
            angular_z = self.turn_direction() * self.circle_linear_speed / self.circle_radius
        else:
            linear_x, angular_z = 0.0, 0.0

        self.publish_cmd(linear_x, angular_z)

    def square_command(self):
        forward_duration = self.square_length / self.square_linear_speed
        rotate_duration = (math.pi / 2.0) / self.square_angular_speed
        elapsed = self.elapsed_in_segment()

        if self.square_state == SquareState.FORWARD:
            if elapsed >= forward_duration:
                self.square_state = SquareState.ROTATE
                self.start_next_segment()
                return 0.0, self.turn_direction() * self.square_angular_speed
            return self.square_linear_speed, 0.0

        if self.square_state == SquareState.ROTATE:
            if elapsed >= rotate_duration:
                self.square_edge_count += 1
                if self.square_edge_count >= 4:
                    # Completed 4 straight segments and 4 turns; force idle state
                    self.trajectory_type = 'idle'
                    self.square_state = SquareState.IDLE
                    self.set_parameters([
                        Parameter('trajectory_type', Parameter.Type.STRING, 'idle')
                    ])
                    return 0.0, 0.0
                
                self.square_state = SquareState.FORWARD
                self.start_next_segment()
                return self.square_linear_speed, 0.0
            return 0.0, self.turn_direction() * self.square_angular_speed

        return 0.0, 0.0

    def publish_cmd(self, linear_x, angular_z):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        for _ in range(3):
            self.publish_cmd(0.0, 0.0)

    def destroy_node(self):
        self.stop_robot()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()