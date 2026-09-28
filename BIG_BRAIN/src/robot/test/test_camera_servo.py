from unittest.mock import Mock

import pytest

from robot.utilities.camera_servo import CameraServo


def servo():
    return CameraServo(Mock(), Mock())


def test_set_angles_clamps_and_publishes_absolute_pose():
    camera = servo()

    camera.set_angles(200.0, 10.0)

    assert camera.pan_angle == 160.0
    assert camera.tilt_angle == 30.0
    assert camera.servo_pan_pub.publish.call_args.args[0].data == 160.0
    assert camera.servo_tilt_pub.publish.call_args.args[0].data == 30.0


def test_tracking_command_applies_proportional_bounded_step():
    camera = servo()
    camera.set_error(100.0, -100.0, tracking_sequence=1)

    remaining = camera.publish_servo_command(tracking=True)

    assert remaining == pytest.approx(0.0)
    assert camera.pan_angle == pytest.approx(99.0)
    assert camera.tilt_angle == pytest.approx(86.0)
