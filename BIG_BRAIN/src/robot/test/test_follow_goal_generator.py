import math

import pytest

from robot.follow_goal_generator import FollowGoalGenerator


def test_generates_standoff_goal_without_unneeded_rotation():
    generator = FollowGoalGenerator()
    goal = generator.generate(
        person_x=2.0,
        person_y=0.0,
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        camera_pan_deg=95.0,
        camera_fresh=True,
    )

    assert goal is not None
    assert goal.x == pytest.approx(2.0 - generator.settings.standoff_m)
    assert goal.y == pytest.approx(0.0)
    assert goal.yaw == pytest.approx(0.0)
    assert goal.used_camera


def test_comfort_cone_absorbs_small_lateral_deviation():
    bearing = math.radians(15.0)
    goal = FollowGoalGenerator().generate(
        person_x=2.0 * math.cos(bearing),
        person_y=2.0 * math.sin(bearing),
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        camera_pan_deg=110.0,
        camera_fresh=True,
    )

    assert goal is not None
    assert goal.yaw == pytest.approx(0.0)
    assert goal.heading_correction == pytest.approx(0.0)


def test_critical_deviation_makes_base_face_person():
    bearing = math.radians(45.0)
    goal = FollowGoalGenerator().generate(
        person_x=2.0 * math.cos(bearing),
        person_y=2.0 * math.sin(bearing),
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        camera_pan_deg=140.0,
        camera_fresh=True,
    )

    assert goal is not None
    assert math.degrees(goal.yaw) == pytest.approx(45.0)
    assert math.degrees(goal.person_bearing - goal.yaw) == pytest.approx(0.0)


def test_rejects_camera_bearing_that_disagrees_with_lidar():
    goal = FollowGoalGenerator().generate(
        person_x=2.0,
        person_y=0.0,
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        camera_pan_deg=140.0,
        camera_fresh=True,
    )

    assert goal is not None
    assert not goal.used_camera
    assert goal.y == pytest.approx(0.0)
    assert goal.yaw == pytest.approx(0.0)


def test_camera_correction_never_moves_lidar_based_goal_position():
    bearing = math.radians(20.0)
    generator = FollowGoalGenerator()
    lidar_only = generator.generate(
        person_x=4.0 * math.cos(bearing),
        person_y=4.0 * math.sin(bearing),
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
    )
    fused = generator.generate(
        person_x=4.0 * math.cos(bearing),
        person_y=4.0 * math.sin(bearing),
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        camera_pan_deg=125.0,
        camera_fresh=True,
    )

    assert lidar_only is not None
    assert fused is not None
    assert fused.x == pytest.approx(lidar_only.x)
    assert fused.y == pytest.approx(lidar_only.y)


def test_stale_camera_falls_back_to_lidar_heading():
    bearing = math.radians(-45.0)
    goal = FollowGoalGenerator().generate(
        person_x=2.0 * math.cos(bearing),
        person_y=2.0 * math.sin(bearing),
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        camera_pan_deg=95.0,
        camera_fresh=False,
    )

    assert goal is not None
    assert not goal.used_camera
    assert math.degrees(goal.yaw) == pytest.approx(-45.0)
