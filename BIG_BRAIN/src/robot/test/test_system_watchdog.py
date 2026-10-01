import pytest

from robot.watchdog.system_watchdog import (
    health_conclusion,
    parse_node_spec,
    parse_service_spec,
    parse_topic_spec,
    topic_status,
)


def spec(mode="stream"):
    return parse_topic_spec(
        f"micro|imu|/stm32/imu_msg|geometry_msgs/msg/Point32|{mode}|1.0|1|1"
    )


def test_parses_topic_and_node_specs():
    topic = spec()
    node = parse_node_spec("core|agent|micro_ros_agent")

    assert topic.alias == "imu"
    assert topic.stale_after == 1.0
    assert node.name == "/micro_ros_agent"
    assert parse_service_spec("layers|stvl|local/toggle").name == "/local/toggle"


@pytest.mark.parametrize(
    "publishers,subscribers,age,grace,expected",
    [
        (0, 1, 0.1, False, "no_pub"),
        (1, 0, 0.1, False, "no_sub"),
        (1, 1, None, False, "no_msg"),
        (1, 1, 1.24, False, "stale:1.2s"),
        (1, 1, 0.1, False, "ok"),
        (0, 0, None, True, "wait_pub"),
    ],
)
def test_stream_status(publishers, subscribers, age, grace, expected):
    assert topic_status(spec(), publishers, subscribers, age, grace) == expected


def test_endpoint_does_not_require_a_message():
    assert topic_status(spec("endpoint"), 1, 1, None, False) == "ok"


def test_rejects_malformed_topic_spec():
    with pytest.raises(ValueError):
        parse_topic_spec("too|short")


def test_health_conclusion_reports_missing_check():
    groups = {
        "bridge": {"imu": "ok"},
        "core": {"stm32": "ok", "micro_agent": "no_node"},
    }

    assert health_conclusion(groups) == "missing/unhealthy: core.micro_agent=no_node"


def test_health_conclusion_is_deterministic_for_multiple_failures():
    groups = {
        "nav": {"scan": "no_msg"},
        "bridge": {"imu": "stale:1.2s"},
    }

    assert health_conclusion(groups) == (
        "missing/unhealthy: bridge.imu=stale:1.2s, nav.scan=no_msg"
    )


def test_health_conclusion_reports_healthy_system():
    assert health_conclusion({"core": {"stm32": "ok"}}) == "all checks healthy"
