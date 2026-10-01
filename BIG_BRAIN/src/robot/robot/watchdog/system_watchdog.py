"""Aggregate ROS graph and topic-liveness checks into one compact message."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Callable

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String


@dataclass(frozen=True)
class TopicSpec:
    group: str
    alias: str
    topic: str
    type_name: str
    mode: str
    stale_after: float
    publishers: int
    subscribers: int


@dataclass(frozen=True)
class NodeSpec:
    group: str
    alias: str
    name: str


@dataclass(frozen=True)
class ServiceSpec:
    group: str
    alias: str
    name: str


def parse_topic_spec(value: str) -> TopicSpec:
    """Parse group|alias|topic|type|mode|timeout|publishers|subscribers."""
    fields = [field.strip() for field in value.split("|")]
    if len(fields) != 8:
        raise ValueError("topic spec must contain 8 pipe-separated fields")
    group, alias, topic, type_name, mode, stale, publishers, subscribers = fields
    if mode not in {"stream", "endpoint"}:
        raise ValueError(f"unsupported mode {mode!r}")
    if not topic.startswith("/"):
        raise ValueError(f"topic must be absolute: {topic!r}")
    result = TopicSpec(
        group=group,
        alias=alias,
        topic=topic,
        type_name=type_name,
        mode=mode,
        stale_after=float(stale),
        publishers=int(publishers),
        subscribers=int(subscribers),
    )
    if not group or not alias or result.stale_after < 0.0:
        raise ValueError("group/alias must be set and timeout must not be negative")
    if result.publishers < 0 or result.subscribers < 0:
        raise ValueError("endpoint counts must not be negative")
    return result


def parse_node_spec(value: str) -> NodeSpec:
    """Parse group|alias|fully-qualified-node-name."""
    fields = [field.strip() for field in value.split("|")]
    if len(fields) != 3:
        raise ValueError("node spec must contain 3 pipe-separated fields")
    group, alias, name = fields
    if not group or not alias or not name:
        raise ValueError("node spec fields must not be empty")
    if not name.startswith("/"):
        name = "/" + name
    return NodeSpec(group, alias, name.rstrip("/") or "/")


def parse_service_spec(value: str) -> ServiceSpec:
    """Parse group|alias|fully-qualified-service-name."""
    node_like = parse_node_spec(value)
    return ServiceSpec(node_like.group, node_like.alias, node_like.name)


def topic_status(
    spec: TopicSpec,
    publisher_count: int,
    external_subscriber_count: int,
    last_message_age: float | None,
    in_startup_grace: bool,
) -> str:
    """Return the compact status value for one topic."""
    if publisher_count < spec.publishers:
        return "wait_pub" if in_startup_grace else "no_pub"
    if external_subscriber_count < spec.subscribers:
        return "wait_sub" if in_startup_grace else "no_sub"
    if spec.mode == "stream":
        if last_message_age is None:
            return "wait_msg" if in_startup_grace else "no_msg"
        if last_message_age > spec.stale_after:
            return f"stale:{last_message_age:.1f}s"
    return "ok"


def health_conclusion(groups: dict[str, dict[str, str]]) -> str:
    """Summarize every unhealthy check as ``group.alias=status``."""
    failures = [
        f"{group}.{alias}={status}"
        for group, checks in sorted(groups.items())
        for alias, status in sorted(checks.items())
        if status != "ok"
    ]
    if not failures:
        return "all checks healthy"
    return "missing/unhealthy: " + ", ".join(failures)


class SystemWatchdog(Node):
    """Watch configured ROS topics/nodes and publish one compact JSON report."""

    def __init__(self) -> None:
        super().__init__("system_watchdog")
        self.declare_parameter("output_topic", "/system/watchdog")
        self.declare_parameter("publish_period", 1.0)
        self.declare_parameter("startup_grace", 10.0)
        self.declare_parameter("topic_specs", [""])
        self.declare_parameter("node_specs", [""])
        self.declare_parameter("service_specs", [""])

        self._started = time.monotonic()
        self._sequence = 0
        self._last_received: dict[str, float] = {}
        self._message_counts: dict[str, int] = {}
        self._stream_subscriptions = []
        self._last_report = ""

        self._topic_specs = self._load_specs(
            "topic_specs", parse_topic_spec
        )
        self._node_specs = self._load_specs("node_specs", parse_node_spec)
        self._service_specs = self._load_specs(
            "service_specs", parse_service_spec
        )

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        for spec in self._topic_specs:
            if spec.mode != "stream":
                continue
            try:
                message_type = get_message(spec.type_name)
                subscription = self.create_subscription(
                    message_type,
                    spec.topic,
                    self._message_callback(spec.topic),
                    qos,
                )
                self._stream_subscriptions.append(subscription)
            except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as error:
                self.get_logger().error(
                    f"Cannot monitor {spec.topic} ({spec.type_name}): {error}"
                )

        output_topic = str(self.get_parameter("output_topic").value)
        self._publisher = self.create_publisher(String, output_topic, 1)
        period = float(self.get_parameter("publish_period").value)
        if period <= 0.0:
            raise ValueError("publish_period must be positive")
        self.create_timer(period, self._publish_report)
        self.get_logger().info(
            f"Monitoring {len(self._topic_specs)} topics and "
            f"{len(self._node_specs)} nodes and {len(self._service_specs)} "
            f"services on {output_topic}"
        )

    def _load_specs(self, parameter: str, parser: Callable):
        parsed = []
        for raw in self.get_parameter(parameter).value:
            if not str(raw).strip():
                continue
            try:
                parsed.append(parser(str(raw)))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid {parameter} entry {raw!r}: {error}") from error
        return parsed

    def _message_callback(self, topic: str):
        def callback(_message) -> None:
            self._last_received[topic] = time.monotonic()
            self._message_counts[topic] = self._message_counts.get(topic, 0) + 1

        return callback

    def _available_nodes(self) -> set[str]:
        available = set()
        for name, namespace in self.get_node_names_and_namespaces():
            namespace = namespace.rstrip("/")
            available.add(f"{namespace}/{name}" if namespace else f"/{name}")
        return available

    def _publish_report(self) -> None:
        now = time.monotonic()
        grace = float(self.get_parameter("startup_grace").value)
        in_startup_grace = now - self._started < grace
        groups: dict[str, dict[str, str]] = {}
        healthy = 0
        total = 0

        for spec in self._topic_specs:
            publishers = self.count_publishers(spec.topic)
            # A stream has one subscription owned by this watchdog.  Only count
            # other subscribers when validating the real processing pipeline.
            subscribers = self.count_subscribers(spec.topic)
            if spec.mode == "stream":
                subscribers = max(0, subscribers - 1)
            received_at = self._last_received.get(spec.topic)
            age = None if received_at is None else max(0.0, now - received_at)
            status = topic_status(
                spec, publishers, subscribers, age, in_startup_grace
            )
            groups.setdefault(spec.group, {})[spec.alias] = status
            healthy += status == "ok"
            total += 1

        available_nodes = self._available_nodes()
        for spec in self._node_specs:
            if spec.name in available_nodes:
                status = "ok"
            else:
                status = "wait_node" if in_startup_grace else "no_node"
            groups.setdefault(spec.group, {})[spec.alias] = status
            healthy += status == "ok"
            total += 1

        available_services = {
            name for name, _types in self.get_service_names_and_types()
        }
        for spec in self._service_specs:
            if spec.name in available_services:
                status = "ok"
            else:
                status = "wait_service" if in_startup_grace else "no_service"
            groups.setdefault(spec.group, {})[spec.alias] = status
            healthy += status == "ok"
            total += 1

        self._sequence += 1
        report = {
            "ok": int(healthy == total and total > 0),
            "seq": self._sequence,
            "up": f"{healthy}/{total}",
            "conclusion": health_conclusion(groups),
            "g": groups,
        }
        encoded = json.dumps(report, separators=(",", ":"), sort_keys=True)
        message = String()
        message.data = encoded
        self._publisher.publish(message)

        state = f"{healthy}/{total}"
        if state != self._last_report:
            log = self.get_logger().info if healthy == total else self.get_logger().warn
            log(f"System health: {state} checks healthy")
            self._last_report = state


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SystemWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
