from __future__ import annotations

from dataclasses import dataclass, field, fields
import copy
import json
import math
import threading
import time

import numpy as np
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point, PoseStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
import zmq


@dataclass
class Cluster:
    points: np.ndarray
    center: np.ndarray
    width: float


@dataclass
class Detection:
    position: np.ndarray
    kind: str


@dataclass(frozen=True)
class DetectionModel:
    """How much one kind of lidar detection should influence the track."""

    reliability: float
    selection_penalty_m: float


# A pair of legs is the strongest observation. A single leg is less reliable,
# and a partial cluster may include a nearby chair or wall. Reliability scales
# both motion-filter gains; the penalty only breaks close association ties.
DETECTION_MODELS = {
    "pair": DetectionModel(reliability=1.00, selection_penalty_m=0.00),
    "single": DetectionModel(reliability=0.58, selection_penalty_m=0.08),
    "partial": DetectionModel(reliability=0.36, selection_penalty_m=0.16),
}

# Adjacent laser beams naturally spread apart with range. This multiplier
# prevents that beam spacing from splitting one physical surface.
BEAM_SPACING_ALLOWANCE = 1.5


@dataclass(frozen=True)
class TrackerSettings:
    """Runtime tuning, loaded once from ROS parameters at startup."""

    # Lidar clustering and human geometry (metres).
    cluster_break_distance: float = 0.10
    min_cluster_points: int = 3
    single_leg_width: float = 0.35
    min_leg_pair_distance: float = 0.10
    leg_pair_distance: float = 0.75
    partial_cluster_radius: float = 0.30

    # Search and confirmation.
    initial_search_radius: float = 0.60
    tracking_search_radius: float = 0.35
    initialization_match_radius: float = 0.30
    confirmation_scans: int = 3
    seed_association_weight: float = 0.50
    recovery_search_growth: float = 0.45
    recovery_search_limit: float = 0.45

    # ToF seed filtering and correction.
    seed_max_age_seconds: float = 0.75
    seed_smoothing_gain: float = 0.50
    seed_velocity_gain: float = 0.50
    seed_jump_tolerance: float = 0.20
    seed_confirmation_count: int = 3
    seed_correction_radius: float = 0.70
    seed_correction_gain: float = 0.12
    seed_override_count: int = 3

    # Lidar motion filter. A gain of 0 ignores the new measurement; 1 snaps
    # directly to it. Detection reliability scales both gains.
    lidar_position_gain: float = 0.55
    lidar_velocity_gain: float = 0.12
    max_person_speed: float = 2.50
    max_prediction_seconds: float = 0.30
    coast_velocity_retention: float = 0.95

    # State timing.
    coast_seconds: float = 0.75
    lost_seconds: float = 1.75
    initialization_timeout_seconds: float = 2.0
    tracking_alive_timeout_seconds: float = 3.0

    # Radius around the active lidar track removed from SLAM's private scan.
    # Navigation continues to consume the unmodified /scan, so the person is
    # still a collision obstacle even though they are not written into the map.
    slam_person_mask_radius: float = 0.45

    @classmethod
    def from_node(cls, node: Node) -> TrackerSettings:
        defaults = cls()
        for item in fields(defaults):
            node.declare_parameter(item.name, getattr(defaults, item.name))
        return cls(**{
            item.name: node.get_parameter(item.name).value
            for item in fields(defaults)
        })


@dataclass
class SeedFilter:
    """Reject isolated ToF jumps and smooth a plausible moving trajectory."""

    smoothing_gain: float = 0.50
    velocity_gain: float = 0.50
    max_speed: float = 2.50
    max_prediction_seconds: float = 0.30
    jump_tolerance: float = 0.20
    confirmation_count: int = 3
    position: np.ndarray | None = None
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    last_measurement: np.ndarray | None = None
    last_update_at: float | None = None
    consistent_hits: int = 0

    def reset(self) -> None:
        self.position = None
        self.velocity = np.zeros(2, dtype=float)
        self.last_measurement = None
        self.last_update_at = None
        self.consistent_hits = 0

    def update(
        self,
        measurement: np.ndarray,
        now: float,
    ) -> tuple[np.ndarray, bool]:
        measurement = measurement.astype(float, copy=True)
        if self.last_measurement is None or self.last_update_at is None:
            self.position = measurement
            self.last_measurement = measurement
            self.last_update_at = now
            self.consistent_hits = 1
            return self.position.copy(), self.confirmation_count <= 1

        dt = now - self.last_update_at
        if not math.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            self.reset()
            return self.update(measurement, now)

        jump = float(np.linalg.norm(measurement - self.last_measurement))
        if jump > self.jump_tolerance + self.max_speed * dt:
            # Start a new candidate sequence. If this is the real person after
            # a bad depth return, later consistent seeds will confirm it.
            self.position = measurement
            self.velocity = np.zeros(2, dtype=float)
            self.last_measurement = measurement
            self.last_update_at = now
            self.consistent_hits = 1
            return self.position.copy(), self.confirmation_count <= 1

        measured_velocity = (measurement - self.last_measurement) / dt
        predicted = self.position + self.velocity * min(
            dt, self.max_prediction_seconds
        )
        self.position = predicted + self.smoothing_gain * (
            measurement - predicted
        )
        self.velocity += self.velocity_gain * (
            measured_velocity - self.velocity
        )
        speed = float(np.linalg.norm(self.velocity))
        if speed > self.max_speed:
            self.velocity *= self.max_speed / speed

        self.last_measurement = measurement
        self.last_update_at = now
        self.consistent_hits += 1
        return (
            self.position.copy(),
            self.consistent_hits >= self.confirmation_count,
        )


@dataclass
class PersonTrack:
    position: np.ndarray
    velocity: np.ndarray
    stamp_seconds: float
    last_hit_at: float
    max_prediction_seconds: float = 0.30
    state: str = "tracking"

    def predict(self, stamp_seconds: float) -> tuple[np.ndarray, float]:
        dt = stamp_seconds - self.stamp_seconds
        if not math.isfinite(dt) or dt <= 0.0:
            dt = 0.1
        dt = min(dt, self.max_prediction_seconds)
        return self.position + self.velocity * dt, dt

    def update(
        self,
        measurement: np.ndarray,
        stamp_seconds: float,
        position_gain: float,
        velocity_gain: float,
        max_speed: float,
    ) -> None:
        predicted, dt = self.predict(stamp_seconds)
        error = measurement - predicted
        self.position = predicted + position_gain * error
        self.velocity = self.velocity + (velocity_gain / dt) * error

        speed = float(np.linalg.norm(self.velocity))
        if speed > max_speed:
            self.velocity *= max_speed / speed

        self.stamp_seconds = stamp_seconds
        self.last_hit_at = time.monotonic()
        self.state = "tracking"


def cluster_scan(
    scan: LaserScan,
    base_break_distance: float,
    min_points: int,
) -> list[np.ndarray]:
    """Split consecutive valid laser beams into Cartesian point clusters."""
    clusters: list[np.ndarray] = []
    current: list[tuple[float, float]] = []
    previous = None
    angle = float(scan.angle_min)

    def finish_cluster() -> None:
        nonlocal current
        if len(current) >= min_points:
            clusters.append(np.asarray(current, dtype=float))
        current = []

    for distance in scan.ranges:
        valid = (
            math.isfinite(distance)
            and float(scan.range_min) <= distance <= float(scan.range_max)
        )
        if not valid:
            finish_cluster()
            previous = None
            angle += float(scan.angle_increment)
            continue

        point = np.array(
            [distance * math.cos(angle), distance * math.sin(angle)],
            dtype=float,
        )
        if previous is not None:
            adaptive_break = (
                base_break_distance
                + BEAM_SPACING_ALLOWANCE
                * distance
                * abs(float(scan.angle_increment))
            )
            if float(np.linalg.norm(point - previous)) > adaptive_break:
                finish_cluster()

        current.append((float(point[0]), float(point[1])))
        previous = point
        angle += float(scan.angle_increment)

    finish_cluster()
    return clusters


def transform_points(points: np.ndarray, transform) -> np.ndarray:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    yaw = math.atan2(
        2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    matrix = np.array([[cosine, -sine], [sine, cosine]])
    offset = np.array([translation.x, translation.y])
    return points @ matrix.T + offset


def mask_person_returns(
    scan: LaserScan,
    scan_to_tracking_transform,
    person_position: np.ndarray,
    mask_radius: float,
) -> list[float]:
    """Return scan ranges with endpoints near the tracked person invalidated.

    NaN is intentional: it makes SLAM ignore those beams instead of treating
    them as maximum-range rays that could incorrectly clear a wall behind the
    person. The source LaserScan is never mutated.
    """
    filtered_ranges = list(scan.ranges)
    valid_indices = [
        index
        for index, distance in enumerate(scan.ranges)
        if (
            math.isfinite(distance)
            and float(scan.range_min) <= distance <= float(scan.range_max)
        )
    ]
    if not valid_indices or mask_radius <= 0.0:
        return filtered_ranges

    angles = np.asarray([
        float(scan.angle_min) + index * float(scan.angle_increment)
        for index in valid_indices
    ])
    distances = np.asarray(
        [float(scan.ranges[index]) for index in valid_indices]
    )
    scan_points = np.column_stack((
        distances * np.cos(angles),
        distances * np.sin(angles),
    ))
    tracking_points = transform_points(
        scan_points, scan_to_tracking_transform
    )
    person_position = np.asarray(person_position, dtype=float)
    near_person = (
        np.linalg.norm(tracking_points - person_position, axis=1)
        <= mask_radius
    )
    for index, should_mask in zip(valid_indices, near_person):
        if should_mask:
            filtered_ranges[index] = math.nan
    return filtered_ranges


def describe_clusters(point_groups: list[np.ndarray]) -> list[Cluster]:
    clusters = []
    for points in point_groups:
        center = np.median(points, axis=0)
        width = float(np.linalg.norm(points[-1] - points[0]))
        clusters.append(Cluster(points=points, center=center, width=width))
    return clusters


def person_detections(
    clusters: list[Cluster],
    reference: np.ndarray,
    search_radius: float,
    single_leg_width: float,
    min_pair_distance: float,
    pair_distance: float,
    partial_radius: float,
    min_points: int,
) -> list[Detection]:
    """Build person-center candidates near a seed or predicted track."""
    nearby = [
        cluster
        for cluster in clusters
        if np.linalg.norm(cluster.center - reference)
        <= search_radius + cluster.width * 0.5
    ]

    detections: list[Detection] = []
    leg_clusters = [c for c in nearby if c.width <= single_leg_width]

    for cluster in leg_clusters:
        detections.append(Detection(cluster.center.copy(), "single"))

    for index, first in enumerate(leg_clusters):
        for second in leg_clusters[index + 1:]:
            separation = float(np.linalg.norm(first.center - second.center))
            if min_pair_distance <= separation <= pair_distance:
                midpoint = (first.center + second.center) * 0.5
                if np.linalg.norm(midpoint - reference) <= search_radius:
                    detections.append(Detection(midpoint, "pair"))

    # A person beside an obstacle can be part of one wide cluster. Only use
    # the local portion around the predicted person instead of its centroid.
    for cluster in nearby:
        if cluster.width <= single_leg_width:
            continue
        distances = np.linalg.norm(cluster.points - reference, axis=1)
        local_points = cluster.points[distances <= partial_radius]
        if len(local_points) >= min_points:
            detections.append(
                Detection(np.median(local_points, axis=0), "partial")
            )

    return detections


class PersonLidarTracker(Node):
    def __init__(self):
        super().__init__("person_lidar_tracker")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("slam_scan_topic", "/scan_slam")
        self.declare_parameter("tracking_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("tof_endpoint", "tcp://*:5560")
        self.settings = TrackerSettings.from_node(self)

        self.tracking_frame = str(self.get_parameter("tracking_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        scan_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self.scan_callback,
            scan_qos,
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped, "/person_pose", 10
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray, "/person_tracker/markers", 10
        )
        self.status_publisher = self.create_publisher(
            String, "/person_tracker/status", 10
        )
        self.slam_scan_publisher = self.create_publisher(
            LaserScan,
            str(self.get_parameter("slam_scan_topic").value),
            scan_qos,
        )

        self.track: PersonTrack | None = None
        self.pending_position: np.ndarray | None = None
        self.pending_hits = 0
        self.latest_seed = None
        self.seed_lock = threading.Lock()
        self.last_seed_position: np.ndarray | None = None
        self.last_seed_at: float | None = None
        self.seed_filter = SeedFilter(
            smoothing_gain=self.settings.seed_smoothing_gain,
            velocity_gain=self.settings.seed_velocity_gain,
            max_speed=self.settings.max_person_speed,
            max_prediction_seconds=self.settings.max_prediction_seconds,
            jump_tolerance=self.settings.seed_jump_tolerance,
            confirmation_count=self.settings.seed_confirmation_count,
        )
        self.seed_mismatch_hits = 0
        self.handoff_position: np.ndarray | None = None
        self.handoff_velocity = np.zeros(2, dtype=float)
        self.handoff_stamp_seconds: float | None = None
        self.handoff_hits = 0
        self.handoff_started_at: float | None = None
        self.acquisition_state = "disabled"
        self.acquisition_started_at: float | None = None
        self.last_detections: list[Detection] = []
        self.enabled = False
        self.session_id: str | None = None
        self.last_alive_at: float | None = None
        self.pending_control = None

        self.zmq_running = True
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 200)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        endpoint = str(self.get_parameter("tof_endpoint").value)
        self.zmq_socket.bind(endpoint)
        self.zmq_thread = threading.Thread(
            target=self._receive_tof_seeds,
            name="person-tof-seeds",
            daemon=True,
        )
        self.zmq_thread.start()

        self.get_logger().info(
            f"Tracking people from /scan in {self.tracking_frame}; "
            f"waiting for stable ToF seeds on {endpoint}"
        )

    def _receive_tof_seeds(self) -> None:
        while self.zmq_running:
            try:
                message = self.zmq_socket.recv_json()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                return

            message_type = message.get("type")
            action_id = message.get("action_id")
            if message_type == "person_tracking_state":
                with self.seed_lock:
                    self.pending_control = (
                        bool(message.get("active")), action_id
                    )
                    if message.get("active"):
                        self.last_alive_at = time.monotonic()
                    if not message.get("active"):
                        self.latest_seed = None
                continue
            if message_type != "person_tof_position":
                continue
            try:
                seed = {
                    "x": float(message["x"]),
                    "y": float(message["y"]),
                    "frame_id": str(message.get("frame_id") or self.base_frame),
                    "received_at": time.monotonic(),
                }
            except (KeyError, TypeError, ValueError):
                continue

            with self.seed_lock:
                self.pending_control = (True, action_id)
                self.last_alive_at = time.monotonic()
                self.latest_seed = seed

    def _reset_tracking_state(self) -> None:
        self.track = None
        self.pending_position = None
        self.pending_hits = 0
        self.latest_seed = None
        self.last_seed_position = None
        self.last_seed_at = None
        self.seed_filter.reset()
        self.seed_mismatch_hits = 0
        self._clear_seed_handoff()
        self.acquisition_state = "waiting"
        self.acquisition_started_at = None
        self.last_detections = []

    def _apply_pending_control(self) -> None:
        with self.seed_lock:
            control = self.pending_control
            self.pending_control = None
        if control is None:
            return

        active, action_id = control
        if active:
            if not self.enabled or action_id != self.session_id:
                self._reset_tracking_state()
            self.enabled = True
            self.session_id = action_id
            return

        if action_id is None or action_id == self.session_id:
            self.enabled = False
            self.session_id = None
            self.last_alive_at = None
            self._reset_tracking_state()
            self.acquisition_state = "disabled"

    def _take_seed(self):
        with self.seed_lock:
            seed = self.latest_seed
            self.latest_seed = None
        return seed

    def _lookup_transform(self, source_frame: str, stamp) -> object | None:
        try:
            return self.tf_buffer.lookup_transform(
                self.tracking_frame,
                source_frame,
                Time.from_msg(stamp),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as error:
            self.get_logger().warning(
                f"Cannot transform {source_frame} to {self.tracking_frame}: "
                f"{error}",
                throttle_duration_sec=2.0,
            )
            return None

    def _seed_in_tracking_frame(self, seed, stamp) -> np.ndarray | None:
        transform = self._lookup_transform(seed["frame_id"], stamp)
        if transform is None:
            return None
        point = np.array([[seed["x"], seed["y"]]], dtype=float)
        return transform_points(point, transform)[0]

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0

    def scan_callback(self, scan: LaserScan) -> None:
        self._apply_pending_control()
        if (
            self.enabled
            and self.last_alive_at is not None
            and time.monotonic() - self.last_alive_at
            > self.settings.tracking_alive_timeout_seconds
        ):
            self.enabled = False
            self.session_id = None
            self.last_alive_at = None
            self._reset_tracking_state()
        if not self.enabled:
            self.slam_scan_publisher.publish(scan)
            self._publish_disabled(scan.header.stamp)
            return

        transform = self._lookup_transform(scan.header.frame_id, scan.header.stamp)
        if transform is None:
            self.slam_scan_publisher.publish(scan)
            return

        point_groups = cluster_scan(
            scan,
            self.settings.cluster_break_distance,
            self.settings.min_cluster_points,
        )
        transformed_groups = [
            transform_points(points, transform) for points in point_groups
        ]
        clusters = describe_clusters(transformed_groups)
        stamp_seconds = self._stamp_seconds(scan.header.stamp)

        seed = self._take_seed()
        if seed is not None:
            seed_position = self._seed_in_tracking_frame(seed, scan.header.stamp)
            if seed_position is not None:
                filtered_seed, trusted = self.seed_filter.update(
                    seed_position,
                    float(seed.get("received_at", time.monotonic())),
                )
                if trusted:
                    self._accept_trusted_seed(filtered_seed)

        handoff_completed = self._update_seed_handoff(
            clusters, stamp_seconds
        )
        if self.track is None or self.track.state == "lost":
            self._try_initialization(clusters, stamp_seconds)
        elif not handoff_completed:
            self._update_track(clusters, stamp_seconds)

        self._publish_slam_scan(scan, transform)
        self._publish(scan.header.stamp, clusters)

    def _publish_slam_scan(self, scan: LaserScan, transform) -> None:
        if self.track is None or self.track.state == "lost":
            self.slam_scan_publisher.publish(scan)
            return

        filtered_scan = copy.deepcopy(scan)
        filtered_scan.ranges = mask_person_returns(
            scan,
            transform,
            self.track.position,
            self.settings.slam_person_mask_radius,
        )
        self.slam_scan_publisher.publish(filtered_scan)

    def _publish_disabled(self, stamp) -> None:
        status_message = String()
        status_message.data = json.dumps(
            {"state": "disabled"},
            separators=(",", ":"),
        )
        self.status_publisher.publish(status_message)

        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.tracking_frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        self.marker_publisher.publish(markers)

    @staticmethod
    def _best_detection(
        detections: list[Detection],
        reference: np.ndarray,
        trusted_seed: np.ndarray | None = None,
        seed_weight: float = 0.0,
    ) -> Detection | None:
        if not detections:
            return None
        return min(
            detections,
            key=lambda detection: (
                float(np.linalg.norm(detection.position - reference))
                + DETECTION_MODELS[detection.kind].selection_penalty_m
                + (
                    seed_weight
                    * float(
                        np.linalg.norm(detection.position - trusted_seed)
                    )
                    if trusted_seed is not None else 0.0
                )
            ),
        )

    def _fresh_trusted_seed(self) -> np.ndarray | None:
        if self.last_seed_position is None or self.last_seed_at is None:
            return None
        if (
            time.monotonic() - self.last_seed_at
            > self.settings.seed_max_age_seconds
        ):
            return None
        return self.last_seed_position

    def _begin_reacquisition(self, seed_position: np.ndarray) -> None:
        self._clear_seed_handoff()
        self.track = None
        self.pending_position = None
        self.pending_hits = 0
        self.last_detections = []
        self.last_seed_position = seed_position.copy()
        self.last_seed_at = time.monotonic()
        self.seed_mismatch_hits = 0
        self.acquisition_state = "reacquiring"
        self.acquisition_started_at = time.monotonic()
        self.get_logger().info(
            "Trusted ToF seed invalidated the lidar association; reacquiring"
        )

    def _clear_seed_handoff(self) -> None:
        self.handoff_position = None
        self.handoff_velocity = np.zeros(2, dtype=float)
        self.handoff_stamp_seconds = None
        self.handoff_hits = 0
        self.handoff_started_at = None

    def _begin_seed_handoff(self) -> None:
        if self.handoff_started_at is not None:
            return
        self.handoff_started_at = time.monotonic()
        self.handoff_position = None
        self.handoff_velocity = np.zeros(2, dtype=float)
        self.handoff_stamp_seconds = None
        self.handoff_hits = 0
        self.get_logger().info(
            "Trusted ToF seed disagrees with the active track; "
            "confirming a replacement while retaining the current track"
        )

    def _accept_trusted_seed(self, seed_position: np.ndarray) -> None:
        self.last_seed_position = seed_position.copy()
        self.last_seed_at = time.monotonic()

        if self.track is None:
            if self.acquisition_state == "waiting":
                self.acquisition_state = "initializing"
                self.acquisition_started_at = time.monotonic()
            return
        if self.track.state == "lost":
            self._begin_reacquisition(seed_position)
            return

        distance = float(np.linalg.norm(seed_position - self.track.position))
        if distance <= self.settings.seed_correction_radius:
            self.seed_mismatch_hits = 0
            self._clear_seed_handoff()
            self.track.position += self.settings.seed_correction_gain * (
                seed_position - self.track.position
            )
            return

        self.seed_mismatch_hits += 1
        if self.seed_mismatch_hits >= self.settings.seed_override_count:
            self._begin_seed_handoff()

    def _update_seed_handoff(
        self, clusters: list[Cluster], stamp_seconds: float
    ) -> bool:
        """Confirm a seed-associated replacement without dropping the track."""
        if self.handoff_started_at is None or self.track is None:
            return False

        trusted_seed = self._fresh_trusted_seed()
        if trusted_seed is None:
            if time.monotonic() - self.handoff_started_at > (
                self.settings.initialization_timeout_seconds
            ):
                self.seed_mismatch_hits = 0
                self._clear_seed_handoff()
            return False

        detections = person_detections(
            clusters,
            trusted_seed,
            self.settings.initial_search_radius,
            self.settings.single_leg_width,
            self.settings.min_leg_pair_distance,
            self.settings.leg_pair_distance,
            self.settings.partial_cluster_radius,
            self.settings.min_cluster_points,
        )
        selected = self._best_detection(detections, trusted_seed)
        if selected is None:
            self.handoff_position = None
            self.handoff_velocity = np.zeros(2, dtype=float)
            self.handoff_stamp_seconds = None
            self.handoff_hits = 0
            return False

        if (
            self.handoff_position is None
            or np.linalg.norm(selected.position - self.handoff_position)
            > self.settings.initialization_match_radius
        ):
            self.handoff_position = selected.position.copy()
            self.handoff_velocity = np.zeros(2, dtype=float)
            self.handoff_stamp_seconds = stamp_seconds
            self.handoff_hits = 1
            return False

        dt = stamp_seconds - float(self.handoff_stamp_seconds)
        if math.isfinite(dt) and dt > 0.0:
            measured_velocity = (
                selected.position - self.handoff_position
            ) / dt
            self.handoff_velocity += 0.25 * (
                measured_velocity - self.handoff_velocity
            )
            speed = float(np.linalg.norm(self.handoff_velocity))
            if speed > self.settings.max_person_speed:
                self.handoff_velocity *= self.settings.max_person_speed / speed

        self.handoff_position = selected.position.copy()
        self.handoff_stamp_seconds = stamp_seconds
        self.handoff_hits += 1
        if self.handoff_hits < self.settings.confirmation_scans:
            return False

        self.track = PersonTrack(
            position=self.handoff_position.copy(),
            velocity=self.handoff_velocity.copy(),
            stamp_seconds=stamp_seconds,
            last_hit_at=time.monotonic(),
            max_prediction_seconds=self.settings.max_prediction_seconds,
        )
        self.seed_mismatch_hits = 0
        self.acquisition_state = "tracking"
        self.acquisition_started_at = None
        self._clear_seed_handoff()
        self.get_logger().info("Confirmed handoff to seed-associated lidar track")
        return True

    def _try_initialization(
        self, clusters: list[Cluster], stamp_seconds: float
    ) -> None:
        trusted_seed = self._fresh_trusted_seed()
        if trusted_seed is None:
            self.last_detections = []
            self.pending_position = None
            self.pending_hits = 0
            self._check_acquisition_timeout()
            return

        detections = person_detections(
            clusters,
            trusted_seed,
            self.settings.initial_search_radius,
            self.settings.single_leg_width,
            self.settings.min_leg_pair_distance,
            self.settings.leg_pair_distance,
            self.settings.partial_cluster_radius,
            self.settings.min_cluster_points,
        )
        self.last_detections = detections
        selected = self._best_detection(detections, trusted_seed)
        if selected is None:
            self.pending_position = None
            self.pending_hits = 0
            self._check_acquisition_timeout()
            return

        if self.acquisition_state == "lost":
            self.acquisition_state = "reacquiring"
            self.acquisition_started_at = time.monotonic()

        if (
            self.pending_position is None
            or np.linalg.norm(selected.position - self.pending_position)
            > self.settings.initialization_match_radius
        ):
            self.pending_position = selected.position.copy()
            self.pending_hits = 1
            return

        self.pending_position = (
            self.pending_position * self.pending_hits + selected.position
        ) / (self.pending_hits + 1)
        self.pending_hits += 1
        if self.pending_hits < self.settings.confirmation_scans:
            self._check_acquisition_timeout()
            return

        self.track = PersonTrack(
            position=self.pending_position.copy(),
            velocity=np.zeros(2),
            stamp_seconds=stamp_seconds,
            last_hit_at=time.monotonic(),
            max_prediction_seconds=self.settings.max_prediction_seconds,
        )
        self.pending_position = None
        self.pending_hits = 0
        self.seed_mismatch_hits = 0
        self.acquisition_state = "tracking"
        self.acquisition_started_at = None
        self.get_logger().info("Initialized lidar person track")

    def _check_acquisition_timeout(self) -> None:
        if self.acquisition_started_at is None:
            return
        if time.monotonic() - self.acquisition_started_at <= (
            self.settings.initialization_timeout_seconds
        ):
            return
        self.acquisition_state = "lost"
        self.acquisition_started_at = None
        self.pending_position = None
        self.pending_hits = 0

    def _update_track(
        self, clusters: list[Cluster], stamp_seconds: float
    ) -> None:
        predicted, _ = self.track.predict(stamp_seconds)
        missed_for = time.monotonic() - self.track.last_hit_at
        search_radius = self.settings.tracking_search_radius + min(
            self.settings.recovery_search_limit,
            missed_for * self.settings.recovery_search_growth,
        )

        detections = person_detections(
            clusters,
            predicted,
            search_radius,
            self.settings.single_leg_width,
            self.settings.min_leg_pair_distance,
            self.settings.leg_pair_distance,
            self.settings.partial_cluster_radius,
            self.settings.min_cluster_points,
        )
        self.last_detections = detections
        trusted_seed = self._fresh_trusted_seed()
        if self.seed_mismatch_hits > 0:
            # A conflicting seed is evaluated by the parallel handoff guard;
            # it must not pull the still-valid active association sideways.
            trusted_seed = None
        selected = self._best_detection(
            detections,
            predicted,
            trusted_seed=trusted_seed,
            seed_weight=self.settings.seed_association_weight,
        )

        if selected is not None:
            reliability = DETECTION_MODELS[selected.kind].reliability
            self.track.update(
                selected.position,
                stamp_seconds,
                self.settings.lidar_position_gain * reliability,
                self.settings.lidar_velocity_gain * reliability,
                self.settings.max_person_speed,
            )
            return

        self.track.position = predicted
        self.track.velocity *= self.settings.coast_velocity_retention
        self.track.stamp_seconds = stamp_seconds

        if missed_for <= self.settings.coast_seconds:
            self.track.state = "coasting"
        elif missed_for <= self.settings.lost_seconds:
            self.track.state = "recovering"
        else:
            if self.track.state != "lost":
                self.get_logger().info(
                    "Lidar person track lost; waiting for a new stable ToF seed"
                )
            self.track.state = "lost"
            self.acquisition_state = "lost"
            self.acquisition_started_at = None

    def _publish(self, stamp, clusters: list[Cluster]) -> None:
        if self.track is not None and self.track.state != "lost":
            pose = PoseStamped()
            pose.header.frame_id = self.tracking_frame
            pose.header.stamp = stamp
            pose.pose.position.x = float(self.track.position[0])
            pose.pose.position.y = float(self.track.position[1])
            speed = float(np.linalg.norm(self.track.velocity))
            if speed > 0.05:
                yaw = math.atan2(self.track.velocity[1], self.track.velocity[0])
                pose.pose.orientation.z = math.sin(yaw * 0.5)
                pose.pose.orientation.w = math.cos(yaw * 0.5)
            else:
                pose.pose.orientation.w = 1.0
            self.pose_publisher.publish(pose)

        published_state = (
            self.track.state
            if self.track is not None and self.track.state != "lost"
            else self.acquisition_state
        )
        status = {
            "state": published_state,
            "candidate_count": len(self.last_detections),
            "initialization_hits": self.pending_hits,
            "seed_filter_hits": self.seed_filter.consistent_hits,
            "seed_mismatch_hits": self.seed_mismatch_hits,
            "handoff_hits": self.handoff_hits,
        }
        status_message = String()
        status_message.data = json.dumps(status, separators=(",", ":"))
        self.status_publisher.publish(status_message)
        self.marker_publisher.publish(self._markers(stamp, clusters))

    def _markers(self, stamp, clusters: list[Cluster]) -> MarkerArray:
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        lifetime = DurationMsg(sec=0, nanosec=300_000_000)

        centers = Marker()
        centers.header.frame_id = self.tracking_frame
        centers.header.stamp = stamp
        centers.ns = "lidar_clusters"
        centers.id = 0
        centers.type = Marker.SPHERE_LIST
        centers.action = Marker.ADD
        centers.scale.x = centers.scale.y = centers.scale.z = 0.05
        centers.color.r = centers.color.g = centers.color.b = 0.65
        centers.color.a = 0.65
        centers.lifetime = lifetime
        centers.points = [
            Point(x=float(c.center[0]), y=float(c.center[1]), z=0.08)
            for c in clusters
        ]
        markers.markers.append(centers)

        candidates = Marker()
        candidates.header = centers.header
        candidates.ns = "person_candidates"
        candidates.id = 1
        candidates.type = Marker.SPHERE_LIST
        candidates.action = Marker.ADD
        candidates.scale.x = candidates.scale.y = candidates.scale.z = 0.10
        candidates.color.b = 1.0
        candidates.color.g = 0.8
        candidates.color.a = 0.9
        candidates.lifetime = lifetime
        candidates.points = [
            Point(x=float(d.position[0]), y=float(d.position[1]), z=0.12)
            for d in self.last_detections
        ]
        markers.markers.append(candidates)

        trusted_seed = self._fresh_trusted_seed()
        if trusted_seed is not None:
            seed = Marker()
            seed.header = centers.header
            seed.ns = "tof_seed"
            seed.id = 2
            seed.type = Marker.SPHERE
            seed.action = Marker.ADD
            seed.pose.position.x = float(trusted_seed[0])
            seed.pose.position.y = float(trusted_seed[1])
            seed.pose.position.z = 0.18
            seed.pose.orientation.w = 1.0
            seed.scale.x = seed.scale.y = seed.scale.z = 0.16
            seed.color.r = seed.color.g = 1.0
            seed.color.a = 0.95
            markers.markers.append(seed)

        if self.track is not None:
            tracked = Marker()
            tracked.header = centers.header
            tracked.ns = "tracked_person"
            tracked.id = 3
            tracked.type = Marker.CYLINDER
            tracked.action = Marker.ADD
            tracked.pose.position.x = float(self.track.position[0])
            tracked.pose.position.y = float(self.track.position[1])
            tracked.pose.position.z = 0.5
            tracked.pose.orientation.w = 1.0
            tracked.scale.x = tracked.scale.y = 0.30
            tracked.scale.z = 1.0
            if self.track.state == "tracking":
                tracked.color.g = 1.0
            elif self.track.state in {"coasting", "recovering"}:
                tracked.color.r = 1.0
                tracked.color.g = 0.55
            else:
                tracked.color.r = 1.0
            tracked.color.a = 0.75
            markers.markers.append(tracked)

            label = Marker()
            label.header = centers.header
            label.ns = "tracked_person_label"
            label.id = 4
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(self.track.position[0])
            label.pose.position.y = float(self.track.position[1])
            label.pose.position.z = 1.2
            label.pose.orientation.w = 1.0
            label.scale.z = 0.18
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text = f"person: {self.track.state}"
            markers.markers.append(label)

        return markers

    def destroy_node(self):
        self.zmq_running = False
        self.zmq_thread.join(timeout=1.0)
        self.zmq_socket.close(linger=0)
        self.zmq_context.term()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PersonLidarTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
