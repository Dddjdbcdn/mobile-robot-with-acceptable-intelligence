from __future__ import annotations

import copy
import json
import threading
import time

import numpy as np
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String
import tf2_geometry_msgs
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray
import zmq

from robot.person_pose.lidar_clusters import (
    Cluster,
    clusters_from_scan,
    filter_scan_around_person,
    transform_points,
)
from robot.person_pose.person_tracker_state import (
    PersonTrackerState,
    TrackerSettings,
)
from robot.person_pose.person_tracker_renderer import PersonTrackerRenderer
from robot.person_pose.yolo_correction import YoloCorrection, YoloSettings


class PersonLidarTracker(Node):
    """Fuse lidar clusters, ToF seeds, and YOLO motion into a person pose."""

    TF_HISTORY_MARGIN_SECONDS = 0.01

    def __init__(self):
        super().__init__("person_lidar_tracker")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("filtered_scan_topic", "/scan_filtered")
        self.declare_parameter("tracking_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("tof_endpoint", "tcp://*:5560")
        self.declare_parameter("camera_fov_marker_range", 3.0)

        self.settings = TrackerSettings.from_node(self)
        self.tracking_frame = str(self.get_parameter("tracking_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.state = PersonTrackerState(self.settings)
        self.renderer = PersonTrackerRenderer(
            self.tracking_frame,
            float(self.get_parameter("camera_fov_marker_range").value),
        )
        self.yolo = YoloCorrection(
            YoloSettings.from_node(self),
            self.base_frame,
        )
        self.latest_tof_seed = None
        self.processed_tof_seed = None
        self.processed_tof_seed_key = None
        self.tof_position = None
        self.tof_position_at = None
        self.last_filtered_beam_count = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        scan_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        scan_topic = str(self.get_parameter("scan_topic").value)
        filtered_topic = str(
            self.get_parameter("filtered_scan_topic").value
        )
        if filtered_topic == scan_topic:
            raise ValueError(
                "filtered_scan_topic must differ from the raw scan_topic"
            )
        self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, scan_qos
        )
        self.filtered_scan_publisher = self.create_publisher(
            LaserScan, filtered_topic, scan_qos
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped, "/person_pose", 10
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray, "/person_tracker/markers", 10
        )
        self.person_marker_publisher = self.create_publisher(
            MarkerArray, "/person_tracker/person_marker", 10
        )
        self.status_publisher = self.create_publisher(
            String, "/person_tracker/status", 10
        )
        self.create_subscription(
            Empty, "/big_brain/reset", self.reset_callback, 10
        )
        self.zmq_running = True
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 200)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        endpoint = str(self.get_parameter("tof_endpoint").value)
        self.zmq_socket.bind(endpoint)
        self.zmq_thread = threading.Thread(
            target=self._receive_observations,
            name="person-observations",
            daemon=True,
        )
        self.zmq_thread.start()
        self.get_logger().info(
            f"Person lidar tracker idle in {self.tracking_frame}; "
            f"waiting for a ToF seed on {endpoint}"
        )

    def reset(self) -> None:
        """Clear all state belonging to the current person track."""
        self.state.reset()
        self.yolo.reset()
        self.latest_tof_seed = None
        self.processed_tof_seed = None
        self.processed_tof_seed_key = None
        self.tof_position = None
        self.tof_position_at = None
        self.last_filtered_beam_count = 0
        self._publish_disabled(self.get_clock().now().to_msg())

    def reset_callback(self, _message: Empty) -> None:
        self.reset()
        self.get_logger().info("Person tracker reset")

    def _receive_observations(self) -> None:
        while self.zmq_running:
            try:
                message = self.zmq_socket.recv_json()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                return
            if message.get("type") == "person_tof_position":
                self.latest_tof_seed = message
            else:
                self.yolo.accept(message)

    def _lookup_transform(
        self,
        source_frame: str,
        stamp,
    ):
        try:
            return self.tf_buffer.lookup_transform(
                self.tracking_frame,
                source_frame,
                Time.from_msg(stamp),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as error:
            self.get_logger().warning(
                f"Cannot transform {source_frame} to {self.tracking_frame}: "
                f"{error}",
                throttle_duration_sec=2.0,
            )
            return None

    def _lookup_transform_at_monotonic(
        self,
        source_frame: str,
        captured_at: float,
    ):
        age = max(0.0, time.monotonic() - captured_at)
        stamp = (
            self.get_clock().now()
            - Duration(seconds=age + self.TF_HISTORY_MARGIN_SECONDS)
        ).to_msg()
        return self._lookup_transform(source_frame, stamp)

    def scan_callback(self, scan: LaserScan) -> None:
        seed = self.latest_tof_seed
        seed_key = None
        seed_is_fresh = False
        if seed is not None:
            try:
                captured_at = float(seed["captured_at_monotonic"])
                seed_key = (
                    str(seed["session_id"]),
                    str(seed["target"]),
                    captured_at,
                )
                seed_is_fresh = (
                    time.monotonic() - captured_at
                    <= self.settings.seed_max_age_seconds
                )
            except (KeyError, TypeError, ValueError):
                seed_key = None

        seed_is_new = (
            seed_key is not None
            and seed_key != self.processed_tof_seed_key
        )
        if seed_is_new and not seed_is_fresh:
            self.processed_tof_seed = seed
            self.processed_tof_seed_key = seed_key
            seed_is_new = False

        if (
            self.state.acquisition_state == "disabled"
            and seed_is_new
        ):
            self.state.activate()
            self.get_logger().info(
                "Received first ToF seed; lidar person tracking enabled"
            )
        if self.state.acquisition_state == "disabled":
            self._publish_filtered_scan(scan)
            self._publish_disabled(scan.header.stamp)
            return

        scan_transform = self._lookup_transform(
            scan.header.frame_id, scan.header.stamp
        )
        if scan_transform is None:
            self._publish_filtered_scan(scan)
            return

        clusters = clusters_from_scan(
            scan,
            scan_transform,
            self.settings.cluster_break_distance,
            self.settings.min_cluster_points,
        )
        stamp_seconds = (
            float(scan.header.stamp.sec)
            + float(scan.header.stamp.nanosec) / 1_000_000_000.0
        )
        visual_context = self.yolo.context(
            self._lookup_transform_at_monotonic
        )
        visual_constraint = (
            visual_context.constraint if visual_context is not None else None
        )
        camera_view = (
            visual_context.camera_view if visual_context is not None else None
        )

        tof_position = None
        if seed_is_new:
            seed_transform = self._lookup_transform_at_monotonic(
                seed["frame_id"], captured_at
            )
            if seed_transform is not None:
                seed_position = transform_points(
                    np.array([[seed["x"], seed["y"]]], dtype=float),
                    seed_transform,
                )[0]
                acceptance = self.state.accept_tof(
                    seed_position,
                    captured_at,
                )
                if acceptance != "deferred":
                    self.processed_tof_seed = seed
                    self.processed_tof_seed_key = seed_key
                    self.tof_position = seed_position
                    self.tof_position_at = captured_at
                    tof_position = seed_position

        events = self.state.update(
            clusters,
            stamp_seconds,
            tof_position,
            visual_context,
        )
        if "initialized" in events:
            self.get_logger().info("Initialized person pose from ToF seed")
        if "leg_acquired" in events:
            self.get_logger().info("Acquired clean lidar leg support")
        if "visual_recovery_started" in events:
            self.get_logger().warning(
                "Visual evidence rejected the current lidar identity"
            )
        if "visual_found" in events:
            self.get_logger().info(
                "YOLO target found; centering robot during validation"
            )
        if "visual_centered" in events:
            self.get_logger().info(
                "YOLO target centered; holding still for visual validation"
            )
        if "visual_validated" in events:
            self.get_logger().info(
                "YOLO target validated; waiting for fresh ToF seed"
            )
        if "visual_recovery_failed" in events:
            self.get_logger().error("Visual person recovery timed out")

        seed_marker_position = (
            self.tof_position
            if self.tof_position_at is not None
            and time.monotonic() - self.tof_position_at
            <= self.settings.seed_max_age_seconds
            else None
        )
        self._publish_filtered_scan(scan, scan_transform)
        self._publish_tracking(
            scan.header.stamp,
            clusters,
            seed_marker_position,
            visual_context,
        )

    def _publish_filtered_scan(self, scan: LaserScan, transform=None) -> None:
        if (
            transform is not None
            and self.state.track is not None
            and self.state.published_state == "tracking"
        ):
            filtered, count = filter_scan_around_person(
                scan,
                transform,
                self.state.track.position,
                self.settings.person_filter_radius,
            )
        else:
            filtered, count = copy.deepcopy(scan), 0
        self.last_filtered_beam_count = count
        self.filtered_scan_publisher.publish(filtered)

    def _publish_disabled(self, stamp) -> None:
        self.status_publisher.publish(String(data='{"state":"disabled"}'))
        self.marker_publisher.publish(self.renderer.clear(stamp))
        self.person_marker_publisher.publish(self.renderer.clear(stamp))

    def _publish_tracking(
        self,
        stamp,
        clusters: list[Cluster],
        tof_position,
        visual_context,
    ) -> None:
        camera_view = (
            visual_context.camera_view if visual_context is not None else None
        )
        visual_constraint = (
            visual_context.constraint if visual_context is not None else None
        )
        if self.state.track is not None and self.state.published_state in {
            "seeded",
            "tracking",
            "recovering",
        }:
            pose = PoseStamped()
            pose.header.frame_id = self.tracking_frame
            pose.header.stamp = stamp
            pose.pose.position.x = float(self.state.track.position[0])
            pose.pose.position.y = float(self.state.track.position[1])
            pose.pose.orientation.w = 1.0
            try:
                transform = self.tf_buffer.lookup_transform(
                    "map", self.tracking_frame, Time()
                )
                pose_map = tf2_geometry_msgs.do_transform_pose_stamped(
                    pose, transform
                )
            except TransformException as error:
                self.get_logger().warning(
                    f"Cannot publish /person_pose in map: {error}",
                    throttle_duration_sec=2.0,
                )
            else:
                pose_map.header.stamp = self.get_clock().now().to_msg()
                self.pose_publisher.publish(pose_map)

        observation = (
            visual_context.observation if visual_context is not None else None
        )
        status = {
            "state": self.state.published_state,
            "seed_accepted_age_seconds": (
                time.monotonic() - self.tof_position_at
                if self.tof_position_at is not None else None
            ),
            "candidate_count": len(self.state.last_detections),
            "cluster_count": len(clusters),
            "leg_cluster_count": len(self.state.leg_clusters),
            "clear_leg_cluster_count": sum(
                id(cluster) in self.state.clear_cluster_ids
                for cluster in self.state.leg_clusters
            ),
            "anchor_owned_leg_cluster_count": sum(
                id(cluster) in self.state.stationary_cluster_ids
                for cluster in self.state.leg_clusters
            ),
            "near_anchor_leg_cluster_count": sum(
                id(cluster) in self.state.near_anchor_cluster_ids
                for cluster in self.state.leg_clusters
            ),
            "selected_leg_cluster_count": len(
                self.state.selected_clusters
            ),
            "stationary_anchor_count": len(
                self.state.anchor_map.anchors
            ),
            "confirmed_stationary_anchor_count": sum(
                anchor.observations
                >= self.settings.stationary_confirm_scans
                for anchor in self.state.anchor_map.anchors
            ),
            "frozen_stationary_anchor_count": sum(
                anchor.frozen for anchor in self.state.anchor_map.anchors
            ),
            "seed_mismatch_hits": self.state.seed_mismatch_hits,
            "filtered_beam_count": self.last_filtered_beam_count,
            "target_observation_kind": (
                self.state.track.observation_kind
                if self.state.track is not None else None
            ),
            "search_radius": self.state.active_search_radius,
            "search_center": (
                self.state.active_search_center.tolist()
                if self.state.active_search_center is not None else None
            ),
            "search_fallback_center": (
                self.state.active_fallback_center.tolist()
                if self.state.active_fallback_center is not None else None
            ),
            "search_phase": self.state.search_phase,
            "search_expanded": self.state.search_expanded,
            "camera_view_active": observation is not None,
            "visual_bearing_active": (
                observation is not None and observation.person_visible
            ),
            "visual_search_direction": self.state.visual_search_direction,
            "last_visual_base_bearing_rad": (
                self.state.last_visual_base_bearing_rad
            ),
            "last_visual_camera_bearing_rad": (
                self.state.last_visual_camera_bearing_rad
            ),
        }
        self.status_publisher.publish(
            String(data=json.dumps(status, separators=(",", ":")))
        )
        self.marker_publisher.publish(
            self.renderer.markers(
                stamp,
                clusters,
                self.state,
                tof_position,
                camera_view,
                visual_constraint,
            )
        )
        self.person_marker_publisher.publish(
            self.renderer.person_markers(stamp, self.state)
        )

    def destroy_node(self):
        self.zmq_running = False
        self.zmq_thread.join(timeout=1.0)
        self.zmq_socket.close(linger=0)
        self.zmq_context.term()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PersonLidarTracker()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
