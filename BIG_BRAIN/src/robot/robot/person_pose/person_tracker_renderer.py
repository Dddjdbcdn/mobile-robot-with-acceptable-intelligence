from __future__ import annotations

import math

import numpy as np
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from robot.person_pose.lidar_clusters import Cluster
from robot.person_pose.yolo_correction import CameraView, VisualConstraint


class PersonTrackerRenderer:
    """Build RViz markers without coupling rendering to the ROS node."""

    def __init__(
        self,
        tracking_frame: str,
        camera_fov_range: float = 3.0,
    ) -> None:
        self.tracking_frame = tracking_frame
        self.camera_fov_range = camera_fov_range

    def clear(self, stamp) -> MarkerArray:
        markers = MarkerArray()
        marker = Marker()
        marker.header.frame_id = self.tracking_frame
        marker.header.stamp = stamp
        marker.action = Marker.DELETEALL
        markers.markers.append(marker)
        return markers

    def markers(
        self,
        stamp,
        clusters: list[Cluster],
        state,
        seed_position,
        camera_view: CameraView | None,
        visual_constraint: VisualConstraint | None = None,
    ) -> MarkerArray:
        markers = self.clear(stamp)
        lifetime = DurationMsg(sec=0, nanosec=300_000_000)

        raw = self._cluster_points_marker(
            stamp,
            "lidar_clusters",
            clusters,
            (0.55, 0.55, 0.55, 0.55),
            0.025,
            lifetime,
        )
        markers.markers.append(raw)
        markers.markers.append(self._cluster_points_marker(
            stamp,
            "leg_clusters",
            state.leg_clusters,
            (1.0, 0.75, 0.0, 0.90),
            0.045,
            lifetime,
        ))
        markers.markers.append(self._cluster_points_marker(
            stamp,
            "selected_leg_clusters",
            state.selected_clusters,
            (0.0, 1.0, 0.2, 1.0),
            0.075,
            lifetime,
        ))

        self._append_anchor_debug_markers(
            markers,
            raw,
            state,
            lifetime,
        )

        if camera_view is not None:
            markers.markers.append(self._camera_fov_marker(
                stamp,
                camera_view,
                lifetime,
            ))
        if visual_constraint is not None:
            markers.markers.append(self._person_bearing_fov_marker(
                stamp,
                visual_constraint,
                lifetime,
            ))

        if seed_position is not None:
            seed = Marker()
            seed.header = raw.header
            seed.ns = "tof_seed"
            seed.id = 2
            seed.type = Marker.SPHERE
            seed.action = Marker.ADD
            seed.pose.position.x = float(seed_position[0])
            seed.pose.position.y = float(seed_position[1])
            seed.pose.position.z = 0.18
            seed.pose.orientation.w = 1.0
            seed.scale.x = seed.scale.y = seed.scale.z = 0.16
            seed.color.r = seed.color.g = 1.0
            seed.color.a = 0.95
            markers.markers.append(seed)

        if state.track is not None:
            self._append_track_debug_markers(
                markers, raw, state, lifetime
            )
        return markers

    def person_markers(self, stamp, state) -> MarkerArray:
        """Build the person body/label layer for its dedicated RViz topic."""
        markers = self.clear(stamp)
        if state.track is None:
            return markers

        lifetime = DurationMsg(sec=0, nanosec=300_000_000)
        header = markers.markers[0].header
        markers.markers.append(self._tracked_person_marker(
            header, state, lifetime
        ))
        markers.markers.append(self._tracked_person_label(
            header, state, lifetime
        ))
        return markers

    def _camera_fov_marker(
        self,
        stamp,
        camera_view: CameraView,
        lifetime: DurationMsg,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.tracking_frame
        marker.header.stamp = stamp
        marker.ns = "camera_fov"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.lifetime = lifetime

        marker.color.r = 0.05
        marker.color.g = 0.55
        marker.color.b = 1.0
        marker.color.a = 0.24

        angles = np.linspace(
            camera_view.bearing - camera_view.half_fov,
            camera_view.bearing + camera_view.half_fov,
            25,
        )
        origin = Point(
            x=float(camera_view.origin[0]),
            y=float(camera_view.origin[1]),
            z=0.025,
        )
        edge = [
            Point(
                x=float(
                    camera_view.origin[0]
                    + self.camera_fov_range * math.cos(angle)
                ),
                y=float(
                    camera_view.origin[1]
                    + self.camera_fov_range * math.sin(angle)
                ),
                z=0.025,
            )
            for angle in angles
        ]
        marker.points = [
            point
            for start, end in zip(edge, edge[1:])
            for point in (origin, start, end)
        ]
        return marker

    def _person_bearing_fov_marker(
        self,
        stamp,
        visual_constraint: VisualConstraint,
        lifetime: DurationMsg,
    ) -> Marker:
        """Draw the same bearing gate used by the visual correction."""
        marker = Marker()
        marker.header.frame_id = self.tracking_frame
        marker.header.stamp = stamp
        marker.ns = "person_bearing_fov"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.lifetime = lifetime

        marker.color.r = 1.0
        marker.color.g = 0.35
        marker.color.b = 0.05
        marker.color.a = 0.38

        angles = np.linspace(
            visual_constraint.bearing - visual_constraint.gate,
            visual_constraint.bearing + visual_constraint.gate,
            9,
        )
        origin = Point(
            x=float(visual_constraint.origin[0]),
            y=float(visual_constraint.origin[1]),
            z=0.035,
        )
        edge = [
            Point(
                x=float(
                    visual_constraint.origin[0]
                    + self.camera_fov_range * math.cos(angle)
                ),
                y=float(
                    visual_constraint.origin[1]
                    + self.camera_fov_range * math.sin(angle)
                ),
                z=0.035,
            )
            for angle in angles
        ]
        marker.points = [
            point
            for start, end in zip(edge, edge[1:])
            for point in (origin, start, end)
        ]
        return marker

    def _append_track_debug_markers(
        self,
        markers: MarkerArray,
        raw: Marker,
        state,
        lifetime: DurationMsg,
    ) -> None:
        velocity_marker = self._velocity_marker(raw, state, lifetime)
        if velocity_marker is not None:
            markers.markers.append(velocity_marker)

        search_radius = float(getattr(state, "active_search_radius", 0.0))
        search_expanded = bool(getattr(state, "search_expanded", False))
        search_phase = getattr(state, "search_phase", "inactive")
        search_center = getattr(state, "active_search_center", None)
        fallback_center = getattr(state, "active_fallback_center", None)
        if search_center is None:
            search_center = state.track.position
        if search_radius > 0.0:
            if search_expanded:
                primary_color = (1.0, 0.10, 0.85, 0.90)
            elif search_phase == "fallback":
                primary_color = (0.20, 1.0, 1.0, 0.75)
            else:
                primary_color = (0.10, 0.55, 1.0, 0.85)
            markers.markers.append(self._search_radius_marker(
                raw,
                marker_id=5,
                center=search_center,
                radius=search_radius,
                color=primary_color,
                line_width=0.030,
                lifetime=lifetime,
            ))
            if fallback_center is not None:
                markers.markers.append(self._search_radius_marker(
                    raw,
                    marker_id=6,
                    center=fallback_center,
                    radius=search_radius,
                    color=(0.20, 1.0, 1.0, 0.55),
                    line_width=0.018,
                    lifetime=lifetime,
                ))

    @staticmethod
    def _velocity_marker(raw, state, lifetime) -> Marker | None:
        """Draw velocity only while directional association is active."""
        track = getattr(state, "track", None)
        settings = getattr(state, "settings", None)
        velocity = getattr(track, "velocity", None)
        if velocity is None or settings is None:
            return None

        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
            return None
        speed = float(np.linalg.norm(velocity))
        threshold = max(
            0.0,
            float(getattr(settings, "motion_direction_min_speed", math.inf)),
        )
        if speed < max(threshold, 1e-6):
            return None

        start = np.asarray(track.position, dtype=float)
        arrow_length = float(np.clip(speed * 0.75, 0.12, 0.40))
        end = start + velocity * (arrow_length / speed)

        marker = Marker()
        marker.header = raw.header
        marker.ns = "person_velocity"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.025
        marker.scale.y = 0.065
        marker.scale.z = 0.080
        marker.color.r = 1.0
        marker.color.g = 0.85
        marker.color.b = 0.05
        marker.color.a = 0.95
        marker.lifetime = lifetime
        marker.points = [
            Point(x=float(start[0]), y=float(start[1]), z=0.20),
            Point(x=float(end[0]), y=float(end[1]), z=0.20),
        ]
        return marker

    @staticmethod
    def _tracked_person_marker(header, state, lifetime) -> Marker:
        tracked = Marker()
        tracked.header = header
        tracked.ns = "tracked_person"
        tracked.id = 3
        tracked.type = Marker.CYLINDER
        tracked.action = Marker.ADD
        tracked.pose.position.x = float(state.track.position[0])
        tracked.pose.position.y = float(state.track.position[1])
        tracked.pose.position.z = 0.5
        tracked.pose.orientation.w = 1.0
        tracked.scale.x = tracked.scale.y = 0.30
        tracked.scale.z = 1.0
        if state.track.state == "tracking":
            tracked.color.g = 1.0
        elif state.track.state == "recovering":
            tracked.color.r = 1.0
            tracked.color.g = 0.55
        else:
            tracked.color.r = 1.0
        tracked.color.a = 0.75
        tracked.lifetime = lifetime
        return tracked

    @staticmethod
    def _tracked_person_label(header, state, lifetime) -> Marker:
        label = Marker()
        label.header = header
        label.ns = "tracked_person_label"
        label.id = 4
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(state.track.position[0])
        label.pose.position.y = float(state.track.position[1])
        label.pose.position.z = 1.2
        label.pose.orientation.w = 1.0
        label.scale.z = 0.18
        label.color.r = label.color.g = label.color.b = 1.0
        label.color.a = 1.0
        label.lifetime = lifetime
        search_radius = float(getattr(state, "active_search_radius", 0.0))
        search_phase = getattr(state, "search_phase", "inactive")
        search_expanded = bool(getattr(state, "search_expanded", False))
        if search_expanded:
            search_phase = "expanded"
        label.text = (
            f"person: {state.track.state} "
            f"({state.track.observation_kind})\n"
            f"search: {search_phase} "
            f"{search_radius:.2f} m"
        )
        return label

    def _append_anchor_debug_markers(
        self,
        markers: MarkerArray,
        raw: Marker,
        state,
        lifetime: DurationMsg,
    ) -> None:
        """Show why each leg cluster is accepted or rejected near anchors."""
        stationary_ids = set(
            getattr(state, "stationary_cluster_ids", set())
        )
        near_anchor_ids = set(
            getattr(state, "near_anchor_cluster_ids", set())
        )
        clear_ids = set(getattr(state, "clear_cluster_ids", set()))
        anchor_owned = [
            cluster for cluster in state.leg_clusters
            if id(cluster) in stationary_ids
        ]
        near_anchor = [
            cluster for cluster in state.leg_clusters
            if id(cluster) in near_anchor_ids
        ]
        clear = [
            cluster for cluster in state.leg_clusters
            if id(cluster) in clear_ids
        ]

        # These overlays are deliberately larger than the generic orange leg
        # points so the anchor classifier's decision remains obvious in RViz.
        markers.markers.append(self._cluster_points_marker(
            raw.header.stamp,
            "clear_leg_clusters",
            clear,
            (0.0, 0.80, 1.0, 1.0),
            0.060,
            lifetime,
        ))
        markers.markers.append(self._cluster_points_marker(
            raw.header.stamp,
            "near_anchor_leg_clusters",
            near_anchor,
            (1.0, 0.0, 0.85, 1.0),
            0.070,
            lifetime,
        ))
        markers.markers.append(self._cluster_points_marker(
            raw.header.stamp,
            "anchor_owned_leg_clusters",
            anchor_owned,
            (1.0, 0.05, 0.05, 1.0),
            0.070,
            lifetime,
        ))

        anchors = list(state.anchor_map.anchors)
        frozen = Marker()
        frozen.header = raw.header
        frozen.ns = "frozen_stationary_anchors"
        frozen.id = 0
        frozen.type = Marker.SPHERE_LIST
        frozen.action = Marker.ADD
        frozen.scale.x = 0.075
        frozen.scale.y = 0.075
        frozen.scale.z = 0.075
        frozen.color.r = 0.0
        frozen.color.g = 0.80
        frozen.color.b = 1.0
        frozen.color.a = 0.90
        frozen.lifetime = lifetime
        frozen.points = [
            Point(
                x=float(anchor.position[0]),
                y=float(anchor.position[1]),
                z=0.08,
            )
            for anchor in anchors
            if bool(getattr(anchor, "frozen", False))
        ]
        markers.markers.append(frozen)

        confirmed = list(getattr(state.anchor_map, "confirmed", []))
        match_radius = float(getattr(
            getattr(state, "settings", None),
            "stationary_match_radius",
            0.0,
        ))
        near_radius = max(match_radius, float(getattr(
            getattr(state, "settings", None),
            "near_anchor_radius",
            match_radius,
        )))
        if confirmed and match_radius > 0.0:
            markers.markers.append(self._anchor_radius_marker(
                raw,
                confirmed,
                match_radius,
                lifetime,
            ))
        if confirmed and near_radius > match_radius:
            markers.markers.append(self._anchor_radius_marker(
                raw,
                confirmed,
                near_radius,
                lifetime,
                namespace="stationary_anchor_near_radius",
                color=(1.0, 0.0, 0.85, 0.55),
            ))
        self._append_anchor_relation_marker(
            markers,
            raw,
            "anchor_owned_relations",
            anchor_owned,
            confirmed,
            (1.0, 0.05, 0.05, 0.95),
            lifetime,
        )
        self._append_anchor_relation_marker(
            markers,
            raw,
            "near_anchor_relations",
            near_anchor,
            confirmed,
            (1.0, 0.0, 0.85, 0.95),
            lifetime,
        )

    @staticmethod
    def _anchor_radius_marker(
        raw: Marker,
        anchors,
        radius: float,
        lifetime: DurationMsg,
        *,
        namespace: str = "stationary_anchor_match_radius",
        color: tuple[float, float, float, float] = (
            1.0, 0.2, 0.05, 0.75,
        ),
    ) -> Marker:
        marker = Marker()
        marker.header = raw.header
        marker.ns = namespace
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.012
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.lifetime = lifetime
        angles = np.linspace(0.0, 2.0 * np.pi, 37)
        for anchor in anchors:
            points = [
                Point(
                    x=float(anchor.position[0] + radius * np.cos(angle)),
                    y=float(anchor.position[1] + radius * np.sin(angle)),
                    z=0.06,
                )
                for angle in angles
            ]
            marker.points.extend(
                point
                for start, end in zip(points, points[1:])
                for point in (start, end)
            )
        return marker

    @staticmethod
    def _append_anchor_relation_marker(
        markers: MarkerArray,
        raw: Marker,
        namespace: str,
        clusters: list[Cluster],
        anchors,
        color: tuple[float, float, float, float],
        lifetime: DurationMsg,
    ) -> None:
        if not clusters or not anchors:
            return
        marker = Marker()
        marker.header = raw.header
        marker.ns = namespace
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.018
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.lifetime = lifetime
        for cluster in clusters:
            anchor = min(
                anchors,
                key=lambda item: float(np.linalg.norm(
                    cluster.center - item.position
                )),
            )
            marker.points.extend((
                Point(
                    x=float(cluster.center[0]),
                    y=float(cluster.center[1]),
                    z=0.09,
                ),
                Point(
                    x=float(anchor.position[0]),
                    y=float(anchor.position[1]),
                    z=0.09,
                ),
            ))
        markers.markers.append(marker)

    @staticmethod
    def _search_radius_marker(
        raw: Marker,
        *,
        marker_id: int,
        center: np.ndarray,
        radius: float,
        color: tuple[float, float, float, float],
        line_width: float,
        lifetime: DurationMsg,
    ) -> Marker:
        """Build one circle for an active tracker search gate."""
        marker = Marker()
        marker.header = raw.header
        marker.ns = "person_search_radius"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = line_width
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.lifetime = lifetime
        angles = np.linspace(0.0, 2.0 * np.pi, 73)
        marker.points = [
            Point(
                x=float(center[0] + radius * np.cos(angle)),
                y=float(center[1] + radius * np.sin(angle)),
                z=0.04,
            )
            for angle in angles
        ]
        return marker

    def _cluster_points_marker(
        self,
        stamp,
        namespace: str,
        clusters: list[Cluster],
        color: tuple[float, float, float, float],
        point_size: float,
        lifetime: DurationMsg,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.tracking_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = point_size
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.lifetime = lifetime
        marker.points = [
            Point(x=float(point[0]), y=float(point[1]), z=0.05)
            for cluster in clusters
            for point in cluster.points
        ]
        return marker
