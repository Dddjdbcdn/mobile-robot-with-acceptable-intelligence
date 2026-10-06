"""ROS/TF orchestration and ZeroMQ publication for map snapshots."""

import json
import threading
import time
import uuid
import math

import cv2
import zmq

from robot.map.map_logic import MapLogic
from robot.map.map_renderer import MapRenderer
from robot.map.room_registry import RoomRegistry

BASE_FRAME = "base_footprint"
IMAGE_ENDPOINT = "tcp://127.0.0.1:5559"


class MapImageStream:
    def __init__(self, node, tf_buffer, context):
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

        self.node = node
        self.tf = tf_buffer
        self.base_frame = BASE_FRAME

        self.logic = MapLogic()
        self.room_registry = RoomRegistry(self.logic)
        self.renderer = MapRenderer()
        self.context = context
        self.socket = None

        self.state_lock = threading.Lock()
        self.map_msg = None
        self.cost_msg = None
        self._map_revision = 0
        self._map_received_at_unix_ns = 0
        self.analysis_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.sequence = 0

        self.search_overlay = None
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        node.create_subscription(OccupancyGrid, "/map", self._map, qos)
        node.create_subscription(
            OccupancyGrid, "/global_costmap/costmap", self._cost, qos
        )

        self.stream_thread = threading.Thread(
            target=self._stream_loop,
            name="map-stream",
            daemon=True,
        )
        self.stream_thread.start()

    def _map(self, msg):
        with self.state_lock:
            self.map_msg = msg
            self._map_revision += 1
            self._map_received_at_unix_ns = time.time_ns()

    def _cost(self, msg):
        with self.state_lock:
            self.cost_msg = msg

    def map_revision(self):
        """Return the revision of the latest received /map message."""
        with self.state_lock:
            return self._map_revision

    def _pose(self, frame_id):
        import rclpy.time
        try:
            transform = self.tf.lookup_transform(
                frame_id,
                self.base_frame,
                rclpy.time.Time(),
            )
            position = transform.transform.translation
            orientation = transform.transform.rotation
        except Exception:
            return None

        def yaw_of(q):
            return math.atan2(
                2 * (q.w * q.z + q.x * q.y),
                1 - 2 * (q.y * q.y + q.z * q.z),
            )

        return {
            "x": float(position.x),
            "y": float(position.y),
            "yaw": yaw_of(orientation),
            "frame_id": frame_id,
        }

    @staticmethod
    def _overlay_pose(item):
        try:
            yaw = item["yaw"] if "yaw" in item else item["angle"]
            return {
                "x": float(item["x"]),
                "y": float(item["y"]),
                "yaw": float(yaw),
                "frame_id": str(item.get("frame_id") or "map"),
            }
        except (KeyError, TypeError, ValueError):
            return None

    def _apply_command(self, message):
        """Apply one ordered request and return its operation."""
        if not isinstance(message, dict) or message.get("schema_version") != 1:
            raise ValueError("unsupported map request schema")
        action_id = str(message.get("action_id") or "")
        operation = str(message.get("operation") or "")
        if operation == "snapshot":
            return operation
        if operation == "clear":
            if self.search_overlay and action_id == self.search_overlay.get("action_id"):
                self.search_overlay = None
            return operation
        if operation != "set" or not action_id:
            raise ValueError(f"unsupported map operation: {operation!r}")

        revision = int(message.get("revision", 0))
        if (
            self.search_overlay
            and action_id == self.search_overlay.get("action_id")
            and revision < int(self.search_overlay.get("revision", 0))
        ):
            return operation
        mode = str(message.get("mode") or "exploration")
        observation = message.get("observation")
        normalized_observation = None
        if isinstance(observation, dict):
            observation_pose = self._overlay_pose(observation.get("robot_pose"))
            if observation_pose is not None:
                normalized_observation = {
                    "robot_pose": observation_pose,
                    "pan_angle": float(observation.get("pan_angle", 95.0)),
                    "tilt_angle": float(observation.get("tilt_angle", 90.0)),
                    "contextual_clue": str(observation.get("contextual_clue") or ""),
                    "candidate_type": str(observation.get("candidate_type") or ""),
                    "movement_limit": int(observation.get("movement_limit") or 0),
                    "movements_used": int(observation.get("movements_used") or 0),
                    "remaining_waypoints": int(
                        observation.get("remaining_waypoints") or 0
                    ),
                    "reassessment_limit": int(
                        observation.get("reassessment_limit") or 0
                    ),
                    "reassessments_used": int(
                        observation.get("reassessments_used") or 0
                    ),
                }
        search_poses = []
        for item in message.get("search_poses") or []:
            pose = self._overlay_pose(item)
            if pose is None:
                continue
            pose.update({
                "kind": str(item.get("kind") or "pose"),
                "reason": str(item.get("reason") or ""),
                "step": int(item.get("step", 0)),
                "views": [self._normalize_search_view(view)
                          for view in item.get("views") or []],
            })
            search_poses.append(pose)
        self.search_overlay = {
            "action_id": action_id,
            "revision": revision,
            "frame_id": str(message.get("frame_id") or "map"),
            "mode": mode,
            "observation": normalized_observation,
            "search_poses": search_poses,
            "camera_horizontal_fov_deg": float(message.get("camera_horizontal_fov_deg", 85.0)),
            "camera_reliable_range_m": float(message.get("camera_reliable_range_m", 2.0)),
        }
        return operation

    @staticmethod
    def _normalize_search_view(view):
        normalized = {
            "pan": float(view["pan"]),
            "tilt": float(view["tilt"]),
        }
        for key in (
            "candidate_type", "contextual_clue",
            "movement_limit", "movements_used",
            "remaining_waypoints", "reassessment_limit",
            "reassessments_used",
        ):
            if view.get(key) is not None:
                normalized[key] = view[key]
        return normalized

    def prepare_map(self):
        """Analyze the latest messages into one map-and-pose context."""
        with self.analysis_lock:
            with self.state_lock:
                map_msg = self.map_msg
                cost_msg = self.cost_msg
                map_revision = self._map_revision
                map_received_at_unix_ns = self._map_received_at_unix_ns
            if map_msg is None:
                raise ValueError("Waiting for OccupancyGrid on /map")
            if cost_msg is None:
                raise ValueError(
                    "Waiting for OccupancyGrid on /global_costmap/costmap"
                )

            pose = self._pose(map_msg.header.frame_id)
            if pose is None:
                raise ValueError(
                    f"TF unavailable: {map_msg.header.frame_id} -> {self.base_frame}"
                )

            started_at = time.monotonic()
            try:
                prepared = self.logic.prepare(map_msg, cost_msg, pose)
            except Exception as error:
                self.node.get_logger().warning(f"Map analysis: {error}")
                raise ValueError(
                    f"Map analysis failed: {type(error).__name__}: {error}"
                ) from error

            finished_at = time.monotonic()
            prepared["analysis_ms"] = round(
                (finished_at - started_at) * 1000, 1
            )
            prepared["prepared_at_monotonic"] = finished_at
            prepared["prepared_at_unix_ns"] = time.time_ns()
            prepared["map_revision"] = map_revision
            prepared["map_received_at_unix_ns"] = map_received_at_unix_ns
            self.room_registry.ensure_startup_room(prepared, pose)
            return prepared, pose

    def _stream_loop(self):
        """Serve ordered map requests without background map analysis."""
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)

        try:
            self.socket.bind(IMAGE_ENDPOINT)
            poller = zmq.Poller()
            poller.register(self.socket, zmq.POLLIN)

            while not self.stop_event.is_set():
                events = dict(poller.poll(timeout=100))
                if self.socket in events:
                    self._handle_request(self.socket.recv_json())
        except Exception as error:
            if not self.stop_event.is_set():
                self.node.get_logger().error(f"Map stream: {error}")
        finally:
            self.socket.close(0)
            self.socket = None

    def _handle_request(self, message):
        try:
            operation = self._apply_command(message)
            if operation != "snapshot":
                self.socket.send_json({"ok": True, "operation": operation})
                return
            crop_size_m = message.get("crop_size_m")
            render_frontiers = bool(message.get("render_frontiers", False))
            pending_navigation_pose = self._overlay_pose(
                message.get("pending_navigation_pose")
            )
            try:
                prepared, pose = self.prepare_map()
            except ValueError as error:
                self.socket.send_json({
                    "ok": False,
                    "error": str(error),
                    "retryable": True,
                })
                return
            metadata, jpeg = self._encode_snapshot(
                prepared,
                pose,
                crop_size_m=crop_size_m,
                render_frontiers=render_frontiers,
                pending_navigation_pose=pending_navigation_pose,
            )
            response_metadata = dict(metadata)
            response_metadata["snapshot_request_id"] = str(
                message.get("request_id") or ""
            )
            self.socket.send_multipart([
                json.dumps(response_metadata).encode("utf-8"), jpeg,
            ])
        except Exception as error:
            self.node.get_logger().warning(f"Map request: {error}")
            self.socket.send_json({
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "retryable": False,
            })

    def _compose_snapshot(
        self, prepared, pose, camera_pan_angle, crop_size_m=None,
        render_frontiers=False, pending_navigation_pose=None,
    ):
        """Coordinate planning, visibility analysis, rendering, and metadata."""
        overlay = self.search_overlay
        grid = prepared["grid"]
        coverage_counts = self.logic.coverage_counts(grid, overlay)
        candidates = self.logic.plan_candidates(
            prepared,
            pose,
            overlay,
            coverage=coverage_counts > 0,
        )

        live_fov = self.logic.live_camera_fov(pose, camera_pan_angle)
        live_mask = (
            self.logic.visibility_mask(
                grid, pose, live_fov["center_yaw_rad"],
                live_fov["horizontal_fov_deg"], live_fov["max_range_m"],
            )
            if live_fov is not None and self.renderer.cfg.live_fov_enabled
            else None
        )
        observation = self.logic.observation_fov(overlay)
        observation_mask = (
            self.logic.visibility_mask(
                grid, observation["pose"], observation["center_yaw_rad"],
                observation["horizontal_fov_deg"], observation["max_range_m"],
            )
            if observation is not None else None
        )
        def render_view(view_size_m):
            # Render directly into the requested robot-relative view. This
            # keeps a crop crisp even when the global explored map is large.
            rendered = dict(prepared)
            rendered["background"], rendered["view"] = (
                self.renderer.render_background(
                    prepared, pose, view_size_m=view_size_m
                )
            )
            marker_scale = self.renderer.pose_marker_scale_for_crop(
                rendered["view"], view_size_m
            )
            image = self.renderer.render_snapshot(
                rendered, pose, candidates, coverage_counts,
                search_overlay=overlay,
                live_fov_mask=live_mask,
                observation=observation,
                observation_mask=observation_mask,
                pose_marker_scale=marker_scale,
                frontiers=(prepared["frontiers"] if render_frontiers else None),
                pending_navigation_pose=pending_navigation_pose,
            )
            return image, rendered["view"]

        image, image_view = render_view(crop_size_m)

        mode = str((overlay or {}).get("mode") or "goal")
        search_poses = (overlay or {}).get("search_poses") or []
        metadata = {
            "schema_version": 1,
            "snapshot_id": uuid.uuid4().hex,
            "snapshot_request_id": None,
            "frame_id": pose["frame_id"],
            "robot_pose": pose,
            "candidates": candidates,
            "frontiers": prepared["frontiers"],
            "pending_navigation_pose": pending_navigation_pose,
            "sample_radius_m": self.logic.cfg.sample_radius,
            "selection_mode": mode,
            "selection_policy": "vision",
            "selected_pose_id": None,
            "live_camera_fov": live_fov,
            "observation_fov": observation,
            "search_overlay": (
                {
                    "action_id": overlay.get("action_id"),
                    "revision": int(overlay.get("revision", 0)),
                    "mode": mode,
                    "candidate_type": str(
                        (overlay.get("observation") or {}).get("candidate_type") or ""
                    ),
                    "coverage_observation_count": sum(
                        len(item.get("views") or []) for item in search_poses
                    ),
                    "pose_count": len(search_poses),
                }
                if isinstance(overlay, dict) else None
            ),
            "camera_reliable_range_m": float((overlay or {}).get(
                "camera_reliable_range_m",
                self.logic.cfg.camera_fov_max_range_m,
            )),
            "image_robot_view": {
                key: image_view[key]
                for key in (
                    "xmin", "xmax", "ymin", "ymax", "origin_x",
                    "origin_y", "heading_yaw",
                )
            },
        }
        if crop_size_m is not None:
            size = float(crop_size_m)
            half = size / 2.0
            yaw = float(pose["yaw"])

            def inside_robot_crop(item):
                dx = float(item["x"]) - float(pose["x"])
                dy = float(item["y"]) - float(pose["y"])
                right = dx * math.sin(yaw) - dy * math.cos(yaw)
                forward = dx * math.cos(yaw) + dy * math.sin(yaw)
                return abs(right) <= half and abs(forward) <= half

            metadata["candidates"] = [
                item for item in metadata["candidates"]
                if inside_robot_crop(item)
            ]
            metadata["map_crop_size_m"] = size
        return image, metadata

    def _encode_snapshot(
        self, prepared, pose, crop_size_m=None, render_frontiers=False,
        pending_navigation_pose=None,
    ):
        started_at = time.monotonic()
        captured_at_unix_ns = time.time_ns()
        camera_pan_angle = self.logic.cfg.camera_center_pan_deg
        image, metadata = self._compose_snapshot(
            prepared, pose, camera_pan_angle, crop_size_m=crop_size_m,
            render_frontiers=render_frontiers,
            pending_navigation_pose=pending_navigation_pose,
        )
        ok, jpeg = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not ok:
            raise RuntimeError("OpenCV failed to encode the map JPEG")
        encoded_at = time.monotonic()
        self.sequence += 1
        metadata.update({
            "sequence": self.sequence,
            "publish_mode": "on_request",
            "analysis_ms": prepared["analysis_ms"],
            "analysis_age_ms": round(
                (started_at - prepared["prepared_at_monotonic"]) * 1000, 1
            ),
            "captured_at_unix_ns": captured_at_unix_ns,
            "generated_at_unix_ns": time.time_ns(),
            "render_ms": round((encoded_at - started_at) * 1000, 1),
        })
        return metadata, jpeg.tobytes()

    def close(self):
        self.stop_event.set()
        if self.stream_thread is not threading.current_thread():
            self.stream_thread.join()
