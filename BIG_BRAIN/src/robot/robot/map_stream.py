"""ROS/TF orchestration and ZeroMQ publication for map snapshots."""

import json
import threading
import time
import uuid
import math

import cv2
import zmq

from robot.map_logic import MapLogic
from robot.map_renderer import MapRenderer

BASE_FRAME = "base_footprint"
IMAGE_ENDPOINT = "tcp://127.0.0.1:5559"
ANALYSIS_HZ = 0.5
# "on_request" renders only when a snapshot is requested. "fixed_hz" keeps
# an encoded snapshot warm at FIXED_HZ and returns it to requesters.
DELIVERY_MODE = "on_request"
FIXED_HZ = 2.0


class MapImageStream:
    def __init__(self, node, tf_buffer, context):
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

        self.node = node
        self.tf = tf_buffer
        self.base_frame = BASE_FRAME
        self.analysis_hz = ANALYSIS_HZ
        self.delivery_mode = DELIVERY_MODE
        self.fixed_hz = FIXED_HZ
        if self.delivery_mode not in {"on_request", "fixed_hz"}:
            raise ValueError(f"Unknown map delivery mode: {self.delivery_mode}")
        if self.fixed_hz <= 0.0:
            raise ValueError("FIXED_HZ must be positive")

        self.logic = MapLogic()
        self.renderer = MapRenderer()
        self.context = context
        self.socket = None

        self.state_lock = threading.Lock()
        self.map_msg = None
        self.cost_msg = None
        self.prepared = None
        self._not_ready_reason = "Waiting for /map and /global_costmap/costmap"
        self._last_render_error = None
        self._coverage_key = None
        self._coverage_counts = None
        self._coverage_mask = None
        self.stop_event = threading.Event()
        self.sequence = 0

        self.search_overlay = None
        self._encoded_cache = None
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

    def _cost(self, msg):
        with self.state_lock:
            self.cost_msg = msg

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
            return {
                "x": float(item["x"]),
                "y": float(item["y"]),
                "yaw": float(item["yaw"]),
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
                self._encoded_cache = None
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
        mode = str(message.get("mode") or "goal")
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
                    "original_candidate_type": str(
                        observation.get("original_candidate_type") or ""
                    ),
                    "original_contextual_clue": str(
                        observation.get("original_contextual_clue")
                        or observation.get("contextual_clue") or ""
                    ),
                    "hypothesis_id": str(observation.get("hypothesis_id") or ""),
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
                    "allow_frontier": bool(
                        observation.get("allow_frontier", False)
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
            "allow_frontier": bool(message.get("allow_frontier", True)),
            "observation": normalized_observation,
            "search_poses": search_poses,
            "camera_horizontal_fov_deg": float(message.get("camera_horizontal_fov_deg", 60.0)),
            "camera_reliable_range_m": float(message.get("camera_reliable_range_m", 2.0)),
        }
        self._encoded_cache = None
        return operation

    @staticmethod
    def _normalize_search_view(view):
        normalized = {
            "pan": float(view["pan"]),
            "tilt": float(view["tilt"]),
        }
        for key in (
            "candidate_type", "hypothesis_id", "original_candidate_type",
            "contextual_clue", "original_contextual_clue",
            "movement_limit", "movements_used",
            "remaining_waypoints", "reassessment_limit",
            "reassessments_used", "allow_frontier",
        ):
            if view.get(key) is not None:
                normalized[key] = view[key]
        return normalized

    def _refresh_analysis(self):
        """Refresh the prepared map and return the delay before retrying."""
        period = 1.0 / self.analysis_hz
        with self.state_lock:
            map_msg = self.map_msg
            cost_msg = self.cost_msg
        if map_msg is None:
            self._not_ready_reason = "Waiting for OccupancyGrid on /map"
            return 0.1
        if cost_msg is None:
            self._not_ready_reason = (
                "Waiting for OccupancyGrid on /global_costmap/costmap"
            )
            return 0.1

        pose = self._pose(map_msg.header.frame_id)
        if pose is None:
            self._not_ready_reason = (
                f"TF unavailable: {map_msg.header.frame_id} -> {self.base_frame}"
            )
            return 0.1

        started_at = time.monotonic()
        try:
            prepared = self.logic.prepare(map_msg, cost_msg, pose)
            prepared["background"], prepared["view"] = (
                self.renderer.render_background(prepared, pose)
            )
        except Exception as error:
            self._not_ready_reason = (
                f"Map analysis failed: {type(error).__name__}: {error}"
            )
            self.node.get_logger().warning(f"Map analysis: {error}")
            return period

        finished_at = time.monotonic()
        prepared["analysis_ms"] = round(
            (finished_at - started_at) * 1000, 1
        )
        prepared["prepared_at_monotonic"] = finished_at
        prepared["prepared_at_unix_ns"] = time.time_ns()
        with self.state_lock:
            self.prepared = prepared
        self._not_ready_reason = None
        return period

    def _stream_loop(self):
        """Analyze continuously and serve ordered map requests."""
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)

        try:
            self.socket.bind(IMAGE_ENDPOINT)
            next_analysis = time.monotonic()
            next_fixed_render = time.monotonic()

            while not self.stop_event.is_set():
                try:
                    message = self.socket.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    message = None
                if message is not None:
                    self._handle_request(message)

                now = time.monotonic()
                if now >= next_analysis:
                    retry_delay = self._refresh_analysis()
                    next_analysis = time.monotonic() + retry_delay

                if self.delivery_mode == "fixed_hz" and now >= next_fixed_render:
                    self._encoded_cache = self._encode_snapshot()
                    next_fixed_render = time.monotonic() + 1.0 / self.fixed_hz

                delay = max(
                    0.0,
                    min(next_analysis - time.monotonic(), 0.02),
                )
                self.stop_event.wait(delay)
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
            encoded = self._encoded_cache if self.delivery_mode == "fixed_hz" else None
            if encoded is None:
                encoded = self._encode_snapshot()
                # At startup a request can arrive before the first periodic
                # analysis. Prepare immediately once instead of making the
                # client wait for the background schedule.
                if encoded is None and self.prepared is None:
                    self._refresh_analysis()
                    encoded = self._encode_snapshot()
                if self.delivery_mode == "fixed_hz":
                    self._encoded_cache = encoded
            if encoded is None:
                self.socket.send_json({
                    "ok": False,
                    "error": (
                        self._last_render_error
                        or self._not_ready_reason or "Map snapshot is not ready"
                    ),
                    "retryable": self._last_render_error is None,
                })
                return
            metadata, jpeg = encoded
            response_metadata = dict(metadata)
            response_metadata["snapshot_request_id"] = str(message.get("request_id") or "")
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

    def _compose_snapshot(self, prepared, pose, camera_pan_angle):
        """Coordinate planning, visibility analysis, rendering, and metadata."""
        overlay = self.search_overlay
        grid = prepared["grid"]
        overlay_key = (
            (overlay.get("action_id"), int(overlay.get("revision", 0)))
            if isinstance(overlay, dict)
            else (None, None)
        )
        coverage_key = (id(grid), *overlay_key)
        if coverage_key != self._coverage_key:
            self._coverage_counts = self.logic.coverage_counts(grid, overlay)
            self._coverage_mask = self._coverage_counts > 0
            self._coverage_key = coverage_key

        coverage_counts = self._coverage_counts
        candidates = self.logic.plan_candidates(
            prepared,
            pose,
            overlay,
            coverage=self._coverage_mask,
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
        image = self.renderer.render_snapshot(
            prepared, pose, candidates, coverage_counts,
            search_overlay=overlay,
            live_fov_mask=live_mask,
            observation=observation,
            observation_mask=observation_mask,
        )

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
            "doors": prepared["doors"],
            "rooms": prepared["rooms"],
            "sample_radius_m": self.logic.cfg.sample_radius,
            "selection_mode": mode,
            "selection_policy": (
                "deterministic" if mode == "exploration" else "vision"
            ),
            "selected_pose_id": (
                candidates[0]["id"]
                if mode == "exploration" and len(candidates) == 1 else None
            ),
            "live_camera_fov": live_fov,
            "observation_fov": observation,
            "search_overlay": (
                {
                    "action_id": overlay.get("action_id"),
                    "revision": int(overlay.get("revision", 0)),
                    "mode": mode,
                    "allow_frontier": bool(overlay.get("allow_frontier", True)),
                    "candidate_type": str(
                        (overlay.get("observation") or {}).get("candidate_type") or ""
                    ),
                    "hypothesis_id": str(
                        (overlay.get("observation") or {}).get("hypothesis_id") or ""
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
            "image_world_bounds": {
                key: prepared["view"][key]
                for key in ("xmin", "xmax", "ymin", "ymax")
            },
        }
        return image, metadata

    def _encode_snapshot(self):
        self._last_render_error = None
        with self.state_lock:
            prepared = self.prepared
        if prepared is None:
            return None

        pose = self._pose(prepared["frame_id"])
        if pose is None:
            self._not_ready_reason = (
                f"TF unavailable: {prepared['frame_id']} -> {self.base_frame}"
            )
            return None

        started_at = time.monotonic()
        captured_at_unix_ns = time.time_ns()
        try:
            camera_pan_angle = self.logic.cfg.camera_center_pan_deg
            image, metadata = self._compose_snapshot(
                prepared, pose, camera_pan_angle
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
                "publish_mode": self.delivery_mode,
                "analysis_hz": self.analysis_hz,
                "analysis_ms": prepared["analysis_ms"],
                "analysis_age_ms": round(
                    (started_at - prepared["prepared_at_monotonic"]) * 1000, 1
                ),
                "captured_at_unix_ns": captured_at_unix_ns,
                "generated_at_unix_ns": time.time_ns(),
                "render_ms": round((encoded_at - started_at) * 1000, 1),
            })
            self._not_ready_reason = None
            return metadata, jpeg.tobytes()
        except Exception as error:
            self._last_render_error = (
                f"Map render failed: {type(error).__name__}: {error}"
            )
            self.node.get_logger().warning(self._last_render_error)
            return None

    def close(self):
        self.stop_event.set()
        if self.stream_thread is not threading.current_thread():
            self.stream_thread.join()
