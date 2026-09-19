"""ROS/TF orchestration and ZeroMQ publication for map snapshots."""

import json
import struct
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
OVERLAY_ENDPOINT = "tcp://127.0.0.1:5560"
ANALYSIS_HZ = 0.5


class MapImageStream:
    def __init__(self, node, tf_buffer, context):
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

        self.node = node
        self.tf = tf_buffer
        self.base_frame = BASE_FRAME
        self.analysis_hz = ANALYSIS_HZ

        self.logic = MapLogic()
        self.renderer = MapRenderer()
        self.context = context
        self.socket = None
        self.overlay_socket = None

        self.state_lock = threading.Lock()
        self.map_msg = None
        self.cost_msg = None
        self.prepared = None
        self._coverage_key = None
        self._coverage_counts = None
        self._coverage_mask = None
        self.stop_event = threading.Event()
        self.sequence = 0

        self.search_overlay = None
        self.snapshot_request_id = None
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

    def _drain_overlay_commands(self):
        """Apply queued commands and report whether one snapshot was requested."""
        publish_requested = False
        while True:
            try:
                message = self.overlay_socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                return publish_requested
            if not isinstance(message, dict) or message.get("schema_version") != 1:
                continue
            action_id = str(message.get("action_id") or "")
            operation = message.get("operation")
            if operation == "snapshot":
                request_id = str(message.get("request_id") or "")
                if request_id:
                    self.snapshot_request_id = request_id
                    publish_requested = True
                continue
            if operation == "clear":
                if self.search_overlay and action_id == self.search_overlay.get("action_id"):
                    self.search_overlay = None
                continue
            if operation != "set" or not action_id:
                continue
            revision = int(message.get("revision", 0))
            if (
                self.search_overlay
                and action_id == self.search_overlay.get("action_id")
                and revision < int(self.search_overlay.get("revision", 0))
            ):
                continue
            mode = str(message.get("mode") or "goal")
            observation = message.get("observation")
            normalized_observation = None
            if isinstance(observation, dict):
                observation_pose = self._overlay_pose(
                    observation.get("robot_pose")
                )
                if observation_pose is not None:
                    normalized_observation = {
                        "robot_pose": observation_pose,
                        "pan_angle": float(observation.get("pan_angle", 95.0)),
                        "tilt_angle": float(observation.get("tilt_angle", 90.0)),
                        "contextual_clue": str(
                            observation.get("contextual_clue") or ""
                        ),
                        "clue_confidence": float(
                            observation.get("clue_confidence") or 0.0
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
                    "views": [],
                })
                pose["views"] = [
                    {"pan": float(view["pan"]), "tilt": float(view["tilt"])}
                    for view in item.get("views") or []
                ]
                search_poses.append(pose)
            self.search_overlay = {
                "action_id": action_id,
                "revision": revision,
                "frame_id": str(message.get("frame_id") or "map"),
                "mode": mode,
                "observation": normalized_observation,
                "search_poses": search_poses,
                "camera_horizontal_fov_deg": float(
                    message.get("camera_horizontal_fov_deg", 60.0)
                ),
                "camera_reliable_range_m": float(
                    message.get("camera_reliable_range_m", 2.0)
                ),
            }
            publish_requested = True

    def _refresh_analysis(self):
        """Refresh the prepared map and return the delay before retrying."""
        period = 1.0 / self.analysis_hz
        with self.state_lock:
            map_msg = self.map_msg
            cost_msg = self.cost_msg
        if map_msg is None or cost_msg is None:
            return 0.1

        pose = self._pose(map_msg.header.frame_id)
        if pose is None:
            return 0.1

        started_at = time.monotonic()
        try:
            prepared = self.logic.prepare(map_msg, cost_msg, pose)
            prepared["background"], prepared["view"] = (
                self.renderer.render_background(prepared, pose)
            )
        except Exception as error:
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
        return period

    def _stream_loop(self):
        """Analyze continuously, but render only for an explicit request."""
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.overlay_socket = self.context.socket(zmq.PULL)
        self.overlay_socket.setsockopt(zmq.RCVHWM, 8)
        self.overlay_socket.setsockopt(zmq.LINGER, 0)

        try:
            self.socket.bind(IMAGE_ENDPOINT)
            self.overlay_socket.bind(OVERLAY_ENDPOINT)
            next_analysis = time.monotonic()
            publish_pending = False

            while not self.stop_event.is_set():
                try:
                    publish_pending = (
                        self._drain_overlay_commands() or publish_pending
                    )
                except Exception as error:
                    self.node.get_logger().warning(
                        f"Map overlay command: {error}"
                    )
                now = time.monotonic()

                if now >= next_analysis:
                    retry_delay = self._refresh_analysis()
                    next_analysis = time.monotonic() + retry_delay

                if publish_pending:
                    with self.state_lock:
                        prepared = self.prepared
                    analysis_period = 1.0 / self.analysis_hz
                    analysis_is_fresh = (
                        prepared is not None
                        and time.monotonic()
                        - prepared["prepared_at_monotonic"] <= analysis_period
                    )
                    if analysis_is_fresh and self.publish():
                        publish_pending = False
                        self.snapshot_request_id = None

                delay = max(
                    0.0,
                    min(next_analysis - time.monotonic(), 0.05),
                )
                self.stop_event.wait(delay)
        except Exception as error:
            if not self.stop_event.is_set():
                self.node.get_logger().error(f"Map stream: {error}")
        finally:
            self.overlay_socket.close(0)
            self.socket.close(0)
            self.overlay_socket = None
            self.socket = None

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
            "snapshot_request_id": self.snapshot_request_id,
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

    def publish(self):
        with self.state_lock:
            prepared = self.prepared
        if prepared is None:
            return False

        pose = self._pose(prepared["frame_id"])
        if pose is None:
            return False

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
                return False
            encoded_at = time.monotonic()
            self.sequence += 1
            metadata.update({
                "sequence": self.sequence,
                "publish_mode": "on_request",
                "analysis_hz": self.analysis_hz,
                "analysis_ms": prepared["analysis_ms"],
                "analysis_age_ms": round(
                    (started_at - prepared["prepared_at_monotonic"]) * 1000, 1
                ),
                "captured_at_unix_ns": captured_at_unix_ns,
                "generated_at_unix_ns": time.time_ns(),
                "render_ms": round((encoded_at - started_at) * 1000, 1),
            })
            header = json.dumps(metadata).encode()
            payload = struct.pack("!I", len(header)) + header + jpeg.tobytes()
            self.socket.send_multipart(
                [b"map/image", payload], flags=zmq.NOBLOCK
            )
            return True

        except zmq.Again:
            self.node.get_logger().warning(
                "Map image dropped: subscriber is too slow"
            )
            return False
        except Exception as error:
            self.node.get_logger().warning(f"Map image: {error}")
            return False

    def close(self):
        self.stop_event.set()
        if self.stream_thread is not threading.current_thread():
            self.stream_thread.join()
