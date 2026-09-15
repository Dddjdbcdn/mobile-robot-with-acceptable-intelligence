import threading
import time
import struct
import cv2
import json
import copy
from collections import deque
from pathlib import Path
import numpy as np
import zmq

class CameraSnapshot:
    def __init__(
        self,
        sequence,
        captured_at,
        full_bgr,
        tracking_bgr,
        source="usb",
        source_generation=0,
    ):
        self.sequence = sequence
        self.captured_at = captured_at
        self.full_bgr = full_bgr
        self.tracking_bgr = tracking_bgr
        self.source = source
        self.source_generation = source_generation

class CameraStream:
    SOURCES = ("usb", "astra")

    def __init__(self, camera_index=0, capture_width=1280, capture_height=720, 
                 tracking_width=640, tracking_height=360, history_frames=60, fps=30,
                 astra_endpoint="tcp://127.0.0.1:5558", initial_source="usb",
                 map_endpoint="tcp://127.0.0.1:5559", map_save_path=None):
        if initial_source not in self.SOURCES:
            raise ValueError(f"Unknown camera source: {initial_source}")
        self.camera_index = camera_index
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.tracking_size = (tracking_width, tracking_height)
        self.fps = fps
        self.current_fps = 0.0
        self.astra_endpoint = astra_endpoint
        self.map_endpoint = map_endpoint
        self.map_save_path = Path(map_save_path) if map_save_path else (
            Path(__file__).resolve().parents[1] / "results" / "map" / "latest.jpg"
        )
        self.map_thread = None
        self._map_latest = None
        self._map_save_override = None
        self._map_save_lock = threading.Lock()

        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()

        self.thread = None
        self.astra_thread = None
        self.latest = None
        self.history = deque(maxlen=history_frames)
        self.sequence = 0
        self.history_sequence = 0
        self.startup_error = None
        self._source = initial_source
        self._source_generation = 0
        self._source_latest = {}
        self._astra_depth = None
        self._astra_camera_info = None
        self._source_change_callbacks = []
        self._switch_lock = threading.Lock()
        self._zmq_context = zmq.Context()

    @property
    def source(self):
        with self.condition:
            return self._source

    @property
    def source_generation(self):
        with self.condition:
            return self._source_generation

    def add_source_change_callback(self, callback):
        self._source_change_callbacks.append(callback)

    def switch_source(self, source):
        source = str(source).strip().lower()
        if source not in self.SOURCES:
            raise ValueError(
                f"Unknown camera source {source!r}; expected one of {self.SOURCES}"
            )

        with self._switch_lock:
            with self.condition:
                if source == self._source:
                    return source

                cached = self._source_latest.get(source)
                if cached is None or time.monotonic() - cached[0] > 1.0:
                    raise RuntimeError(
                        f"Camera source {source!r} has no recent frames"
                    )
                previous_source = self._source
                callbacks = list(self._source_change_callbacks)

            # Reset source-dependent consumers before the new image is visible.
            for callback in callbacks:
                try:
                    callback(previous_source, source)
                except Exception as error:
                    print(f"[Camera source callback error: {error}]")

            with self.condition:
                self._source = source
                self._source_generation += 1

                # Frames and tracker history from different cameras must not mix.
                self.latest = None
                self.history.clear()
                self.history_sequence = self.sequence
                self.current_fps = 0.0

                # The availability check guarantees this is a live frame and
                # avoids a blank display while the next frame is in flight.
                self._publish_frame_locked(source, cached[0], cached[1])
                self.condition.notify_all()
        return source

    def toggle_source(self):
        next_source = "astra" if self.source == "usb" else "usb"
        return self.switch_source(next_source)

    def start(self, timeout=5.0):
        if self.thread and self.thread.is_alive():
            return

        self.stop_event.clear()
        self.ready_event.clear()
        self.startup_error = None

        self.astra_thread = threading.Thread(
            target=self._astra_receive_loop,
            name="astra-camera-receiver",
            daemon=True,
        )
        self.astra_thread.start()
        self.map_thread = threading.Thread(
            target=self._map_receive_loop, name="map-image-receiver", daemon=True
        )
        self.map_thread.start()

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        if not self.ready_event.wait(timeout):
            raise TimeoutError(f"Timed out waiting for the {self.source} camera.")

        if self.startup_error:
            raise RuntimeError("Camera startup failed.") from self.startup_error

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.astra_thread:
            self.astra_thread.join(timeout=2.0)
        if self.map_thread:
            self.map_thread.join(timeout=2.0)
        self._zmq_context.term()

    def _record_frame(self, source, frame):
        captured_at = time.monotonic()
        with self.condition:
            self._source_latest[source] = (captured_at, frame)
            if source != self._source:
                return
            self._publish_frame_locked(source, captured_at, frame)

    def _publish_frame_locked(self, source, captured_at, frame):
        tracking_frame = cv2.resize(
            frame,
            self.tracking_size,
            interpolation=cv2.INTER_AREA,
        )
        self.sequence += 1
        self.latest = CameraSnapshot(
            self.sequence,
            captured_at,
            frame,
            tracking_frame,
            source=source,
            source_generation=self._source_generation,
        )

        if (self.sequence - self.history_sequence) >= 6:
            self.history_sequence = self.sequence
            self.history.append({
                "sequence": self.sequence,
                "captured_at": captured_at,
                "source": source,
                "source_generation": self._source_generation,
                "bgr": tracking_frame,
            })

        self.ready_event.set()
        self.condition.notify_all()

    def _capture_loop(self):
        camera = None
        try:
            camera = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if not camera.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera_index}.")

            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            camera.set(cv2.CAP_PROP_FPS, self.fps)

            first_frame_received = False

            last_time = time.monotonic()
            frame_count = 0

            while not self.stop_event.is_set():
                ok, full_frame = camera.read()
                if not ok or full_frame is None:
                    time.sleep(0.01)
                    continue

                now = time.monotonic()
                frame_count += 1
                if now - last_time >= 1.0: 
                    self.current_fps = frame_count / (now - last_time)
                    frame_count = 0
                    last_time = now

                self._record_frame("usb", full_frame)

                if not first_frame_received:
                    first_frame_received = True
                    
        except Exception as error:
            self.startup_error = error
            self.ready_event.set()
        finally:
            if camera:
                camera.release()
            with self.condition:
                self.condition.notify_all()

    def _astra_receive_loop(self):
        socket = self._zmq_context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 6)
        socket.setsockopt(zmq.RCVTIMEO, 200)
        # Empty subscription is intentional: receive all three Astra topics.
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.connect(self.astra_endpoint)

        try:
            while not self.stop_event.is_set():
                try:
                    parts = socket.recv_multipart()
                except zmq.Again:
                    continue

                if len(parts) != 2:
                    print("[Invalid Astra stream message: expected topic and payload]")
                    continue
                topic, payload = parts
                try:
                    if topic == b"astra/color":
                        self._record_frame(
                            "astra", self._decode_ros_image(payload)
                        )
                    elif topic == b"astra/depth":
                        depth, metadata = self._decode_depth_image(payload)
                        with self.condition:
                            self._astra_depth = {
                                "captured_at": time.monotonic(),
                                "image": depth,
                                "metadata": metadata,
                            }
                    elif topic == b"astra/camera_info":
                        info = json.loads(payload.decode("utf-8"))
                        k = [float(value) for value in info["k"]]
                        if len(k) != 9 or k[0] <= 0.0 or k[4] <= 0.0:
                            raise ValueError("invalid Astra color intrinsics")
                        with self.condition:
                            self._astra_camera_info = {
                                **info,
                                "k": k,
                                "received_at": time.monotonic(),
                            }
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    print(f"[Invalid Astra stream payload: {error}]")
        finally:
            socket.close(linger=0)

    @staticmethod
    def _split_ros_image_payload(payload):
        if len(payload) < 4:
            raise ValueError("message is shorter than its metadata header")
        metadata_size = struct.unpack("!I", payload[:4])[0]
        metadata_end = 4 + metadata_size
        if metadata_end > len(payload):
            raise ValueError("metadata length exceeds message size")
        metadata = json.loads(payload[4:metadata_end].decode("utf-8"))
        return metadata, payload[metadata_end:]

    @staticmethod
    def _decode_ros_image(payload):
        metadata, image_bytes = CameraStream._split_ros_image_payload(payload)
        width = int(metadata["width"])
        height = int(metadata["height"])
        step = int(metadata["step"])
        encoding = str(metadata["encoding"]).lower()

        format_info = {
            "bgr8": (3, None),
            "rgb8": (3, cv2.COLOR_RGB2BGR),
            "bgra8": (4, cv2.COLOR_BGRA2BGR),
            "rgba8": (4, cv2.COLOR_RGBA2BGR),
            "mono8": (1, cv2.COLOR_GRAY2BGR),
        }.get(encoding)
        if format_info is None:
            raise ValueError(f"unsupported ROS image encoding {encoding!r}")
        channels, conversion = format_info
        row_bytes = width * channels
        if width <= 0 or height <= 0 or step < row_bytes:
            raise ValueError("invalid ROS image dimensions or step")
        if len(image_bytes) < step * height:
            raise ValueError("image data is shorter than height * step")

        rows = np.frombuffer(image_bytes, dtype=np.uint8, count=step * height)
        rows = rows.reshape(height, step)[:, :row_bytes]
        if channels == 1:
            frame = rows.reshape(height, width)
        else:
            frame = rows.reshape(height, width, channels)
        if conversion is not None:
            frame = cv2.cvtColor(frame, conversion)
        else:
            frame = frame.copy()
        return frame

    @staticmethod
    def _decode_depth_image(payload):
        metadata, image_bytes = CameraStream._split_ros_image_payload(payload)
        width = int(metadata["width"])
        height = int(metadata["height"])
        step = int(metadata["step"])
        encoding = str(metadata["encoding"]).lower()
        bigendian = bool(metadata.get("is_bigendian", 0))

        if encoding in {"16uc1", "mono16"}:
            item_size = 2
            dtype = np.dtype(">u2" if bigendian else "<u2")
            scale = 0.001
        elif encoding == "32fc1":
            item_size = 4
            dtype = np.dtype(">f4" if bigendian else "<f4")
            scale = 1.0
        else:
            raise ValueError(f"unsupported depth encoding {encoding!r}")

        row_bytes = width * item_size
        if width <= 0 or height <= 0 or step < row_bytes:
            raise ValueError("invalid depth image dimensions or step")
        if len(image_bytes) < step * height:
            raise ValueError("depth data is shorter than height * step")

        rows = np.frombuffer(
            image_bytes, dtype=np.uint8, count=step * height
        ).reshape(height, step)[:, :row_bytes]
        depth = rows.copy().view(dtype).reshape(height, width).astype(np.float32)
        if scale != 1.0:
            depth *= scale
        return depth, metadata

    def astra_depth_snapshot(self):
        # Return the latest stored depth frame and matching calibration.
        with self.condition:
            if self._astra_depth is None:
                raise RuntimeError("No Astra depth frame is available")
            if self._astra_camera_info is None:
                raise RuntimeError(
                    "No Astra color camera calibration is available"
                )
            return {
                "captured_at": self._astra_depth["captured_at"],
                "image": self._astra_depth["image"],
                "metadata": dict(self._astra_depth["metadata"]),
                "camera_info": dict(self._astra_camera_info),
            }

    def snapshot(self):
        with self.condition:
            if not self.latest:
                raise RuntimeError("No camera frame is available.")
            return self.latest
        
    def jpeg_bytes_snapshot(self, jpeg_quality=70,tracking_bgr=False,save_path=None):
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

        snapshot = self.snapshot()
        bgr = snapshot.full_bgr if not tracking_bgr else snapshot.tracking_bgr
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise RuntimeError("Failed to encode camera frame as JPEG.")

        jpeg_bytes = encoded.tobytes()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(jpeg_bytes)
                    
        return jpeg_bytes

    def history_frames_after(self, sequence):
        """Returns tracking frames newer than the supplied sequence."""
        with self.condition:
            return [frame for frame in self.history if frame["sequence"] > sequence]

    def wait_for_frame_after(self, sequence, timeout=0.2):
        with self.condition:
            return self.condition.wait_for(
                lambda: (self.latest and self.latest.sequence > sequence) or self.stop_event.is_set(),
                timeout=timeout
            )

    def _map_receive_loop(self):
        socket = self._zmq_context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 2)
        socket.setsockopt(zmq.RCVTIMEO, 200)
        socket.setsockopt(zmq.SUBSCRIBE, b"map/image")
        socket.connect(self.map_endpoint)
        last_saved = 0.0
        try:
            while not self.stop_event.is_set():
                try:
                    parts = socket.recv_multipart()
                except zmq.Again:
                    continue
                try:
                    if len(parts) != 2 or parts[0] != b"map/image":
                        raise ValueError("expected map/image topic and payload")
                    metadata, jpeg = self._split_ros_image_payload(parts[1])
                    if metadata.get("schema_version") != 1 or not isinstance(metadata.get("candidates"), list):
                        raise ValueError("unsupported map metadata")
                    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None or not jpeg.startswith(bytes.fromhex("ffd8")):
                        raise ValueError("invalid map JPEG")
                    now = time.monotonic()
                    with self.condition:
                        self._map_latest = dict(received_at=now, jpeg_bytes=jpeg,
                                                image=frame, metadata=metadata)
                except (KeyError, TypeError, ValueError, OSError, cv2.error) as error:
                    print(f"[Invalid map stream or save error: {error}]")
        finally:
            socket.close(linger=0)

    def map_snapshot(self, max_age=3.0):
        """Return a matching image/candidate set, refusing a stopped stream."""
        with self.condition:
            if self._map_latest is None:
                raise RuntimeError("No map image available; check BIG BRAIN map/costmap/pose")
            age = time.monotonic() - self._map_latest["received_at"]
            if age > max_age:
                raise RuntimeError(f"Map image is stale ({age:.1f} seconds)")
            return {**self._map_latest, "age_seconds": age,
                    "image": self._map_latest["image"].copy(),
                    "metadata": copy.deepcopy(self._map_latest["metadata"])}
