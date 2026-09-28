import threading
import time
import os
import shutil
import subprocess
import cv2
from collections import deque
from pathlib import Path

class CameraSnapshot:
    def __init__(self, sequence, captured_at, full_bgr, tracking_bgr):
        self.sequence = sequence
        self.captured_at = captured_at
        self.full_bgr = full_bgr
        self.tracking_bgr = tracking_bgr

class CameraStream:
    USB_RECONNECT_DELAY = 0.5
    USB_READ_FAILURE_LIMIT = 5

    USB_CONTROL_ENV = {
        "USB_CAMERA_BRIGHTNESS": "brightness",
        "USB_CAMERA_CONTRAST": "contrast",
        "USB_CAMERA_SATURATION": "saturation",
        "USB_CAMERA_HUE": "hue",
        "USB_CAMERA_GAMMA": "gamma",
        "USB_CAMERA_SHARPNESS": "sharpness",
        "USB_CAMERA_BACKLIGHT_COMPENSATION": "backlight_compensation",
        "USB_CAMERA_AUTO_EXPOSURE": "auto_exposure",
        "USB_CAMERA_EXPOSURE_TIME_ABSOLUTE": "exposure_time_absolute",
        "USB_CAMERA_EXPOSURE_DYNAMIC_FRAMERATE": "exposure_dynamic_framerate",
        "USB_CAMERA_WHITE_BALANCE_AUTOMATIC": "white_balance_automatic",
        "USB_CAMERA_WHITE_BALANCE_TEMPERATURE": "white_balance_temperature",
        "USB_CAMERA_POWER_LINE_FREQUENCY": "power_line_frequency",
    }
    USB_CONTROL_ORDER = (
        "auto_exposure",
        "exposure_dynamic_framerate",
        "exposure_time_absolute",
        "white_balance_automatic",
        "white_balance_temperature",
        "power_line_frequency",
        "backlight_compensation",
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "gamma",
        "sharpness",
    )

    def __init__(self, camera_index=0, capture_width=1280, capture_height=720, 
                 tracking_width=640, tracking_height=360, history_frames=60, fps=30,
                 usb_controls=None, usb_device=None):
        self.camera_index = camera_index
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.tracking_size = (tracking_width, tracking_height)
        self.fps = fps
        self.current_fps = 0.0
        self.usb_device = usb_device or f"/dev/video{camera_index}"
        self.usb_controls = dict(usb_controls or {})
        self.applied_usb_controls = {}

        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()

        self.thread = None
        self.latest = None
        self.history = deque(maxlen=history_frames)
        self.sequence = 0
        self.history_sequence = 0
        self.startup_error = None

    @classmethod
    def usb_controls_from_env(cls, environ=None):
        """Read optional native UVC controls without changing driver defaults."""
        environ = os.environ if environ is None else environ
        controls = {}
        for env_name, control_name in cls.USB_CONTROL_ENV.items():
            raw_value = environ.get(env_name)
            if raw_value is None or not str(raw_value).strip():
                continue
            controls[control_name] = cls._parse_usb_control(
                control_name, raw_value
            )

        # These controls are inactive while their automatic mode is enabled.
        if "exposure_time_absolute" in controls:
            controls.setdefault("auto_exposure", 1)
        if "white_balance_temperature" in controls:
            controls.setdefault("white_balance_automatic", 0)
        return controls

    @staticmethod
    def _parse_usb_control(name, value):
        text = str(value).strip().lower()
        aliases = {
            "auto_exposure": {
                "manual": 1,
                "auto": 3,
                "aperture_priority": 3,
                "aperture-priority": 3,
            },
            "power_line_frequency": {
                "disabled": 0,
                "off": 0,
                "50hz": 1,
                "50_hz": 1,
                "60hz": 2,
                "60_hz": 2,
            },
        }
        if name in aliases and text in aliases[name]:
            return aliases[name][text]
        if name in {
            "exposure_dynamic_framerate", "white_balance_automatic"
        }:
            if text in {"1", "true", "yes", "on"}:
                return 1
            if text in {"0", "false", "no", "off"}:
                return 0
            raise ValueError(
                f"{name} must be true/false, on/off, yes/no, or 1/0"
            )
        try:
            return int(text)
        except ValueError as error:
            raise ValueError(
                f"Invalid USB camera value for {name}: {value!r}"
            ) from error

    def _apply_usb_controls(self):
        """Apply configured controls exactly by their V4L2 driver names."""
        self.applied_usb_controls = {}
        if not self.usb_controls:
            return
        executable = shutil.which("v4l2-ctl")
        if executable is None:
            raise RuntimeError(
                "USB camera controls were configured but v4l2-ctl is unavailable"
            )

        unknown = set(self.usb_controls) - set(self.USB_CONTROL_ORDER)
        if unknown:
            raise ValueError(
                "Unknown USB camera controls: " + ", ".join(sorted(unknown))
            )
        for name in self.USB_CONTROL_ORDER:
            if name not in self.usb_controls:
                continue
            value = int(self.usb_controls[name])
            completed = subprocess.run(
                [
                    executable,
                    "-d", self.usb_device,
                    "--set-ctrl", f"{name}={value}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(
                    f"Failed to set USB camera {name}={value}: {detail}"
                )
            self.applied_usb_controls[name] = value
        print(f"[USB camera controls: {self.applied_usb_controls}]")

    def start(self, timeout=5.0):
        if self.thread and self.thread.is_alive():
            return

        self.stop_event.clear()
        self.ready_event.clear()
        self.startup_error = None

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        if not self.ready_event.wait(timeout):
            raise TimeoutError("Timed out waiting for the USB camera.")

        if self.startup_error:
            raise RuntimeError("Camera startup failed.") from self.startup_error

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=2.0)

    def _record_frame(self, frame):
        captured_at = time.monotonic()
        with self.condition:
            self._publish_frame_locked(captured_at, frame)

    def _publish_frame_locked(self, captured_at, frame):
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
        )

        if (self.sequence - self.history_sequence) >= 6:
            self.history_sequence = self.sequence
            self.history.append({
                "sequence": self.sequence,
                "captured_at": captured_at,
                "bgr": tracking_frame,
            })

        self.ready_event.set()
        self.condition.notify_all()

    def _capture_loop(self):
        camera = None
        connected_once = False
        try:
            while not self.stop_event.is_set():
                try:
                    camera = self._open_usb_camera()
                except Exception as error:
                    if connected_once:
                        print(f"[USB camera reconnect failed: {error}]")
                    else:
                        print(f"[Waiting for USB camera {self.usb_device}: {error}]")
                    self._mark_usb_unavailable()
                    if self.stop_event.wait(self.USB_RECONNECT_DELAY):
                        break
                    continue

                if connected_once:
                    print(f"[USB camera reconnected: {self.usb_device}]")
                connected_once = True
                last_time = time.monotonic()
                frame_count = 0
                read_failures = 0

                while not self.stop_event.is_set():
                    ok, full_frame = camera.read()
                    if not ok or full_frame is None:
                        read_failures += 1
                        if read_failures >= self.USB_READ_FAILURE_LIMIT:
                            print(
                                "[USB camera stopped delivering frames; "
                                "reopening device]"
                            )
                            break
                        self.stop_event.wait(0.01)
                        continue

                    read_failures = 0
                    now = time.monotonic()
                    frame_count += 1
                    if now - last_time >= 1.0:
                        self.current_fps = frame_count / (now - last_time)
                        frame_count = 0
                        last_time = now

                    self._record_frame(full_frame)

                camera.release()
                camera = None
                self._mark_usb_unavailable()
                if not self.stop_event.is_set():
                    self.stop_event.wait(self.USB_RECONNECT_DELAY)
        except Exception as error:
            self.startup_error = error
            self.ready_event.set()
        finally:
            if camera:
                camera.release()
            with self.condition:
                self.condition.notify_all()

    def _open_usb_camera(self):
        """Open the stable V4L2 path and restore its capture configuration."""
        camera = cv2.VideoCapture(self.usb_device, cv2.CAP_V4L2)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(f"Could not open camera {self.usb_device}.")

        try:
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            camera.set(cv2.CAP_PROP_FPS, self.fps)
            self._apply_usb_controls()
        except Exception:
            camera.release()
            raise
        return camera

    def _mark_usb_unavailable(self):
        """Prevent consumers from using a stale frame while USB reconnects."""
        with self.condition:
            self.latest = None
            self.history.clear()
            self.history_sequence = self.sequence
            self.current_fps = 0.0
            self.condition.notify_all()

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

    def wait_for_frame_captured_after(self, captured_at, timeout=3.0):
        """Wait for a live frame captured after a camera movement completed."""
        with self.condition:
            ready = self.condition.wait_for(
                lambda: (
                    self.latest is not None
                    and self.latest.captured_at > captured_at
                ) or self.stop_event.is_set(),
                timeout=timeout,
            )
            return bool(
                ready
                and self.latest is not None
                and self.latest.captured_at > captured_at
            )
