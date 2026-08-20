import threading
import time
import cv2
import base64
import json
from collections import deque
from pathlib import Path
import asyncio
import zmq

from database.state import robot_state

class CameraSnapshot:
    def __init__(self, sequence, captured_at, full_bgr, tracking_bgr):
        self.sequence = sequence
        self.captured_at = captured_at
        self.full_bgr = full_bgr
        self.tracking_bgr = tracking_bgr

class CameraStream:
    def __init__(self, camera_index=0, capture_width=1280, capture_height=720, 
                 tracking_width=640, tracking_height=360, history_frames=60, fps=30):
        self.camera_index = camera_index
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.tracking_size = (tracking_width, tracking_height)
        self.fps = fps
        self.current_fps = 0.0

        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()

        self.thread = None
        self.latest = None
        self.history = deque(maxlen=history_frames)
        self.sequence = 0
        self.history_sequence = 0
        self.startup_error = None

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

                tracking_frame = cv2.resize(full_frame, self.tracking_size, interpolation=cv2.INTER_AREA)
                captured_at = time.monotonic()

                with self.condition:
                    self.sequence += 1
                    self.latest = CameraSnapshot(self.sequence, captured_at, full_frame, tracking_frame)

                    if (self.sequence - self.history_sequence) >= 6:
                        self.history_sequence = self.sequence
                        self.history.append({
                            "sequence": self.sequence,
                            "captured_at": captured_at,
                            "bgr": tracking_frame
                        })

                    self.condition.notify_all()

                if not first_frame_received:
                    first_frame_received = True
                    self.ready_event.set()
                    
        except Exception as error:
            self.startup_error = error
            self.ready_event.set()
        finally:
            if camera:
                camera.release()
            with self.condition:
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            if not self.latest:
                raise RuntimeError("No camera frame is available.")
            return self.latest
        
    def jpeg_bytes_snapshot(self, jpeg_quality=70,tracking_bgr=False,save_path=None):
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

        bgr = self.latest.full_bgr if not tracking_bgr else self.latest.tracking_bgr
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

# Standalone helper functions 
async def send_camera_image(ws, jpeg_bytes, instruction):
    jpeg_bytes = bytes(jpeg_bytes)
    if not jpeg_bytes or not jpeg_bytes.startswith(b"\xff\xd8"):
        raise ValueError("Invalid JPEG bytes. Encode the OpenCV frame with cv2.imencode('.jpg', frame).")

    encoded_image = base64.b64encode(jpeg_bytes).decode("ascii")
    image_url = f"data:image/jpeg;base64,{encoded_image}"

    image_event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": instruction.strip()},
                {"type": "input_image", "image_url": image_url},
            ],
        },
    }

    await ws.send(json.dumps(image_event))
    print(f"\n[Camera] IMAGE ADDED")

    return {"jpeg_bytes": len(jpeg_bytes), "base64_characters": len(encoded_image)}

def clear_images_folder(folder_path="results/search_results", extensions=(".jpg", ".jpeg", ".png", ".webp")):
    target_dir = Path(folder_path)
    if not target_dir.is_dir():
        print(f"[Cleanup] Folder '{folder_path}' does not exist.")
        return 0

    deleted_count = 0
    for item in target_dir.iterdir():
        if item.is_file() and (extensions is None or item.suffix.lower() in extensions):
            try:
                item.unlink()
                deleted_count += 1
            except OSError as e:
                print(f"[Cleanup] Failed to delete {item.name}: {e}")

    return deleted_count