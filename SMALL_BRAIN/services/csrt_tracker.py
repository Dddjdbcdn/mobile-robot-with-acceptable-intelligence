import queue
import threading
import cv2
import time

# ---------------------------------------------------------
# DATA MODELS & SETUP (Remains the same)
# ---------------------------------------------------------
class StartTrackingCommand:
    def __init__(self, detection_sequence, initialization_frame, bbox_xywh, target):
        self.detection_sequence = detection_sequence
        self.initialization_frame = initialization_frame
        self.bbox_xywh = bbox_xywh
        self.target = target

class TrackingUpdate:
    def __init__(self, is_tracking, target, sequence, success, bbox_xywh=None, normalized_x=None, normalized_y=None):
        self.is_tracking = is_tracking
        self.target = target
        self.sequence = sequence
        self.success = success
        self.bbox_xywh = bbox_xywh
        self.normalized_x = normalized_x
        self.normalized_y = normalized_y

def create_csrttracker():
    if not hasattr(cv2, "TrackerCSRT_Params"):
        # Older/legacy builds may not expose configurable parameters.
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()

        if hasattr(cv2, "legacy") and hasattr(
            cv2.legacy, "TrackerCSRT_create"
        ):
            return cv2.legacy.TrackerCSRT_create()

        raise RuntimeError(
            "CSRT is unavailable. Install opencv-contrib-python."
        )

    params = cv2.TrackerCSRT_Params()

    # Feature extraction
    params.use_hog = True
    params.use_color_names = True
    params.use_gray = True
    params.use_rgb = False
    params.use_channel_weights = True
    params.use_segmentation = True

    # Scale estimation
    params.number_of_scales = 45
    params.scale_step = 1.035
    params.scale_lr = 0.04
    params.scale_sigma_factor = 0.25
    params.scale_model_max_area = 768.0

    # Search and model size
    params.padding = 2.5
    params.template_size = 200.0

    # Learnings
    params.filter_lr = 0.02
    params.weights_lr = 0.02
    params.histogram_lr = 0.03

    # Optimization and loss detection
    params.admm_iterations = 3
    params.psr_threshold = 0.08

    # Segmentation
    params.histogram_bins = 16
    params.background_ratio = 2

    try:
        return cv2.TrackerCSRT_create(params)
    except (TypeError, AttributeError):
        return cv2.TrackerCSRT.create(params)

# ---------------------------------------------------------
# REFACTORED TRACKER MANAGER
# ---------------------------------------------------------

class CSRTTrackingManager:
    def __init__(self, camera, max_initial_replay_frames=15):
        self.camera = camera
        self.tracking_update = None
        self.max_initial_replay_frames = max_initial_replay_frames
        
        self._commands = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = None
        
        # Internal state variables for the background thread
        self.tracker = None
        self.active = False
        self.target = ""
        self.current_sequence = -1

        self.current_fps = 0.0 
        self._last_track_time = time.monotonic() 
    def start_worker(self):
        if self._thread and self._thread.is_alive(): return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="csrt-tracker", daemon=True)
        self._thread.start()

    def stop_worker(self):
        self._stop_event.set()
        self.stop_tracking()
        if self._thread:
            self._thread.join(timeout=2.0)

    def begin_tracking(self, detection_sequence, initialization_frame, bbox_xywh, target):
        command = StartTrackingCommand(detection_sequence, initialization_frame.copy(), bbox_xywh, target)
        try: self._commands.get_nowait()
        except queue.Empty: pass
        self._commands.put_nowait(command)

    def stop_tracking(self):
        self.active = False
        self.tracking_update = None

        try: 
            self._commands.get_nowait()
        except queue.Empty: pass
        self._commands.put_nowait(None)

    def _publish_lost(self, target, sequence):
        self.tracking_update = TrackingUpdate(
            is_tracking=False, target=target, sequence=sequence, success=False
        )
        self.stop_tracking()

    # ==========================================
    # CLEAN RUN LOOP
    # ==========================================
    def _run(self):
        """The main background loop is now a simple, readable state machine."""
        while not self._stop_event.is_set():
            # 1. Always check the mailbox for new targets or stop commands
            self._process_commands()

            # 2. If we have a target, track it on the newest frame
            if self.active and self.tracker is not None:
                self._track_live_frame()

    # ==========================================
    # HELPER 1: INITIALIZATION BLOCK
    # ==========================================
    def _process_commands(self):
        try:
            # Block for 200ms if idle, or just peek (10ms) if busy tracking
            command = self._commands.get(timeout=0.01 if self.active else 0.2)
        except queue.Empty:
            return # Mailbox is empty, do nothing

        # If the command is None, the LLM told us to stop tracking
        if command is None:
            self.tracker = None
            self.active = False
            self.tracking_update = None
            self.target = ""
            return

        # --- A new target has arrived! Initialize it. ---
        self.tracker = create_csrttracker()
        self.tracker.init(command.initialization_frame, command.bbox_xywh)
        
        self.target = command.target
        self.current_sequence = command.detection_sequence
        self.active = True

        # Fast-forward through the history buffer to catch up
        buffered = self.camera.history_frames_after(self.current_sequence)
        if len(buffered) > self.max_initial_replay_frames:
            buffered = buffered[-self.max_initial_replay_frames:]

        for frame in buffered:
            # Note: Assuming your history buffer saves frames as dictionaries
            ok, bbox = self.tracker.update(frame["bgr"]) 
            self.current_sequence = frame["sequence"]
            if not ok:
                self.active = False
                self._publish_lost(self.target, self.current_sequence)
                break

    # ==========================================
    # HELPER 2: LIVE TRACKING BLOCK
    # ==========================================
    def _track_live_frame(self):
        if not self.camera.wait_for_frame_after(self.current_sequence, timeout=0.2):
            return

        now = time.monotonic()
        dt = now - self._last_track_time
        if dt > 0: self.current_fps = 1.0 / dt
        self._last_track_time = now

        newest = self.camera.snapshot()
        ok, bbox = self.tracker.update(newest.tracking_bgr)
        self.current_sequence = newest.sequence

        if not ok:
            self.active = False
            self._publish_lost(self.target, self.current_sequence)
            return

        # Calculate math on success
        x, y, width, height = map(float, bbox)
        frame_height, frame_width = newest.tracking_bgr.shape[:2]

        center_x = x + width / 2.0
        center_y = y + height / 2.0

        normalized_x = center_x / frame_width 
        normalized_y = center_y / frame_height

        # Expose the state to the main thread
        self.tracking_update = TrackingUpdate(
            is_tracking=True,
            target=self.target,
            sequence=self.current_sequence,
            success=True,
            bbox_xywh=(x, y, width, height),
            normalized_x=normalized_x,
            normalized_y=normalized_y,
        )