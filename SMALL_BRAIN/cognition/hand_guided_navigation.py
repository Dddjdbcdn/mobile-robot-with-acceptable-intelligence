"""Automatic hand-guided approach and return while a person is tracked."""

from __future__ import annotations

import asyncio
from collections import deque
import math
from pathlib import Path
from statistics import median
import time
import uuid

from actions.tracking.target_catalog import normalize_human_target
from cognition.state import robot_state


FINGERS = ("index", "middle", "ring", "pinky")


class HandGuidedNavigation:
    """Track a welcomed hand, approach its ToF seed, then return on a push."""

    MONITOR_HZ = 10.0
    STATUS_MAX_AGE_SECONDS = 1.0
    WELCOME_CONFIRM_FRAMES = 4
    UP_FLICK_CONFIRM_FRAMES = 2
    OPEN_FINGER_MINIMUM = 3
    MAX_FINGER_BEND_DEG = 75.0
    MIN_FINGER_EXTENSION_RATIO = 0.65
    WELCOME_AXIS_Y_MIN = 0.05
    UPWARD_AXIS_Y_MAX = -0.30
    TOF_MAX_AGE_SECONDS = 0.35
    APPROACH_STANDOFF_M = 0.05
    PUSH_BASELINE_SAMPLES = 4
    PUSH_DROP_M = 0.05
    PUSH_CONFIRM_FRAMES = 2
    NAVIGATION_TIMEOUT = 90.0
    HAND_LOST_RESET_SECONDS = 1.0

    def __init__(self, hand_landmarks, track_action, send_robot_command):
        self.hand_landmarks = hand_landmarks
        self.track_action = track_action
        self.send_robot_command = send_robot_command
        self.state = "watching_person"
        self.active = False
        self.action_id = None
        self._running = False
        self._last_yolo_sequence = None
        self._latest_status = None
        self._welcome_frames = 0
        self._up_frames = 0
        self._last_hand_at = None
        self._stable_seed = None
        self._push_samples = deque(maxlen=8)
        self._push_baseline = None
        self._push_frames = 0
        self._navigation_task = None
        self._navigation_future = None
        self._navigation_action_id = None
        self._motion_allowed = lambda: True

    def set_motion_allowed(self, callback):
        self._motion_allowed = callback

    @staticmethod
    def _xy(point):
        return float(point["normalized_x"]), float(point["normalized_y"])

    @classmethod
    def _finger_bend(cls, landmarks, name):
        mcp = landmarks.get(f"{name}_mcp")
        pip = landmarks.get(f"{name}_pip")
        dip = landmarks.get(f"{name}_dip")
        tip = landmarks.get(f"{name}_tip")
        if not all((mcp, pip, dip, tip)):
            return None

        mcp_xy, pip_xy, dip_xy, tip_xy = map(cls._xy, (mcp, pip, dip, tip))
        first = (dip_xy[0] - pip_xy[0], dip_xy[1] - pip_xy[1])
        second = (tip_xy[0] - dip_xy[0], tip_xy[1] - dip_xy[1])
        first_length = math.hypot(*first)
        second_length = math.hypot(*second)
        if first_length <= 1e-6 or second_length <= 1e-6:
            return None
        cosine = (
            first[0] * second[0] + first[1] * second[1]
        ) / (first_length * second_length)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    @classmethod
    def classify_hand_pose(cls, hand):
        landmarks = (hand or {}).get("landmarks") or {}
        required_palm = [
            landmarks.get(name)
            for name in (
                "wrist", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"
            )
        ]
        if not all(required_palm):
            return {
                "gesture": "unavailable", "open_fingers": 0,
                "palm_center": None, "finger_axis_y": None,
            }

        palm_scale = math.dist(
            cls._xy(landmarks["index_mcp"]),
            cls._xy(landmarks["pinky_mcp"]),
        )
        if palm_scale <= 1e-6:
            return {
                "gesture": "unavailable", "open_fingers": 0,
                "palm_center": None, "finger_axis_y": None,
            }

        finger_debug = {}
        open_fingers = 0
        tips = []
        mcps = []
        for name in FINGERS:
            mcp = landmarks.get(f"{name}_mcp")
            tip = landmarks.get(f"{name}_tip")
            bend = cls._finger_bend(landmarks, name)
            if not mcp or not tip or bend is None:
                finger_debug[name] = {"extended": False, "bend_deg": bend}
                continue
            extension = math.dist(cls._xy(mcp), cls._xy(tip)) / palm_scale
            extended = (
                bend <= cls.MAX_FINGER_BEND_DEG
                and extension >= cls.MIN_FINGER_EXTENSION_RATIO
            )
            open_fingers += int(extended)
            tips.append(cls._xy(tip))
            mcps.append(cls._xy(mcp))
            finger_debug[name] = {
                "extended": extended,
                "bend_deg": bend,
                "extension_ratio": extension,
            }

        palm_points = [cls._xy(point) for point in required_palm]
        palm_center = (
            sum(point[0] for point in palm_points) / len(palm_points),
            sum(point[1] for point in palm_points) / len(palm_points),
        )
        finger_axis_y = None
        if tips and mcps:
            tip_y = sum(point[1] for point in tips) / len(tips)
            mcp_y = sum(point[1] for point in mcps) / len(mcps)
            finger_axis_y = (tip_y - mcp_y) / palm_scale

        gesture = "relaxed"
        if open_fingers >= cls.OPEN_FINGER_MINIMUM and finger_axis_y is not None:
            if finger_axis_y <= cls.UPWARD_AXIS_Y_MAX:
                gesture = "fingers_up"
            elif finger_axis_y >= cls.WELCOME_AXIS_Y_MIN:
                gesture = "welcome"
            else:
                gesture = "open_horizontal"
        return {
            "gesture": gesture,
            "open_fingers": open_fingers,
            "palm_center": palm_center,
            "finger_axis_y": finger_axis_y,
            "palm_scale": palm_scale,
            "fingers": finger_debug,
        }

    def gesture_status(self, max_age_seconds=None):
        status = self._latest_status
        if not isinstance(status, dict):
            return None
        max_age = (
            self.STATUS_MAX_AGE_SECONDS
            if max_age_seconds is None else float(max_age_seconds)
        )
        if time.monotonic() - float(status.get("updated_at") or 0.0) > max_age:
            return None
        return dict(status)

    def _person_tracking_active(self):
        return bool(
            self.track_action.active
            and normalize_human_target(self.track_action.target) is not None
        )

    @staticmethod
    def _tracked_person(detections):
        people = [
            item for item in detections
            if item.get("class") == "person" and item.get("keypoints")
        ]
        return max(
            people,
            key=lambda item: float(item.get("confidence") or 0.0),
            default=None,
        )

    def _publish_status(self, sequence, person, hand, pose):
        self._latest_status = {
            "updated_at": time.monotonic(),
            "sequence": sequence,
            "state": self.state,
            "gesture": pose.get("gesture"),
            "person": person,
            "hand": hand,
            "debug": pose,
            "stable_seed": (
                None if self._stable_seed is None else dict(self._stable_seed)
            ),
        }

    def _clear_seed(self):
        self._stable_seed = None

    def _set_hand_target(self, palm_center):
        if palm_center is None:
            return
        setter = getattr(self.track_action, "set_visual_target_override", None)
        if setter is not None:
            setter("hand_guided_navigation", *palm_center)

    def _clear_hand_target(self):
        clearer = getattr(self.track_action, "clear_visual_target_override", None)
        if clearer is not None:
            clearer("hand_guided_navigation")

    def _read_stable_hand_seed(self):
        getter = getattr(self.track_action, "get_stable_target_seed", None)
        if getter is None:
            return None
        seed = getter(
            "hand_guided_navigation",
            target="hand",
            session_id=getattr(self.track_action, "action_id", None),
        )
        return dict(seed) if isinstance(seed, dict) else None

    async def _sample(self, yolo):
        sequence, detections, frame_bgr, _ = yolo.detection_snapshot_with_frame()
        if frame_bgr is None or sequence == self._last_yolo_sequence:
            return
        self._last_yolo_sequence = sequence
        person = self._tracked_person(detections)
        hand = None
        if person is not None:
            hand = await self.hand_landmarks.detect_right_hand(
                frame_bgr,
                person,
                debug_path=str(
                    Path("results/action_results") / "hand_landmarks_latest.jpg"
                ),
            )
        pose = self.classify_hand_pose(hand)
        self._publish_status(sequence, person, hand, pose)

        now = time.monotonic()
        if hand is None:
            self._welcome_frames = 0
            self._up_frames = 0
            if (
                self.state == "tracking_hand"
                and self._last_hand_at is not None
                and now - self._last_hand_at >= self.HAND_LOST_RESET_SECONDS
            ):
                self._reset_to_person()
            return
        self._last_hand_at = now

        gesture = pose["gesture"]
        palm_center = pose["palm_center"]
        if self.state == "watching_person":
            self._welcome_frames = (
                self._welcome_frames + 1 if gesture == "welcome" else 0
            )
            if (
                self._welcome_frames >= self.WELCOME_CONFIRM_FRAMES
                and self._motion_allowed()
            ):
                self.state = "tracking_hand"
                self.active = True
                self.action_id = f"hand-guided-{uuid.uuid4().hex}"
                self._clear_seed()
                self._set_hand_target(palm_center)
            return

        if self.state in {"tracking_hand", "approaching", "awaiting_push"}:
            self._set_hand_target(palm_center)

        if self.state == "tracking_hand":
            self._stable_seed = self._read_stable_hand_seed()
            self._up_frames = self._up_frames + 1 if gesture == "fingers_up" else 0
            if (
                self._up_frames >= self.UP_FLICK_CONFIRM_FRAMES
                and self._stable_seed is not None
                and self._motion_allowed()
                and self._navigation_task is None
            ):
                self._navigation_task = asyncio.create_task(
                    self._approach_hand(), name=f"{self.action_id}:approach"
                )
            return

        if self.state == "awaiting_push":
            self._observe_push(gesture)

    def _observe_push(self, gesture):
        camera = robot_state.get("camera") or {}
        distance = camera.get("camera_tof_range")
        timestamp = camera.get("timestamp")
        if (
            gesture != "fingers_up"
            or not isinstance(distance, (int, float))
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(distance))
            or float(distance) <= 0.01
            or time.monotonic() - float(timestamp) > self.TOF_MAX_AGE_SECONDS
        ):
            self._push_samples.clear()
            self._push_baseline = None
            self._push_frames = 0
            return

        distance = float(distance)
        self._push_samples.append(distance)
        if self._push_baseline is None:
            if len(self._push_samples) >= self.PUSH_BASELINE_SAMPLES:
                self._push_baseline = median(self._push_samples)
            return
        pushed = self._push_baseline - distance >= self.PUSH_DROP_M
        self._push_frames = self._push_frames + 1 if pushed else 0
        if self._push_frames >= self.PUSH_CONFIRM_FRAMES and self._navigation_task is None:
            self._navigation_task = asyncio.create_task(
                self._return_and_face_person(), name=f"{self.action_id}:return"
            )

    async def _navigate(self, payload, stage):
        action_id = f"{self.action_id}:{stage}"
        self._navigation_action_id = action_id
        self._navigation_future = asyncio.get_running_loop().create_future()
        feedback = await self.send_robot_command({**payload, "action_id": action_id})
        if not isinstance(feedback, dict) or feedback.get("status") != "accepted":
            return False
        try:
            event = await asyncio.wait_for(
                self._navigation_future, self.NAVIGATION_TIMEOUT
            )
        except TimeoutError:
            await self.send_robot_command({
                "command": "stop_moving", "action_id": action_id,
            })
            return False
        return event.get("status") == "Goal Reached"

    async def _approach_hand(self):
        try:
            seed = dict(self._stable_seed or {})
            if not seed:
                return
            self.state = "approaching"
            succeeded = await self._navigate({
                "command": "navigate_to_approach",
                "frame_id": "base_footprint",
                "x": seed["x"],
                "y": seed["y"],
                "angle": seed["angle"],
                "tof_range": seed["tof_range"],
                "standoff_m": self.APPROACH_STANDOFF_M,
            }, "approach")
            if succeeded:
                self.state = "awaiting_push"
                self._push_samples.clear()
                self._push_baseline = None
                self._push_frames = 0
            else:
                self._reset_to_person()
        finally:
            self._navigation_task = None

    async def _return_and_face_person(self):
        try:
            self.state = "returning"
            self._clear_hand_target()
            returned = await self._navigate({
                "command": "navigate_local",
                "local_command": "previous_position",
            }, "return")
            if returned:
                self.state = "looking_at_person"
                await self._navigate({
                    "command": "navigate_local",
                    "local_command": "face_person",
                }, "face-person")
            self._reset_to_person()
        finally:
            self._navigation_task = None

    def handle_navigation_event(self, payload):
        if (
            payload.get("event") == "navigation"
            and payload.get("action_id") == self._navigation_action_id
            and self._navigation_future is not None
            and not self._navigation_future.done()
        ):
            self._navigation_future.set_result(dict(payload))
            return True
        return False

    def _reset_to_person(self):
        self._clear_hand_target()
        self.state = "watching_person"
        self.active = False
        self.action_id = None
        self._welcome_frames = 0
        self._up_frames = 0
        self._last_hand_at = None
        self._clear_seed()
        self._push_samples.clear()
        self._push_baseline = None
        self._push_frames = 0
        self._navigation_future = None
        self._navigation_action_id = None

    async def run(self, yolo):
        self._running = True
        interval = 1.0 / self.MONITOR_HZ
        try:
            while self._running:
                started_at = time.monotonic()
                if self._person_tracking_active():
                    try:
                        await self._sample(yolo)
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        print(
                            "[HAND GUIDANCE ERROR] "
                            f"{type(error).__name__}: {error}"
                        )
                else:
                    self._latest_status = None
                    self._last_yolo_sequence = None
                    if self.state != "watching_person":
                        self._reset_to_person()
                elapsed = time.monotonic() - started_at
                await asyncio.sleep(max(0.01, interval - elapsed))
        finally:
            self._running = False
            self._reset_to_person()

    async def stop(self, reason_code="STOPPED"):
        if reason_code == "SHUTDOWN":
            self._running = False
        task = self._navigation_task
        self._navigation_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self.active:
            await self.send_robot_command({
                "command": "stop_moving",
                "action_id": self.action_id or "hand-guided-stop",
            })
        self._reset_to_person()
