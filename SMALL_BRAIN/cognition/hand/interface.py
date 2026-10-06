"""Always-on vision interface that turns hand poses into commands."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
import time

from cognition.hand.gestures import FrameConfirmation, HandGestureClassifier


class HandGestureInterface:
    """Recognize gestures without owning any robot action state."""

    MONITOR_HZ = 15.0
    STATUS_MAX_AGE_SECONDS = 1.0
    IDLE_WELCOME_CONFIRM_FRAMES = 10
    TRACKING_STOP_CONFIRM_FRAMES = 5
    MOVEMENT_STOP_CONFIRM_FRAMES = 10
    MODIFIER_CONFIRM_FRAMES = 3
    COMMAND_CONFIRM_FRAMES = 2
    COMMAND_TIMEOUT_FRAMES = 45
    MISS_TOLERANCE = 3
    WELCOME_PERSON_MIN_CONFIDENCE = 0.50
    WELCOME_WRIST_MIN_CONFIDENCE = 0.50
    WELCOME_PALM_WRIST_MAX_DISTANCE = 0.18

    def __init__(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks
        self._dispatch_handler = None
        self._running = False
        self._last_frame_sequence = None
        self._inference_sequence = 0
        self._latest_status = None
        self._armed_gesture = None
        self._armed_frames_remaining = 0

        self._idle_welcome = FrameConfirmation(
            "welcome",
            self.IDLE_WELCOME_CONFIRM_FRAMES,
            miss_tolerance=self.MISS_TOLERANCE,
        )
        self._tracking_thumb_up = FrameConfirmation(
            "thumb_up",
            self.TRACKING_STOP_CONFIRM_FRAMES,
            miss_tolerance=self.MISS_TOLERANCE,
        )
        self._tracking_direction_confirmations = {
            name: FrameConfirmation(
                name,
                self.COMMAND_CONFIRM_FRAMES,
                miss_tolerance=self.MISS_TOLERANCE,
            )
            for name in ("thumb_left", "thumb_right")
        }
        self._movement_push = FrameConfirmation(
            "push",
            self.MOVEMENT_STOP_CONFIRM_FRAMES,
            miss_tolerance=self.MISS_TOLERANCE,
        )
        self._modifier_confirmations = {
            name: FrameConfirmation(
                name,
                self.MODIFIER_CONFIRM_FRAMES,
                miss_tolerance=self.MISS_TOLERANCE,
            )
            for name in ("welcome", "push")
        }
        self._command_confirmations = {
            name: FrameConfirmation(
                name,
                self.COMMAND_CONFIRM_FRAMES,
                miss_tolerance=self.MISS_TOLERANCE,
            )
            for name in (
                "finger_curl_up",
                "finger_curl_down",
                "follow",
                "get_space",
            )
        }

        self._debug_image_interval = 1.0
        self._last_debug_image_at = None

    def set_dispatcher(self, dispatch):
        """Connect recognized commands to CognitionManager."""
        self._dispatch_handler = dispatch

    async def _dispatch(self, command, **data):
        """Send semantic input without knowing or owning robot actions."""
        if self._dispatch_handler is None:
            raise RuntimeError("Hand gesture dispatcher is not configured")
        return await self._dispatch_handler(command, data)

    @staticmethod
    def classify_hand_pose(hand):
        return HandGestureClassifier.classify(hand)

    def gesture_status(self, max_age_seconds=None):
        status = self._latest_status
        if not isinstance(status, dict):
            return None
        max_age = (
            self.STATUS_MAX_AGE_SECONDS
            if max_age_seconds is None
            else float(max_age_seconds)
        )
        if time.monotonic() - float(status.get("updated_at") or 0.0) > max_age:
            return None
        return dict(status)

    def _reset_sequence(self):
        self._armed_gesture = None
        self._armed_frames_remaining = 0
        for confirmation in self._modifier_confirmations.values():
            confirmation.reset()
        for confirmation in self._command_confirmations.values():
            confirmation.reset()
        self._tracking_thumb_up.reset()
        for confirmation in self._tracking_direction_confirmations.values():
            confirmation.reset()

    def _reset_all(self):
        self._reset_sequence()
        self._idle_welcome.reset()
        self._movement_push.reset()

    async def _sample(self, yolo):
        frame_source = None
        try:
            snapshot = self.hand_landmarks.camera.snapshot()
            frame_bgr = snapshot.full_bgr
            sequence = snapshot.sequence
            if frame_bgr.ndim == 3 and isinstance(sequence, int):
                _, detections = yolo.detection_snapshot()
                frame_source = "camera"
        except (RuntimeError, TypeError, ValueError):
            frame_source = None

        if frame_source is None:
            sequence, detections, frame_bgr, _ = yolo.detection_snapshot_with_frame()
            frame_source = "yolo"

        frame_sequence = (frame_source, sequence)
        if frame_bgr is None or frame_sequence == self._last_frame_sequence:
            return
        self._last_frame_sequence = frame_sequence

        person = max(
            (
                item
                for item in detections
                if item.get("class") == "person" and item.get("keypoints")
            ),
            key=lambda item: float(item.get("confidence") or 0.0),
            default=None,
        )
        hand = None
        if person is not None:
            debug_path = None
            now = time.monotonic()
            debug_due = self._debug_image_interval > 0.0 and (
                self._last_debug_image_at is None
                or now - self._last_debug_image_at >= self._debug_image_interval
            )
            if debug_due:
                self._last_debug_image_at = now
                debug_path = str(
                    Path("results/action_results") / "hand_landmarks_latest.jpg"
                )
            hand = await self.hand_landmarks.detect_right_hand(
                frame_bgr, person, debug_path=debug_path
            )
            self._inference_sequence += 1

        pose = HandGestureClassifier.classify(hand)
        context = await self._observe_gesture(
            pose["gesture"], pose["palm_center"], detections
        )
        self._latest_status = {
            "updated_at": time.monotonic(),
            "sequence": sequence,
            "inference_sequence": self._inference_sequence,
            "state": context.get("mode", "idle"),
            "action": dict(context.get("action") or {}),
            "gesture": pose.get("gesture"),
            "person": person,
            "hand": hand,
            "debug": pose,
            "armed_gesture": self._armed_gesture,
        }

    @classmethod
    def _welcome_person_candidate(cls, detections, palm_center):
        if palm_center is None:
            return None
        candidates = []
        for person in detections or ():
            if not isinstance(person, dict) or person.get("class") != "person":
                continue
            try:
                person_confidence = float(person.get("confidence") or 0.0)
                wrist = (person.get("keypoints") or {})["right_wrist"]
                wrist_confidence = float(
                    (person.get("keypoint_confidences") or {}).get(
                        "right_wrist",
                        wrist.get("confidence") or 0.0,
                    )
                )
                distance = math.hypot(
                    float(palm_center[0]) - float(wrist["normalized_x"]),
                    float(palm_center[1]) - float(wrist["normalized_y"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if (
                person_confidence < cls.WELCOME_PERSON_MIN_CONFIDENCE
                or wrist_confidence < cls.WELCOME_WRIST_MIN_CONFIDENCE
                or distance > cls.WELCOME_PALM_WRIST_MAX_DISTANCE
            ):
                continue
            candidates.append((distance, -person_confidence, person))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[:2])[2]

    async def _observe_gesture(self, gesture, palm_center, detections=None):
        context = await self._dispatch(
            "observe_gesture", gesture=gesture, palm_center=palm_center
        )
        tracking_person = bool(context.get("tracking_person"))
        if context.get("movement_active"):
            self._reset_sequence()
            if self._movement_push.observe(gesture):
                await self._dispatch("stop_movement", reason="HAND_PUSH")
            return context

        self._movement_push.reset()
        if not tracking_person:
            self._reset_sequence()
            welcome_person = self._welcome_person_candidate(
                detections,
                palm_center,
            )
            observed_gesture = gesture if welcome_person is not None else None
            if self._idle_welcome.observe(observed_gesture):
                await self._dispatch(
                    "watch_target",
                    target="person",
                    target_kind="person",
                    initial_person=welcome_person,
                )
                self._idle_welcome.reset()
            return context

        if self._tracking_thumb_up.observe(gesture):
            await self._dispatch(
                "stop_watching_target", reason="HAND_THUMB_UP"
            )
            self._reset_all()
            return context

        person_movements = {
            "thumb_left": "move_to_person_left",
            "thumb_right": "move_to_person_right",
        }
        for name, confirmation in (
            self._tracking_direction_confirmations.items()
        ):
            if confirmation.observe(gesture):
                await self._dispatch(
                    "explicit_navigation",
                    local_command=person_movements[name],
                    room_id=None,
                )
                self._reset_all()
                return context

        if self._armed_gesture is None:
            for name, confirmation in self._modifier_confirmations.items():
                if confirmation.observe(gesture):
                    self._armed_gesture = name
                    self._armed_frames_remaining = self.COMMAND_TIMEOUT_FRAMES
                    for item in self._command_confirmations.values():
                        item.reset()
                    break
            return context

        self._armed_frames_remaining -= 1
        if self._armed_frames_remaining <= 0:
            self._reset_sequence()
            return context

        for name, confirmation in self._command_confirmations.items():
            if confirmation.observe(gesture):
                await self._execute_sequence(self._armed_gesture, name)
                break
        return context

    async def _execute_sequence(self, modifier, command):
        self._reset_sequence()

        if modifier == "welcome" and command == "follow":
            await self._dispatch("follow_person", target="person")
            return

        if modifier == "welcome" and command == "finger_curl_up":
            await self._dispatch("approach_target", target="hand")
            return

        if modifier == "push" and command == "get_space":
            await self._dispatch(
                "explicit_navigation",
                local_command="open_space_middle",
                room_id=None,
                face_person=True,
            )
            return

        if modifier == "push" and command == "finger_curl_down":
            await self._dispatch(
                "explicit_navigation",
                local_command="nudge_backward",
                room_id=None,
                face_person=True,
            )

    async def run(self, yolo):
        self._running = True
        interval = 1.0 / self.MONITOR_HZ
        try:
            while self._running:
                started_at = time.monotonic()
                try:
                    await self._sample(yolo)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    print(f"[HAND INTERFACE ERROR] {type(error).__name__}: {error}")
                elapsed = time.monotonic() - started_at
                await asyncio.sleep(max(0.01, interval - elapsed))
        finally:
            self._running = False
            self._reset_all()

    async def stop(self, reason_code="STOPPED"):
        if reason_code == "SHUTDOWN":
            self._running = False
        self._reset_all()
