"""Always-on vision interface that turns hand poses into commands."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import time

from cognition.hand.gestures import FrameConfirmation, HandGestureClassifier
from services.diagnostic_session import smallbrain_diagnostics


class HandGestureInterface:
    """Recognize gestures without owning any robot action state."""

    MONITOR_HZ = 15.0
    STATUS_MAX_AGE_SECONDS = 1.0
    IDLE_WELCOME_CONFIRM_FRAMES = 10
    TRACKING_STOP_CONFIRM_FRAMES = 10
    TRACKING_DIRECTION_CONFIRM_FRAMES = 5
    MOVEMENT_STOP_CONFIRM_FRAMES = 5
    MODIFIER_CONFIRM_FRAMES = 3
    COMMAND_CONFIRM_FRAMES = 3
    COMMAND_TIMEOUT_FRAMES = 45
    MISS_TOLERANCE = 3
    WELCOME_PERSON_MIN_CONFIDENCE = 0.50

    def __init__(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks
        self._dispatch_handler = None
        self._running = False
        self._last_frame_sequence = None
        self._inference_sequence = 0
        self._latest_status = None
        self._armed_gesture = None
        self._armed_frames_remaining = 0
        self._diagnostic_decision = "starting"
        self._command_dispatch_count = 0
        self._last_command = None
        self._last_sequence = None

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
            name:
            FrameConfirmation(
                name,
                self.TRACKING_DIRECTION_CONFIRM_FRAMES,
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
            name:
            FrameConfirmation(
                name,
                self.MODIFIER_CONFIRM_FRAMES,
                miss_tolerance=self.MISS_TOLERANCE,
            )
            for name in ("welcome", "push")
        }
        self._command_confirmations = {
            name:
            FrameConfirmation(
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
        result = await self._dispatch_handler(command, data)
        if command != "observe_gesture":
            self._command_dispatch_count += 1
            self._last_command = {
                "name": command,
                "data": data,
                "count": self._command_dispatch_count,
                "timestamp": datetime.now().astimezone().isoformat(),
            }
        return result

    @staticmethod
    def classify_hand_pose(hand):
        return HandGestureClassifier.classify(hand)

    def gesture_status(self, max_age_seconds=None):
        status = self._latest_status
        if not isinstance(status, dict):
            return None
        max_age = (self.STATUS_MAX_AGE_SECONDS
                   if max_age_seconds is None else float(max_age_seconds))
        if time.monotonic() - float(status.get("updated_at") or 0.0) > max_age:
            return None
        return dict(status)

    def _reset_command_sequence(self):
        self._armed_gesture = None
        self._armed_frames_remaining = 0
        for confirmation in self._modifier_confirmations.values():
            confirmation.reset()
        for confirmation in self._command_confirmations.values():
            confirmation.reset()

    def _reset_sequence(self):
        self._reset_command_sequence()
        self._tracking_thumb_up.reset()
        for confirmation in self._tracking_direction_confirmations.values():
            confirmation.reset()

    def _reset_all(self):
        self._reset_sequence()
        self._idle_welcome.reset()
        self._movement_push.reset()

    @staticmethod
    def _confirmation_status(confirmation):
        return {
            "count": confirmation.count,
            "required": confirmation.required_frames,
            "misses": confirmation.misses,
            "miss_tolerance": confirmation.miss_tolerance,
            "latched": confirmation.latched,
        }

    def _write_diagnostic_status(self, status):
        """Update the gesture section of the central diagnostics snapshot."""
        confirmations = {
            "idle_welcome": self._confirmation_status(self._idle_welcome),
            "tracking_stop": self._confirmation_status(
                self._tracking_thumb_up),
            "movement_stop": self._confirmation_status(self._movement_push),
            "tracking_direction": {
                name: self._confirmation_status(item)
                for name, item in self._tracking_direction_confirmations.items()
            },
            "modifier": {
                name: self._confirmation_status(item)
                for name, item in self._modifier_confirmations.items()
            },
            "command": {
                name: self._confirmation_status(item)
                for name, item in self._command_confirmations.items()
            },
        }
        debug = status.get("debug") or {}
        event = {
            "sequence": status.get("sequence"),
            "inference_sequence": status.get("inference_sequence"),
            "state": status.get("state"),
            "gesture": status.get("gesture"),
            "modifier": self._armed_gesture,
            "modifier_frames_remaining": self._armed_frames_remaining,
            "decision": self._diagnostic_decision,
            "person_detected": status.get("person") is not None,
            "hand_detected": status.get("hand") is not None,
            "palm_center": debug.get("palm_center"),
            "gesture_checks": debug.get("gesture_checks"),
            "confirmations": confirmations,
            "commands_dispatched": self._command_dispatch_count,
            "last_command": self._last_command,
            "last_sequence": self._last_sequence,
        }
        smallbrain_diagnostics.update(
            "hand_gesture",
            event,
            tally_group="gestures",
            tally_value=status.get("gesture") or "unavailable",
        )

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
            sequence, detections, frame_bgr, _ = yolo.detection_snapshot_with_frame(
            )
            frame_source = "yolo"

        frame_sequence = (frame_source, sequence)
        if frame_bgr is None or frame_sequence == self._last_frame_sequence:
            return
        self._last_frame_sequence = frame_sequence

        person = max(
            (item for item in detections
             if item.get("class") == "person" and item.get("keypoints")),
            key=lambda item: float(item.get("confidence") or 0.0),
            default=None,
        )
        hand = None
        if person is not None:
            debug_path = None
            now = time.monotonic()
            debug_due = self._debug_image_interval > 0.0 and (
                self._last_debug_image_at is None or
                now - self._last_debug_image_at >= self._debug_image_interval)
            if debug_due:
                self._last_debug_image_at = now
                debug_path = str(
                    Path("results/action_results") /
                    "hand_landmarks_latest.jpg")
            hand = await self.hand_landmarks.detect_right_hand(
                frame_bgr, person, debug_path=debug_path)
            self._inference_sequence += 1

        pose = HandGestureClassifier.classify(hand)
        context = await self._observe_gesture(pose["gesture"],
                                              pose["palm_center"], detections)
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
        self._write_diagnostic_status(self._latest_status)

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
                bbox = person["bbox"]
                palm_x = float(palm_center[0])
                palm_y = float(palm_center[1])
                x1 = float(bbox["normalized_x1"])
                y1 = float(bbox["normalized_y1"])
                x2 = float(bbox["normalized_x2"])
                y2 = float(bbox["normalized_y2"])
            except (KeyError, TypeError, ValueError):
                continue
            if (person_confidence < cls.WELCOME_PERSON_MIN_CONFIDENCE or
                    not (x1 <= palm_x <= x2 and y1 <= palm_y <= y2)):
                continue
            candidates.append((person_confidence, person))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    async def _observe_gesture(self, gesture, palm_center, detections=None):
        context = await self._dispatch("observe_gesture",
                                       gesture=gesture,
                                       palm_center=palm_center)
        tracking_person = bool(context.get("tracking_person"))
        if context.get("movement_active"):
            if context.get("mode") == "following":
                self._diagnostic_decision = "following_waiting_for_thumb_up"
                self._reset_command_sequence()
                # During follow, push is a continuous hand-position override
                # that asks the base to restore its normal standoff.
                self._movement_push.reset()
                if self._tracking_thumb_up.observe(gesture):
                    self._diagnostic_decision = "stop_follow_dispatched"
                    await self._dispatch("stop_follow", reason="HAND_THUMB_UP")
                    self._reset_all()
            else:
                self._diagnostic_decision = "movement_waiting_for_push"
                self._reset_sequence()
                if self._movement_push.observe(gesture):
                    self._diagnostic_decision = "stop_movement_dispatched"
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
            self._diagnostic_decision = (
                "idle_confirming_welcome" if welcome_person is not None else
                "idle_no_person_bound_to_hand")
            if self._idle_welcome.observe(observed_gesture):
                self._diagnostic_decision = "watch_target_dispatched"
                await self._dispatch(
                    "watch_target",
                    target="person",
                    target_kind="person",
                    initial_person=welcome_person,
                )
                self._idle_welcome.reset()
            return context

        if self._tracking_thumb_up.observe(gesture):
            self._diagnostic_decision = "stop_watching_dispatched"
            await self._dispatch("stop_watching_target",
                                 reason="HAND_THUMB_UP")
            self._reset_all()
            return context

        person_movements = {
            "thumb_left": "move_to_person_left",
            "thumb_right": "move_to_person_right",
        }
        for name, confirmation in (
                self._tracking_direction_confirmations.items()):
            if confirmation.observe(gesture):
                self._diagnostic_decision = f"{name}_navigation_dispatched"
                await self._dispatch(
                    "explicit_navigation",
                    local_command=person_movements[name],
                    room_id=None,
                )
                self._reset_all()
                return context

        if self._armed_gesture is None:
            self._diagnostic_decision = "tracking_waiting_for_modifier"
            for name, confirmation in self._modifier_confirmations.items():
                if confirmation.observe(gesture):
                    self._armed_gesture = name
                    self._armed_frames_remaining = self.COMMAND_TIMEOUT_FRAMES
                    for item in self._command_confirmations.values():
                        item.reset()
                    self._diagnostic_decision = f"modifier_{name}_armed"
                    break
            return context

        self._armed_frames_remaining -= 1
        self._diagnostic_decision = "modifier_armed_waiting_for_command"
        if self._armed_frames_remaining <= 0:
            self._diagnostic_decision = "modifier_timed_out"
            self._reset_sequence()
            return context

        for name, confirmation in self._command_confirmations.items():
            if confirmation.observe(gesture):
                self._diagnostic_decision = f"command_{name}_confirmed"
                await self._execute_sequence(self._armed_gesture, name)
                break
        return context

    async def _execute_sequence(self, modifier, command):
        self._last_sequence = {
            "modifier": modifier,
            "command_gesture": command,
            "modifier_confirmation": self._confirmation_status(
                self._modifier_confirmations[modifier]),
            "command_confirmation": self._confirmation_status(
                self._command_confirmations[command]),
            "timestamp": datetime.now().astimezone().isoformat(),
            "outcome": "unsupported_modifier_command_pair",
        }
        self._reset_sequence()

        if modifier == "welcome" and command == "follow":
            self._diagnostic_decision = "follow_person_dispatched"
            self._last_sequence["outcome"] = "follow_person_dispatched"
            await self._dispatch("follow_person", target="person")
            return

        if modifier == "welcome" and command == "finger_curl_up":
            self._diagnostic_decision = "approach_target_dispatched"
            self._last_sequence["outcome"] = "approach_target_dispatched"
            await self._dispatch("approach_target", target="hand")
            return

        if modifier == "push" and command == "get_space":
            self._diagnostic_decision = "open_space_navigation_dispatched"
            self._last_sequence["outcome"] = (
                "open_space_navigation_dispatched")
            await self._dispatch(
                "explicit_navigation",
                local_command="open_space_middle",
                room_id=None,
                face_person=True,
            )
            return

        if modifier == "push" and command == "finger_curl_down":
            self._diagnostic_decision = "backward_navigation_dispatched"
            self._last_sequence["outcome"] = (
                "backward_navigation_dispatched")
            await self._dispatch(
                "explicit_navigation",
                local_command="nudge_backward",
                room_id=None,
                face_person=True,
            )
            return

        self._diagnostic_decision = "unsupported_modifier_command_pair"

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
                    print(
                        f"[HAND INTERFACE ERROR] {type(error).__name__}: {error}"
                    )
                elapsed = time.monotonic() - started_at
                await asyncio.sleep(max(0.01, interval - elapsed))
        finally:
            self._running = False
            self._reset_all()

    async def stop(self, reason_code="STOPPED"):
        if reason_code == "SHUTDOWN":
            self._running = False
        self._reset_all()
