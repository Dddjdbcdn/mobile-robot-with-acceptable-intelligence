"""Hand-pose classification and temporal gesture confirmation.

The classifier intentionally reads from top to bottom: extract landmarks,
derive reusable geometry, evaluate the named poses, then choose the first
matching pose. Image coordinates increase downwards on the y axis.
"""

from __future__ import annotations

import math


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
LONG_FINGERS = ("index", "middle", "ring", "pinky")
MAX_FINGER_BEND_DEG = 75.0


class HandGestureClassifier:
    @staticmethod
    def _xy(point):
        return float(point["normalized_x"]), float(point["normalized_y"])

    @classmethod
    def _points(cls, landmarks, *names):
        points = [landmarks.get(name) for name in names]
        return points if all(points) else None

    @classmethod
    def _finger_points(cls, landmarks, finger):
        joints = (
            ("cmc", "mcp", "ip", "tip")
            if finger == "thumb"
            else ("mcp", "pip", "dip", "tip")
        )
        return cls._points(landmarks, *(f"{finger}_{joint}" for joint in joints))

    @classmethod
    def _finger_bend(cls, landmarks, finger):
        points = cls._finger_points(landmarks, finger)
        if points is None:
            return None

        _, proximal, distal, tip = map(cls._xy, points)
        before_joint = (distal[0] - proximal[0], distal[1] - proximal[1])
        after_joint = (tip[0] - distal[0], tip[1] - distal[1])
        before_length = math.hypot(*before_joint)
        after_length = math.hypot(*after_joint)
        if before_length <= 1e-6 or after_length <= 1e-6:
            return None

        cosine = (
            before_joint[0] * after_joint[0]
            + before_joint[1] * after_joint[1]
        ) / (before_length * after_length)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    @classmethod
    def _joints_point(cls, landmarks, finger, direction):
        """Return whether a finger's joints progress up or down the image."""
        points = cls._finger_points(landmarks, finger)
        if points is None:
            return False
        ys = [cls._xy(point)[1] for point in points]
        compare = (lambda first, second: first < second) if direction == "down" else (
            lambda first, second: first > second
        )
        return all(compare(first, second) for first, second in zip(ys, ys[1:]))

    @classmethod
    def _tips_follow(cls, landmarks, finger_order):
        points = cls._points(
            landmarks, *(f"{finger}_tip" for finger in finger_order)
        )
        if points is None:
            return False
        xs = [cls._xy(point)[0] for point in points]
        return all(left < right for left, right in zip(xs, xs[1:]))

    @classmethod
    def _finger_is_folded(cls, landmarks, finger, axis):
        """Return whether the PIP extends past both the MCP and tip."""
        points = cls._points(
            landmarks, f"{finger}_mcp", f"{finger}_pip", f"{finger}_tip"
        )
        if points is None:
            return False
        coordinate = 0 if axis == "x" else 1
        mcp, pip, tip = (cls._xy(point)[coordinate] for point in points)
        return pip > max(mcp, tip) if axis == "x" else pip < min(mcp, tip)

    @classmethod
    def classify(cls, hand):
        landmarks = (hand or {}).get("landmarks") or {}
        palm = cls._points(
            landmarks, "wrist", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"
        )
        if palm is None:
            return {
                "gesture": "unavailable",
                "open_fingers": 0,
                "palm_center": None,
                "gesture_checks": None,
            }

        # Derive the small set of facts shared by all pose rules.
        bends = {finger: cls._finger_bend(landmarks, finger) for finger in FINGERS}
        fingers = {
            finger: {
                "straight": bend is not None and bend <= MAX_FINGER_BEND_DEG,
                "bend_deg": bend,
            }
            for finger, bend in bends.items()
        }
        all_straight = all(item["straight"] for item in fingers.values())
        long_fingers_down = all(
            cls._joints_point(landmarks, finger, "down") for finger in LONG_FINGERS
        )
        long_fingers_up = all(
            cls._joints_point(landmarks, finger, "up") for finger in LONG_FINGERS
        )
        tips_left_to_right = cls._tips_follow(landmarks, LONG_FINGERS)
        tips_right_to_left = cls._tips_follow(landmarks, reversed(LONG_FINGERS))

        thumb = cls._points(landmarks, "thumb_tip", "thumb_ip")
        thumb_above_palm = False
        if thumb is not None:
            thumb_tip_y, thumb_ip_y = cls._xy(thumb[0])[1], cls._xy(thumb[1])[1]
            palm_top_y = min(cls._xy(point)[1] for point in palm[1:])
            thumb_above_palm = thumb_tip_y < thumb_ip_y < palm_top_y

        # Pose definitions live together so precedence and differences are visible.
        pose_matches = {
            "thumb_up": (
                fingers["thumb"]["straight"]
                and thumb_above_palm
                and all(
                    cls._finger_is_folded(landmarks, finger, "x")
                    for finger in LONG_FINGERS
                )
            ),
            "index_finger": (
                fingers["thumb"]["straight"]
                and fingers["index"]["straight"]
                and cls._joints_point(landmarks, "index", "up")
                and all(
                    cls._finger_is_folded(landmarks, finger, "y")
                    for finger in ("middle", "ring", "pinky")
                )
            ),
            "welcome": all_straight and long_fingers_down and tips_left_to_right,
            "push": (
                all_straight
                and long_fingers_up
                and cls._joints_point(landmarks, "thumb", "up")
                and tips_right_to_left
            ),
            "fingers_down": (
                all_straight and long_fingers_down and tips_right_to_left
            ),
            "fingers_up": all_straight and long_fingers_up and tips_left_to_right,
        }
        gesture = next(
            (name for name, matches in pose_matches.items() if matches),
            (
                "open_other"
                if any(item["straight"] for item in fingers.values())
                else "relaxed"
            ),
        )

        # Keep detailed facts for the live debug HUD and gesture tuning.
        gesture_checks = {
            **pose_matches,
            "handedness": "Right",
            "all_fingers_straight": all_straight,
            "welcome_y_order": long_fingers_down,
            "fingers_up_y_order": long_fingers_up,
            "push_thumb_y_order": cls._joints_point(landmarks, "thumb", "up"),
            "hand_tip_order": tips_left_to_right,
            "reverse_tip_order": tips_right_to_left,
        }
        palm_xy = [cls._xy(point) for point in palm]
        return {
            "gesture": gesture,
            "open_fingers": sum(item["straight"] for item in fingers.values()),
            "palm_center": (
                sum(point[0] for point in palm_xy) / len(palm_xy),
                sum(point[1] for point in palm_xy) / len(palm_xy),
            ),
            "fingers": fingers,
            "gesture_checks": gesture_checks,
        }


class FrameConfirmation:
    def __init__(self, gesture, required_frames, miss_tolerance=2):
        self.gesture = gesture
        self.required_frames = int(required_frames)
        self.miss_tolerance = int(miss_tolerance)
        self.reset()

    def observe(self, gesture):
        if gesture == self.gesture:
            self.misses = 0
            self.count += 1
            if self.count >= self.required_frames and not self.latched:
                self.latched = True
                return True
        else:
            self.misses += 1
            if self.misses > self.miss_tolerance:
                self.reset()
        return False

    def reset(self):
        self.count = 0
        self.misses = 0
        self.latched = False
