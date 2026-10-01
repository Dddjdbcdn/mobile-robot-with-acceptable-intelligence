"""Hand-pose classification and temporal gesture confirmation.

The classifier intentionally reads from top to bottom: extract landmarks,
derive reusable geometry, evaluate the named poses, then choose the first
matching pose. Image coordinates increase downwards on the y axis.
"""

from __future__ import annotations

LONG_FINGERS = ("index", "middle", "ring", "pinky")
THREE_FINGERS = ("middle", "ring", "pinky")


class HandGestureClassifier:
    @staticmethod
    def _xy(point):
        return float(point["normalized_x"]), float(point["normalized_y"])

    @classmethod
    def _points(cls, landmarks, *names):
        points = [landmarks.get(name) for name in names]
        return points if all(points) else None

    @classmethod
    def _axis_order(cls, landmarks, axis, *levels, relation="<"):
        """Compare arbitrary landmark groups along one image axis.

        A string is one ordering level. An iterable groups landmarks whose
        internal order does not matter. Every value in a level must satisfy
        ``relation`` against every value in the following level. ``relation``
        may be ``"<"``, ``">"``, or a two-argument comparison function.

        Examples::

            # index_mcp.y < index_pip.y < index_dip.y < index_tip.y
            _axis_order(landmarks, "y", "index_mcp", "index_pip",
                        "index_dip", "index_tip")

            # index_dip.y > both index_pip.y and middle_tip.y
            _axis_order(landmarks, "y", "index_dip",
                        ("index_pip", "middle_tip"), relation=">")

            # index_tip.x must be at least 0.05 beyond middle_tip.x
            _axis_order(landmarks, "x", "index_tip", "middle_tip",
                        relation=lambda left, right: left > right + 0.05)
        """
        if axis not in ("x", "y"):
            raise ValueError(f"Unsupported axis: {axis!r}")
        if len(levels) < 2:
            return False
        if relation == "<":
            compare = lambda left, right: left < right
        elif relation == ">":
            compare = lambda left, right: left > right
        elif callable(relation):
            compare = relation
        else:
            raise ValueError(f"Unsupported relation: {relation!r}")

        coordinate = 0 if axis == "x" else 1
        coordinate_levels = []
        for level in levels:
            names = (level,) if isinstance(level, str) else tuple(level)
            if not names:
                return False
            points = cls._points(landmarks, *names)
            if points is None:
                return False
            coordinate_levels.append(
                [cls._xy(point)[coordinate] for point in points]
            )

        return all(
            compare(left, right)
            for left_level, right_level in zip(
                coordinate_levels, coordinate_levels[1:]
            )
            for left in left_level
            for right in right_level
        )

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

        # Direction and ordering are more stable than landmark-derived bend angles.
        finger_joints = {
            "thumb": ("cmc", "mcp", "ip", "tip"),
            **{
                finger: ("mcp", "pip", "dip", "tip")
                for finger in LONG_FINGERS
            },
        }
        fingers = {
            finger: {
                "up": cls._axis_order(
                    landmarks,
                    "y",
                    *(f"{finger}_{joint}" for joint in joints),
                    relation=">",
                ),
                "down": cls._axis_order(
                    landmarks,
                    "y",
                    *(f"{finger}_{joint}" for joint in joints),
                ),
                "curl_up": cls._axis_order(
                    landmarks,
                    "y",
                    f"{finger}_pip",
                    tuple(
                        f"{finger}_{joint}"
                        for joint in ("mcp", "dip", "tip")
                    ),
                    relation=">",
                ),
                "curl_down": cls._axis_order(
                    landmarks,
                    "y",
                    f"{finger}_pip",
                    tuple(
                        f"{finger}_{joint}"
                        for joint in ("mcp", "dip", "tip")
                    ),
                    relation="<",
                ),
            }
            for finger, joints in finger_joints.items()
        }
        long_fingers_down = all(
            fingers[finger]["down"] for finger in LONG_FINGERS
        )
        long_fingers_up = all(
            fingers[finger]["up"] for finger in LONG_FINGERS
        )
        long_fingers_curl_up = all(
            fingers[finger]["curl_up"] for finger in LONG_FINGERS
        )
        long_fingers_curl_down = all(
            fingers[finger]["curl_down"] for finger in LONG_FINGERS
        )

        tips_left_to_right = cls._axis_order(
            landmarks, "x", *(f"{finger}_tip" for finger in LONG_FINGERS)
        )
        tips_right_to_left = cls._axis_order(
            landmarks,
            "x",
            *(f"{finger}_tip" for finger in reversed(LONG_FINGERS)),
        )

        thumb_tip_left = cls._axis_order(
            landmarks, "x", "thumb_tip", "thumb_ip"
        )
        thumb_tip_right = cls._axis_order(
            landmarks, "x", "thumb_tip", "thumb_ip", relation=">"
        )
        thumb_above_palm = cls._axis_order(
            landmarks,
            "y",
            "thumb_tip",
            "thumb_ip",
            ("index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"),
        )

        # Pose definitions live together so precedence and differences are visible.
        pose_matches = {
            "thumb_up": (
                thumb_above_palm
                and all(
                    cls._axis_order(
                        landmarks,
                        "x",
                        f"{finger}_pip",
                        (f"{finger}_mcp", f"{finger}_tip"),
                        relation=">",
                    )
                    for finger in LONG_FINGERS
                )
            ),
            "welcome": (
                long_fingers_down and tips_left_to_right and thumb_tip_left
            ),
            "follow": (
                long_fingers_down and tips_left_to_right and thumb_tip_right
            ),
            "push": (
                long_fingers_up and tips_right_to_left and thumb_tip_right
            ),
            "get_space": (
                long_fingers_up and tips_right_to_left and thumb_tip_left
            ),
            "finger_curl_down": (
                long_fingers_curl_down and tips_right_to_left and thumb_tip_right
            ),
            "finger_curl_up": (
                long_fingers_curl_up and tips_left_to_right and thumb_tip_left
            )
        }
        gesture = next(
            (name for name, matches in pose_matches.items() if matches),
            (
                "open_other"
                if any(
                    item[direction]
                    for item in fingers.values()
                    for direction in ("up", "down")
                )
                else "relaxed"
            ),
        )

        # Keep detailed facts for the live debug HUD and gesture tuning.
        gesture_checks = {
            **pose_matches,
            "handedness": "Right",
            "welcome_y_order": long_fingers_down,
            "fingers_up_y_order": long_fingers_up,
            "long_fingers_curl_down": long_fingers_curl_down,
            "long_fingers_curl_up": long_fingers_curl_up,
            "push_thumb_y_order": fingers["thumb"]["up"],
            "thumb_tip_left_of_ip": thumb_tip_left,
            "thumb_tip_right_of_ip": thumb_tip_right,
            "hand_tip_order": tips_left_to_right,
            "reverse_tip_order": tips_right_to_left,
        }
        palm_xy = [cls._xy(point) for point in palm]
        return {
            "gesture": gesture,
            "open_fingers": sum(
                item["up"] or item["down"] for item in fingers.values()
            ),
            "palm_center": (
                sum(point[0] for point in palm_xy) / len(palm_xy),
                sum(point[1] for point in palm_xy) / len(palm_xy),
            ),
            "fingers": fingers,
            "gesture_checks": gesture_checks,
        }


class FrameConfirmation:
    def __init__(self, gesture, required_frames, miss_tolerance):
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
