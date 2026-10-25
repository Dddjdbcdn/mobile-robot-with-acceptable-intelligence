"""Person-pose keypoint, torso-target, and skeleton-path helpers."""

from collections import deque

from actions.tracking.target_catalog import HUMAN_RETARGETS


SKELETON_GRAPH = {
    "nose": ["left_eye", "right_eye"],
    "left_eye": ["nose", "left_ear"],
    "right_eye": ["nose", "right_ear"],
    "left_ear": ["left_eye", "left_shoulder"],
    "right_ear": ["right_eye", "right_shoulder"],
    "left_shoulder": [
        "left_ear", "right_shoulder", "left_elbow", "left_hip",
        "torso_center",
    ],
    "right_shoulder": [
        "right_ear", "left_shoulder", "right_elbow", "right_hip",
        "torso_center",
    ],
    "left_elbow": ["left_shoulder", "left_wrist"],
    "right_elbow": ["right_shoulder", "right_wrist"],
    "left_wrist": ["left_elbow"],
    "right_wrist": ["right_elbow"],
    "left_hip": [
        "left_shoulder", "right_hip", "left_knee", "torso_center",
    ],
    "right_hip": [
        "right_shoulder", "left_hip", "right_knee", "torso_center",
    ],
    "left_knee": ["left_hip", "left_ankle"],
    "right_knee": ["right_hip", "right_ankle"],
    "left_ankle": ["left_knee"],
    "right_ankle": ["right_knee"],
    "torso_center": [
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    ],
}


def find_skeleton_path(start, target):
    queue = deque([(start, [start])])
    visited = set()
    while queue:
        current, path = queue.popleft()
        if current == target:
            return path
        if current in visited:
            continue
        visited.add(current)
        for neighbor in SKELETON_GRAPH.get(current, ()):
            queue.append((neighbor, path + [neighbor]))
    return None


def find_best_person_path(person, target):
    target_keypoints = HUMAN_RETARGETS.get(target, ())
    visible_keypoints = (person.get("keypoints") or {}).keys()
    best_path = None
    for visible_keypoint in visible_keypoints:
        for target_keypoint in target_keypoints:
            path = find_skeleton_path(visible_keypoint, target_keypoint)
            if path and (best_path is None or len(path) < len(best_path)):
                best_path = path
    return best_path


def _visible_torso_center(person):
    keypoints = person.get("keypoints") or {}
    shoulders = [
        keypoints[name]
        for name in ("left_shoulder", "right_shoulder")
        if name in keypoints
    ]
    hips = [
        keypoints[name]
        for name in ("left_hip", "right_hip")
        if name in keypoints
    ]
    if shoulders and hips:
        shoulder_x = sum(p["normalized_x"] for p in shoulders) / len(shoulders)
        shoulder_y = sum(p["normalized_y"] for p in shoulders) / len(shoulders)
        hip_x = sum(p["normalized_x"] for p in hips) / len(hips)
        hip_y = sum(p["normalized_y"] for p in hips) / len(hips)
        return {
            "normalized_x": (shoulder_x + hip_x) * 0.5,
            "normalized_y": (shoulder_y + hip_y) * 0.5,
        }
    return None


def person_keypoint(person, name, torso_target=None):
    if name != "torso_center":
        return (person.get("keypoints") or {}).get(name)
    if torso_target is not None:
        return torso_target
    return _visible_torso_center(person)


class TorsoTargetEstimator:
    """Keep one torso target while shoulder and hip landmarks flicker.

    Before a shoulder is visible, a configured box-relative offset above the
    hips supplies the target. Once a visible shoulder/hip center is available,
    offsets from each visible torso anchor are calibrated. Later estimates can
    therefore follow any calibrated shoulder or hip without aiming directly at
    that joint.

    Temporal target smoothing intentionally remains in ``track_action``. This
    class only combines simultaneous geometric observations.
    """

    ANCHOR_NAMES = (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )
    HIP_NAMES = ("left_hip", "right_hip")
    DEFAULT_OFFSET_X_BOX_RATIO = 0.0
    DEFAULT_OFFSET_Y_BOX_RATIO = -0.35

    def __init__(
        self,
        *,
        bootstrap_offset_x=DEFAULT_OFFSET_X_BOX_RATIO,
        bootstrap_offset_y=DEFAULT_OFFSET_Y_BOX_RATIO,
    ):
        self.bootstrap_offset_x = float(bootstrap_offset_x)
        self.bootstrap_offset_y = float(bootstrap_offset_y)
        self._anchor_offsets = {}

    @staticmethod
    def _bbox_height(person):
        bbox = person.get("bbox") or {}
        y1 = bbox.get("normalized_y1")
        y2 = bbox.get("normalized_y2")
        if y1 is None or y2 is None:
            return None
        height = float(y2) - float(y1)
        return height if height > 0.0 else None

    @staticmethod
    def _mean_points(points):
        if not points:
            return None
        return {
            "normalized_x": sum(
                point["normalized_x"] for point in points
            ) / len(points),
            "normalized_y": sum(
                point["normalized_y"] for point in points
            ) / len(points),
        }

    @staticmethod
    def _target(x, y, source):
        return {
            "normalized_x": max(0.0, min(1.0, x)),
            "normalized_y": max(0.0, min(1.0, y)),
            "source": source,
        }

    def _calibrated_estimate(self, keypoints, bbox_height):
        estimates = []
        for name in self.ANCHOR_NAMES:
            anchor = keypoints.get(name)
            offset = self._anchor_offsets.get(name)
            if anchor is None or offset is None:
                continue
            estimates.append({
                "normalized_x": (
                    anchor["normalized_x"] + offset[0] * bbox_height
                ),
                "normalized_y": (
                    anchor["normalized_y"] + offset[1] * bbox_height
                ),
            })
        return self._mean_points(estimates)

    def _update_anchor_offsets(
        self,
        keypoints,
        torso_center,
        bbox_height,
    ):
        for name in self.ANCHOR_NAMES:
            anchor = keypoints.get(name)
            if anchor is None:
                continue
            self._anchor_offsets[name] = (
                (torso_center["normalized_x"]
                 - anchor["normalized_x"]) / bbox_height,
                (torso_center["normalized_y"]
                 - anchor["normalized_y"]) / bbox_height,
            )

    def update(self, person, *, allow_bootstrap=True):
        """Return the observed, fused, calibrated, or bootstrap torso target."""
        keypoints = person.get("keypoints") or {}
        bbox_height = self._bbox_height(person)
        visible_center = _visible_torso_center(person)

        if bbox_height is None:
            return visible_center

        calibrated_center = self._calibrated_estimate(
            keypoints,
            bbox_height,
        )

        if visible_center is not None:
            if calibrated_center is None:
                target = visible_center
                source = "observed"
            else:
                target = self._mean_points(
                    (visible_center, calibrated_center)
                )
                source = "fused"
            self._update_anchor_offsets(
                keypoints,
                visible_center,
                bbox_height,
            )
            return self._target(
                target["normalized_x"],
                target["normalized_y"],
                source,
            )

        if calibrated_center is not None:
            return self._target(
                calibrated_center["normalized_x"],
                calibrated_center["normalized_y"],
                "calibrated_offset",
            )

        if not allow_bootstrap:
            return None

        hips = [
            keypoints[name]
            for name in self.HIP_NAMES
            if name in keypoints
        ]
        hip_center = self._mean_points(hips)
        if hip_center is None:
            return None
        return self._target(
            hip_center["normalized_x"]
            + self.bootstrap_offset_x * bbox_height,
            hip_center["normalized_y"]
            + self.bootstrap_offset_y * bbox_height,
            "bootstrap_offset",
        )


def predict_next_keypoint(person, path, path_index):
    if path_index <= 0 or path_index >= len(path):
        return None
    keypoints = person.get("keypoints") or {}
    previous = keypoints.get(path[path_index - 1])
    current = keypoints.get(path[path_index])
    if previous is None or current is None:
        return None

    predicted_x = 2.0 * current["normalized_x"] - previous["normalized_x"]
    predicted_y = 2.0 * current["normalized_y"] - previous["normalized_y"]
    return (
        max(0.0, min(1.0, predicted_x)),
        max(0.0, min(1.0, predicted_y)),
    )
