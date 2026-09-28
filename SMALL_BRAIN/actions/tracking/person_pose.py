"""Pure person-pose keypoint and skeleton-path helpers."""

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


def person_keypoint(person, name):
    if name != "torso_center":
        return (person.get("keypoints") or {}).get(name)

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

    bbox = person.get("bbox") or {}
    center_x = bbox.get("normalized_center_x")
    center_y = bbox.get("normalized_center_y")
    if isinstance(center_x, (int, float)) and isinstance(center_y, (int, float)):
        return {
            "normalized_x": float(center_x),
            "normalized_y": float(center_y),
        }
    return None


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
