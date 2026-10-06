"""Temporal geometry helpers for keeping one visual person target."""

import math


PERSON_MATCH_MIN_IOU = 0.10
PERSON_MATCH_MAX_CENTER_DISTANCE = 0.20
PERSON_MATCH_MIN_AREA_RATIO = 0.35
PERSON_BBOX_SMOOTHING_ALPHA = 0.35


def _normalized_box(value):
    bbox = value.get("bbox") if isinstance(value, dict) else None
    if isinstance(bbox, dict):
        value = bbox
    if not isinstance(value, dict):
        return None
    try:
        box = (
            float(value["normalized_x1"]),
            float(value["normalized_y1"]),
            float(value["normalized_x2"]),
            float(value["normalized_y2"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _box_metrics(previous_bbox, detection):
    previous = _normalized_box(previous_bbox)
    current = _normalized_box(detection)
    if previous is None or current is None:
        return None

    px1, py1, px2, py2 = previous
    cx1, cy1, cx2, cy2 = current
    intersection_width = max(0.0, min(px2, cx2) - max(px1, cx1))
    intersection_height = max(0.0, min(py2, cy2) - max(py1, cy1))
    intersection = intersection_width * intersection_height
    previous_area = (px2 - px1) * (py2 - py1)
    current_area = (cx2 - cx1) * (cy2 - cy1)
    union = previous_area + current_area - intersection
    iou = intersection / union if union > 0.0 else 0.0
    center_distance = math.hypot(
        (px1 + px2 - cx1 - cx2) * 0.5,
        (py1 + py2 - cy1 - cy2) * 0.5,
    )
    area_ratio = min(previous_area, current_area) / max(
        previous_area, current_area
    )
    return iou, center_distance, area_ratio


def _confidence(detection):
    try:
        value = float(detection.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def select_tracked_person(detections, previous_bbox):
    """Select only the person geometrically continuous with the saved target."""
    people = [
        detection
        for detection in detections
        if detection.get("class") == "person" and detection.get("keypoints")
    ]
    if not people:
        return None
    if _normalized_box(previous_bbox) is None:
        return max(people, key=_confidence)

    matches = []
    for detection in people:
        metrics = _box_metrics(previous_bbox, detection)
        if metrics is None:
            continue
        iou, center_distance, area_ratio = metrics
        if iou < PERSON_MATCH_MIN_IOU and not (
            center_distance <= PERSON_MATCH_MAX_CENTER_DISTANCE
            and area_ratio >= PERSON_MATCH_MIN_AREA_RATIO
        ):
            continue

        center_similarity = max(
            0.0,
            1.0 - center_distance / PERSON_MATCH_MAX_CENTER_DISTANCE,
        )
        score = (
            0.55 * iou
            + 0.25 * center_similarity
            + 0.15 * area_ratio
            + 0.05 * _confidence(detection)
        )
        matches.append((score, detection))

    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def update_tracked_bbox(previous_bbox, detection, alpha=None):
    """Maintain a smoothed association box without modifying the detection."""
    current = _normalized_box(detection)
    if current is None:
        return previous_bbox
    previous = _normalized_box(previous_bbox)
    if previous is None:
        blended = current
    else:
        resolved_alpha = (
            PERSON_BBOX_SMOOTHING_ALPHA if alpha is None else float(alpha)
        )
        resolved_alpha = max(0.0, min(1.0, resolved_alpha))
        blended = tuple(
            old + resolved_alpha * (new - old)
            for old, new in zip(previous, current)
        )

    x1, y1, x2, y2 = blended
    return {
        "normalized_x1": x1,
        "normalized_y1": y1,
        "normalized_x2": x2,
        "normalized_y2": y2,
        "normalized_center_x": (x1 + x2) * 0.5,
        "normalized_center_y": (y1 + y2) * 0.5,
    }
