from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import numpy as np
from sensor_msgs.msg import LaserScan


BEAM_SPACING_ALLOWANCE = 1.5


@dataclass
class Cluster:
    points: np.ndarray
    center: np.ndarray
    width: float


@dataclass
class Detection:
    position: np.ndarray
    kind: str
    clusters: tuple[Cluster, ...] = ()


def transform_yaw(transform) -> float:
    rotation = transform.transform.rotation
    return math.atan2(
        2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )


def transform_points(points: np.ndarray, transform) -> np.ndarray:
    translation = transform.transform.translation
    yaw = transform_yaw(transform)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    matrix = np.array([[cosine, -sine], [sine, cosine]])
    offset = np.array([translation.x, translation.y])
    return points @ matrix.T + offset


def clusters_from_scan(
    scan: LaserScan,
    transform,
    break_distance: float,
    min_points: int,
) -> list[Cluster]:
    """Cluster valid adjacent beams and transform them to the tracking frame."""
    groups: list[np.ndarray] = []
    current: list[tuple[float, float]] = []
    previous = None
    angle = float(scan.angle_min)

    def finish() -> None:
        nonlocal current
        if len(current) >= min_points:
            groups.append(np.asarray(current, dtype=float))
        current = []

    for distance in scan.ranges:
        valid = (
            math.isfinite(distance)
            and float(scan.range_min) <= distance <= float(scan.range_max)
        )
        if not valid:
            finish()
            previous = None
            angle += float(scan.angle_increment)
            continue

        point = np.array(
            [distance * math.cos(angle), distance * math.sin(angle)],
            dtype=float,
        )
        if previous is not None:
            adaptive_break = (
                break_distance
                + BEAM_SPACING_ALLOWANCE
                * distance
                * abs(float(scan.angle_increment))
            )
            if float(np.linalg.norm(point - previous)) > adaptive_break:
                finish()
        current.append((float(point[0]), float(point[1])))
        previous = point
        angle += float(scan.angle_increment)
    finish()

    clusters = []
    for points in groups:
        transformed = transform_points(points, transform)
        clusters.append(
            Cluster(
                points=transformed,
                center=np.median(transformed, axis=0),
                width=float(np.linalg.norm(transformed[-1] - transformed[0])),
            )
        )
    return clusters


def leg_detections(
    clusters: list[Cluster],
    settings,
) -> tuple[list[Cluster], list[Detection]]:
    """Return global single-leg and compatible leg-pair observations."""
    legs = [
        cluster for cluster in clusters
        if settings.min_leg_width
        <= cluster.width
        <= settings.single_leg_width
    ]
    detections = [
        Detection(cluster.center.copy(), "single", (cluster,))
        for cluster in legs
    ]
    for index, first in enumerate(legs):
        for second in legs[index + 1:]:
            separation = float(np.linalg.norm(first.center - second.center))
            if not (
                settings.min_leg_pair_distance
                <= separation
                <= settings.leg_pair_distance
            ):
                continue
            detections.append(Detection(
                (first.center + second.center) * 0.5,
                "pair",
                (first, second),
            ))
    return legs, detections


def closest_detection(
    detections: list[Detection],
    reference: np.ndarray,
) -> Detection | None:
    """Return the observation nearest a reference point."""
    if not detections:
        return None
    return min(
        detections,
        key=lambda detection: float(
            np.linalg.norm(detection.position - reference)
        ),
    )


def filter_scan_around_person(
    scan: LaserScan,
    transform,
    person_position: np.ndarray,
    radius: float,
) -> tuple[LaserScan, int]:
    """Copy a scan and omit returns within radius of a tracked person."""
    filtered = copy.deepcopy(scan)
    ranges = np.asarray(scan.ranges, dtype=float)
    angles = (
        float(scan.angle_min)
        + np.arange(ranges.size, dtype=float) * float(scan.angle_increment)
    )
    valid = (
        np.isfinite(ranges)
        & (ranges >= float(scan.range_min))
        & (ranges <= float(scan.range_max))
    )
    valid_indices = np.flatnonzero(valid)
    scan_points = np.column_stack((
        ranges[valid] * np.cos(angles[valid]),
        ranges[valid] * np.sin(angles[valid]),
    ))
    tracking_points = transform_points(scan_points, transform)
    distances = np.linalg.norm(tracking_points - person_position, axis=1)
    masked_indices = valid_indices[distances <= radius]

    output_ranges = list(filtered.ranges)
    for index in masked_indices:
        output_ranges[int(index)] = math.nan
    filtered.ranges = output_ranges
    return filtered, int(masked_indices.size)
