from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot.person_pose.lidar_clusters import Cluster


@dataclass
class StationaryLegAnchor:
    """Short-lived averaged location of a non-person leg-like return."""

    position: np.ndarray
    first_seen_at: float
    last_seen_at: float
    observations: int = 1
    frozen: bool = False
    frozen_at: float | None = None

    def observe(self, position: np.ndarray, average_window: int) -> None:
        self.observations += 1
        weight = 1.0 / min(self.observations, max(1, average_window))
        self.position += weight * (position - self.position)


class StationaryLegAnchorMap:
    """Create and maintain stationary leg-like background anchors.

    The caller owns person-track confidence.  Passing a person position says
    that every cluster inside ``person_filter_radius`` is uncertain person
    evidence: anchors in that area freeze and clusters in that area cannot
    create or update anchors.  Passing ``None`` disables that protection.
    """

    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self) -> None:
        self.anchors: list[StationaryLegAnchor] = []
        self.enabled = False

    @property
    def confirmed(self) -> list[StationaryLegAnchor]:
        return [
            anchor for anchor in self.anchors
            if anchor.observations >= self.settings.stationary_confirm_scans
        ]

    def prepare(
        self,
        stamp_seconds: float,
        person_position: np.ndarray | None,
    ) -> None:
        """Freeze anchors near a trusted person and discard stale memory."""
        if not self.enabled:
            return

        retained = []
        for anchor in self.anchors:
            should_freeze = (
                person_position is not None
                and np.linalg.norm(anchor.position - person_position)
                <= self.settings.person_filter_radius
            )

            if should_freeze and not anchor.frozen:
                anchor.frozen = True
                anchor.frozen_at = stamp_seconds
            elif not should_freeze and anchor.frozen:
                if anchor.frozen_at is not None:
                    anchor.last_seen_at += max(
                        0.0, stamp_seconds - anchor.frozen_at
                    )
                anchor.frozen = False
                anchor.frozen_at = None

            stale = (
                not anchor.frozen
                and stamp_seconds - anchor.last_seen_at
                > self.settings.stationary_timeout_seconds
            )
            if not stale:
                retained.append(anchor)
        self.anchors = retained

    def update(
        self,
        leg_clusters: list[Cluster],
        selected_clusters: list[Cluster],
        stamp_seconds: float,
        person_position: np.ndarray | None,
        *,
        allow_new: bool,
    ) -> None:
        """Update existing centroid anchors and optionally create new ones."""
        if allow_new:
            self.enabled = True
        if not self.enabled:
            return

        # The accepted pose may be newer than the pose used at scan start.
        # Re-evaluate freezing here so lifecycle safety does not depend on the
        # caller remembering a separate prepare call after person selection.
        self.prepare(stamp_seconds, person_position)

        selected_ids = {id(cluster) for cluster in selected_clusters}
        available = [
            cluster for cluster in leg_clusters
            if id(cluster) not in selected_ids
            and not self._inside_person_uncertainty(
                cluster.center,
                person_position,
            )
        ]
        unused = set(range(len(available)))
        for anchor in self.anchors:
            if anchor.frozen or not unused:
                continue
            index = min(
                unused,
                key=lambda item: float(np.linalg.norm(
                    available[item].center - anchor.position
                )),
            )
            distance = float(np.linalg.norm(
                available[index].center - anchor.position
            ))
            if distance > self.settings.stationary_match_radius:
                continue
            anchor.observe(
                available[index].center,
                self.settings.stationary_average_window,
            )
            anchor.last_seen_at = stamp_seconds
            unused.remove(index)

        if allow_new:
            for index in unused:
                position = available[index].center
                if any(
                    np.linalg.norm(position - anchor.position)
                    <= self.settings.stationary_match_radius
                    for anchor in self.anchors
                ):
                    continue
                self.anchors.append(StationaryLegAnchor(
                    position=position.copy(),
                    first_seen_at=stamp_seconds,
                    last_seen_at=stamp_seconds,
                ))

    def _inside_person_uncertainty(
        self,
        position: np.ndarray,
        person_position: np.ndarray | None,
    ) -> bool:
        return (
            person_position is not None
            and np.linalg.norm(position - person_position)
            <= self.settings.person_filter_radius
        )
