from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
import time

import numpy as np

from robot.person_pose.lidar_clusters import (
    Cluster,
    Detection,
    leg_detections,
)
from robot.person_pose.stationary_leg_anchors import StationaryLegAnchorMap
from robot.person_pose.yolo_correction import VisualContext


@dataclass(frozen=True)
class TrackerSettings:
    cluster_break_distance: float = 0.10
    min_cluster_points: int = 3
    min_leg_width: float = 0.02
    single_leg_width: float = 0.20
    min_leg_pair_distance: float = 0.10
    leg_pair_distance: float = 0.6
    leg_motion_radius: float = 0.35
    velocity_alpha: float = 0.65
    velocity_beta: float = 0.20
    motion_direction_min_speed: float = 0.15
    lateral_motion_penalty_gain: float = 0.6
    reverse_motion_penalty_gain: float = 0.8
    prediction_max_dt_seconds: float = 0.50
    prediction_max_shift_m: float = 1.0
    prediction_fallback_delay_seconds: float = 0.50
    recovery_expand_delay_seconds: float = 0.50
    velocity_decay_time_seconds: float = 1.0
    visual_gate_confirm_seconds: float = 1.0
    visual_missing_in_fov_timeout_seconds: float = 5.0
    visual_recovery_confirm_seconds: float = 1.0
    visual_recovery_timeout_seconds: float = 8.0
    visual_centered_gate_deg: float = 10.0

    seed_max_age_seconds: float = 0.75
    seed_correction_radius: float = 0.70
    seed_correction_gain: float = 0.12
    seed_override_count: int = 3

    recovery_delay_seconds: float = 1.0
    person_filter_radius: float = 0.35
    recovery_leg_search_radius: float = 0.8

    stationary_match_radius: float = 0.08
    near_anchor_radius: float = 0.16
    anchor_clear_bonus_m: float = 0.25
    leg_pair_bonus_m: float = 0.20
    leg_pair_bonus_m_cluster: float = 0.05
    stationary_confirm_scans: int = 5
    stationary_timeout_seconds: float = 2.0
    stationary_average_window: int = 20

    @property
    def leg_search_radius(self) -> float:
        """Compatibility name for the independently configured motion gate."""
        return self.leg_motion_radius

    @classmethod
    def from_node(cls, node) -> TrackerSettings:
        defaults = cls()
        for item in fields(defaults):
            node.declare_parameter(item.name, getattr(defaults, item.name))
        return cls(**{
            item.name: node.get_parameter(item.name).value
            for item in fields(defaults)
        })


@dataclass
class PersonTrack:
    position: np.ndarray
    stamp_seconds: float
    last_hit_at: float
    state: str = "seeded"
    observation_kind: str = "seed"
    filtered_position: np.ndarray = field(init=False)
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    prediction_stamp_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.filtered_position = self.position.copy()
        self.prediction_stamp_seconds = self.stamp_seconds

    def update_prediction(
        self,
        stamp_seconds: float,
        settings: TrackerSettings,
    ) -> np.ndarray:
        """Return the clamped constant-velocity centroid prediction."""
        dt = max(0.0, stamp_seconds - self.prediction_stamp_seconds)
        dt = min(dt, max(0.0, settings.prediction_max_dt_seconds))
        offset = self.velocity * dt
        offset_length = float(np.linalg.norm(offset))
        max_shift = max(0.0, settings.prediction_max_shift_m)
        if offset_length > max_shift and offset_length > 0.0:
            offset *= max_shift / offset_length
        return self.filtered_position + offset

    def update(
        self,
        detection: Detection,
        stamp_seconds: float,
        settings: TrackerSettings,
        *,
        reset_velocity: bool = False,
    ) -> None:
        """Commit an observation and correct its alpha-beta prediction."""
        if reset_velocity:
            self.reset_prediction(detection.position, stamp_seconds)
        else:
            raw_dt = stamp_seconds - self.prediction_stamp_seconds
            prediction = self.update_prediction(stamp_seconds, settings)
            innovation = detection.position - prediction
            alpha = float(np.clip(settings.velocity_alpha, 0.0, 1.0))
            beta = float(np.clip(settings.velocity_beta, 0.0, 1.0))
            self.filtered_position = prediction + alpha * innovation
            max_dt = max(0.0, settings.prediction_max_dt_seconds)
            if raw_dt > 1e-6 and max_dt > 1e-6:
                self.velocity += beta * innovation / min(raw_dt, max_dt)
            self.prediction_stamp_seconds = stamp_seconds

        self.position = detection.position.copy()
        self.stamp_seconds = stamp_seconds
        self.last_hit_at = time.monotonic()
        self.state = "tracking"
        self.observation_kind = detection.kind

    def reset_prediction(
        self,
        position: np.ndarray,
        stamp_seconds: float,
    ) -> None:
        """Restart the motion model at a stationary trusted position."""
        self.filtered_position = position.copy()
        self.velocity = np.zeros(2, dtype=float)
        self.prediction_stamp_seconds = stamp_seconds

    def decay_velocity(self, elapsed: float, decay_time: float) -> None:
        """Reduce velocity while its prediction has no observation support."""
        if decay_time <= 1e-6:
            self.velocity.fill(0.0)
            return
        self.velocity *= float(np.exp(-max(0.0, elapsed) / decay_time))

    def correct_position(self, correction: np.ndarray) -> None:
        """Apply an external positional correction without changing velocity."""
        self.position += correction
        self.filtered_position += correction


class PersonTrackerState:
    """Track a person from a trusted seed and clean leg observations only."""

    def __init__(self, settings: TrackerSettings):
        self.settings = settings
        self.anchor_map = StationaryLegAnchorMap(settings)
        self.reset()

    def reset(self) -> None:
        self.track: PersonTrack | None = None
        self.initialization_seed: np.ndarray | None = None
        self.acquisition_state = "disabled"
        self.last_detections: list[Detection] = []
        self.leg_clusters: list[Cluster] = []
        self.selected_clusters: list[Cluster] = []
        self.anchor_map.reset()
        self.stationary_cluster_ids: set[int] = set()
        self.near_anchor_cluster_ids: set[int] = set()
        self.clear_cluster_ids: set[int] = set()
        self.prediction_miss_started_at: float | None = None
        self.velocity_decay_stamp_seconds: float | None = None
        self.fallback_active = False
        self.selection_resets_velocity = False
        self.active_search_radius = 0.0
        self.active_search_center: np.ndarray | None = None
        self.active_fallback_center: np.ndarray | None = None
        self.search_phase = "inactive"
        self.search_expanded = False
        self.seed_mismatch_hits = 0

        # Visual recovery is deliberately kept separate from the lidar track
        # lifecycle.  The last lidar track remains available as a recovery
        # prior, but is not published while visual identity is contradictory.
        self.visual_mode = "inactive"
        self.visual_visible_started_at: float | None = None
        self.visual_missing_inside_started_at: float | None = None
        self.visual_recovery_started_at: float | None = None
        self.active_visual_constraint = None
        self.last_visual_base_bearing_rad: float | None = None
        self.last_visual_camera_bearing_rad: float | None = None
        self.visual_search_direction = 0
        self.visual_seed_not_before: float | None = None

    def activate(self) -> None:
        if self.acquisition_state == "disabled":
            self.acquisition_state = "waiting"

    def accept_tof(
        self,
        position: np.ndarray,
        captured_at: float | None = None,
    ) -> str:
        """Accept a trusted 2-D seed without requiring immediate lidar legs."""
        if self.visual_mode in {
            "visual_recovery", "reacquiring", "validating",
        }:
            return "deferred"

        if self.visual_mode == "awaiting_seed":
            if (
                captured_at is None
                or self.visual_seed_not_before is None
                or captured_at < self.visual_seed_not_before
            ):
                return "deferred"
            self.initialization_seed = position.copy()
            self.track = None
            self._clear_visual_recovery()
            self.anchor_map.reset()
            self.seed_mismatch_hits = 0
            self.acquisition_state = "seeded"
            return "reseed"

        if self.track is None:
            self._clear_visual_recovery()
            self.initialization_seed = position.copy()
            self.acquisition_state = "seeded"
            return "initialize"

        if self.track.observation_kind == "seed":
            self._clear_visual_recovery()
            self.track.position = position.copy()
            self.track.reset_prediction(position, self.track.stamp_seconds)
            self.seed_mismatch_hits = 0
            return "corrected"

        if np.linalg.norm(position - self.track.position) <= (
            self.settings.seed_correction_radius
        ):
            self._clear_visual_recovery()
            self.seed_mismatch_hits = 0
            correction = self.settings.seed_correction_gain * (
                position - self.track.position
            )
            self.track.correct_position(correction)
            return "corrected"

        self.seed_mismatch_hits += 1
        if self.seed_mismatch_hits < self.settings.seed_override_count:
            return "mismatch"

        self.initialization_seed = position.copy()
        self.track = None
        self._clear_visual_recovery()
        self.anchor_map.reset()
        self.seed_mismatch_hits = 0
        self.acquisition_state = "seeded"
        return "reseed"

    def update(
        self,
        clusters: list[Cluster],
        stamp_seconds: float,
        tof_position: np.ndarray | None,
        visual_context: VisualContext | None = None,
        *,
        monotonic_now: float | None = None,
    ) -> list[str]:
        """Run one lidar scan through initialization, selection, and handoff."""
        self._begin_scan(clusters)
        events: list[str] = []

        if self._initialize_track(tof_position, stamp_seconds):
            events.append("initialized")
        if self.track is None:
            return events

        now = time.monotonic() if monotonic_now is None else monotonic_now
        events.extend(self._update_visual_evidence(
            visual_context,
            now,
        ))
        if self.visual_mode in {
            "visual_recovery",
            "reacquiring",
            "validating",
            "awaiting_seed",
            "lost",
        }:
            return events

        self.anchor_map.prepare(
            stamp_seconds,
            self._trusted_person_position(),
        )
        self._prepare_cluster_anchor_relations()

        seeded = self.track.observation_kind == "seed"
        previous_position = self.track.position.copy()
        predicted_position = self.track.update_prediction(
            stamp_seconds,
            self.settings,
        )
        selected = self._select_leg(
            previous_position,
            predicted_position,
            stamp_seconds,
            allow_expansion=not seeded,
        )
        if seeded:
            if selected is not None:
                self._handoff(selected, stamp_seconds)
                events.append("leg_acquired")
            else:
                self.track.state = "seeded"
            return events

        self._handoff(selected, stamp_seconds)
        return events

    # Visual identity and recovery ------------------------------------

    def _update_visual_evidence(
        self,
        context: VisualContext | None,
        now: float,
    ) -> list[str]:
        """Apply delayed visual gating and detect lidar identity conflicts."""
        events: list[str] = []
        self.active_visual_constraint = None
        if (
            self.visual_mode in {
                "visual_recovery",
                "reacquiring",
                "validating",
                "awaiting_seed",
            }
            and self.visual_recovery_started_at is not None
            and now - self.visual_recovery_started_at
            >= max(0.0, self.settings.visual_recovery_timeout_seconds)
        ):
            self.visual_mode = "lost"
            events.append("visual_recovery_failed")
            return events
        if self.visual_mode == "lost":
            return events
        if context is None:
            self.visual_visible_started_at = None
            self.visual_missing_inside_started_at = None
            return events

        observation = context.observation
        camera_view = context.camera_view
        visual_constraint = context.constraint
        if observation.person_visible and visual_constraint is not None:
            self.visual_missing_inside_started_at = None
            if self.visual_visible_started_at is None:
                self.visual_visible_started_at = now

            if observation.base_bearing_rad is not None:
                bearing = float(observation.base_bearing_rad)
                self.last_visual_base_bearing_rad = bearing
                if abs(bearing) > 1e-3:
                    self.visual_search_direction = 1 if bearing > 0.0 else -1
            self.last_visual_camera_bearing_rad = float(
                observation.camera_bearing_rad
            )

            if self.visual_mode == "visual_recovery":
                self.visual_mode = "reacquiring"
                self.visual_visible_started_at = now
                events.append("visual_found")

            visible_for = now - self.visual_visible_started_at
            gate_confirmed = visible_for >= max(
                0.0,
                self.settings.visual_gate_confirm_seconds,
            )
            recovery_confirmed = visible_for >= max(
                0.0,
                self.settings.visual_recovery_confirm_seconds,
            )
            if self.visual_mode == "reacquiring":
                centered_gate = math.radians(max(
                    0.0,
                    self.settings.visual_centered_gate_deg,
                ))
                centered = (
                    observation.base_bearing_rad is not None
                    and abs(float(observation.base_bearing_rad))
                    <= centered_gate
                )
                if centered:
                    self.visual_mode = "validating"
                    events.append("visual_centered")
            if self.visual_mode == "validating" and recovery_confirmed:
                self.visual_mode = "awaiting_seed"
                self.visual_seed_not_before = now
                events.append("visual_validated")
            if gate_confirmed:
                self.active_visual_constraint = visual_constraint
            return events

        self.visual_visible_started_at = None
        if self.visual_mode in {
            "visual_recovery",
            "reacquiring",
            "validating",
            "awaiting_seed",
        }:
            if self.visual_mode != "visual_recovery":
                self.visual_mode = "visual_recovery"
                self.visual_seed_not_before = None
                events.append("visual_recovery_resumed")
            return events

        pose_is_visible = camera_view.contains(self.track.position)
        if not pose_is_visible:
            self.visual_missing_inside_started_at = None
            return events

        if self.visual_missing_inside_started_at is None:
            self.visual_missing_inside_started_at = now
            return events

        if now - self.visual_missing_inside_started_at < max(
            0.0,
            self.settings.visual_missing_in_fov_timeout_seconds,
        ):
            return events

        self.visual_mode = "visual_recovery"
        self.visual_recovery_started_at = now
        self.visual_seed_not_before = None
        if self.visual_search_direction == 0:
            bearing = self.last_visual_base_bearing_rad or 0.0
            self.visual_search_direction = 1 if bearing >= 0.0 else -1
        events.append("visual_recovery_started")
        return events

    def _clear_visual_recovery(self) -> None:
        self.visual_mode = "inactive"
        self.visual_visible_started_at = None
        self.visual_missing_inside_started_at = None
        self.visual_recovery_started_at = None
        self.active_visual_constraint = None
        self.visual_seed_not_before = None

    def _begin_scan(self, clusters: list[Cluster]) -> None:
        """Reset scan-local output and derive all leg observations."""
        self.active_search_radius = 0.0
        self.active_search_center = None
        self.active_fallback_center = None
        self.search_phase = "inactive"
        self.search_expanded = False
        self.selection_resets_velocity = False
        self.stationary_cluster_ids = set()
        self.near_anchor_cluster_ids = set()
        self.clear_cluster_ids = set()
        self.leg_clusters, self.last_detections = leg_detections(
            clusters,
            self.settings,
        )
        self.selected_clusters = []

    def _initialize_track(
        self,
        tof_position: np.ndarray | None,
        stamp_seconds: float,
    ) -> bool:
        """Create a seeded track when a trusted ToF position is available."""
        if (
            tof_position is not None
            and self.initialization_seed is None
            and self.track is None
        ):
            self.initialization_seed = tof_position.copy()
        if self.track is not None or self.initialization_seed is None:
            return False

        self.track = PersonTrack(
            position=self.initialization_seed.copy(),
            stamp_seconds=stamp_seconds,
            last_hit_at=time.monotonic(),
        )
        self.initialization_seed = None
        self.acquisition_state = "seeded"
        return True

    def _select_leg(
        self,
        previous_reference: np.ndarray,
        predicted_reference: np.ndarray,
        stamp_seconds: float,
        *,
        allow_expansion: bool,
    ) -> Detection | None:
        """Advance from prediction to timed fallback and recovery gates."""
        normal_radius = self.settings.leg_search_radius
        if not self.fallback_active:
            self._show_search("prediction", predicted_reference, normal_radius)
            selected = self._best_leg_in_radius(
                predicted_reference,
                normal_radius,
            )
            if selected is not None:
                self._clear_prediction_miss()
                return selected

        if not allow_expansion:
            self._clear_prediction_miss()
            return None

        missed_for = self._record_prediction_miss(stamp_seconds)
        if not self.fallback_active:
            self._decay_motion_velocity(stamp_seconds)
            fallback_delay = max(
                0.0,
                self.settings.prediction_fallback_delay_seconds,
            )
            if missed_for + 1e-9 < fallback_delay:
                return None
            self.fallback_active = True
            self.track.reset_prediction(previous_reference, stamp_seconds)

        self._show_search("fallback", previous_reference, normal_radius)
        selected = self._best_leg_in_radius(
            previous_reference,
            normal_radius,
        )
        if selected is not None:
            self.selection_resets_velocity = True
            self._clear_prediction_miss()
            return selected

        expand_after = (
            max(
                0.0,
                self.settings.prediction_fallback_delay_seconds,
            )
            + max(0.0, self.settings.recovery_expand_delay_seconds)
        )
        if missed_for + 1e-9 < expand_after:
            return None

        recovery_radius = max(
            normal_radius,
            self.settings.recovery_leg_search_radius,
        )
        self.search_expanded = True
        self._show_search("recovery", previous_reference, recovery_radius)
        selected = self._best_leg_in_radius(
            previous_reference,
            recovery_radius,
            require_clear=True,
        )
        if selected is not None:
            self.selection_resets_velocity = True
            self._clear_prediction_miss()
        return selected

    def _show_search(
        self,
        phase: str,
        center: np.ndarray,
        radius: float,
    ) -> None:
        """Expose the single gate used by the current search phase."""
        self.search_phase = phase
        self.active_search_center = center.copy()
        self.active_fallback_center = None
        self.active_search_radius = radius

    def _record_prediction_miss(self, stamp_seconds: float) -> float:
        """Return consecutive prediction-miss time in scan time."""
        if (
            self.prediction_miss_started_at is None
            or stamp_seconds < self.prediction_miss_started_at
        ):
            self.prediction_miss_started_at = stamp_seconds
            self.velocity_decay_stamp_seconds = stamp_seconds
        return max(0.0, stamp_seconds - self.prediction_miss_started_at)

    def _clear_prediction_miss(self) -> None:
        """Return selection to prediction-first operation."""
        self.prediction_miss_started_at = None
        self.velocity_decay_stamp_seconds = None
        self.fallback_active = False

    def _decay_motion_velocity(self, stamp_seconds: float) -> None:
        """Gradually weaken unsupported velocity during prediction timeout."""
        if self.velocity_decay_stamp_seconds is None:
            self.velocity_decay_stamp_seconds = stamp_seconds
            return
        dt = max(0.0, stamp_seconds - self.velocity_decay_stamp_seconds)
        self.velocity_decay_stamp_seconds = stamp_seconds
        decay_time = max(0.0, self.settings.velocity_decay_time_seconds)
        self.track.decay_velocity(dt, decay_time)

    def _best_leg_in_radius(
        self,
        reference: np.ndarray,
        search_radius: float,
        *,
        require_clear: bool = False,
    ) -> Detection | None:
        """Rank eligible detections inside one active search circle."""
        eligible: list[tuple[Detection, float]] = []
        for detection in self.last_detections:
            distance = self._detection_motion_distance(
                detection,
                reference,
            )
            if distance > search_radius:
                continue
            if not self._inside_visual_gate(detection):
                continue
            if any(
                id(cluster) in self.stationary_cluster_ids
                for cluster in detection.clusters
            ):
                continue
            is_clear = self._detection_is_clear(detection)
            if (
                (detection.kind == "pair" or require_clear)
                and not is_clear
            ):
                continue
            eligible.append((detection, distance))

        if not eligible:
            return None

        eligible_pair_count = sum(
            detection.kind == "pair"
            for detection, _ in eligible
        )
        pair_bonus = (
            self.settings.leg_pair_bonus_m
            if eligible_pair_count <= 1
            else self.settings.leg_pair_bonus_m_cluster
        )

        def ranking_key(
            item: tuple[Detection, float],
        ) -> tuple[float, float]:
            """Score continuity, motion direction, clearance, and pairs."""
            detection, distance = item
            is_clear = self._detection_is_clear(detection)
            clear_bonus = (
                self.settings.anchor_clear_bonus_m if is_clear else 0.0
            )
            detection_pair_bonus = (
                pair_bonus
                if detection.kind == "pair"
                else 0.0
            )
            continuity_margin = max(0.0, search_radius - distance)
            direction_penalty = self._directional_motion_penalty(detection)
            primary_score = (
                continuity_margin
                + clear_bonus
                + detection_pair_bonus
                - direction_penalty
            )
            proximity_tiebreak = -distance
            return (
                primary_score,
                proximity_tiebreak,
            )

        selected, _ = max(eligible, key=ranking_key)
        return selected

    def _directional_motion_penalty(self, detection: Detection) -> float:
        """Penalize lateral and reverse jumps once velocity is reliable."""
        if self.track is None:
            return 0.0
        velocity = self.track.velocity
        speed = float(np.linalg.norm(velocity))
        min_speed = max(0.0, self.settings.motion_direction_min_speed)
        if speed < max(min_speed, 1e-6):
            return 0.0

        direction = velocity / speed
        motion = detection.position - self.track.filtered_position
        forward_motion = float(np.dot(motion, direction))
        lateral_motion = abs(float(
            direction[0] * motion[1] - direction[1] * motion[0]
        ))
        lateral_penalty = (
            max(0.0, self.settings.lateral_motion_penalty_gain)
            * lateral_motion
        )
        reverse_penalty = (
            max(0.0, self.settings.reverse_motion_penalty_gain)
            * max(0.0, -forward_motion)
        )
        return lateral_penalty + reverse_penalty

    def _inside_visual_gate(self, detection: Detection) -> bool:
        constraint = self.active_visual_constraint
        if constraint is None:
            return True
        offset = detection.position - constraint.origin
        bearing = math.atan2(float(offset[1]), float(offset[0]))
        error = (bearing - constraint.bearing + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        return abs(error) <= constraint.gate

    def _detection_motion_distance(
        self,
        detection: Detection,
        reference: np.ndarray,
    ) -> float:
        """Measure motion of the position committed by this detection."""
        return float(
            np.linalg.norm(detection.position - reference)
        )

    def _prepare_cluster_anchor_relations(self) -> None:
        """Classify clusters as anchor-owned, near-anchor, or clear."""
        self.stationary_cluster_ids = set()
        self.near_anchor_cluster_ids = set()
        self.clear_cluster_ids = set()
        confirmed = self.anchor_map.confirmed
        match_radius = max(0.0, self.settings.stationary_match_radius)
        near_radius = max(match_radius, self.settings.near_anchor_radius)

        for cluster in self.leg_clusters:
            cluster_id = id(cluster)
            if not confirmed:
                self.clear_cluster_ids.add(cluster_id)
                continue
            clearance = min(
                float(np.linalg.norm(cluster.center - anchor.position))
                for anchor in confirmed
            )
            if clearance <= match_radius:
                self.stationary_cluster_ids.add(cluster_id)
            elif clearance <= near_radius:
                self.near_anchor_cluster_ids.add(cluster_id)
            else:
                self.clear_cluster_ids.add(cluster_id)

    def _detection_is_clear(self, detection: Detection) -> bool:
        """Return whether every component is clear of confirmed anchors."""
        return all(
            id(cluster) in self.clear_cluster_ids
            for cluster in detection.clusters
        )

    def _handoff(
        self,
        detection: Detection | None,
        stamp_seconds: float,
    ) -> None:
        """Commit selection, maintain anchors, and advance tracking state."""
        if detection is not None:
            self.track.update(
                detection,
                stamp_seconds,
                self.settings,
                reset_velocity=self.selection_resets_velocity,
            )
            self.selected_clusters = list(detection.clusters)
            self.acquisition_state = "tracking"

        self._update_anchors(detection, stamp_seconds)
        if detection is not None:
            return

        missed_for = time.monotonic() - self.track.last_hit_at
        self.track.state = (
            "tracking"
            if missed_for < self.settings.recovery_delay_seconds
            else "recovering"
        )

    def _update_anchors(
        self,
        detection: Detection | None,
        stamp_seconds: float,
    ) -> None:
        """Apply all post-selection anchor creation and centroid updates."""
        self.anchor_map.update(
            self.leg_clusters,
            self.selected_clusters,
            stamp_seconds,
            self._trusted_person_position(),
            allow_new=(
                detection is not None
                and detection.kind == "pair"
                and not self.search_expanded
            ),
        )

    def _trusted_person_position(self) -> np.ndarray | None:
        """Return a recent lidar-supported person pose for anchor protection."""
        if (
            self.track is None
            or self.track.observation_kind not in {"single", "pair"}
            or time.monotonic() - self.track.last_hit_at
            > self.settings.recovery_delay_seconds
        ):
            return None
        return self.track.position

    @property
    def published_state(self) -> str:
        if self.visual_mode != "inactive":
            return self.visual_mode
        if self.track is not None:
            return self.track.state
        return self.acquisition_state
