"""Target-centric person tracking with geometry and appearance memory.

YOLO detections are ephemeral observations, not identities. This service keeps
one persistent target: center/hand selection happens once, geometry maintains
continuity, and ReID resolves ambiguous multi-person motion and loss recovery.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np


_SHOULDERS = frozenset(("left_shoulder", "right_shoulder"))
_HIPS = frozenset(("left_hip", "right_hip"))
_DEBUG_TILE_WIDTH = 112
_DEBUG_TILE_HEIGHT = 196
_DEBUG_GALLERY_COLUMNS = 5
_DEBUG_TRACKING_LOG_MAX_BYTES = 5 * 1024 * 1024
_SMALL_BRAIN_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class PersonTrackingConfig:
    embedding_interval_frames: int = 3
    gallery_size: int = 10
    confirmation_hits: int = 5
    tentative_timeout_seconds: float = 0.5
    # Ordinary visual tracking and TrackAction share this loss window. During
    # following, the confirmed target is pinned for the lifetime of the follow
    # session so its appearance gallery remains available for validation.
    lost_timeout_seconds: float = 3.0
    yolo_wait_timeout_seconds: float = 0.5
    reid_crop_padding: float = 0.05

    # Pose is an acquisition/learning gate, not a permanent requirement for
    # every person in the scene.
    credible_person_confidence: float = 0.75
    credible_person_keypoints: int = 5
    credible_pair_maximum_iou: float = 0.15

    # Geometry is temporary after loss. Trusted appearance lasts until expiry.
    geometry_memory_seconds: float = 1.0
    recent_geometry_minimum_iou: float = 0.02
    recent_geometry_maximum_center_distance: float = 0.35
    multi_person_minimum_similarity: float = 0.60
    single_person_reacquisition_minimum_similarity: float = 0.45
    multi_person_reacquisition_minimum_similarity: float = 0.50
    reid_winner_margin: float = 0.12
    reacquisition_confirmation_frames: int = 3

    maximum_prediction_seconds: float = 1.0
    previous_velocity_weight: float = 0.65
    measured_velocity_weight: float = 0.35

    minimum_detection_confidence: float = 0.40
    minimum_box_width: float = 0.04
    minimum_box_height: float = 0.14

    prototype_previous_weight: float = 0.90
    prototype_new_weight: float = 0.10
    gallery_not_full_similarity: float = 0.80
    gallery_full_similarity: float = 0.70

    save_gallery_debug_images: bool = True
    gallery_debug_directory: str = "results/reid"
    lookup_minimum_iou: float = 0.10
    lookup_maximum_center_distance: float = 0.12


DEFAULT_PERSON_TRACKING_CONFIG = PersonTrackingConfig()


def _normalized_box(person: dict) -> np.ndarray | None:
    try:
        bbox = person["bbox"]
        box = np.asarray((
            bbox["normalized_x1"], bbox["normalized_y1"],
            bbox["normalized_x2"], bbox["normalized_y2"],
        ), dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _box_center(box: np.ndarray) -> np.ndarray:
    return np.asarray(((box[0] + box[2]) * 0.5,
                       (box[1] + box[3]) * 0.5), dtype=np.float32)


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 1e-12 else 0.0


def _area_ratio(first: np.ndarray, second: np.ndarray) -> float:
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1]))
    largest = max(first_area, second_area)
    return min(first_area, second_area) / largest if largest > 1e-12 else 0.0


def _confidence(person: dict) -> float:
    try:
        value = float(person.get("confidence") or 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _geometry_cost(predicted, person) -> float:
    box = _normalized_box(person)
    iou = _box_iou(predicted, box)
    center_distance = float(np.linalg.norm(
        _box_center(predicted) - _box_center(box)))
    return (
        (1.0 - iou)
        + min(1.0, center_distance / 0.30)
        + (1.0 - _area_ratio(predicted, box))
    ) / 3.0


def _embedding_quality(person, config) -> float | None:
    box = _normalized_box(person)
    if box is None:
        return None
    width, height = box[2:] - box[:2]
    confidence = _confidence(person)
    if (confidence < config.minimum_detection_confidence
            or width < config.minimum_box_width
            or height < config.minimum_box_height):
        return None
    return confidence


def _person_debug_thumbnail(frame_bgr, person, padding):
    box = _normalized_box(person)
    if frame_bgr is None or box is None or frame_bgr.size == 0:
        return None
    height, width = frame_bgr.shape[:2]
    box_width, box_height = box[2:] - box[:2]
    x1 = max(0, int((box[0] - box_width * padding) * width))
    y1 = max(0, int((box[1] - box_height * padding) * height))
    x2 = min(width, int(math.ceil((box[2] + box_width * padding) * width)))
    y2 = min(height, int(math.ceil((box[3] + box_height * padding) * height)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    scale = min(_DEBUG_TILE_WIDTH / crop.shape[1],
                _DEBUG_TILE_HEIGHT / crop.shape[0])
    resized_width = max(1, int(round(crop.shape[1] * scale)))
    resized_height = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (resized_width, resized_height))
    tile = np.zeros((_DEBUG_TILE_HEIGHT, _DEBUG_TILE_WIDTH, 3), dtype=np.uint8)
    left = (_DEBUG_TILE_WIDTH - resized_width) // 2
    top = (_DEBUG_TILE_HEIGHT - resized_height) // 2
    tile[top:top + resized_height, left:left + resized_width] = resized
    return tile


@dataclass(frozen=True, slots=True)
class PersonTrackSnapshot:
    track_id: int
    state: str
    visible: bool
    person: dict | None
    bbox: tuple[float, float, float, float]
    association_confidence: float
    appearance_samples: int
    age_seconds: float
    missing_seconds: float
    appearance_match_source: str = "WAIT"
    reid_similarity: float | None = None
    association_cost: float | None = None
    geometry_cost: float | None = None
    appearance_cost: float | None = None
    pose_cost: float | None = None
    pose_cost_is_fallback: bool = False
    gallery_update_age_seconds: float | None = None


@dataclass(slots=True)
class _AppearanceSample:
    embedding: np.ndarray
    quality: float
    image: np.ndarray | None = field(compare=False, repr=False)
    admission_similarity: float | None = None
    admission_kind: str = "NEW"
    added_sequence: int = 0
    last_supported_sequence: int = 0
    support_count: int = 1


@dataclass(frozen=True, slots=True)
class _AppearanceMatch:
    similarity: float
    source: str


@dataclass(slots=True)
class _TargetTrack:
    candidate_id: int
    bbox: np.ndarray
    person: dict
    created_at: float
    last_seen_at: float
    last_sequence: int
    track_id: int | None = None
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32))
    consecutive_hits: int = 1
    misses: int = 0
    state: str = "tentative"
    visible: bool = True
    association_confidence: float = 1.0
    gallery: list[_AppearanceSample] = field(default_factory=list)
    appearance_prototype: np.ndarray | None = None
    gallery_last_updated_at: float | None = None
    appearance_match_source: str = "WAIT"
    reid_similarity: float | None = None
    association_cost: float | None = None
    geometry_cost: float | None = None
    appearance_cost: float | None = None
    pose_cost: float | None = None
    pose_cost_is_fallback: bool = False

    def predict(self, now, config):
        elapsed = max(0.0, min(
            config.maximum_prediction_seconds, now - self.last_seen_at))
        predicted = self.bbox + self.velocity * elapsed
        predicted[0::2] = np.clip(predicted[0::2], 0.0, 1.0)
        predicted[1::2] = np.clip(predicted[1::2], 0.0, 1.0)
        if predicted[2] <= predicted[0] or predicted[3] <= predicted[1]:
            return self.bbox.copy()
        return predicted

    def appearance_match(self, embedding):
        references = [(sample.embedding, f"G{index}")
                      for index, sample in enumerate(self.gallery)]
        if self.appearance_prototype is not None:
            references.append((self.appearance_prototype, "PROTO"))
        if not references:
            return None
        similarity, source = max(
            (float(np.clip(np.dot(embedding, reference), -1.0, 1.0)), source)
            for reference, source in references
        )
        return _AppearanceMatch(similarity, source)

    def mark_gallery_support(self, embedding, sequence, config):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12 or not np.isfinite(vector).all():
            return
        vector /= norm
        for sample in self.gallery:
            if float(np.dot(vector, sample.embedding)) >= config.gallery_full_similarity:
                sample.last_supported_sequence = sequence
                sample.support_count += 1

    def add_embedding(self, embedding, *, quality, image, sequence, config):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12 or not np.isfinite(vector).all():
            return False
        vector = np.ascontiguousarray(vector / norm, dtype=np.float32)
        if self.appearance_prototype is None:
            self.appearance_prototype = vector.copy()
        else:
            prototype = (config.prototype_previous_weight
                         * self.appearance_prototype
                         + config.prototype_new_weight * vector)
            prototype_norm = float(np.linalg.norm(prototype))
            if prototype_norm > 1e-12:
                self.appearance_prototype = np.ascontiguousarray(
                    prototype / prototype_norm, dtype=np.float32)

        if not self.gallery:
            self.gallery.append(_AppearanceSample(
                vector.copy(), quality, image,
                admission_similarity=None, admission_kind="FIRST",
                added_sequence=sequence, last_supported_sequence=sequence,
            ))
            return True
        similarities = [float(np.dot(vector, sample.embedding))
                        for sample in self.gallery]
        nearest_index = int(np.argmax(similarities))
        nearest_similarity = similarities[nearest_index]
        nearest = self.gallery[nearest_index]
        candidate = _AppearanceSample(
            vector.copy(), quality, image,
            admission_similarity=nearest_similarity,
            added_sequence=sequence, last_supported_sequence=sequence,
        )
        if len(self.gallery) < config.gallery_size:
            if nearest_similarity < config.gallery_not_full_similarity:
                self.gallery.append(candidate)
                return True
            if quality > nearest.quality:
                candidate.admission_kind = "UP"
                candidate.support_count = nearest.support_count
                self.gallery[nearest_index] = candidate
                return True
            return False
        if nearest_similarity < config.gallery_full_similarity:
            replace_index = min(
                range(len(self.gallery)),
                key=lambda index: (
                    self.gallery[index].last_supported_sequence,
                    self.gallery[index].quality,
                    self.gallery[index].added_sequence,
                ),
            )
            self.gallery[replace_index] = candidate
            return True
        if quality > nearest.quality:
            candidate.admission_kind = "UP"
            candidate.support_count = nearest.support_count
            self.gallery[nearest_index] = candidate
            return True
        return False


class PersonTrackingService:
    """Maintain one persistent visual target from ephemeral observations."""

    def __init__(
        self,
        yolo,
        reid_service,
        *,
        config: PersonTrackingConfig | None = None,
        embedding_interval_frames: int | None = None,
        gallery_size: int | None = None,
        confirmation_hits: int | None = None,
        tentative_timeout_seconds: float | None = None,
        lost_timeout_seconds: float | None = None,
        save_gallery_debug_images: bool | None = None,
    ):
        resolved = config or DEFAULT_PERSON_TRACKING_CONFIG
        overrides = {name: value for name, value in {
            "embedding_interval_frames": embedding_interval_frames,
            "gallery_size": gallery_size,
            "confirmation_hits": confirmation_hits,
            "tentative_timeout_seconds": tentative_timeout_seconds,
            "lost_timeout_seconds": lost_timeout_seconds,
            "save_gallery_debug_images": save_gallery_debug_images,
        }.items() if value is not None}
        resolved = replace(resolved, **overrides)
        not_full = max(-1.0, min(1.0, resolved.gallery_not_full_similarity))
        full = min(not_full, max(-1.0, min(
            1.0, resolved.gallery_full_similarity)))
        self.config = replace(
            resolved,
            embedding_interval_frames=max(
                1, int(resolved.embedding_interval_frames)),
            gallery_size=max(1, int(resolved.gallery_size)),
            confirmation_hits=max(1, int(resolved.confirmation_hits)),
            tentative_timeout_seconds=max(
                0.0, float(resolved.tentative_timeout_seconds)),
            lost_timeout_seconds=max(0.1, float(resolved.lost_timeout_seconds)),
            geometry_memory_seconds=max(
                0.0, float(resolved.geometry_memory_seconds)),
            reacquisition_confirmation_frames=max(
                1, int(resolved.reacquisition_confirmation_frames)),
            gallery_not_full_similarity=not_full,
            gallery_full_similarity=full,
        )
        self.yolo = yolo
        self.reid_service = reid_service
        self.embedding_interval_frames = self.config.embedding_interval_frames
        self.gallery_size = self.config.gallery_size
        self.confirmation_hits = self.config.confirmation_hits
        self.tentative_timeout_seconds = self.config.tentative_timeout_seconds
        self.lost_timeout_seconds = self.config.lost_timeout_seconds

        self._tracks: dict[int, _TargetTrack] = {}
        # Keep the most recently retired appearance memory until a replacement
        # target has learned its first embedding.  This avoids throwing away
        # the only useful ReID gallery during the gap between targets.
        self._retired_gallery_track: _TargetTrack | None = None
        self._next_candidate_id = 1
        self._next_track_id = 1
        self._last_embedding_sequence: int | None = None
        self._reacquisition_hits = 0
        self._last_scene_mode = "acquiring"
        self._multi_last_seen_at: float | None = None
        self._preferred_detection: dict | None = None
        self._snapshot_sequence = 0
        self._snapshot_captured_at: float | None = None
        self._snapshot: tuple[PersonTrackSnapshot, ...] = ()
        self._snapshot_lock = threading.Lock()
        self._running = False
        self._target_tracking_active = False
        self._target_persistent = False
        self._debug_decision = "idle"
        self._debug_gallery_decision = "not_checked"
        self._debug_geometry_index: int | None = None
        self._debug_geometry_score: float | None = None
        self._debug_reid_best: float | None = None
        self._debug_reid_margin: float | None = None
        self._debug_reid_source: str | None = None
        self._debug_reid_threshold: float | None = None
        self._debug_reid_role = "--"
        self._debug_candidates = "--"
        self._last_console_debug_key = None
        self._debug_log_path = _SMALL_BRAIN_ROOT / "results/person_tracking.log"
        try:
            self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._debug_log_path = None
        self._debug_directory = None
        if self.config.save_gallery_debug_images:
            configured = Path(self.config.gallery_debug_directory)
            self._debug_directory = (configured if configured.is_absolute()
                                     else _SMALL_BRAIN_ROOT / configured)
            try:
                self._debug_directory.mkdir(parents=True, exist_ok=True)
                for path in self._debug_directory.glob("id_*_gallery.jpg"):
                    path.unlink(missing_ok=True)
            except OSError:
                self._debug_directory = None

    @property
    def active(self):
        return self._running

    @property
    def scene_mode(self):
        if not self._target_tracking_active:
            return "idle"
        track = self._target()
        if track is not None and track.track_id is None:
            return "acquiring"
        return ("recovery" if track is not None and not track.visible
                else self._last_scene_mode)

    def set_target_persistent(self, persistent):
        """Retain a confirmed lost target until its owning session ends."""
        self._target_persistent = bool(persistent)

    def start_target_tracking(self):
        """Allow target selection for a newly requested tracking action."""
        if self._target_tracking_active:
            return
        self._reset_target()
        self._target_tracking_active = True
        self._last_console_debug_key = None
        self._write_debug_line(
            f"[PERSON_TRACK] session=start wall_time="
            f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}"
        )

    def stop_target_tracking(self):
        """Discard identity state when no tracking action owns a target."""
        self._write_debug_line(
            f"[PERSON_TRACK] session=stop wall_time="
            f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}"
        )
        self._target_tracking_active = False
        self._reset_target()
        with self._snapshot_lock:
            self._snapshot = ()

    def _target(self):
        return next(iter(self._tracks.values()), None)

    def _debug_path(self, track):
        if self._debug_directory is None or track.track_id is None:
            return None
        return self._debug_directory / f"id_{track.track_id:04d}_gallery.jpg"

    def _delete_debug_image(self, track):
        path = self._debug_path(track)
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _has_appearance_memory(track):
        return bool(track.gallery or track.appearance_prototype is not None)

    def _retire_gallery(self, track):
        if not self._has_appearance_memory(track):
            return
        previous = self._retired_gallery_track
        if previous is not None and previous is not track:
            self._delete_debug_image(previous)
        self._retired_gallery_track = track

    def _clear_retired_gallery(self):
        retired = self._retired_gallery_track
        if retired is None:
            return
        self._delete_debug_image(retired)
        self._retired_gallery_track = None

    def _save_debug_image(self, track):
        path = self._debug_path(track)
        if path is None:
            return
        samples = [sample for sample in track.gallery if sample.image is not None]
        try:
            if not samples:
                path.unlink(missing_ok=True)
                return
            columns = min(_DEBUG_GALLERY_COLUMNS, len(samples))
            rows = int(math.ceil(len(samples) / columns))
            sheet = np.full((rows * _DEBUG_TILE_HEIGHT,
                             columns * _DEBUG_TILE_WIDTH, 3), 20, dtype=np.uint8)
            for index, sample in enumerate(samples):
                row, column = divmod(index, columns)
                top, left = row * _DEBUG_TILE_HEIGHT, column * _DEBUG_TILE_WIDTH
                sheet[top:top + _DEBUG_TILE_HEIGHT,
                      left:left + _DEBUG_TILE_WIDTH] = sample.image
                cv2.putText(sheet, f"G{index} Q{sample.quality:.2f}",
                            (left + 3, top + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.36, (50, 255, 50), 1, cv2.LINE_AA)
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except (OSError, cv2.error):
            pass

    def tracking_snapshot(self):
        with self._snapshot_lock:
            return self._snapshot_sequence, self._snapshot

    def tracking_snapshot_with_time(self):
        with self._snapshot_lock:
            return (self._snapshot_sequence, self._snapshot,
                    self._snapshot_captured_at)

    def get_track(self, track_id):
        _, tracks = self.tracking_snapshot()
        return next((track for track in tracks
                     if track.track_id == int(track_id)), None)

    def _detection_matches_target(self, person):
        track = self._target()
        box = _normalized_box(person)
        if track is None or not track.visible or box is None:
            return False
        return (_box_iou(box, track.bbox) >= self.config.lookup_minimum_iou
                or np.linalg.norm(_box_center(box) - _box_center(track.bbox))
                <= self.config.lookup_maximum_center_distance)

    async def wait_for_confirmed_track(self, person=None, *, timeout=1.0):
        if not self._target_tracking_active:
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, float(timeout))
        bound_candidate_id = None
        if person is not None:
            target = self._target()
            if target is not None and self._detection_matches_target(person):
                bound_candidate_id = target.candidate_id
            else:
                # An explicit hand/search handoff owns initial acquisition.
                # Drop an unrelated automatically centered target and consume
                # this box as the reference on the next YOLO frame.
                self._preferred_detection = copy.deepcopy(person)
                self._reset_target(keep_preference=True)
        while True:
            target = self._target()
            if person is None:
                if target is not None and target.track_id is not None and target.visible:
                    snapshot = self.get_track(target.track_id)
                    if snapshot is not None and snapshot.visible:
                        return snapshot
            else:
                if bound_candidate_id is None and target is not None:
                    bound_candidate_id = target.candidate_id
                if (target is not None
                        and target.candidate_id == bound_candidate_id
                        and target.track_id is not None and target.visible):
                    snapshot = self.get_track(target.track_id)
                    if snapshot is not None and snapshot.visible:
                        return snapshot
            if loop.time() >= deadline:
                return None
            await asyncio.sleep(0.01)

    def track_for_detection(self, person):
        target = self._target()
        if (target is None or target.track_id is None
                or not self._detection_matches_target(person)):
            return None
        snapshot = self.get_track(target.track_id)
        return snapshot if snapshot is not None and snapshot.visible else None

    async def close(self):
        self._running = False

    async def run(self):
        self._running = True
        last_sequence = -1
        try:
            while self._running:
                sequence, detections, frame_bgr, timing = (
                    self.yolo.detection_snapshot_with_frame())
                if sequence <= last_sequence or frame_bgr is None:
                    await self.yolo.wait_for_inference_after(
                        sequence if frame_bgr is None else last_sequence,
                        timeout=self.config.yolo_wait_timeout_seconds)
                    continue
                last_sequence = sequence
                captured_at = timing.get("source_captured_at")
                await self.update(
                    sequence, frame_bgr, detections,
                    captured_at=(time.monotonic() if captured_at is None
                                 else float(captured_at)),
                )
        finally:
            self._running = False

    def _credible_person(self, person):
        """Return whether pose makes this a clear whole-person observation."""
        keypoints = set(person.get("keypoints") or {})
        return (
            _confidence(person) >= self.config.credible_person_confidence
            and len(keypoints) >= self.config.credible_person_keypoints
            and bool(keypoints & _SHOULDERS)
            and bool(keypoints & _HIPS)
        )

    def _separate_observations(self, first, second):
        return (
            _box_iou(_normalized_box(first), _normalized_box(second))
            <= self.config.credible_pair_maximum_iou
        )

    def _select_initial_person(self, people):
        if not people:
            return None
        preferred = self._preferred_detection
        if preferred is not None and _normalized_box(preferred) is not None:
            reference = _normalized_box(preferred)
            selected = min(
                people,
                key=lambda person: (
                    -_box_iou(reference, _normalized_box(person)),
                    float(np.linalg.norm(
                        _box_center(reference)
                        - _box_center(_normalized_box(person)))),
                ),
            )
            self._preferred_detection = None
            return selected
        image_center = np.asarray((0.5, 0.5), dtype=np.float32)
        return min(
            people,
            key=lambda person: (
                float(np.linalg.norm(
                    _box_center(_normalized_box(person)) - image_center)),
                -_confidence(person),
            ),
        )

    def _geometry_candidate(self, track, people, now, *, strict=False):
        predicted = track.predict(now, self.config)
        minimum_iou = (self.config.lookup_minimum_iou if strict
                       else self.config.recent_geometry_minimum_iou)
        maximum_center_distance = (
            self.config.lookup_maximum_center_distance if strict
            else self.config.recent_geometry_maximum_center_distance
        )
        candidates = []
        for index, person in enumerate(people):
            box = _normalized_box(person)
            iou = _box_iou(predicted, box)
            center_distance = float(np.linalg.norm(
                _box_center(predicted) - _box_center(box)))
            if (iou < minimum_iou
                    and center_distance > maximum_center_distance):
                continue
            candidates.append((
                _geometry_cost(predicted, person),
                -iou,
                center_distance,
                index,
            ))
        if not candidates:
            self._debug_geometry_index = None
            self._debug_geometry_score = None
            return None
        selected = min(candidates)
        self._debug_geometry_score = float(selected[0])
        self._debug_geometry_index = int(selected[3])
        return selected[3]

    def _has_multi_evidence(self, target_index, people):
        target = people[target_index]
        target_is_clear = self._credible_person(target)
        for index, other in enumerate(people):
            if index == target_index or not self._separate_observations(
                    target, other):
                continue
            other_is_credible = (
                _confidence(other) >= self.config.credible_person_confidence
                if target_is_clear else self._credible_person(other)
            )
            if other_is_credible:
                return True
        return False

    def _scene_mode_for(self, target_index, people, now):
        if self._has_multi_evidence(target_index, people):
            self._multi_last_seen_at = now
            return "multi"
        if self._last_scene_mode == "multi":
            # Once two people are established, overlapping boxes or one merged
            # box are an occlusion, not proof that the scene became solo.
            if len(people) >= 2:
                self._multi_last_seen_at = now
                return "multi"
            if (self._multi_last_seen_at is not None
                    and now - self._multi_last_seen_at
                    <= self.config.geometry_memory_seconds):
                return "multi"
        return "solo"

    def _boxes_are_crossing(self, target_index, people, mode):
        if mode != "multi":
            return False
        if len(people) < 2:
            return True
        target = people[target_index]
        return any(
            index != target_index
            and not self._separate_observations(target, person)
            for index, person in enumerate(people)
        )

    async def _embedding(self, frame_bgr, person):
        if _embedding_quality(person, self.config) is None:
            return None
        try:
            return await self.reid_service.extract_embedding_async(
                frame_bgr, person, padding=self.config.reid_crop_padding)
        except (RuntimeError, TypeError, ValueError):
            return None

    async def _all_embeddings(self, frame_bgr, people):
        return [await self._embedding(frame_bgr, person) for person in people]

    def _appearance_winner(self, track, embeddings, threshold):
        self._debug_reid_threshold = float(threshold)
        candidates = []
        for index, embedding in enumerate(embeddings):
            if embedding is None:
                continue
            match = track.appearance_match(embedding)
            if match is not None:
                candidates.append((match.similarity, index, match))
        if not candidates:
            return None
        candidates.sort(reverse=True, key=lambda item: item[0])
        best = candidates[0]
        self._debug_reid_best = float(best[0])
        self._debug_reid_source = best[2].source
        self._debug_reid_margin = (math.inf if len(candidates) == 1
                                   else float(best[0] - candidates[1][0]))
        if best[0] < threshold:
            return None
        margin = self._debug_reid_margin
        if margin < self.config.reid_winner_margin:
            return None
        return best[1], best[2], margin

    @staticmethod
    def _debug_number(value):
        if value is None:
            return "--"
        if math.isinf(value):
            return "inf"
        return f"{float(value):.3f}"

    def _write_debug_line(self, line):
        """Persist the full-rate trace while keeping its disk use bounded."""
        path = self._debug_log_path
        if path is None:
            return
        try:
            if (path.exists()
                    and path.stat().st_size >= _DEBUG_TRACKING_LOG_MAX_BYTES):
                path.replace(path.with_suffix(".log.1"))
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except OSError:
            # Tracking must never fail because debug storage is unavailable.
            self._debug_log_path = None

    def _print_debug_state(self, sequence, people, now):
        track = self._target()
        if track is None:
            line = (
                f"[PERSON_TRACK] seq={sequence} state=none mode={self.scene_mode} "
                f"people={len(people)} decision={self._debug_decision}"
            )
            console_key = ("none", self.scene_mode, self._debug_decision)
        else:
            person = track.person if track.visible else None
            confidence = _confidence(person) if person is not None else None
            keypoints = len((person or {}).get("keypoints") or {})
            clear = bool(person is not None and self._credible_person(person))
            target_id = "--" if track.track_id is None else str(track.track_id)
            missing = max(0.0, now - track.last_seen_at)
            line = (
                f"[PERSON_TRACK] seq={sequence} state={track.state} "
                f"mode={self.scene_mode} visible={int(track.visible)} "
                f"people={len(people)} id={target_id} candidate={track.candidate_id} "
                f"hits={track.consecutive_hits}/{self.config.confirmation_hits} "
                f"missing={missing:.2f}s conf={self._debug_number(confidence)} "
                f"kp={keypoints} clear={int(clear)} "
                f"geom_i={self._debug_geometry_index if self._debug_geometry_index is not None else '--'} "
                f"geom={self._debug_number(self._debug_geometry_score)} "
                f"reid={self._debug_number(self._debug_reid_best)} "
                f"reid_min={self._debug_number(self._debug_reid_threshold)} "
                f"margin={self._debug_number(self._debug_reid_margin)} "
                f"source={self._debug_reid_source or '--'} "
                f"reid_role={self._debug_reid_role} "
                f"gallery={len(track.gallery)} gallery_action={self._debug_gallery_decision} "
                f"recovery_hits={self._reacquisition_hits}/"
                f"{self.config.reacquisition_confirmation_frames} "
                f"decision={self._debug_decision} "
                f"candidates={self._debug_candidates}"
            )
            console_key = (
                track.state,
                self.scene_mode,
                track.visible,
                self._debug_decision,
                self._debug_gallery_decision,
            )
        self._write_debug_line(line)
        if console_key != self._last_console_debug_key:
            print(line, flush=True)
            self._last_console_debug_key = console_key

    async def update(self, sequence, frame_bgr, detections, *, captured_at=None):
        now = time.monotonic() if captured_at is None else float(captured_at)
        if not self._target_tracking_active:
            return self._publish_snapshot(sequence, now)
        people = [detection for detection in detections
                  if detection.get("class") == "person"
                  and _normalized_box(detection) is not None]
        self._debug_decision = "pending"
        self._debug_gallery_decision = "not_checked"
        self._debug_geometry_index = None
        self._debug_geometry_score = None
        self._debug_reid_best = None
        self._debug_reid_margin = None
        self._debug_reid_source = None
        self._debug_reid_threshold = None
        self._debug_reid_role = "--"
        self._debug_candidates = "--"
        track = self._target()
        if track is None:
            await self._start_target(sequence, frame_bgr, people, now)
        elif track.track_id is None:
            if people:
                await self._update_acquiring(
                    sequence, frame_bgr, people, now)
            else:
                self._debug_decision = "acquiring_no_detections"
                self._mark_missing(track)
        elif not people:
            self._debug_decision = "lost_no_detections"
            self._mark_missing(track)
        elif not track.visible:
            self._last_scene_mode = "recovery"
            await self._recover_target(sequence, frame_bgr, people, now)
        else:
            await self._update_confirmed(
                sequence, frame_bgr, people, now)
        self._expire_target(now)
        self._print_debug_state(sequence, people, now)
        return self._publish_snapshot(sequence, now)

    async def _start_target(self, sequence, frame_bgr, people, now):
        person = self._select_initial_person(people)
        if person is None:
            self._debug_decision = "acquiring_no_people"
            return
        track = _TargetTrack(
            candidate_id=self._next_candidate_id,
            bbox=_normalized_box(person).copy(),
            person=copy.deepcopy(person), created_at=now,
            last_seen_at=now, last_sequence=sequence,
        )
        self._next_candidate_id += 1
        self._tracks[track.candidate_id] = track
        self._last_scene_mode = "acquiring"
        self._multi_last_seen_at = None
        self._debug_decision = "acquiring_selected"
        if self.config.confirmation_hits <= 1:
            self._confirm_target(track)
            target_index = next(
                index for index, candidate in enumerate(people)
                if candidate is person)
            self._last_scene_mode = self._scene_mode_for(
                target_index, people, now)
            self._debug_decision = "acquisition_confirmed_geometry"
            await self._learn_target(
                track, sequence, frame_bgr, person,
                frozen=self._boxes_are_crossing(
                    target_index, people, self._last_scene_mode),
            )

    async def _update_acquiring(self, sequence, frame_bgr, people, now):
        track = self._target()
        self._last_scene_mode = "acquiring"
        index = self._geometry_candidate(
            track, people, now, strict=True)
        if index is None:
            self._debug_decision = "acquiring_geometry_miss"
            self._mark_missing(track)
            return
        person = people[index]
        self._accept(track, sequence, person, now, appearance_match=None)
        if track.consecutive_hits < self.config.confirmation_hits:
            self._debug_decision = "acquiring_geometry_wait"
            return
        self._confirm_target(track)
        self._last_scene_mode = self._scene_mode_for(index, people, now)
        self._debug_decision = "acquisition_confirmed_geometry"
        await self._learn_target(
            track, sequence, frame_bgr, person,
            frozen=self._boxes_are_crossing(
                index, people, self._last_scene_mode),
        )

    async def _update_confirmed(self, sequence, frame_bgr, people, now):
        track = self._target()
        geometry_index = self._geometry_candidate(track, people, now)
        if geometry_index is None:
            self._debug_decision = "lost_geometry_miss"
            self._mark_missing(track)
            return

        mode = self._scene_mode_for(geometry_index, people, now)
        if mode == "solo":
            self._last_scene_mode = "solo"
            self._debug_decision = "solo_geometry"
            person = people[geometry_index]
            self._accept(
                track, sequence, person, now, appearance_match=None)
            await self._learn_target(
                track, sequence, frame_bgr, person, frozen=False,
                debug_similarity=True)
            return

        self._last_scene_mode = "multi"
        has_appearance_memory = bool(
            track.gallery or track.appearance_prototype is not None)
        embeddings = (
            await self._all_embeddings(frame_bgr, people)
            if has_appearance_memory else [None] * len(people)
        )
        winner = (
            self._appearance_winner(
                track, embeddings,
                self.config.multi_person_minimum_similarity)
            if has_appearance_memory else None
        )
        self._debug_reid_role = "multi_tracking"
        selected_index = geometry_index
        appearance_match = None
        if winner is not None:
            selected_index, appearance_match, _margin = winner
            self._debug_decision = "multi_reid"
        elif not has_appearance_memory:
            self._debug_decision = "multi_geometry_no_gallery"
        elif self._debug_reid_best is None:
            self._debug_decision = "multi_geometry_no_embedding"
        elif self._debug_reid_best < self.config.multi_person_minimum_similarity:
            self._debug_decision = "multi_geometry_reid_below_threshold"
        else:
            self._debug_decision = "multi_geometry_reid_margin_rejected"

        crossing = self._boxes_are_crossing(
            selected_index, people, "multi")
        if crossing:
            self._debug_decision += "_crossing"
        person = people[selected_index]
        self._accept(
            track, sequence, person, now,
            appearance_match=appearance_match)
        await self._learn_target(
            track, sequence, frame_bgr, person,
            frozen=crossing,
            embedding=embeddings[selected_index],
        )

    async def _learn_target(
        self,
        track,
        sequence,
        frame_bgr,
        person,
        *,
        frozen,
        embedding=None,
        debug_similarity=False,
    ):
        if track.track_id is None:
            self._debug_gallery_decision = "skip_no_id"
            return
        if frozen:
            self._debug_gallery_decision = "freeze_crossing"
            return
        if not self._credible_person(person):
            self._debug_gallery_decision = "skip_person_gate"
            return
        due = (self._last_embedding_sequence is None
               or sequence - self._last_embedding_sequence
               >= self.config.embedding_interval_frames)
        if not due:
            self._debug_gallery_decision = "skip_interval"
            return
        if embedding is None:
            embedding = await self._embedding(frame_bgr, person)
        if embedding is None:
            self._debug_gallery_decision = "skip_no_embedding"
            return
        if debug_similarity:
            match = track.appearance_match(embedding)
            self._debug_reid_role = "solo_debug"
            if match is not None:
                self._debug_reid_best = match.similarity
                self._debug_reid_source = match.source
        self._last_embedding_sequence = sequence
        track.mark_gallery_support(embedding, sequence, self.config)
        activating_gallery = not self._has_appearance_memory(track)
        changed = track.add_embedding(
            embedding,
            quality=_confidence(person),
            image=_person_debug_thumbnail(
                frame_bgr, person, self.config.reid_crop_padding),
            sequence=sequence,
            config=self.config,
        )
        if activating_gallery and self._has_appearance_memory(track):
            self._clear_retired_gallery()
        if changed:
            track.gallery_last_updated_at = track.last_seen_at
            self._debug_gallery_decision = "updated"
            self._save_debug_image(track)
        else:
            self._debug_gallery_decision = "supported_existing"

    async def _recover_target(self, sequence, frame_bgr, people, now):
        track = self._target()
        missing_seconds = max(0.0, now - track.last_seen_at)
        has_appearance_memory = bool(
            track.gallery or track.appearance_prototype is not None)
        embeddings = (
            await self._all_embeddings(frame_bgr, people)
            if has_appearance_memory else [None] * len(people)
        )
        predicted = track.predict(now, self.config)
        candidate_debug = []
        for index, (person, embedding) in enumerate(zip(people, embeddings)):
            appearance = (
                track.appearance_match(embedding)
                if embedding is not None else None
            )
            candidate_debug.append(
                f"{index}:g={_geometry_cost(predicted, person):.3f},"
                f"r={self._debug_number(None if appearance is None else appearance.similarity)},"
                f"src={appearance.source if appearance is not None else '--'},"
                f"conf={_confidence(person):.2f},"
                f"kp={len(person.get('keypoints') or {})}"
            )
        self._debug_candidates = "[" + ";".join(candidate_debug) + "]"
        multi_person_recovery = any(
            self._credible_person(first)
            and self._credible_person(second)
            and self._separate_observations(first, second)
            for first_index, first in enumerate(people)
            for second in people[first_index + 1:]
        )
        recovery_threshold = (
            self.config.multi_person_reacquisition_minimum_similarity
            if multi_person_recovery
            else self.config.single_person_reacquisition_minimum_similarity
        )
        self._debug_reid_role = (
            "multi_recovery" if multi_person_recovery
            else "single_recovery"
        )
        winner = (
            self._appearance_winner(
                track, embeddings, recovery_threshold)
            if has_appearance_memory else None
        )
        match = None
        if winner is not None:
            index, match, _margin = winner
            self._debug_decision = "recovery_reid_candidate"
        elif (track.gallery or track.appearance_prototype is not None) and any(
                embedding is not None for embedding in embeddings):
            # Explicit appearance disagreement cannot be overridden by a
            # nearby box.
            self._reacquisition_hits = 0
            if (self._debug_reid_best is not None
                    and self._debug_reid_best < recovery_threshold):
                self._debug_decision = "recovery_reid_below_threshold"
            else:
                self._debug_decision = "recovery_reid_margin_rejected"
            return
        elif missing_seconds <= self.config.geometry_memory_seconds:
            index = self._geometry_candidate(track, people, now)
            if index is None:
                self._reacquisition_hits = 0
                self._debug_decision = "recovery_geometry_miss"
                return
            self._debug_decision = "recovery_geometry_candidate"
        else:
            self._reacquisition_hits = 0
            self._debug_decision = "recovery_wait_reid"
            return
        self._reacquisition_hits += 1
        if (self._reacquisition_hits
                < self.config.reacquisition_confirmation_frames):
            self._debug_decision += "_confirming"
            return
        self._accept(
            track, sequence, people[index], now,
            appearance_match=match)
        self._reacquisition_hits = 0
        self._last_scene_mode = self._scene_mode_for(index, people, now)
        self._debug_decision = (
            "recovered_reid" if match is not None else "recovered_geometry")

    def _accept(self, track, sequence, person, now, *, appearance_match):
        box = _normalized_box(person)
        predicted = track.predict(now, self.config)
        geometry_cost = _geometry_cost(predicted, person)
        elapsed = max(1e-3, now - track.last_seen_at)
        measured_velocity = (box - track.bbox) / elapsed
        track.velocity = np.ascontiguousarray(
            self.config.previous_velocity_weight * track.velocity
            + self.config.measured_velocity_weight * measured_velocity,
            dtype=np.float32)
        track.bbox = box.copy()
        track.person = copy.deepcopy(person)
        track.last_seen_at = now
        track.last_sequence = sequence
        track.visible = True
        track.misses = 0
        track.consecutive_hits += 1
        track.geometry_cost = geometry_cost
        track.pose_cost = None
        track.pose_cost_is_fallback = True
        if appearance_match is None:
            track.appearance_cost = None
            track.association_cost = geometry_cost
            track.association_confidence = max(0.0, 1.0 - geometry_cost)
        else:
            track.appearance_match_source = appearance_match.source
            track.reid_similarity = appearance_match.similarity
            track.appearance_cost = max(
                0.0, 1.0 - appearance_match.similarity)
            track.association_cost = track.appearance_cost
            track.association_confidence = max(
                0.0, appearance_match.similarity)
        if track.track_id is not None:
            track.state = "confirmed"

    def _mark_missing(self, track):
        self._reacquisition_hits = 0
        if not track.visible:
            return
        track.visible = False
        track.misses += 1
        track.consecutive_hits = 0
        if track.track_id is not None:
            track.state = "lost"

    def _confirm_target(self, track):
        if track.track_id is None:
            track.track_id = self._next_track_id
            self._next_track_id += 1
        track.state = "confirmed"
        self._save_debug_image(track)

    def _reset_target(self, *, keep_preference=False):
        track = self._target()
        if track is not None:
            self._retire_gallery(track)
        self._tracks.clear()
        self._reacquisition_hits = 0
        self._last_embedding_sequence = None
        self._last_scene_mode = "acquiring"
        self._multi_last_seen_at = None
        if not keep_preference:
            self._preferred_detection = None

    def _expire_target(self, now):
        track = self._target()
        if track is None:
            return
        if self._target_persistent and track.track_id is not None:
            return
        timeout = (self.config.lost_timeout_seconds
                   if track.track_id is not None
                   else self.config.tentative_timeout_seconds)
        if now - track.last_seen_at > timeout:
            self._debug_decision = "target_expired"
            self._reset_target()

    def _publish_snapshot(self, sequence, now):
        track = self._target()
        snapshots = ()
        if track is not None and track.track_id is not None:
            snapshots = (PersonTrackSnapshot(
                track_id=track.track_id, state=track.state,
                visible=track.visible,
                person=copy.deepcopy(track.person) if track.visible else None,
                bbox=tuple(float(value) for value in track.bbox),
                association_confidence=track.association_confidence,
                appearance_samples=len(track.gallery),
                age_seconds=max(0.0, now - track.created_at),
                missing_seconds=max(0.0, now - track.last_seen_at),
                appearance_match_source=track.appearance_match_source,
                reid_similarity=track.reid_similarity,
                association_cost=track.association_cost,
                geometry_cost=track.geometry_cost,
                appearance_cost=track.appearance_cost,
                pose_cost=track.pose_cost,
                pose_cost_is_fallback=track.pose_cost_is_fallback,
                gallery_update_age_seconds=(
                    None
                    if track.gallery_last_updated_at is None
                    else max(0.0, now - track.gallery_last_updated_at)
                ),
            ),)
        with self._snapshot_lock:
            self._snapshot_sequence = int(sequence)
            self._snapshot_captured_at = float(now)
            self._snapshot = snapshots
        return snapshots
