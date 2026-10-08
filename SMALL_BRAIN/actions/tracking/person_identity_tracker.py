"""Stateful geometric person association with reID-only recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2

from actions.tracking.person_association import (
    select_tracked_person,
    update_tracked_bbox,
)
from actions.tracking.person_embeddings import (
    PERSON_VIEWS,
    PersonEmbeddingRecord,
)


@dataclass(frozen=True, slots=True)
class PersonTrackDecision:
    person: dict | None
    status: str
    identity: str
    allow_lidar_seed: bool
    reason: str
    reid_similarity: float | None = None
    candidate_margin: float | None = None


@dataclass(frozen=True, slots=True)
class EnrollmentCandidate:
    frame_bgr: object
    person: dict
    quality: float
    captured_at: float


class PersonIdentityTracker:
    """Preserve geometric tracking and use appearance only after it fails."""

    MIN_PERSON_CONFIDENCE = 0.50
    MIN_NORMALIZED_WIDTH = 0.06
    MIN_NORMALIZED_HEIGHT = 0.18
    MIN_VISIBLE_KEYPOINTS = 5
    MIN_BLUR_VARIANCE = 35.0
    EDGE_MARGIN = 0.005

    def __init__(
        self,
        reid_service,
        *,
        profile_path: str | Path,
        artifact_directory: str | Path,
        person_id: str = "follow_target",
        samples_per_view: int = 3,
        reid_threshold: float = 0.70,
        reid_margin: float = 0.05,
        recovery_confirmations: int = 2,
        recovery_interval_seconds: float = 0.25,
    ):
        self.reid_service = reid_service
        self.profile_path = Path(profile_path)
        self.artifact_directory = Path(artifact_directory)
        self.reid_threshold = float(reid_threshold)
        self.reid_margin = float(reid_margin)
        self.recovery_confirmations = max(1, int(recovery_confirmations))
        self.recovery_interval_seconds = max(0.0,
                                             float(recovery_interval_seconds))

        if self.profile_path.is_file():
            self.embedding_record = PersonEmbeddingRecord.load(
                self.profile_path)
            if self.embedding_record.model_name != reid_service.model_name:
                raise ValueError(
                    "Person reID profile model does not match the active model"
                )
        else:
            self.embedding_record = PersonEmbeddingRecord(
                person_id,
                reid_service.model_name,
                samples_per_view=samples_per_view,
            )

        self._bbox = None
        self._current_person = None
        self._status = "idle"
        self._identity = "unenrolled"
        self._last_reid_at = None
        self._recovery_bbox = None
        self._recovery_hits = 0
        self._next_view_index = self._initial_view_index()

    @property
    def status(self) -> str:
        return self._status

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def current_person(self) -> dict | None:
        return self._current_person

    @property
    def has_embeddings(self) -> bool:
        return any(
            self.embedding_record.samples(view) for view in PERSON_VIEWS)

    def _initial_view_index(self) -> int:
        missing = self.embedding_record.missing_views
        if missing:
            return PERSON_VIEWS.index(missing[0])
        return 0

    def start(self, initial_person: dict) -> None:
        self._bbox = update_tracked_bbox(None, initial_person)
        self._current_person = initial_person
        self._status = "tracked"
        self._identity = "unverified" if self.has_embeddings else "unenrolled"
        self._clear_recovery()

    def reset(self) -> None:
        self._bbox = None
        self._current_person = None
        self._status = "idle"
        self._identity = "unenrolled"
        self._last_reid_at = None
        self._clear_recovery()

    def geometric_candidate(self, detections) -> dict | None:
        return select_tracked_person(detections, self._bbox)

    def _clear_recovery(self) -> None:
        self._recovery_bbox = None
        self._recovery_hits = 0

    @staticmethod
    def _bbox_values(person: dict):
        try:
            bbox = person["bbox"]
            return tuple(
                float(bbox[name]) for name in (
                    "normalized_x1",
                    "normalized_y1",
                    "normalized_x2",
                    "normalized_y2",
                ))
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def candidate_quality(
        cls,
        frame_bgr,
        person: dict,
        *,
        enrollment: bool,
    ) -> float | None:
        values = cls._bbox_values(person)
        if values is None:
            return None
        x1, y1, x2, y2 = values
        width = x2 - x1
        height = y2 - y1
        confidence = float(person.get("confidence") or 0.0)
        visible_keypoints = len(person.get("keypoints") or {})
        if (confidence < cls.MIN_PERSON_CONFIDENCE
                or width < cls.MIN_NORMALIZED_WIDTH
                or height < cls.MIN_NORMALIZED_HEIGHT
                or visible_keypoints < cls.MIN_VISIBLE_KEYPOINTS):
            return None
        if enrollment and (x1 <= cls.EDGE_MARGIN or y1 <= cls.EDGE_MARGIN
                           or x2 >= 1.0 - cls.EDGE_MARGIN
                           or y2 >= 1.0 - cls.EDGE_MARGIN):
            return None

        try:
            crop = cls._crop(frame_bgr, person)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except (TypeError, ValueError, cv2.error):
            return None
        if enrollment and blur_variance < cls.MIN_BLUR_VARIANCE:
            return None

        sharpness = min(1.0, blur_variance / 250.0)
        keypoint_quality = min(1.0, visible_keypoints / 13.0)
        size_quality = min(1.0, height / 0.65)
        return (0.40 * confidence + 0.25 * sharpness +
                0.20 * keypoint_quality + 0.15 * size_quality)

    @staticmethod
    def _crop(frame_bgr, person: dict):
        from services.vision.person_reid_service import PersonReIDService

        return PersonReIDService.crop_person(frame_bgr, person)

    async def observe(
        self,
        *,
        sequence: int,
        frame_bgr,
        detections: list[dict],
    ) -> PersonTrackDecision:
        del sequence  # Reserved for future stale-result protection.
        person = self.geometric_candidate(detections)
        if person is not None:
            self._bbox = update_tracked_bbox(self._bbox, person)
            self._current_person = person
            self._status = "tracked"
            self._clear_recovery()
            return PersonTrackDecision(
                person,
                "tracked",
                self._identity,
                True,
                "geometric_continuity",
            )

        self._current_person = None
        self._status = "recovering"
        if frame_bgr is None or not self.has_embeddings:
            return PersonTrackDecision(
                None,
                "recovering",
                self._identity,
                False,
                "geometric_target_missing",
            )

        now = time.monotonic()
        if (self._last_reid_at is not None
                and now - self._last_reid_at < self.recovery_interval_seconds):
            return PersonTrackDecision(
                None,
                "recovering",
                self._identity,
                False,
                "waiting_for_reid_interval",
            )
        self._last_reid_at = now

        matches = []
        for candidate in detections:
            quality = self.candidate_quality(frame_bgr,
                                             candidate,
                                             enrollment=False)
            if quality is None:
                continue
            try:
                embedding = await self.reid_service.extract_embedding_async(
                    frame_bgr,
                    candidate,
                )
                match = self.embedding_record.compare(
                    embedding,
                    self.reid_threshold,
                )
            except (RuntimeError, TypeError, ValueError):
                continue
            matches.append((match.similarity, candidate))

        if not matches:
            self._identity = "suspect"
            self._clear_recovery()
            return PersonTrackDecision(None, "recovering", "suspect", False,
                                       "no_reid_candidate")

        matches.sort(key=lambda item: item[0], reverse=True)
        best_similarity, best_person = matches[0]
        second_similarity = matches[1][0] if len(matches) > 1 else -1.0
        margin = best_similarity - second_similarity
        if (best_similarity < self.reid_threshold
                or margin < self.reid_margin):
            self._identity = "suspect"
            self._clear_recovery()
            return PersonTrackDecision(
                None,
                "recovering",
                "suspect",
                False,
                "reid_below_threshold_or_margin",
                best_similarity,
                margin,
            )

        same_recovery = select_tracked_person([best_person],
                                              self._recovery_bbox)
        if self._recovery_bbox is None or same_recovery is None:
            self._recovery_bbox = update_tracked_bbox(None, best_person)
            self._recovery_hits = 1
        else:
            self._recovery_bbox = update_tracked_bbox(self._recovery_bbox,
                                                      best_person)
            self._recovery_hits += 1

        if self._recovery_hits < self.recovery_confirmations:
            return PersonTrackDecision(
                None,
                "recovering",
                "candidate",
                False,
                "reid_confirmation_pending",
                best_similarity,
                margin,
            )

        self._bbox = update_tracked_bbox(None, best_person)
        self._current_person = best_person
        self._status = "tracked"
        self._identity = "verified"
        self._clear_recovery()
        return PersonTrackDecision(
            best_person,
            "tracked",
            "verified",
            True,
            "reid_recovered",
            best_similarity,
            margin,
        )

    def next_enrollment_view(self) -> str:
        return PERSON_VIEWS[self._next_view_index]

    async def enroll(
        self,
        candidates: list[EnrollmentCandidate],
    ) -> dict:
        view = self.next_enrollment_view()
        accepted = 0
        saved_images = []
        self.artifact_directory.mkdir(parents=True, exist_ok=True)

        for index, candidate in enumerate(
                sorted(candidates, key=lambda item: item.quality,
                       reverse=True)):
            try:
                embedding = await self.reid_service.extract_embedding_async(
                    candidate.frame_bgr,
                    candidate.person,
                )
                stamp = time.time_ns()
                destination = self.artifact_directory / (
                    f"{view}_{stamp}_{index}.jpg")
                crop = self._crop(candidate.frame_bgr, candidate.person)
                if not cv2.imwrite(str(destination), crop):
                    continue
                changed = self.embedding_record.add(
                    view,
                    embedding,
                    candidate.quality,
                    captured_at=candidate.captured_at,
                    image_path=str(destination),
                )
                if not changed:
                    destination.unlink(missing_ok=True)
                    continue
            except (OSError, RuntimeError, TypeError, ValueError, cv2.error):
                continue
            accepted += 1
            saved_images.append(str(destination))

        if accepted:
            self.embedding_record.save(self.profile_path)
            self._identity = "verified"
            if len(self.embedding_record.samples(view)) >= (
                    self.embedding_record.samples_per_view):
                self._next_view_index = (self._next_view_index +
                                         1) % len(PERSON_VIEWS)

        return {
            "view": view,
            "accepted": accepted,
            "saved_images": saved_images,
            "profile_path": str(self.profile_path),
            "complete": self.embedding_record.complete,
            "missing_views": list(self.embedding_record.missing_views),
        }
