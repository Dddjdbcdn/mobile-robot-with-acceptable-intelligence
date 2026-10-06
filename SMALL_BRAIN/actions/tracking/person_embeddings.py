"""View-labelled person embedding storage and cosine similarity policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np


PERSON_VIEWS = ("front", "side", "back")


def normalize_embedding(embedding: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a finite, one-dimensional, L2-normalized float32 vector."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("embedding must contain finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("embedding norm must be greater than zero")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def cosine_similarity(
    first: np.ndarray | Sequence[float],
    second: np.ndarray | Sequence[float],
) -> float:
    """Compare two embeddings after normalization."""
    first_vector = normalize_embedding(first)
    second_vector = normalize_embedding(second)
    if first_vector.shape != second_vector.shape:
        raise ValueError("embedding dimensions do not match")
    return float(np.clip(np.dot(first_vector, second_vector), -1.0, 1.0))


@dataclass(frozen=True)
class EmbeddingSample:
    embedding: np.ndarray
    quality: float
    captured_at: float


@dataclass(frozen=True)
class EmbeddingMatch:
    matched: bool
    similarity: float
    threshold: float
    best_view: str | None
    per_view_similarity: dict[str, float]


class PersonEmbeddingRecord:
    """Keep the best appearance samples for each enrollment view."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        person_id: str,
        model_name: str,
        samples_per_view: int = 1,
        embedding_size: int | None = None,
    ):
        person_id = str(person_id).strip()
        model_name = str(model_name).strip()
        if not person_id:
            raise ValueError("person_id cannot be empty")
        if not model_name:
            raise ValueError("model_name cannot be empty")
        if int(samples_per_view) < 1:
            raise ValueError("samples_per_view must be at least one")
        if embedding_size is not None and int(embedding_size) < 1:
            raise ValueError("embedding_size must be positive")
        self.person_id = person_id
        self.model_name = model_name
        self.samples_per_view = int(samples_per_view)
        self.embedding_size = (
            None if embedding_size is None else int(embedding_size)
        )
        self._samples: dict[str, list[EmbeddingSample]] = {
            view: [] for view in PERSON_VIEWS
        }

    @property
    def complete(self) -> bool:
        return all(
            len(self._samples[view]) >= self.samples_per_view
            for view in PERSON_VIEWS
        )

    @property
    def missing_views(self) -> tuple[str, ...]:
        return tuple(
            view
            for view in PERSON_VIEWS
            if len(self._samples[view]) < self.samples_per_view
        )

    def samples(self, view: str) -> tuple[EmbeddingSample, ...]:
        self._validate_view(view)
        return tuple(self._samples[view])

    @staticmethod
    def _validate_view(view: str) -> None:
        if view not in PERSON_VIEWS:
            raise ValueError(
                f"view must be one of {', '.join(PERSON_VIEWS)}"
            )

    def _validated_embedding(
        self, embedding: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        vector = normalize_embedding(embedding)
        if self.embedding_size is None:
            self.embedding_size = int(vector.size)
        elif vector.size != self.embedding_size:
            raise ValueError(
                f"expected embedding size {self.embedding_size}, "
                f"received {vector.size}"
            )
        return vector

    def add(
        self,
        view: str,
        embedding: np.ndarray | Sequence[float],
        quality: float,
        captured_at: float | None = None,
    ) -> bool:
        """Add a sample or replace the lowest-quality sample for that view.

        Returns ``True`` when the record changed. This lets future automatic
        enrollment keep submitting candidates without implementing replacement
        policy in the camera loop.
        """
        self._validate_view(view)
        quality = float(quality)
        if not math.isfinite(quality) or quality < 0.0:
            raise ValueError("quality must be finite and non-negative")
        captured_at = time.time() if captured_at is None else float(captured_at)
        if not math.isfinite(captured_at):
            raise ValueError("captured_at must be finite")
        vector = self._validated_embedding(embedding)
        sample = EmbeddingSample(vector, quality, captured_at)
        view_samples = self._samples[view]
        if len(view_samples) < self.samples_per_view:
            view_samples.append(sample)
        else:
            worst_index = min(
                range(len(view_samples)),
                key=lambda index: view_samples[index].quality,
            )
            if quality <= view_samples[worst_index].quality:
                return False
            view_samples[worst_index] = sample
        view_samples.sort(key=lambda item: item.quality, reverse=True)
        return True

    def compare(
        self,
        embedding: np.ndarray | Sequence[float],
        threshold: float,
    ) -> EmbeddingMatch:
        """Match a candidate against the strongest sample from every view."""
        threshold = float(threshold)
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between -1 and 1")
        candidate = normalize_embedding(embedding)
        if (
            self.embedding_size is not None
            and candidate.size != self.embedding_size
        ):
            raise ValueError(
                f"expected embedding size {self.embedding_size}, "
                f"received {candidate.size}"
            )
        per_view: dict[str, float] = {}
        for view, samples in self._samples.items():
            if samples:
                per_view[view] = max(
                    float(np.clip(np.dot(candidate, sample.embedding), -1.0, 1.0))
                    for sample in samples
                )
        if not per_view:
            return EmbeddingMatch(False, -1.0, threshold, None, {})
        best_view, similarity = max(
            per_view.items(), key=lambda item: item[1]
        )
        return EmbeddingMatch(
            similarity >= threshold,
            similarity,
            threshold,
            best_view,
            per_view,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "person_id": self.person_id,
            "model_name": self.model_name,
            "embedding_size": self.embedding_size,
            "samples_per_view": self.samples_per_view,
            "views": {
                view: [
                    {
                        "embedding": sample.embedding.tolist(),
                        "quality": sample.quality,
                        "captured_at": sample.captured_at,
                    }
                    for sample in samples
                ]
                for view, samples in self._samples.items()
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PersonEmbeddingRecord":
        if int(value.get("format_version", -1)) != cls.FORMAT_VERSION:
            raise ValueError("unsupported person embedding record format")
        record = cls(
            person_id=value["person_id"],
            model_name=value["model_name"],
            samples_per_view=int(value["samples_per_view"]),
            embedding_size=value.get("embedding_size"),
        )
        views = value.get("views")
        if not isinstance(views, dict):
            raise ValueError("views must be a mapping")
        unknown_views = set(views) - set(PERSON_VIEWS)
        if unknown_views:
            raise ValueError(f"unknown person views: {sorted(unknown_views)}")
        for view in PERSON_VIEWS:
            samples = views.get(view, [])
            if not isinstance(samples, list):
                raise ValueError(f"samples for {view} must be a list")
            if len(samples) > record.samples_per_view:
                raise ValueError(f"too many stored samples for {view}")
            for sample in samples:
                record.add(
                    view,
                    sample["embedding"],
                    quality=sample["quality"],
                    captured_at=sample["captured_at"],
                )
        return record

    def save(self, path: str | Path) -> Path:
        """Atomically write the record as portable JSON."""
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PersonEmbeddingRecord":
        source = Path(path).expanduser()
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("person embedding record must contain an object")
        return cls.from_dict(value)
