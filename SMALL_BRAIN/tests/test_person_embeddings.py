import tempfile
import unittest
from pathlib import Path

import numpy as np

from actions.tracking.person_embeddings import (
    PersonEmbeddingRecord,
    cosine_similarity,
)


class PersonEmbeddingRecordTests(unittest.TestCase):

    def test_cosine_similarity_normalizes_inputs(self):
        self.assertAlmostEqual(cosine_similarity([2, 0], [5, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_retains_highest_quality_samples_per_view(self):
        record = PersonEmbeddingRecord("owner",
                                       "test-model",
                                       samples_per_view=2)

        self.assertTrue(record.add("front", [1, 0], quality=0.4))
        self.assertTrue(record.add("front", [0.9, 0.1], quality=0.8))
        self.assertFalse(record.add("front", [0.8, 0.2], quality=0.2))
        self.assertTrue(record.add("front", [0.7, 0.3], quality=0.9))

        qualities = [sample.quality for sample in record.samples("front")]
        self.assertEqual(qualities, [0.9, 0.8])

    def test_matches_against_all_recorded_views(self):
        record = PersonEmbeddingRecord("owner",
                                       "test-model",
                                       samples_per_view=1)
        record.add("front", [1, 0, 0], quality=1.0)
        record.add("back", [0, 1, 0], quality=1.0)

        match = record.compare([0.05, 0.95, 0], threshold=0.9)

        self.assertTrue(match.matched)
        self.assertEqual(match.best_view, "back")
        self.assertGreater(match.similarity, 0.99)

    def test_reports_incomplete_views(self):
        record = PersonEmbeddingRecord("owner", "test-model")
        record.add("front", [1, 0], quality=1.0)

        self.assertFalse(record.complete)
        self.assertEqual(record.missing_views, ("side", "back"))

    def test_comparing_empty_record_does_not_set_embedding_size(self):
        record = PersonEmbeddingRecord("owner", "test-model")

        match = record.compare([1, 0], threshold=0.5)

        self.assertFalse(match.matched)
        self.assertIsNone(record.embedding_size)

    def test_json_round_trip_preserves_matching(self):
        record = PersonEmbeddingRecord("owner",
                                       "test-model",
                                       samples_per_view=1)
        record.add(
            "front",
            [1, 2, 3],
            quality=0.75,
            captured_at=123.0,
            image_path="images/front.jpg",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owner.json"
            record.save(path)
            restored = PersonEmbeddingRecord.load(path)

        match = restored.compare([1, 2, 3], threshold=0.99)
        self.assertTrue(match.matched)
        self.assertEqual(restored.person_id, "owner")
        self.assertEqual(restored.embedding_size, 3)
        self.assertEqual(
            restored.samples("front")[0].image_path,
            "images/front.jpg",
        )


if __name__ == "__main__":
    unittest.main()
