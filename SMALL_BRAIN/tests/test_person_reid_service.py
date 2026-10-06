import unittest

import numpy as np

from services.vision.person_reid_service import PersonReIDService


class PersonReIDServiceTests(unittest.TestCase):
    def test_extracts_normalized_embedding_from_normalized_bbox(self):
        received = []

        def infer(tensor):
            received.append(tensor)
            return np.array([[3.0, 4.0]], dtype=np.float32)

        service = PersonReIDService(inference_callable=infer)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[20:80, 50:150] = (10, 20, 30)
        bbox = {
            "bbox": {
                "normalized_x1": 0.25,
                "normalized_y1": 0.20,
                "normalized_x2": 0.75,
                "normalized_y2": 0.80,
            }
        }

        embedding = service.extract_embedding(frame, bbox)

        np.testing.assert_allclose(embedding, [0.6, 0.8])
        self.assertEqual(received[0].shape, (1, 3, 256, 128))
        self.assertEqual(received[0].dtype, np.float32)
        np.testing.assert_allclose(received[0][0, :, 0, 0], [10, 20, 30])

    def test_rejects_bbox_outside_frame(self):
        service = PersonReIDService(
            inference_callable=lambda tensor: np.ones((1, 2))
        )
        frame = np.zeros((20, 20, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            service.extract_embedding(frame, (30, 30, 40, 40))

    def test_rejects_zero_embedding(self):
        service = PersonReIDService(
            inference_callable=lambda tensor: np.zeros((1, 3))
        )

        with self.assertRaises(ValueError):
            service.extract_embedding(np.zeros((20, 20, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
