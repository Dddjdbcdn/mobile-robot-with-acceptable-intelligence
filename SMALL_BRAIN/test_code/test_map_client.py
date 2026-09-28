import tempfile
import unittest
from pathlib import Path

from services.map_client import MapClient


class MapClientSaveTests(unittest.TestCase):
    def test_new_snapshot_overrides_previous_action_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            client = MapClient(context=None, save_dir=directory)

            client.save_snapshot({
                "jpeg_bytes": b"first-crop",
                "full_jpeg_bytes": b"first-full",
            })
            paths = client.save_snapshot({
                "jpeg_bytes": b"newest-crop",
                "full_jpeg_bytes": b"newest-full",
            })

            save_dir = Path(directory)
            self.assertEqual(
                (save_dir / "latest_map_crop.jpg").read_bytes(),
                b"newest-crop",
            )
            self.assertEqual(
                (save_dir / "latest_map_full.jpg").read_bytes(),
                b"newest-full",
            )
            self.assertEqual(
                paths,
                {
                    "map_crop": str(save_dir / "latest_map_crop.jpg"),
                    "map_full": str(save_dir / "latest_map_full.jpg"),
                },
            )

    def test_crop_is_used_when_service_has_no_separate_full_render(self):
        with tempfile.TemporaryDirectory() as directory:
            client = MapClient(context=None, save_dir=directory)

            client.save_snapshot({"jpeg_bytes": b"same-render"})

            self.assertEqual(
                (Path(directory) / "latest_map_full.jpg").read_bytes(),
                b"same-render",
            )


if __name__ == "__main__":
    unittest.main()
