import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services import diagnostic_session


class DiagnosticSessionTests(unittest.TestCase):

    def test_streams_share_session_stamp(self):
        central = diagnostic_session.session_log_path("smallbrain")

        self.assertEqual(central.parent.name, "smallbrain_sessions")
        self.assertTrue(central.name.endswith("-smallbrain.jsonl"))

    def test_output_directory_can_be_configured(self):
        with patch.dict(
            "os.environ",
            {"DJ_SMALLBRAIN_LOG_DIR": "/tmp/dj-smallbrain-diagnostics"},
        ):
            path = diagnostic_session.session_log_path("hand-gesture")

        self.assertEqual(path.parent, Path("/tmp/dj-smallbrain-diagnostics"))

    def test_best_effort_logger_is_rate_limited_jsonl(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"DJ_SMALLBRAIN_LOG_DIR": directory}
        ):
            logger = diagnostic_session.BestEffortSessionLogger(
                "track-action", interval_seconds=1.0)
            self.assertTrue(logger.write({"event": "first"}, now=10.0))
            self.assertFalse(logger.write({"event": "skipped"}, now=10.5))
            self.assertTrue(logger.write({"event": "second"}, now=11.0))
            events = [
                json.loads(line)
                for line in logger.path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([event["event"] for event in events], [
            "first", "second"
        ])

    def test_central_snapshot_combines_components_and_window_counts(self):
        diagnostics = diagnostic_session.SmallBrainDiagnostics()
        diagnostics.update(
            "hand_gesture", {"gesture": "welcome"},
            tally_group="gestures", tally_value="welcome")
        diagnostics.update(
            "track_action", {"seed": {"valid": True}},
            tally_group="pipelines", tally_value="seed_valid")
        diagnostics.record_event(
            "cognition", "hand_command_rejected",
            reason="HAND_LOCATION_UNAVAILABLE")

        snapshot = diagnostics.snapshot()

        self.assertEqual(
            snapshot["components"]["hand_gesture"]["gesture"], "welcome")
        self.assertTrue(
            snapshot["components"]["track_action"]["seed"]["valid"])
        self.assertEqual(
            snapshot["components"]["hand_gesture"]["window"]
            ["gestures"]["welcome"],
            1,
        )
        self.assertEqual(
            snapshot["components"]["cognition"]["window"]["events"][0]
            ["reason"],
            "HAND_LOCATION_UNAVAILABLE",
        )


if __name__ == "__main__":
    unittest.main()
