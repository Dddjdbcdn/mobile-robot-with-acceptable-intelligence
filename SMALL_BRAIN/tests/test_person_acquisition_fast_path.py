import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import AsyncMock


SMALL_BRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(SMALL_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(SMALL_BRAIN_ROOT))

from actions.action_result import ActionResult
from actions.track_action import TrackingService
from cognition.manager.world_state import robot_state
from cognition.sequence.find_target_executor import FindTargetExecutor
from services.vision.person_tracking_service import PersonTrackSnapshot


def person_detection():
    return {
        "class": "person",
        "confidence": 0.9,
        "bbox": {
            "normalized_x1": 0.3,
            "normalized_y1": 0.1,
            "normalized_x2": 0.7,
            "normalized_y2": 0.9,
            "normalized_center_x": 0.5,
            "normalized_center_y": 0.5,
        },
        "keypoints": {
            "nose": {
                "normalized_x": 0.5,
                "normalized_y": 0.25,
            },
        },
    }


class PersistentPersonTracks:
    def __init__(self, tracks):
        self.tracks = tuple(tracks)

    def tracking_snapshot(self):
        return 7, self.tracks

    def track_for_detection(self, detection):
        return next(
            (
                track for track in self.tracks
                if track.person is detection and track.visible
            ),
            None,
        )


def person_track(track_id, detection):
    bbox = detection["bbox"]
    return PersonTrackSnapshot(
        track_id=track_id,
        state="confirmed",
        visible=True,
        person=detection,
        bbox=(
            bbox["normalized_x1"],
            bbox["normalized_y1"],
            bbox["normalized_x2"],
            bbox["normalized_y2"],
        ),
        association_confidence=0.95,
        appearance_samples=3,
        age_seconds=1.0,
        missing_seconds=0.0,
    )


class PersonDetectionWaitTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_person_binds_to_its_confirmed_persistent_id(self):
        selected = person_detection()
        other = person_detection()
        other["confidence"] = 0.99
        other["bbox"] = dict(other["bbox"], normalized_center_x=0.8)
        service = TrackingService.__new__(TrackingService)
        service.person_tracking_service = PersistentPersonTracks((
            person_track(3, other),
            person_track(9, selected),
        ))
        service.PERSON_ID_ACQUISITION_TIMEOUT_SECONDS = 0.1

        track = await service._wait_for_initial_person_track(selected)

        self.assertEqual(track.track_id, 9)
        self.assertIs(track.person, selected)

    async def test_without_handoff_selects_highest_confidence_confirmed_id(
            self):
        lower = person_detection()
        lower["confidence"] = 0.75
        higher = person_detection()
        higher["confidence"] = 0.96
        service = TrackingService.__new__(TrackingService)
        service.person_tracking_service = PersistentPersonTracks((
            person_track(4, lower),
            person_track(6, higher),
        ))
        service.PERSON_ID_ACQUISITION_TIMEOUT_SECONDS = 0.1

        track = await service._wait_for_initial_person_track()

        self.assertEqual(track.track_id, 6)

class PersonSearchHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous_pose = robot_state.get("pose")
        robot_state["pose"] = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "frame_id": "map",
        }

    async def asyncTearDown(self):
        robot_state["pose"] = self.previous_pose
        await asyncio.sleep(0)

    async def test_person_goal_tracks_current_view_before_searching(self):
        track_action = SimpleNamespace(
            start_tracking=AsyncMock(return_value=ActionResult(
                "person-1:direct_person",
                "track_action",
                "running",
                target="person",
                outcome="tracking",
            )),
        )
        inactive_action = SimpleNamespace(active=False)
        executor = FindTargetExecutor(
            search_action=inactive_action,
            track_action=track_action,
            navigation_action=inactive_action,
            see_action=inactive_action,
            map_navigation_action=inactive_action,
            send_map_overlay=AsyncMock(),
        )
        executor._search_environment = AsyncMock(return_value=ActionResult(
            "person-1:search",
            "track_action",
            "succeeded",
            target="person",
            outcome="target_tracking_started",
        ))

        started = await executor.start(
            target="person",
            action_id="person-1",
            target_kind="person",
            approach=False,
        )
        result = await executor.wait_until_finished()

        self.assertEqual(started.status, "running")
        self.assertEqual(result.status, "succeeded")
        track_action.start_tracking.assert_awaited_once_with(
            target="person",
            action_id="person-1:direct_person",
            allow_grounding_dino=False,
        )
        executor._search_environment.assert_not_awaited()

    async def test_person_fast_track_miss_falls_back_to_centered_search(self):
        track_action = SimpleNamespace(
            start_tracking=AsyncMock(return_value=ActionResult(
                "person-miss:direct_person",
                "track_action",
                "failed",
                target="person",
                outcome="target_not_detected",
                reason_code="PERSON_DETECTION_FAILED",
            )),
        )
        inactive_action = SimpleNamespace(active=False)
        executor = FindTargetExecutor(
            search_action=inactive_action,
            track_action=track_action,
            navigation_action=inactive_action,
            see_action=inactive_action,
            map_navigation_action=inactive_action,
            send_map_overlay=AsyncMock(),
        )
        executor._search_environment = AsyncMock(return_value=ActionResult(
            "person-miss:search",
            "track_action",
            "succeeded",
            target="person",
            outcome="target_tracking_started",
        ))

        await executor.start(
            target="person",
            action_id="person-miss",
            target_kind="person",
            approach=False,
        )
        result = await executor.wait_until_finished()

        self.assertEqual(result.status, "succeeded")
        executor._search_environment.assert_awaited_once_with()

    async def test_confirmed_search_person_seeds_tracking_immediately(self):
        detection = person_detection()
        track_action = SimpleNamespace(
            start_tracking=AsyncMock(return_value=ActionResult(
                "person-2:acquire_person",
                "track_action",
                "running",
                target="person",
                outcome="tracking",
            )),
        )
        inactive_action = SimpleNamespace(active=False)
        executor = FindTargetExecutor(
            search_action=inactive_action,
            track_action=track_action,
            navigation_action=inactive_action,
            see_action=inactive_action,
            map_navigation_action=inactive_action,
            send_map_overlay=AsyncMock(),
        )
        executor.action_id = "person-2"
        executor.target = "person"
        executor.target_kind = "person"

        result = await executor._acquire_person_without_approach(
            0, initial_person=detection
        )

        self.assertEqual(result.status, "succeeded")
        track_action.start_tracking.assert_awaited_once_with(
            target="person",
            action_id="person-2:acquire_person",
            allow_grounding_dino=True,
            initial_person=detection,
        )

    async def test_fast_person_tracking_forwards_welcome_person(self):
        detection = person_detection()
        track_action = SimpleNamespace(
            start_tracking=AsyncMock(return_value=ActionResult(
                "welcome:direct_person",
                "track_action",
                "running",
                target="person",
                outcome="tracking",
            )),
        )
        inactive_action = SimpleNamespace(active=False)
        executor = FindTargetExecutor(
            search_action=inactive_action,
            track_action=track_action,
            navigation_action=inactive_action,
            see_action=inactive_action,
            map_navigation_action=inactive_action,
            send_map_overlay=AsyncMock(),
        )
        executor.action_id = "welcome"
        executor.target = "person"
        executor.target_kind = "person"
        executor.initial_person = detection
        executor.approach_target = False

        result = await executor._try_fast_person_tracking()

        self.assertEqual(result.status, "succeeded")
        track_action.start_tracking.assert_awaited_once_with(
            target="person",
            action_id="welcome:direct_person",
            allow_grounding_dino=False,
            initial_person=detection,
        )

    async def test_person_search_centers_camera_before_scanning(self):
        detection = person_detection()
        centered = ActionResult(
            "person-3:initial_center_camera",
            "move_camera",
            "succeeded",
            outcome="camera_positioned",
        )
        camera_motion = SimpleNamespace(
            move_to_region=AsyncMock(return_value=centered),
        )
        inactive_action = SimpleNamespace(active=False)
        search_action = SimpleNamespace(
            active=False,
            last_observation_frame=None,
            last_coverage_frames=[],
        )
        executor = FindTargetExecutor(
            search_action=search_action,
            track_action=inactive_action,
            navigation_action=inactive_action,
            see_action=inactive_action,
            map_navigation_action=inactive_action,
            send_map_overlay=AsyncMock(),
            move_camera_action=camera_motion,
        )
        executor.action_id = "person-3"
        executor.target = "person"
        executor.target_kind = "person"
        executor.approach_target = False
        executor._search_started_at = time.monotonic()
        executor._search = AsyncMock(return_value=ActionResult(
            "person-3:initial_center_scan",
            "search_action",
            "succeeded",
            target="person",
            outcome="found",
            data={"person_detection": detection},
        ))
        executor._acquire_person_without_approach = AsyncMock(
            return_value=ActionResult(
                "person-3:target_acquired",
                "track_action",
                "succeeded",
                target="person",
                outcome="target_tracking_started",
            )
        )

        result = await executor._search_environment()

        self.assertEqual(result.status, "succeeded")
        camera_motion.move_to_region.assert_awaited_once_with(
            region="center",
            action_id="person-3:initial_center_camera",
        )
        executor._search.assert_awaited_once_with(
            "initial_center_scan",
            initial_view_only=False,
            sweep_directions=("left", "right"),
            candidate_context=None,
        )


if __name__ == "__main__":
    unittest.main()
