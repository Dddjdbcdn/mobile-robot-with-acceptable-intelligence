import unittest

from cognition.executive import CognitiveExecutive
from cognition.models import ActionResult, Goal, GoalKind, GoalStatus, ObjectObservation


class ScriptedActions:
    def __init__(self, verification_results: list[bool]) -> None:
        self.verification_results = iter(verification_results)
        self.searches = 0
        self.stops = 0

    async def search_object(self, target: str) -> ActionResult:
        self.searches += 1
        return ActionResult(
            True,
            observation=ObjectObservation(target, 0.9, source="test-search"),
        )

    async def track_object(self, target: str) -> ActionResult:
        return ActionResult(True)

    async def approach_object(
        self, target: str, desired_distance_m: float
    ) -> ActionResult:
        return ActionResult(True)

    async def verify_object_distance(
        self, target: str, desired_distance_m: float
    ) -> ActionResult:
        success = next(self.verification_results)
        return ActionResult(success, "distance mismatch" if not success else "verified")

    async def stop_motion(self) -> None:
        self.stops += 1


class CognitiveExecutiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_closed_loop(self) -> None:
        actions = ScriptedActions([True])
        executive = CognitiveExecutive(actions)
        goal = Goal(GoalKind.FIND_AND_APPROACH, "bottle")

        result = await executive.execute(goal)

        self.assertEqual(result, GoalStatus.SUCCEEDED)
        self.assertIsNotNone(executive.world.get_object("bottle"))

    async def test_retries_after_failed_verification(self) -> None:
        actions = ScriptedActions([False, True])
        executive = CognitiveExecutive(actions)
        goal = Goal(GoalKind.FIND_AND_APPROACH, "bottle", max_attempts=1)

        result = await executive.execute(goal)

        self.assertEqual(result, GoalStatus.SUCCEEDED)
        self.assertEqual(goal.attempts, 1)
        self.assertEqual(actions.stops, 1)

    async def test_non_retryable_failure_stops_immediately(self) -> None:
        actions = ScriptedActions([True])

        async def fail_search(target: str) -> ActionResult:
            return ActionResult(False, "camera unavailable", retryable=False)

        actions.search_object = fail_search
        executive = CognitiveExecutive(actions)
        goal = Goal(GoalKind.FIND_AND_APPROACH, "bottle")

        result = await executive.execute(goal)

        self.assertEqual(result, GoalStatus.FAILED)
        self.assertEqual(goal.attempts, 0)
        self.assertEqual(goal.error, "camera unavailable")


if __name__ == "__main__":
    unittest.main()
