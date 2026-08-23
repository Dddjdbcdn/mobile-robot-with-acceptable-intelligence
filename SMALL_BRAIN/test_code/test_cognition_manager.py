import asyncio
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


SMALL_BRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(SMALL_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(SMALL_BRAIN_ROOT))

from cognition.cognition_manager import CognitionManager


def result(action, message="done"):
    return SimpleNamespace(
        action_type=action,
        event="stop",
        timestamp=1.0,
        status="success",
        msg=message,
    )


class FakeSee:
    camera = None

    def __init__(self):
        self.calls = []

    async def start_seeing(self, query):
        self.calls.append(query)
        return result("see_action")


class FakeSearch:
    def __init__(self):
        self.completion = None
        self.frame_assessments = []
        self.batch_assessments = []

    async def start_visual_search(self, target):
        self.completion = asyncio.get_running_loop().create_future()
        return await self.completion

    async def assess_frame_search(self, args, metadata):
        self.frame_assessments.append((args, metadata))
        self.completion.set_result(result("search_action", {"result": "found"}))

    async def assess_batch_search(self, args, metadata):
        self.batch_assessments.append((args, metadata))

    async def stop_visual_search(self):
        return result("search_action", "cancelled")


class FakeTrack:
    tracking = False

    async def stop_tracking(self, message=None):
        return result("track_action", message)


class FakeApproach:
    async def start_approaching(self):
        return result("approach_action")


class CognitionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.outputs = []
        self.voices = []
        self.contexts = []
        self.search = FakeSearch()

        async def send_output(call_id, output):
            self.outputs.append((call_id, output))

        async def request_voice(**kwargs):
            self.voices.append(kwargs)

        async def send_command(payload):
            return {"status": "accepted", "payload": payload}

        async def inform_llm(message):
            self.contexts.append(message)

        self.manager = CognitionManager(
            approach_action=FakeApproach(),
            search_action=self.search,
            see_action=FakeSee(),
            track_action=FakeTrack(),
            send_tool_output=send_output,
            request_voice=request_voice,
            send_robot_command=send_command,
            inform_llm=inform_llm,
            idle_wakeup_cooldown=0.0,
            active_context_interval=0.0,
        )
        self.loop_task = asyncio.create_task(self.manager.cognition_loop())

    async def asyncTearDown(self):
        await self.manager.shutdown()
        await self.loop_task

    async def wait_until(self, predicate, attempts=50):
        for _ in range(attempts):
            if predicate():
                return
            await asyncio.sleep(0)
        self.fail("condition was not reached")

    async def test_short_action_is_accepted_and_reported(self):
        await self.manager.handle_tool_call(
            function_name="vision_reasoning",
            arguments='{"query": "What is visible?"}',
            call_id="call-see",
        )
        await self.wait_until(lambda: bool(self.voices))

        self.assertEqual(self.outputs[0][1]["status"], "accepted")
        self.assertEqual(self.manager.see_action.calls, ["What is visible?"])
        self.assertEqual(len(self.manager.action_results), 1)

    async def test_assessment_is_routed_while_search_is_running(self):
        await self.manager.handle_tool_call(
            function_name="get_vision",
            arguments='{"action":"find_object","target":"bottle"}',
            call_id="call-search",
        )
        await self.wait_until(lambda: self.search.completion is not None)

        await self.manager.handle_tool_call(
            function_name="assess_frame_search",
            arguments='{"result":"candidate","confidence":0.9}',
            call_id="call-assess",
            response_metadata={"request_id": "one"},
        )
        await self.wait_until(lambda: bool(self.voices))

        self.assertEqual(len(self.search.frame_assessments), 1)
        self.assertIn(("call-assess", {"status": "received"}), self.outputs)

    async def test_normal_actions_are_queued(self):
        await self.manager.handle_tool_call(
            function_name="get_vision",
            arguments='{"action":"find_object","target":"bottle"}',
            call_id="first",
        )
        await self.wait_until(lambda: self.search.completion is not None)

        await self.manager.handle_tool_call(
            function_name="vision_reasoning",
            arguments='{"query":"Describe the room"}',
            call_id="second",
        )

        self.assertEqual(self.outputs[-1][1]["status"], "queued")
        self.search.completion.set_result(result("search_action"))
        await self.wait_until(lambda: len(self.voices) == 2)
        self.assertEqual(self.manager.see_action.calls, ["Describe the room"])


    async def test_salient_idle_state_wakes_the_llm(self):
        await self.manager.publish_world_state({"camera_tof_range": 0.2})
        await self.wait_until(lambda: bool(self.voices))

        self.assertTrue(self.contexts)
        self.assertIn("very close obstacle", self.contexts[-1])
        self.assertIn("SALIENT WORLD EVENT", self.voices[-1]["system_msg"])

    async def test_active_action_gets_context_without_proactive_voice(self):
        await self.manager.handle_tool_call(
            function_name="get_vision",
            arguments='{"action":"find_object","target":"bottle"}',
            call_id="active-search",
        )
        await self.wait_until(lambda: self.search.completion is not None)

        await self.manager.publish_world_state({"camera_tof_range": 0.2})
        await self.wait_until(lambda: bool(self.contexts))

        self.assertEqual(self.voices, [])
        self.assertEqual(self.manager.current_request.call_id, "active-search")


if __name__ == "__main__":
    unittest.main()
