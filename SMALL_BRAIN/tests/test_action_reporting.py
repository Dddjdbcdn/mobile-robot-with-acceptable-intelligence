import unittest

from actions.action_result import ActionResult
from cognition.manager.action_reporting import decision_instruction


class ActionReportingTests(unittest.TestCase):
    def test_explicit_navigation_retry_requires_a_fresh_tool_result(self):
        result = ActionResult(
            action_id="navigation-1",
            action_type="explicit_navigation",
            status="failed",
            target="move_to_person_front",
            outcome="rejected",
            reason_code="NAVIGATION_REJECTED",
            retryable=True,
        )

        instruction = decision_instruction(result)

        self.assertIn("explicitly asks to try again", instruction)
        self.assertIn("immediately call the same navigation action", instruction)
        self.assertIn("do not claim or imply", instruction)
        self.assertIn("environment was rechecked", instruction)


if __name__ == "__main__":
    unittest.main()
