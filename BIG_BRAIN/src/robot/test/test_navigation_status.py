import threading
import time
from types import SimpleNamespace

from robot.bridge.navigation_controller import (
    ActiveNavigation,
    NavigationController,
)


def make_controller():
    controller = NavigationController.__new__(NavigationController)
    controller._navigation_status_lock = threading.Lock()
    controller._navigation_status = {
        "state": "idle",
        "mode": None,
        "distance_remaining": None,
        "navigation_seconds": 0.0,
        "recoveries": 0,
        "error_code": None,
        "error_message": "",
        "updated_at": time.monotonic(),
    }
    controller.active = None
    return controller


def test_navigation_feedback_updates_display_snapshot():
    controller = make_controller()
    navigation = ActiveNavigation("navigate", "action-1")
    controller.active = navigation
    feedback = SimpleNamespace(
        feedback=SimpleNamespace(
            distance_remaining=1.64,
            navigation_time=SimpleNamespace(sec=3, nanosec=500_000_000),
            number_of_recoveries=1,
        )
    )

    controller._navigation_feedback(feedback, navigation)
    status = controller.status_snapshot()

    assert status["state"] == "navigating"
    assert status["mode"] == "navigate"
    assert status["distance_remaining"] == 1.64
    assert status["navigation_seconds"] == 3.5
    assert status["recoveries"] == 1


def test_stale_feedback_cannot_replace_current_navigation_status():
    controller = make_controller()
    current = ActiveNavigation("navigate", "current")
    stale = ActiveNavigation("navigate", "stale")
    controller.active = current
    feedback = SimpleNamespace(
        feedback=SimpleNamespace(
            distance_remaining=9.0,
            navigation_time=SimpleNamespace(sec=1, nanosec=0),
            number_of_recoveries=0,
        )
    )

    controller._navigation_feedback(feedback, stale)

    assert controller.status_snapshot()["state"] == "idle"
