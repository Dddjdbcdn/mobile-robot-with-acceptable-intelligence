import queue
from types import SimpleNamespace

from robot.control.reset_node import LiftDetector, ResetNode


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class Client:
    def __init__(self, ready=True):
        self.ready = ready
        self.requests = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        return object()


def test_lift_detection_does_not_depend_on_cliff_points():
    detector = LiftDetector()
    detector.state = "normal"
    detector.gravity_direction = (0.0, 0.0, 1.0)
    detector.last_time = 0.0

    first = detector.observe(
        0.1,
        (0.0, 0.0, 9.80665),
        (1.0, 0.0, 0.0),
        placement_clear=False,
    )
    second = detector.observe(
        0.2,
        (0.0, 0.0, 9.80665),
        (1.0, 0.0, 0.0),
        placement_clear=False,
    )

    assert first is None
    assert second == "lifted"


def test_placement_requires_zero_points_when_cliff_layer_is_fresh():
    node = ResetNode.__new__(ResetNode)
    node._last_cliff_at = 10.0

    node._cliff_point_count = 0
    assert node._placement_clear(10.1) is True

    node._cliff_point_count = 1
    assert node._placement_clear(10.1) is False


def test_placement_can_proceed_when_cliff_layer_is_stale():
    node = ResetNode.__new__(ResetNode)
    node._last_cliff_at = 10.0
    node._cliff_point_count = 1

    assert node._placement_clear(10.6) is True


def test_only_valid_nonempty_maps_advance_revision():
    node = ResetNode.__new__(ResetNode)
    node._map_revision = 4

    node._map_callback(SimpleNamespace(
        info=SimpleNamespace(width=0, height=0), data=[]
    ))
    node._map_callback(SimpleNamespace(
        info=SimpleNamespace(width=2, height=2), data=[-1]
    ))
    assert node._map_revision == 4

    node._map_callback(SimpleNamespace(
        info=SimpleNamespace(width=2, height=2), data=[-1] * 4
    ))
    assert node._map_revision == 5


def test_reset_stays_active_until_a_fresh_map_arrives():
    node = ResetNode.__new__(ResetNode)
    node.detector = SimpleNamespace(state="normal")
    node._reset_mode = "manual"
    node._reset_futures = (object(),)
    node._map_revision = 7
    node._map_wait_revision = None
    node._retry_after = 0.0
    node._manual_reset = None
    node._accept_manual_reset = lambda: None
    node._publish_stop = lambda: None
    node._reset_succeeded = lambda: True
    node._publish_state = lambda: None
    node._publish_reset_state = lambda: None
    node.get_logger = lambda: SimpleNamespace(info=lambda _message: None)

    node._control_tick()

    assert node._reset_mode == "manual"
    assert node._map_wait_revision == 7

    node._map_revision = 8
    node._control_tick()

    assert node._reset_mode is None
    assert node._map_wait_revision is None


def test_stop_movement_publishes_stop_state_reset_and_reason():
    node = ResetNode.__new__(ResetNode)
    node.reset_publisher = Publisher()
    node.reset_reason_publisher = Publisher()
    stops = []
    node._publish_stop = lambda: stops.append(True)

    node.stop_movement("MANUAL_RESET")

    assert stops == [True]
    assert len(node.reset_publisher.messages) == 1
    assert node.reset_reason_publisher.messages[0].data == "MANUAL_RESET"


def test_reset_map_uses_slam_and_both_costmaps():
    node = ResetNode.__new__(ResetNode)
    node.slam = Client()
    node.local_costmap = Client()
    node.global_costmap = Client()
    node._reset_futures = None

    assert node.reset_map() is True

    assert len(node._reset_futures) == 3
    assert len(node.slam.requests) == 1
    assert len(node.local_costmap.requests) == 1
    assert len(node.global_costmap.requests) == 1
    assert node.slam.requests[0].pause_new_measurements is False


def test_manual_request_uses_shared_stop_and_does_not_change_lift_state():
    node = ResetNode.__new__(ResetNode)
    node._manual_requests = queue.Queue()
    node._reset_mode = None
    node._manual_reset = None
    node._retry_after = 0.0
    node.detector = SimpleNamespace(state="normal")
    node.get_logger = lambda: SimpleNamespace(info=lambda _message: None)
    reasons = []
    node.stop_movement = reasons.append
    completion = queue.Queue(maxsize=1)
    node._manual_requests.put(("request-1", completion))

    node._accept_manual_reset()

    assert node._reset_mode == "manual"
    assert node.detector.state == "normal"
    assert reasons == ["MANUAL_RESET"]
    assert node._manual_reset[0] == "request-1"
    assert completion.empty()


def test_manual_request_is_rejected_during_lift_reset():
    node = ResetNode.__new__(ResetNode)
    node._manual_requests = queue.Queue()
    node._reset_mode = "lift"
    completion = queue.Queue(maxsize=1)
    node._manual_requests.put(("request-2", completion))

    node._accept_manual_reset()

    assert completion.get_nowait() == (
        False,
        "A spatial reset is already running",
    )


def test_reset_status_reports_lift_manual_reset_and_ready():
    node = ResetNode.__new__(ResetNode)
    node.reset_state_publisher = Publisher()
    node._last_reset_state = None
    node.get_logger = lambda: SimpleNamespace(info=lambda _message: None)
    node.detector = SimpleNamespace(state="lifted")
    node._reset_mode = None

    node._publish_reset_state()
    node.detector.state = "resetting"
    node._reset_mode = "lift"
    node._publish_reset_state()
    node.detector.state = "normal"
    node._reset_mode = "manual"
    node._publish_reset_state()
    node._reset_mode = None
    node._publish_reset_state()

    assert [item.data for item in node.reset_state_publisher.messages] == [
        "lifted",
        "resetting_lift",
        "resetting_manual",
        "ready",
    ]
