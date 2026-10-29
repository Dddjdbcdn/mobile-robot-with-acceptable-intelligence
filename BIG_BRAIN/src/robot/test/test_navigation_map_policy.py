from types import SimpleNamespace

from robot.bridge.navigation_controller import NavigationController


POSE = {"x": 0.0, "y": 0.0, "yaw": 0.0, "frame_id": "map"}
DESTINATION = {"x": 0.5, "y": 0.0, "angle": 0.0, "frame_id": "map"}


class MapStream:
    def __init__(self):
        self.window_prepare_count = 0
        self.global_calls = 0
        self.logic = SimpleNamespace(
            resolve_local_destination=lambda *args, **kwargs: dict(DESTINATION),
            resolve_approach_destination=lambda *args, **kwargs: dict(DESTINATION),
        )
        self.room_registry = SimpleNamespace(
            destination=lambda room_id: dict(DESTINATION)
        )

        self.window_calls = []

    def prepare_global_costmap_window(self, size_m, center=None):
        self.window_prepare_count += 1
        self.window_calls.append((float(size_m), center))
        return {}, dict(POSE)

    def prepare_map(self):
        self.global_calls += 1
        return {}, dict(POSE)


def controller_with(stream):
    controller = NavigationController.__new__(NavigationController)
    controller.map_stream = stream
    controller.previous_pose = None
    controller.sequence = None
    controller.get_person_state = lambda: None
    controller.get_robot_pose = lambda: dict(POSE)
    controller.stop = lambda: None
    controller._send_goal = lambda destination, action_id, mode: None
    return controller


def test_nudge_uses_three_meter_robot_centered_global_window():
    stream = MapStream()
    controller = controller_with(stream)
    result = controller.navigate_local({
        "local_command": "nudge_forward", "action_id": "nearby"
    })
    assert result == {"status": "accepted"}
    assert (stream.window_prepare_count, stream.global_calls) == (1, 0)
    assert stream.window_calls == [(3.0, None)]


def test_room_command_uses_global_map():
    stream = MapStream()
    controller = controller_with(stream)
    result = controller.navigate_local({
        "local_command": "go_to_room", "room_id": 2, "action_id": "room"
    })
    assert result == {"status": "accepted"}
    assert (stream.window_prepare_count, stream.global_calls) == (0, 1)


def test_open_space_uses_bounded_window_but_exit_room_uses_full_map():
    local_stream = MapStream()
    local_controller = controller_with(local_stream)
    local_controller._advance_sequence = lambda prepared, pose: None
    assert local_controller.navigate_sequence({
        "sequence_command": "open_space_middle", "action_id": "open"
    }) == {"status": "accepted"}
    assert (local_stream.window_prepare_count, local_stream.global_calls) == (1, 0)
    assert local_stream.window_calls == [(6.0, None)]

    global_stream = MapStream()
    global_controller = controller_with(global_stream)
    global_controller._advance_sequence = lambda prepared, pose: None
    assert global_controller.navigate_sequence({
        "sequence_command": "exit_room", "action_id": "exit"
    }) == {"status": "accepted"}
    assert (global_stream.window_prepare_count, global_stream.global_calls) == (0, 1)


def test_approach_uses_six_meter_robot_centered_global_window():
    stream = MapStream()
    controller = controller_with(stream)
    result = controller.navigate_to_pose({
        "x": 1.0, "y": 0.0, "angle": 0.0,
        "frame_id": "base_footprint", "action_id": "approach",
    }, approach=True)
    assert result == {"status": "accepted"}
    assert (stream.window_prepare_count, stream.global_calls) == (1, 0)
    assert stream.window_calls == [(6.0, None)]


def test_person_command_uses_three_meter_person_centered_global_window():
    stream = MapStream()
    controller = controller_with(stream)
    person = {
        "x": 4.0, "y": 1.0, "frame_id": "map", "age_seconds": 0.1,
    }
    controller.get_person_state = lambda: dict(person)
    result = controller.navigate_local({
        "local_command": "move_to_person_front", "action_id": "person",
    })
    assert result == {"status": "accepted"}
    assert stream.window_calls == [(3.0, person)]
