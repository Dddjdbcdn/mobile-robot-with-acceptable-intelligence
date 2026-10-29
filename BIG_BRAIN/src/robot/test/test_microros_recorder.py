from robot.diagnostics.microros_recorder import (
    kernel_line_is_relevant,
    matching_process_ids,
    stream_state,
)


def test_stream_state_tracks_start_failure_and_recovery():
    assert stream_state(0, None, 2.0, True) == "starting"
    assert stream_state(0, None, 2.0, False) == "no_publisher"
    assert stream_state(1, None, 2.0, False) == "no_messages"
    assert stream_state(1, 2.1, 2.0, False) == "stale"
    assert stream_state(1, 0.2, 2.0, False) == "live"


def test_kernel_filter_keeps_transport_events_only():
    assert kernel_line_is_relevant("usb 3-2: USB disconnect, device number 3")
    assert kernel_line_is_relevant("cdc_acm 3-2:1.2: ttyACM0: USB ACM device")
    assert kernel_line_is_relevant("usb 3-2: reset full-speed USB device")
    assert not kernel_line_is_relevant("wlan0: associated with access point")


def test_process_scan_can_find_the_test_runner():
    assert matching_process_ids((b"pytest",))
