#!/usr/bin/env python3
"""Display and measure the newest map image received from BIG BRAIN."""

from __future__ import annotations

import argparse
from collections import deque
import json
import struct
import time

import cv2
import numpy as np
import zmq


TOPIC = b"map/image"


def decode_payload(payload: bytes) -> tuple[dict, bytes]:
    if len(payload) < 4:
        raise ValueError("payload is shorter than its metadata header")
    metadata_size = struct.unpack("!I", payload[:4])[0]
    metadata_end = 4 + metadata_size
    if metadata_end > len(payload):
        raise ValueError("metadata length exceeds payload size")
    metadata = json.loads(payload[4:metadata_end].decode("utf-8"))
    jpeg = payload[metadata_end:]
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported map schema")
    if not jpeg.startswith(b"\xff\xd8"):
        raise ValueError("payload does not contain a JPEG")
    return metadata, jpeg


def newest_message(socket: zmq.Socket, first: list[bytes]) -> tuple[list[bytes], int]:
    """Drain queued frames so rendering never walks through stale maps."""
    newest = first
    skipped = 0
    while True:
        try:
            newest = socket.recv_multipart(flags=zmq.NOBLOCK)
            skipped += 1
        except zmq.Again:
            return newest, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View and measure BIG BRAIN's latest JPEG map stream."
    )
    parser.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5559",
        help="ZeroMQ endpoint; use localhost with an SSH -L tunnel",
    )
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
    socket.connect(args.endpoint)

    arrivals: deque[float] = deque(maxlen=30)
    previous_sequence = None
    total_queue_skips = 0
    total_sequence_gaps = 0
    last_report = time.monotonic()
    last_notice = last_report

    print(f"Waiting for maps on {args.endpoint}")
    try:
        while True:
            try:
                parts, skipped = newest_message(socket, socket.recv_multipart())
            except zmq.Again:
                if time.monotonic() - last_notice >= 3.0:
                    print("Still waiting for a fresh map...", flush=True)
                    last_notice = time.monotonic()
                continue

            total_queue_skips += skipped
            if len(parts) != 2 or parts[0] != TOPIC:
                print("Ignoring an invalid map message", flush=True)
                continue

            try:
                metadata, jpeg = decode_payload(parts[1])
                image = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is None:
                    raise ValueError("OpenCV could not decode the map JPEG")
            except (ValueError, OSError, json.JSONDecodeError, cv2.error) as error:
                print(f"Ignoring invalid map: {error}", flush=True)
                continue

            now = time.monotonic()
            arrivals.append(now)
            sequence = metadata.get("sequence")
            if isinstance(sequence, int) and isinstance(previous_sequence, int):
                total_sequence_gaps += max(0, sequence - previous_sequence - 1)
            previous_sequence = sequence

            if now - last_report >= 1.0:
                rx_hz = 0.0
                if len(arrivals) >= 2:
                    rx_hz = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
                captured_ns = metadata.get("captured_at_unix_ns")
                age = "unknown"
                if isinstance(captured_ns, int):
                    # End-to-end age is accurate only when both clocks agree.
                    age_ms = max(0.0, (time.time_ns() - captured_ns) / 1e6)
                    age = f"{age_ms:.0f} ms"
                print(
                    f"seq={sequence if sequence is not None else 'n/a'} "
                    f"rx={rx_hz:.2f} Hz age={age} "
                    f"overlay={metadata.get('render_ms', 'n/a')} ms "
                    f"analysis={metadata.get('analysis_ms', 'n/a')} ms "
                    f"cache_age={metadata.get('analysis_age_ms', 'n/a')} ms "
                    f"queue_skips={total_queue_skips} "
                    f"sequence_gaps={total_sequence_gaps}",
                    flush=True,
                )
                last_report = now
            last_notice = now

            if not args.no_window:
                cv2.imshow("Robot Map", image)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
