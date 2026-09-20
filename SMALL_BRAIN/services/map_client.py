"""Reliable request/reply client for BIG_BRAIN map snapshots."""

import asyncio
import json
import time

import zmq


class MapClient:
    def __init__(
        self,
        context,
        endpoint="tcp://127.0.0.1:5559",
        timeout=8.0,
    ):
        self.context = context
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self._lock = asyncio.Lock()
        self._socket = None

    def _reset_socket(self):
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self.context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint)

    async def _exchange(self, payload, timeout):
        if self._socket is None:
            self._reset_socket()
        try:
            await self._socket.send_json(payload)
            return await asyncio.wait_for(
                self._socket.recv_multipart(), timeout=timeout
            )
        except (asyncio.TimeoutError, asyncio.CancelledError, zmq.ZMQError):
            # A REQ socket cannot send again after a missing reply. Recreate it
            # so the next request starts from a valid send state.
            self._reset_socket()
            raise

    async def command(self, payload):
        """Apply an overlay set/clear command and wait for its acknowledgement."""
        async with self._lock:
            parts = await self._exchange(payload, self.timeout)
            if len(parts) != 1:
                raise RuntimeError("Invalid map command acknowledgement")
            response = json.loads(parts[0].decode("utf-8"))
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or "Map command failed")
            return response

    async def request_snapshot(self, payload):
        """Return the snapshot produced for this exact request."""
        deadline = time.monotonic() + self.timeout
        last_error = None
        async with self._lock:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(last_error or "Timed out waiting for the map service")
                try:
                    parts = await self._exchange(payload, remaining)
                except asyncio.TimeoutError as error:
                    if last_error is not None:
                        raise RuntimeError(last_error) from error
                    raise TimeoutError(
                        f"Timed out waiting {self.timeout:.1f}s for the map service"
                    ) from error
                if len(parts) == 2:
                    metadata = json.loads(parts[0].decode("utf-8"))
                    jpeg = bytes(parts[1])
                    if metadata.get("schema_version") != 1:
                        raise RuntimeError("Unsupported map snapshot schema")
                    if not isinstance(metadata.get("candidates"), list):
                        raise RuntimeError("Map snapshot has no candidate list")
                    if not jpeg.startswith(bytes.fromhex("ffd8")):
                        raise RuntimeError("Map service returned an invalid JPEG")
                    return {
                        "received_at": time.monotonic(),
                        "age_seconds": 0.0,
                        "jpeg_bytes": jpeg,
                        "metadata": metadata,
                    }
                if len(parts) != 1:
                    raise RuntimeError("Invalid map service response")
                response = json.loads(parts[0].decode("utf-8"))
                if response.get("ok"):
                    raise RuntimeError("Map service returned no snapshot")
                last_error = response.get("error") or "Map is not ready"
                if response.get("retryable") is False:
                    raise RuntimeError(last_error)
                if time.monotonic() + 0.1 >= deadline:
                    raise RuntimeError(last_error)
                await asyncio.sleep(0.1)

    def close(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
