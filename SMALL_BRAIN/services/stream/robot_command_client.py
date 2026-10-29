"""Reliable request/reply client for BIG_BRAIN robot commands."""

import asyncio

import zmq


class RobotCommandClient:
    """Serialize robot commands and replace poisoned ZeroMQ REQ sockets."""

    def __init__(self, context, endpoint="tcp://127.0.0.1:5555", timeout=5.0):
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

    async def command(self, payload):
        """Send one command and leave the client ready for the next command."""
        async with self._lock:
            if self._socket is None:
                self._reset_socket()
            try:
                await self._socket.send_json(payload)
                return await asyncio.wait_for(
                    self._socket.recv_json(), timeout=self.timeout
                )
            except (asyncio.TimeoutError, asyncio.CancelledError, zmq.ZMQError):
                # REQ cannot send again when its previous reply is missing.
                # Drop that exchange and reconnect with a fresh socket.
                self._reset_socket()
                raise

    def close(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
