import asyncio
import sys
from pathlib import Path
import unittest

import zmq


SMALL_BRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(SMALL_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(SMALL_BRAIN_ROOT))

from services.stream.robot_command_client import RobotCommandClient


class FakeSocket:
    def __init__(self, response=None, send_error=None):
        self.response = response
        self.send_error = send_error
        self.closed = False
        self.sent = []

    def setsockopt(self, option, value):
        pass

    def connect(self, endpoint):
        self.endpoint = endpoint

    def close(self, linger=0):
        self.closed = True

    async def send_json(self, payload):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(payload)

    async def recv_json(self):
        if isinstance(self.response, asyncio.Event):
            await self.response.wait()
        return self.response


class FakeContext:
    def __init__(self, *sockets):
        self.available = list(sockets)
        self.created = []

    def socket(self, socket_type):
        self.assert_req(socket_type)
        socket = self.available.pop(0)
        self.created.append(socket)
        return socket

    @staticmethod
    def assert_req(socket_type):
        if socket_type != zmq.REQ:
            raise AssertionError(f"Expected REQ socket, got {socket_type}")


class RobotCommandClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_replaces_socket_and_next_command_succeeds(self):
        blocked = FakeSocket(response=asyncio.Event())
        recovered = FakeSocket(response={"status": "accepted"})
        context = FakeContext(blocked, recovered)
        client = RobotCommandClient(context, timeout=0.01)

        with self.assertRaises(TimeoutError):
            await client.command({"command": "first"})

        self.assertTrue(blocked.closed)
        result = await client.command({"command": "second"})
        self.assertEqual(result, {"status": "accepted"})
        self.assertEqual(recovered.sent, [{"command": "second"}])

    async def test_cancellation_replaces_socket(self):
        blocked = FakeSocket(response=asyncio.Event())
        recovered = FakeSocket(response={"status": "accepted"})
        context = FakeContext(blocked, recovered)
        client = RobotCommandClient(context)

        command = asyncio.create_task(client.command({"command": "first"}))
        await asyncio.sleep(0)
        command.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await command

        self.assertTrue(blocked.closed)
        result = await client.command({"command": "second"})
        self.assertEqual(result, {"status": "accepted"})

    async def test_zmq_error_replaces_socket(self):
        failed = FakeSocket(send_error=zmq.ZMQError(zmq.EFSM))
        recovered = FakeSocket(response={"status": "accepted"})
        context = FakeContext(failed, recovered)
        client = RobotCommandClient(context)

        with self.assertRaises(zmq.ZMQError):
            await client.command({"command": "first"})

        self.assertTrue(failed.closed)
        result = await client.command({"command": "second"})
        self.assertEqual(result, {"status": "accepted"})


if __name__ == "__main__":
    unittest.main()
