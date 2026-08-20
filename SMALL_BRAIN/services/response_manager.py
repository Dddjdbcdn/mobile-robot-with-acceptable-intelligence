import asyncio
import json
import uuid
from pathlib import Path

from utilities.database_functions import load_json

REPO_ROOT = Path(__file__).resolve().parent.parent
VISION_TOOLS_PATH = str(REPO_ROOT / "tools" / "vision_tools.json")

tools = load_json(VISION_TOOLS_PATH)
vision_tools = {tool["name"]: tool for tool in tools}
INBAND_VISUAL_SEARCH = vision_tools.get("inband_visual_search")
RESPONSE_TIMEOUT = 10.0

class ResponseManager:
    def __init__(self, ws, app):
        self.ws = ws
        self.app = app

        self.active_response_id = None

        self._voice_transition_lock = asyncio.Lock()

        self._voice_idle = asyncio.Event()
        self._voice_idle.set()

        self._voice_created = asyncio.Event()
        self._pending_create_id = None

    async def create_voice_response(self, system_msg=None) -> None:
        async with self._voice_transition_lock:
            await self._cancel_active_response()
            await self._create_response_and_wait(system_msg)

    async def create_tool_response(self,tool=None):
        request_id = uuid.uuid4().hex

        if tool == "INBAND_VISUAL_SEARCH":
            response_content = {
                "event_id": f"create_tool_{request_id}",
                "type": "response.create",
                "response": {
                    "tools": [INBAND_VISUAL_SEARCH],
                    "tool_choice": "required",
                    "metadata": {
                        "kind": "tool",
                        "request_id": request_id,
                    },
                    "output_modalities": ["text"],
                },
            }

        await self.ws.send(json.dumps(response_content))

    async def _cancel_active_response(self) -> None:
        active_response_id = self.active_response_id

        if active_response_id is None: return

        self._voice_idle.clear()

        await self.ws.send(
            json.dumps({
                "event_id": f"cancel_voice_{uuid.uuid4().hex}",
                "type": "response.cancel",
                "response_id": active_response_id,
            })
        )

        try:
            await asyncio.wait_for(
                self._voice_idle.wait(),
                timeout=RESPONSE_TIMEOUT,
            )
        except TimeoutError as error:
            raise RuntimeError(
                f"Timed out cancelling response {active_response_id}"
            ) from error

    async def _create_response_and_wait(self,system_msg) -> None:
        request_id = uuid.uuid4().hex

        self._pending_create_id = request_id
        self._voice_created.clear()

        if system_msg is not None:
            system_msg_content = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_msg 
                        }
                    ]
                }
            }

            await self.ws.send(json.dumps(system_msg_content))

        response_content = {
            "event_id": f"create_voice_{request_id}",
            "type": "response.create",
            "response": {
                "metadata": {
                    "kind": "voice",
                    "request_id": request_id,
                }
            },
        }

        await self.ws.send(json.dumps(response_content))
        try:
            await asyncio.wait_for(
                self._voice_created.wait(),
                timeout=RESPONSE_TIMEOUT,
            )

        except TimeoutError as error:
            if self._pending_create_id == request_id:
                self._pending_create_id = None

            raise RuntimeError(
                f"Timed out waiting for response.created: {request_id}"
            ) from error

    def handle_response_created(self, response: dict) -> None:
        metadata = response.get("metadata") or {}

        if metadata.get("kind") != "voice": return

        response_id = response.get("id")
        request_id = metadata.get("request_id")

        if not response_id: return

        self.active_response_id = response_id
        self._voice_idle.clear()

        if request_id == self._pending_create_id:
            self._pending_create_id = None
            self._voice_created.set()

    def handle_response_done(self, response: dict) -> None:
        response_id = response.get("id")

        if response_id != self.active_response_id:
            return

        self.active_response_id = None
        self._voice_idle.set()

    async def send_function_output(self, call_id, output) -> None:
        if not isinstance(output, str):
            output = json.dumps(output)

        await self.ws.send(
            json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            })
        )