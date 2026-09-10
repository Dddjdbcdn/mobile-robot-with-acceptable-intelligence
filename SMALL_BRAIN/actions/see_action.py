import asyncio
import base64
import json

from actions.action_result import ActionResult

class SeeAction:
    def __init__(self, ws, camera):
        self.ws = ws
        self.camera = camera

    async def see(self, query, action_id, autonomous=False):
        normalized_query = (query or "").strip()
        if not normalized_query:
            return ActionResult(
                action_id=action_id,
                action_type="see_action",
                status="failed",
                outcome="invalid_request",
                reason_code="VISION_QUERY_REQUIRED",
            )

        try:
            jpeg_bytes = await asyncio.to_thread(
                self.camera.jpeg_bytes_snapshot,
                70,
                False,
                "results/action_results/see_snapshot.jpg",
            )
            snapshot = self.camera.snapshot()
            image_instruction = normalized_query
            if autonomous:
                image_instruction = (
                    "[INTERNAL AUTONOMOUS PERCEPTION] DJ initiated this observation "
                    "without a user request. Treat the attached image as DJ's own "
                    "current visual perception. Do not say or imply that the user "
                    f"asked for this inspection. {normalized_query}"
                )
            image_result = await self.send_camera_image(
                self.ws,
                jpeg_bytes,
                instruction=image_instruction,
            )
        except Exception as error:
            return ActionResult(
                action_id=action_id,
                action_type="see_action",
                status="failed",
                outcome="vision_capture_failed",
                reason_code="VISION_CAPTURE_FAILED",
                retryable=True,
                data={"error": f"{type(error).__name__}: {error}"},
            )

        return ActionResult(
            action_id=action_id,
            action_type="see_action",
            status="succeeded",
            outcome="image_context_added",
            data={
                "query": normalized_query,
                "source": "autonomy" if autonomous else "user",
                "frame_sequence": getattr(snapshot, "sequence", None),
                "captured_at": getattr(snapshot, "captured_at", None),
                **image_result,
            },
        )

    async def send_camera_image(self, ws, jpeg_bytes, instruction):
        jpeg_bytes = bytes(jpeg_bytes)
        if not jpeg_bytes or not jpeg_bytes.startswith(b"\xff\xd8"):
            raise ValueError("Invalid JPEG bytes. Encode the OpenCV frame with cv2.imencode('.jpg', frame).")

        encoded_image = base64.b64encode(jpeg_bytes).decode("ascii")
        image_url = f"data:image/jpeg;base64,{encoded_image}"

        image_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction.strip()},
                    {"type": "input_image", "image_url": image_url},
                ],
            },
        }

        await ws.send(json.dumps(image_event))
        print(f"\n[Camera] IMAGE ADDED")

        return {"jpeg_bytes": len(jpeg_bytes), "base64_characters": len(encoded_image)}
