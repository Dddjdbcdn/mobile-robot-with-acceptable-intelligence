import asyncio

from services.camera_stream import send_camera_image

from actions.action_result import ActionResult

class SeeAction:
    def __init__(self, ws, camera):
        self.ws = ws
        self.camera = camera

    async def see(self, query, action_id):
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
            image_result = await send_camera_image(
                self.ws,
                jpeg_bytes,
                instruction=normalized_query,
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
                "frame_sequence": getattr(snapshot, "sequence", None),
                "captured_at": getattr(snapshot, "captured_at", None),
                **image_result,
            },
        )
