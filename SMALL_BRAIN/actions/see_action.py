from __future__ import annotations

import asyncio
import time

from services.camera_stream import send_camera_image


class ActionResult:
    def __init__(self, action_type, event, timestamp, status, msg=None):
        self.action_type = action_type
        self.event = event
        self.timestamp = timestamp
        self.status = status
        self.msg = msg


class SeeAction:
    """Capture a current frame and add it to the model's visual context.

    Object search belongs to SearchAction and object tracking belongs to
    TrackAction. This action only answers visual reasoning questions about the
    robot's current view.
    """

    def __init__(self, ws, camera):
        self.ws = ws
        self.camera = camera
        self.active = False
        self._operation_lock = asyncio.Lock()

    async def start_seeing(self, query):
        normalized_query = self._normalize_query(query)
        if normalized_query is None:
            return self._result("failure", "vision_query_is_required")

        if self._operation_lock.locked():
            return self._result("failure", "see_action_is_busy")

        async with self._operation_lock:
            self.active = True
            try:
                jpeg_bytes = await asyncio.to_thread(
                    self.camera.jpeg_bytes_snapshot,
                    70,
                    False,
                    "results/search_results/get_vision_snapshot.jpg",
                )
                snapshot = self.camera.snapshot()

                image_result = await send_camera_image(
                    self.ws,
                    jpeg_bytes,
                    instruction=normalized_query,
                )

                return self._result(
                    "success",
                    {
                        "result": "image_context_added",
                        "query": normalized_query,
                        "frame_sequence": getattr(snapshot, "sequence", None),
                        "captured_at": getattr(snapshot, "captured_at", None),
                        **image_result,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                return self._result(
                    "failure",
                    {
                        "result": "vision_capture_failed",
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
            finally:
                self.active = False

    async def stop_seeing(self):
        # Frame capture is short-lived and contains no background task. The
        # operation lock prevents a second capture from overlapping it.
        return self._result(
            "success",
            "see_action_has_no_persistent_operation",
            event="stop",
        )

    @staticmethod
    def _normalize_query(query):
        if not isinstance(query, str):
            return None
        normalized = query.strip()
        return normalized or None

    @staticmethod
    def _result(status, msg=None, event="start"):
        return ActionResult(
            "see_action",
            event,
            time.time(),
            status,
            msg,
        )
