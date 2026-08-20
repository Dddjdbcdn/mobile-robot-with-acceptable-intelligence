from __future__ import annotations

import asyncio
from pathlib import Path
import math, time

from database.state import robot_state

class ObjectApproachingManager:
    def __init__(self, camera, csrt_tracker,grounding_dino,depth_anything,object_tracking_manager,response_manager,zmq_req_lock,zmq_req_socket):
        self.csrt_tracker = csrt_tracker
        self.grounding_dino = grounding_dino
        self.depth_anything = depth_anything
        self.camera = camera
        self.object_tracking_manager = object_tracking_manager
        self.zmq_req_lock = zmq_req_lock
        self.zmq_req_socket = zmq_req_socket
        self.response_manager = response_manager

        self.approaching = False
        self.looking_time = 5.0
        self.timeout = 30.0

    async def start_approaching(self,call_id):
        if await self.object_tracking_manager.wait_until_stable():
            await self.approach_action(call_id)
        else:
            await self.response_manager.send_function_output(call_id,{"status": "failed", "message": "The object tracker is not stable enough to start approaching"})
            await self.response_manager.create_voice_response()

    async def approach_action(self,call_id):
        navigate_payload = {
            "command": "navigate_to_pose",
            "x": robot_state["camera"]["object_x"],
            "y": robot_state["camera"]["object_y"],
            "angle": robot_state["camera"]["object_angle"]
        }

        print(f"OBJECT APPROACHING STARTED")

        async with self.zmq_req_lock:
            await self.zmq_req_socket.send_json(navigate_payload)
    
            feedback = await asyncio.wait_for(
                self.zmq_req_socket.recv_json(),
                timeout=5.0)

            if feedback.get("status") != "accepted":
                    raise RuntimeError(f"Navigation rejected: {feedback}")

        await self.response_manager.send_function_output(call_id,{"status": "success", "message": "Navigation has started"})

        loop = asyncio.get_running_loop()
        start_time = loop.time()

        self.approaching = True

        while self.approaching:
            if (loop.time() - start_time) >= self.timeout:
                async with self.zmq_req_lock:
                    await self.zmq_req_socket.send_json({"command": "stop_motors"})
            
                    feedback = await asyncio.wait_for(
                        self.zmq_req_socket.recv_json(),
                        timeout=1.0)
                await self.response_manager.create_voice_response(system_msg=f"Approaching was stopped because it exceeds the timeout duration of {self.timeout} seconds ")

                self.approaching = False

            
            
            await asyncio.sleep(0.1)
