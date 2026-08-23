from __future__ import annotations

import asyncio
from pathlib import Path
import math, time

from database.state import robot_state

class ActionResult:
    def __init__(self,action_type,event,timestamp,status,msg=None):
        self.action_type = action_type
        self.event = event
        self.timestamp = timestamp
        self.status = status
        self.msg = msg

class ApproachAction:
    def __init__(self,zmq_req_lock,zmq_req_socket):
        self.zmq_req_lock = zmq_req_lock
        self.zmq_req_socket = zmq_req_socket

    async def start_approaching(self):
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
                return ActionResult("approach_action","start",time.time(),"failure","navigation_denied")

        return ActionResult("approach_action","start",time.time(),"success",None)

    async def finish_approaching(self):
        async with self.zmq_req_lock:
            await self.zmq_req_socket.send_json({"command": "stop_navigation"})
    
            feedback = await asyncio.wait_for(
                self.zmq_req_socket.recv_json(),
                timeout=5.0)

            if feedback.get("status") != "accepted":
                return ActionResult("approach_action","stop",time.time(),"failure","stopping_denied")

        return ActionResult("approach_action","stop",time.time(),"success",None)
