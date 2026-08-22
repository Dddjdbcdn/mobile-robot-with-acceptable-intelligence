import asyncio
from pathlib import Path
import time

HORIZONTAL_FOV_DEG = 85
VERTICAL_FOV_DEG = 52

import math

dj_yolo_classes = [
    "guitar",
    "chair",
    "table",
    "person",
    "laptop",
    "door",
    "television",
    "fan",
    "bottle",
    "mirror",
    "toolbox",
    "dumbbell",
    "camera",
    "houseplant",
    "curtain",
    "power socket",
    "book",
    "microphone",
    "smartphone",
    "air conditioner",
    "remote control"
]

SKELETON_GRAPH = {
    "nose": ["left_eye", "right_eye"],
    "left_eye": ["nose", "left_ear"],
    "right_eye": ["nose", "right_ear"],
    "left_ear": ["left_eye", "left_shoulder"],
    "right_ear": ["right_eye", "right_shoulder"],
    "left_shoulder": ["left_ear", "right_shoulder", "left_elbow", "left_hip"],
    "right_shoulder": ["right_ear", "left_shoulder", "right_elbow", "right_hip"],
    "left_elbow": ["left_shoulder", "left_wrist"],
    "right_elbow": ["right_shoulder", "right_wrist"],
    "left_wrist": ["left_elbow"],
    "right_wrist": ["right_elbow"],
    "left_hip": ["left_shoulder", "right_hip", "left_knee"],
    "right_hip": ["right_shoulder", "left_hip", "right_knee"],
    "left_knee": ["left_hip", "left_ankle"],
    "right_knee": ["right_hip", "right_ankle"],
    "left_ankle": ["left_knee"],
    "right_ankle": ["right_knee"],
}

HUMAN_RETARGETS = {
    "person": ["nose"],
    "human": ["nose"],
    "face": ["nose"],
    "head": ["nose"],

    "eye": ["left_eye","right_eye"],
    "eyes": ["left_eye","right_eye"],
    "left_eye": ["left_eye"],
    "right_eye": ["right_eye"],

    "hand": ["left_wrist","right_wrist"],
    "left_hand": ["left_wrist"],
    "right_hand": ["right_wrist"],

    "leg": ["left_knee","right_knee"],
    "left_leg": ["left_knee"],
    "right_leg": ["right_knee"],

    "feet": ["left_ankle","right_ankle"],
    "left_feet": ["left_ankle"],
    "right_feet": ["right_ankle"],
}

human_trackable_parts = list(HUMAN_RETARGETS.keys())

def normalize_human_target(target):
    target = target.lower().strip()

    target = target.replace("'s","")
    target = target.replace("-"," ")
    target = target.replace("_"," ")

    words = target.split()

    side = None

    if "left" in words: side = "left"
    elif "right" in words: side = "right"

    if "hand" in words or "wrist" in words: part = "hand"
    elif "leg" in words: part = "leg"
    elif "foot" in words or "feet" in words or "ankle" in words: part = "feet"
    elif "eye" in words or "eyes" in words: part = "eye"
    elif "face" in words: part = "face"
    elif "head" in words: part = "head"
    elif "person" in words or "human" in words or "user" in words or "me" in words or "you" in words: part = "person"

    else:
        return None

    if side is not None and part in ["hand","leg","eye","foot"]:
        return f"{side}_{part}"

    return part

def normalize_object_target(target):
    target = target.lower().strip()
    target = target.replace("'s", "")
    target = target.replace("-", " ")
    target = target.replace("_", " ")
    target = " ".join(target.split())

    aliases = {
        # guitar
        "guitars": "guitar",
        "acoustic guitar": "guitar",
        "electric guitar": "guitar",

        # chair
        "chairs": "chair",
        "seat": "chair",
        "seats": "chair",

        # table
        "tables": "table",
        "desk": "table",
        "desks": "table",

        # laptop
        "laptops": "laptop",
        "notebook": "laptop",
        "notebook computer": "laptop",
        "computer": "laptop",

        # door
        "doors": "door",

        # television
        "tv": "television",
        "t v": "television",
        "television set": "television",
        "screen": "television",

        # fan
        "fans": "fan",
        "electric fan": "fan",

        # bottle
        "bottles": "bottle",
        "water bottle": "bottle",
        "water bottles": "bottle",
        "drink bottle": "bottle",
        "drinking bottle": "bottle",

        # mirror
        "mirrors": "mirror",

        # toolbox
        "tool box": "toolbox",
        "tool boxes": "toolbox",
        "toolboxes": "toolbox",

        # dumbbell
        "dumbbells": "dumbbell",
        "weight": "dumbbell",
        "weights": "dumbbell",
        "hand weight": "dumbbell",

        # camera
        "cameras": "camera",
        "webcam": "camera",

        # houseplant
        "house plant": "houseplant",
        "house plants": "houseplant",
        "plant": "houseplant",
        "plants": "houseplant",
        "potted plant": "houseplant",
        "indoor plant": "houseplant",

        # curtain
        "curtains": "curtain",
        "drape": "curtain",
        "drapes": "curtain",

        # power socket
        "socket": "power socket",
        "sockets": "power socket",
        "power outlet": "power socket",
        "outlet": "power socket",
        "wall outlet": "power socket",
        "electrical outlet": "power socket",
        "plug socket": "power socket",

        # book
        "books": "book",

        # microphone
        "mic": "microphone",
        "mics": "microphone",
        "microphones": "microphone",

        # smartphone
        "phone": "smartphone",
        "phones": "smartphone",
        "smart phone": "smartphone",
        "mobile phone": "smartphone",
        "cell phone": "smartphone",
        "cellphone": "smartphone",

        # air conditioner
        "air conditioning": "air conditioner",
        "aircon": "air conditioner",
        "air con": "air conditioner",
        "ac": "air conditioner",
        "a c": "air conditioner",

        # remote control
        "remote": "remote control",
        "remotes": "remote control",
        "controller": "remote control",
        "tv remote": "remote control",
        "television remote": "remote control",
        "remote controller": "remote control",
    }

    return aliases.get(target, target)


class ObjectTrackingManager():
    def __init__(self, csrt_tracker,yolo,grounding_dino,zmq_req_lock, zmq_req_socket,zmq_pub_socket, STABLE_THRESHOLD = 0.05):
        self.csrt_tracker = csrt_tracker
        self.grounding_dino = grounding_dino
        self.yolo = yolo
        self.zmq_req_lock = zmq_req_lock
        self.zmq_req_socket = zmq_req_socket
        self.zmq_pub_socket = zmq_pub_socket
        self.tracking = False
        self.stable = False
        self.stable_tick = 0.0
        self.stable_threshold = STABLE_THRESHOLD
        self._tracking_task = None 

        self.person_path = []
        self.person_path_index = 0

    async def start_tracking(self,jpeg_bytes,target,snapshot):
        self.stable = False
        self.stable_tick = 0
        target = normalize_human_target(target) or normalize_object_target(target)
        
        if self.tracking: await self.stop_tracking(reset=False)

        if target in human_trackable_parts:
            self.yolo.vision_mode = "pose"
            await asyncio.sleep(0.5)

            return await self._start_person_tracking(target)

        return await self._start_object_tracking(jpeg_bytes,target,snapshot)

    async def _start_object_tracking(self,jpeg_bytes,target,snapshot):
        yolo_detected = False
        groundingdino_detected = False
        tracking_bbox = None

        if target in dj_yolo_classes:
            matching = [
                detection
                for detection in self.yolo.detections
                if detection["class"] == target
            ]
                
            if matching:
                detection = max(matching,key=lambda d: d["confidence"])
                print(
                    f"YOLO TARGET FOUND: {detection['class']}. "
                    f"SCORE: {detection['confidence']}."
                )
                yolo_detected = True

                bbox = detection["bbox"]

                tracking_bbox = (
                    int(bbox["x1"]),
                    int(bbox["y1"]),
                    int(bbox["x2"] - bbox["x1"]),
                    int(bbox["y2"] - bbox["y1"]),
                )
    
        if not yolo_detected:
            self.yolo.vision_mode = "none"
            await asyncio.sleep(0.1)
            groundinngdino_result = await self.grounding_dino.detect(
                    image_source=jpeg_bytes,
                    target=f"{target}",
                    box_threshold=0.25,
                    text_threshold=0.25,
                    nms_threshold=0.80,
                    output_root=Path("results/grounding_results"),
                )
            self.yolo.vision_mode = "dj"

            detection = groundinngdino_result.best

            print(f"GROUNDING DINO FINISHED. LATENCY: {groundinngdino_result.inference_latency_ms}")

            if detection is not None:
                print(f"GROUNDING DINO TARGET FOUND: {detection.label}. SCORE: {detection.score}.")
                groundingdino_detected = True
                orig_x, orig_y, orig_w, orig_h = detection.tracker_box_xywh

                tracking_bbox = (
                    int(orig_x / 2.0),
                    int(orig_y / 2.0),
                    int(orig_w / 2.0),
                    int(orig_h / 2.0)
                )

        if yolo_detected or groundingdino_detected:
            self.csrt_tracker.begin_tracking(
                detection_sequence=snapshot.sequence, 
                initialization_frame=snapshot.tracking_bgr, 
                bbox_xywh=tracking_bbox, 
                target=target
            )

            print(f"CSRT STARTS TRACKING OBJECT: {target}")

            async with self.zmq_req_lock:
                await self.zmq_req_socket.send_json({"command": "track_object"})
                feedback = await asyncio.wait_for(
                    self.zmq_req_socket.recv_json(),
                    timeout=1.0,
                )

            if feedback.get("status") == "accepted":
                self.tracking = True
                self._tracking_task = asyncio.create_task(self._object_tracking_loop())

                return "tracking"
            else:
                return "failed to track"
        else:
            self.csrt_tracker.stop_tracking()
            return "failed to detect"

    async def _object_tracking_loop(self):
        succeeded = False

        while self.tracking:
            tracking_update = self.csrt_tracker.tracking_update
            if tracking_update is not None and tracking_update.is_tracking and tracking_update.success:
                succeeded = True
                target_x = tracking_update.normalized_x
                target_y = tracking_update.normalized_y

                delta_pan_angle, delta_tilt_angle = self.target_to_angles(target_x, target_y)
    
                await self.zmq_pub_socket.send_json({
                    "delta_pan_angle": delta_pan_angle, 
                    "delta_tilt_angle": delta_tilt_angle
                })

                if (abs(target_x-0.5) < self.stable_threshold and abs(target_y-0.5) < self.stable_threshold):
                    self.stable_tick += 1
                    if self.stable_tick >= 10:
                        self.stable = True
                else:
                    self.stable_tick = 0
                    self.stable = False
            else:
                if succeeded: await self.stop_tracking()
            
            await asyncio.sleep(0.05)
    
    async def _start_person_tracking(self,target):
        detections = [
            detection
            for detection in self.yolo.detections
            if detection["class"] == "person"
            and "keypoints" in detection
        ]

        if not detections:
            return "failed to detect"

        person = max(
            detections,
            key=lambda detection: detection["confidence"]
        )

        self.person_path = self.find_best_person_path(person,target)

        if not self.person_path:
            return "failed to detect"

        self.person_path_index = 0
        self.person_stable_count = 0

        print(
            f"TARGET: {target}"
            f"\nSTARTING FROM: {self.person_path[0]}"
            f"\nPATH: {' -> '.join(self.person_path)}"
        )

        async with self.zmq_req_lock:
            await self.zmq_req_socket.send_json({
                "command": "track_object"
            })

            feedback = await asyncio.wait_for(
                self.zmq_req_socket.recv_json(),
                timeout=1.0,
            )

        if feedback.get("status") != "accepted":
            return "failed to track"

        self.tracking = True

        self._tracking_task = asyncio.create_task(
            self._person_tracking_loop()
        )

        return "tracking"

    def find_best_person_path(self,person,target):
        target_keypoints = HUMAN_RETARGETS.get(target)

        if target_keypoints is None:
            return None

        visible_keypoints = [name for name,keypoint in person["keypoints"].items()]

        best_path = None
        shortest_path = float("inf")

        for visible_keypoint in visible_keypoints:
            for target_keypoint in target_keypoints:
                path = self.find_skeleton_path(
                    visible_keypoint,
                    target_keypoint
                )

                if path and len(path) < shortest_path:
                    shortest_path = len(path)
                    best_path = path

        return best_path

    def find_skeleton_path(self,start,target):
        queue = [(start,[start])]
        visited = set()

        while queue:
            current,path = queue.pop(0)

            if current == target:
                return path

            if current in visited:
                continue

            visited.add(current)

            for neighbor in SKELETON_GRAPH[current]:
                queue.append(
                    (
                        neighbor,
                        path + [neighbor]
                    )
                )

        return None

    async def _person_tracking_loop(self):
        current_reached = False
        predicted_target = None

        while self.tracking:
            detections = [
                detection
                for detection in self.yolo.detections
                if detection["class"] == "person"
                and "keypoints" in detection
            ]

            if not detections:
                await asyncio.sleep(0.05)
                continue

            person = max(
                detections,
                key=lambda detection: detection["confidence"]
            )

            current_name = self.person_path[self.person_path_index]
            current_keypoint = person["keypoints"].get(current_name)

            if not current_reached:
                if current_keypoint is None:
                    await asyncio.sleep(0.05)
                    continue

                target_x = current_keypoint["normalized_x"]
                target_y = current_keypoint["normalized_y"]

            if self.person_path_index < len(self.person_path) - 1:
                if current_reached:
                    next_name = self.person_path[self.person_path_index + 1]
                    next_keypoint = person["keypoints"].get(next_name)

                    if next_keypoint is not None:
                        self.person_path_index += 1
                        current_reached = False
                        predicted_target = None

                        print(f"TRACING TO: {next_name}")

                    else:
                        if predicted_target is not None:
                            target_x,target_y = predicted_target
                        else:
                            await asyncio.sleep(0.05)
                            continue

                else: 
                    if (abs(target_x - 0.5) < self.stable_threshold and abs(target_y - 0.5) < self.stable_threshold):
                        predicted_target = self.predict_next_keypoint(person)
                        current_reached = True

            delta_pan_angle,delta_tilt_angle = self.target_to_angles(target_x,target_y)

            await self.zmq_pub_socket.send_json({
                "delta_pan_angle": delta_pan_angle,
                "delta_tilt_angle": delta_tilt_angle
            })

            if self.person_path_index == len(self.person_path) - 1:
                if (abs(target_x-0.5) < self.stable_threshold and abs(target_y-0.5) < self.stable_threshold):
                    self.stable_tick += 1
                    if self.stable_tick >= 10:
                        self.stable = True
                else:
                    self.stable_tick = 0
                    self.stable = False

            await asyncio.sleep(0.05)

    def predict_next_keypoint(self,person):
        if self.person_path_index == 0:
            return None

        previous_name = self.person_path[self.person_path_index - 1]
        current_name = self.person_path[self.person_path_index]

        previous_keypoint = person["keypoints"].get(previous_name)
        current_keypoint = person["keypoints"].get(current_name)

        if previous_keypoint is None or current_keypoint is None:
            return None

        previous_x = previous_keypoint["normalized_x"]
        previous_y = previous_keypoint["normalized_y"]

        current_x = current_keypoint["normalized_x"]
        current_y = current_keypoint["normalized_y"]

        delta_x = current_x - previous_x
        delta_y = current_y - previous_y

        predicted_x = current_x + delta_x
        predicted_y = current_y + delta_y

        predicted_x = max(0.0,min(1.0,predicted_x))
        predicted_y = max(0.0,min(1.0,predicted_y))

        return predicted_x,predicted_y

    async def wait_until_stable(self,check_interval=0.05,timeout=10.0):
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        while self.tracking:
            if self.stable:
                return True

            if (
                timeout is not None
                and loop.time() - start_time >= timeout
            ):
                return False

            await asyncio.sleep(check_interval)

        return False
        

    async def stop_tracking(self,reset=True):
        self.yolo.vision_mode = "dj"
        self.stable = False
        self.stable_tick = 0
        self.tracking = False 
        self.stable = False
        self.csrt_tracker.stop_tracking()

        if reset:
            async with self.zmq_req_lock:
                await self.zmq_req_socket.send_json({"command": "stop_tracking_object"})
                feedback = await asyncio.wait_for(
                    self.zmq_req_socket.recv_json(),
                    timeout=1.0,
                )
        
        return "Tracking stopped."

    def target_to_angles(self, x, y):
        center_x_error = x - 0.5
        center_y_error = y - 0.5

        tan_half_fov_h = math.tan(math.radians(HORIZONTAL_FOV_DEG / 2.0))
        tan_half_fov_v = math.tan(math.radians(VERTICAL_FOV_DEG / 2.0))
        
        pan_angle = math.degrees(math.atan((center_x_error * 2.0) * tan_half_fov_h))
        tilt_angle = math.degrees(math.atan((center_y_error * 2.0) * tan_half_fov_v))
        
        delta_pan = -pan_angle
        delta_tilt = tilt_angle
        
        return delta_pan, delta_tilt