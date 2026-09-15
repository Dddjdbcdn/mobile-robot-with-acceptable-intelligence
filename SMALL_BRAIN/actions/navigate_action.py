"""Use repeated OOB vision assessments and Nav2 waypoints to reach a goal."""
import asyncio
import base64
import copy
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import cv2

from actions.action_result import ActionResult


class NavigationError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class NavigateAction:
    VISION_TIMEOUT = 15.0
    NAVIGATION_TIMEOUT = 90.0
    CAMERA_MAX_AGE = 2.0
    JPEG_QUALITY = 75
    ZOOM_IMAGE_SIZE = 768
    ZOOM_MARGIN_M = 0.25
    HISTORY_COLOR = (180, 0, 180)  # BGR magenta, distinct from candidates and robot.
    START_COLOR = (255, 255, 0)          # BGR cyan, older session starts.
    CURRENT_START_COLOR = (0, 235, 255)  # BGR yellow, active call's start.
    POSITION_TOLERANCE = 0.05  # Meters; small localization updates are allowed.
    YAW_TOLERANCE = 0.15      # Radians.
    MAX_STEPS = 20
    MAX_DURATION = 300.0

    def __init__(self, ws, camera, send_robot_command, debug_dir=None):
        self.ws = ws
        self.camera = camera
        self.send_robot_command = send_robot_command
        path = Path(__file__).resolve().parents[1] / 'tools' / 'vision_oob_tools.json'
        self.selection_tool = next(t for t in json.loads(path.read_text())
                                   if t['name'] == 'select_navigation_pose')
        configured_debug_dir = debug_dir or os.environ.get('NAVIGATION_DEBUG_DIR')
        self.debug_dir = Path(configured_debug_dir) if configured_debug_dir else (
            Path(__file__).resolve().parents[1] / 'results' / 'navigation'
        )
        self.active = False
        self.action_id = self.target = None
        self.stage = 'idle'
        self.completion_future = None
        self._worker = self._selection_future = self._navigation_future = None
        self._request_id = self._snapshot_id = None
        self._pose_history = []
        self._initial_pose = None
        self._session_start_poses = []
        self._action_number = None
        self._candidates = {}
        self._stop_requested = self._command_sent = False
        self._command_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._result_data = {}

    async def start(self, query, action_id):
        if self.active:
            return ActionResult(action_id, 'navigate_action', 'already_running', target=query,
                                reason_code='NAVIGATION_BUSY', retryable=True)
        if not isinstance(query, str) or not query.strip():
            return ActionResult(action_id, 'navigate_action', 'failed',
                                reason_code='NAVIGATION_QUERY_REQUIRED')
        self.active = True
        self.action_id, self.target = action_id, query.strip()
        self._pose_history = []
        self._initial_pose = None
        self._action_number = None
        self._result_data = {}
        self.stage = 'selecting'
        self._stop_requested = self._command_sent = False
        self.completion_future = asyncio.get_running_loop().create_future()
        self._worker = asyncio.create_task(self._run(), name=f'navigate-{action_id}')
        return ActionResult(action_id, 'navigate_action', 'running', target=self.target,
                            outcome='selecting_pose')

    async def _run(self):
        try:
            started_at = time.monotonic()
            completed_steps = []
            self._result_data['steps'] = completed_steps

            for step_number in range(1, self.MAX_STEPS + 1):
                if time.monotonic() - started_at > self.MAX_DURATION:
                    raise NavigationError(
                        'NAVIGATION_DURATION_LIMIT',
                        'The overall navigation time limit was reached before the goal was confirmed',
                    )

                self.stage = 'selecting'
                self._command_sent = False
                snapshot, frame, jpeg = await self._capture()
                self._remember_pose(snapshot['metadata']['robot_pose'], step_number)
                if self._initial_pose is None:
                    self._initial_pose = copy.deepcopy(self._pose_history[0])
                    self._action_number = len(self._session_start_poses) + 1
                    self._session_start_poses.append(copy.deepcopy(self._initial_pose))
                selection = await self._select_pose(
                    snapshot, frame, jpeg, step_number, completed_steps
                )
                decision = selection['decision']
                self._result_data.update(
                    last_selection=selection,
                    last_snapshot_id=self._snapshot_id,
                    step_count=len(completed_steps),
                )

                if decision == 'goal_reached':
                    self._save_goal_reached_map(snapshot, step_number)
                    self._finish('succeeded', 'goal_reached')
                    return
                if decision == 'blocked':
                    raise NavigationError('NAVIGATION_BLOCKED', selection['reason'])
                # IDs are resolved against the exact snapshot shown to vision.
                destination = dict(self._candidates[selection['pose_id']])
                self._revalidate(snapshot, frame, destination)
                self._navigation_future = asyncio.get_running_loop().create_future()
                async with self._command_lock:
                    if self._stop_requested:
                        return
                    self.stage = 'dispatching'
                    self._command_sent = True
                    feedback = await self.send_robot_command({
                        'command': 'navigate_to_pose', 'action_id': self.action_id,
                        'frame_id': destination['frame_id'], 'x': destination['x'],
                        'y': destination['y'], 'angle': destination['yaw'],
                    })
                if self._stop_requested:
                    return
                if not isinstance(feedback, dict) or feedback.get('status') != 'accepted':
                    raise NavigationError('NAVIGATION_REJECTED', str(feedback))
                self.stage = 'navigating'
                try:
                    event = await asyncio.wait_for(
                        self._navigation_future, self.NAVIGATION_TIMEOUT
                    )
                except TimeoutError:
                    raise NavigationError(
                        'NAVIGATION_TIMEOUT', 'No terminal Nav2 event received'
                    )
                if event.get('status') != 'Goal Reached':
                    raise NavigationError('NAVIGATION_FAILED', str(event.get('status')))
                completed_steps.append({
                    'step': step_number,
                    'snapshot_id': self._snapshot_id,
                    'selection': selection,
                    'destination': destination,
                    'robot_status': event.get('status'),
                })
                self._result_data['step_count'] = len(completed_steps)

            raise NavigationError(
                'NAVIGATION_STEP_LIMIT',
                'The waypoint limit was reached before the overall goal was confirmed',
            )
        except asyncio.CancelledError:
            return  # stop() owns cleanup and the terminal result.
        except Exception as error:
            if self._stop_requested:
                return
            if self._command_sent:
                await self._stop_motion()
            if not self._stop_requested:
                self._result_data['error'] = str(error)
                self._finish('failed', 'navigation_failed',
                             getattr(error, 'code', 'NAVIGATION_ERROR'))

    async def _capture(self):
        snapshot = await asyncio.to_thread(self.camera.map_snapshot)
        metadata = snapshot['metadata']
        self._snapshot_id = metadata['snapshot_id']
        self._candidates = {}
        for candidate in metadata['candidates']:
            if (not isinstance(candidate.get('id'), str)
                    or candidate.get('frame_id') != metadata['frame_id']
                    or not all(math.isfinite(float(candidate[k])) for k in ('x', 'y', 'yaw'))
                    or candidate['id'] in self._candidates):
                raise NavigationError('INVALID_MAP_CANDIDATES', 'Invalid candidate coordinates, ID or frame')
            self._candidates[candidate['id']] = dict(candidate)
        if not self._candidates:
            raise NavigationError('NO_NAVIGATION_CANDIDATES', 'No reachable map candidates')
        frame = self.camera.snapshot()
        if time.monotonic() - frame.captured_at > self.CAMERA_MAX_AGE:
            raise NavigationError('STALE_CAMERA', 'Current camera frame is stale')
        ok, jpeg = await asyncio.to_thread(
            cv2.imencode, '.jpg', frame.tracking_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY]
        )
        if not ok:
            raise NavigationError('CAMERA_ENCODING_FAILED', 'Could not encode camera image')
        return snapshot, frame, jpeg.tobytes()

    async def _select_pose(self, snapshot, frame, camera_jpeg,
                           step_number, completed_steps):
        self._request_id = uuid.uuid4().hex
        self._selection_future = asyncio.get_running_loop().create_future()
        tool = copy.deepcopy(self.selection_tool)
        tool['parameters']['properties']['pose_id']['enum'] = [*self._candidates, None]
        map_jpeg = self._map_jpeg_with_history(snapshot, step_number)

        map_metadata = snapshot['metadata']
        context = {
            'query': self.target,
            'step': step_number,
            'valid_candidate_ids': list(self._candidates),
            'candidates': {
                pose_id: {
                    key: candidate.get(key)
                    for key in ('kind', 'distance_m')
                    if candidate.get(key) is not None
                }
                for pose_id, candidate in self._candidates.items()
            },
        }
        prompt = (
            'Choose the next waypoint for the query using the fresh map and camera. '
            'Image 1 is the full map, and image 2 is the front camera. Map +X is right and +Y is up. '
            'Large green diamond F are frontier goals across the full map and blue numbered-circle are nearby poses. The short colored tick on each '
            'marker shows its final heading. Each green F ID is the maximum-information cell '
            'from one connected frontier group; blue numeric IDs are local translation candidates. '
            'Red is the current robot pose. Cyan S-number markers are older navigation-call starts from this session. '
            'The yellow S-number marker is the starting pose for this current navigation call and goal. '
            'The magenta trail shows movement during only the current navigation call. '
            'valid_candidate_ids is the complete set of IDs allowed for this snapshot; '
            'candidates describes each ID and its kind and distance. '
            'Numeric and F IDs move and face the travel/frontier direction; do not rotate first. '
            'For exploration or when the semantic direction is unknown, prefer the F goal '
            'R IDs rotate in place: R+90 turns left 90°, R-90 turns right 90°, '
            'and R+180 turns around. Use R only for a requested turn or when the '
            'goal direction cannot be determined without looking elsewhere. '
            'Pick the best location that makes most progress towards the goal without worrying about obstacles. Nav2 handles path planning and obstacle avoidance. '
            'Do not shorten a move just to reassess or avoid a camera obstacle; '
            'Use the camera for assess meaningful frontier, semantic direction, visibility, and arrival evidence. '
            'If the map has a frontier but the camera vision shows a dead end, you should not pick that frontier. '
            "Assess completion against the original query and the current navigation call's visible S-number marker, not merely "
            'against the current loop position. Do not keep moving after the semantic goal is '
            'reasonably satisfied. Return goal_reached when the complete goal is satisfied, including when the robot is '
            'reasonably within an approximate semantic goal region. Return blocked only when no '
            'listed pose can safely help. For move, select exactly one listed ID; otherwise use null. \n'
            + json.dumps(context, allow_nan=False)
        )
        self._save_debug_inputs(
            map_jpeg, camera_jpeg, map_metadata, context,
            step_number, self._request_id,
        )
        content = [{'type': 'input_text', 'text': prompt}]
        content.extend({'type': 'input_image', 'image_url': 'data:image/jpeg;base64,'
                        + base64.b64encode(jpeg).decode('ascii')}
                       for jpeg in (map_jpeg, camera_jpeg))
        try:
            await self.ws.send(json.dumps({
                'event_id': f'navigate_vision_{self._request_id}', 'type': 'response.create',
                'response': {
                    'conversation': 'none',
                    'metadata': {
                        'kind': 'navigation_selection', 'action_id': self.action_id,
                        'request_id': self._request_id, 'snapshot_id': self._snapshot_id,
                    },
                    'output_modalities': ['text'], 'tools': [tool], 'tool_choice': 'required',
                    'input': [{'type': 'message', 'role': 'user', 'content': content}],
                },
            }))
            try:
                return await asyncio.wait_for(self._selection_future, self.VISION_TIMEOUT)
            except TimeoutError:
                raise NavigationError('NAVIGATION_VISION_TIMEOUT', 'Pose selection timed out')
        finally:
            self._request_id = None
            self._selection_future = None

    async def select_navigation_pose(self, args, response_metadata):
        """Ignore unrelated, late, duplicate, or cancelled OOB responses."""
        if (not self.active or self._stop_requested
                or response_metadata.get('kind') != 'navigation_selection'
                or response_metadata.get('action_id') != self.action_id
                or response_metadata.get('request_id') != self._request_id
                or response_metadata.get('snapshot_id') != self._snapshot_id):
            return
        future = self._selection_future
        if future is None or future.done():
            return
        decision = args.get('decision')
        pose_id = args.get('pose_id')
        if decision not in {'move', 'goal_reached', 'blocked'}:
            future.set_exception(NavigationError(
                'INVALID_NAVIGATION_DECISION', 'Vision returned an unknown decision'
            ))
            return
        if 'pose_id' not in args or (pose_id is not None and (
                not isinstance(pose_id, str) or pose_id not in self._candidates)):
            future.set_exception(NavigationError('INVALID_POSE_ID', 'Vision returned an unknown pose ID'))
            return
        if (decision == 'move') != (pose_id is not None):
            future.set_exception(NavigationError(
                'INVALID_NAVIGATION_DECISION',
                'move requires a pose ID; terminal decisions require pose_id=null',
            ))
            return
        future.set_result({
            'decision': decision,
            'pose_id': pose_id,
            'reason': str(args.get('reason') or ''),
        })


    def _remember_pose(self, pose, before_step):
        """Record one actual pre-move pose for this navigate_action only."""
        try:
            recorded = {
                'before_step': int(before_step),
                'x': float(pose['x']),
                'y': float(pose['y']),
                'yaw': float(pose['yaw']),
                'frame_id': str(pose['frame_id']),
            }
        except (KeyError, TypeError, ValueError):
            return
        if not all(math.isfinite(recorded[key]) for key in ('x', 'y', 'yaw')):
            return
        self._pose_history.append(recorded)

    @staticmethod
    def _atomic_write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)

    def _save_debug_inputs(self, map_jpeg, vision_jpeg, map_metadata, context,
                           step_number, request_id):
        """Persist the exact images and structured context sent to vision."""
        manifest = {
            'action_id': self.action_id,
            'request_id': request_id,
            'snapshot_id': self._snapshot_id,
            'step': step_number,
            'map_file': 'latest_map.jpg',
            'vision_file': 'latest_vision.jpg',
            'map_metadata': map_metadata,
            'selection_context': context,
        }
        try:
            self._atomic_write(self.debug_dir / manifest['map_file'], map_jpeg)
            self._atomic_write(self.debug_dir / manifest['vision_file'], vision_jpeg)
            self._atomic_write(
                self.debug_dir / 'latest_request.json',
                json.dumps(manifest, indent=2, allow_nan=False).encode('utf-8'),
            )
            self._result_data['debug_dir'] = str(self.debug_dir)
            self._result_data['last_debug_step'] = step_number
        except (OSError, TypeError, ValueError) as error:
            # Debug output must never prevent the robot from navigating.
            self._result_data['debug_save_error'] = str(error)

    def _map_jpeg_with_history(self, snapshot, step_number, show_last_label=False):
        """Overlay the action start and each loop start without mutating the map stream."""
        if not self._pose_history:
            return snapshot['jpeg_bytes']
        image = snapshot.get('image')
        metadata = snapshot.get('metadata') or {}
        bounds = metadata.get('image_world_bounds') or {}
        if image is None or not all(key in bounds for key in ('xmin', 'xmax', 'ymin', 'ymax')):
            return snapshot['jpeg_bytes']
        xmin, xmax = float(bounds['xmin']), float(bounds['xmax'])
        ymin, ymax = float(bounds['ymin']), float(bounds['ymax'])
        if xmax <= xmin or ymax <= ymin:
            return snapshot['jpeg_bytes']

        overlay = image.copy()
        height, width = overlay.shape[:2]

        def pixel(pose):
            try:
                col = round((float(pose['x']) - xmin) * (width - 1) / (xmax - xmin))
                row = round((ymax - float(pose['y'])) * (height - 1) / (ymax - ymin))
            except (KeyError, TypeError, ValueError):
                return None
            return (col, row) if 0 <= col < width and 0 <= row < height else None

        trail = self._pose_history
        trail_pixels = [point for point in (pixel(pose) for pose in trail) if point is not None]
        for start, end in zip(trail_pixels, trail_pixels[1:]):
            cv2.line(overlay, start, end, self.HISTORY_COLOR, 2, cv2.LINE_AA)
        for pose in self._pose_history:
            point = pixel(pose)
            if point is None:
                continue
            color, radius = self.HISTORY_COLOR, 6
            cv2.circle(overlay, point, radius, (20, 20, 20), 7, cv2.LINE_AA)
            cv2.circle(overlay, point, radius, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.circle(overlay, point, radius, color, 2, cv2.LINE_AA)
            cv2.circle(overlay, point, 4, color, -1, cv2.LINE_AA)

        for action_number, saved_start in enumerate(self._session_start_poses, 1):
            start_point = pixel(saved_start)
            if start_point is None:
                continue
            radius = 18
            cv2.circle(overlay, start_point, radius, (20, 20, 20), 7, cv2.LINE_AA)
            cv2.circle(overlay, start_point, radius, (255, 255, 255), 4, cv2.LINE_AA)
            color = (self.CURRENT_START_COLOR
                     if action_number == self._action_number else self.START_COLOR)
            cv2.circle(overlay, start_point, radius, color, 2, cv2.LINE_AA)
            text_origin = (start_point[0] + radius + 3, start_point[1] + 4)
            label = f'S{action_number}'
            cv2.putText(overlay, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (20, 20, 20), 5, cv2.LINE_AA)
            cv2.putText(overlay, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2, cv2.LINE_AA)

        if show_last_label and self._pose_history:
            last_point = pixel(self._pose_history[-1])
            if last_point is not None:
                radius = 14
                cv2.circle(overlay, last_point, radius, (20, 20, 20), 7, cv2.LINE_AA)
                cv2.circle(overlay, last_point, radius, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.circle(overlay, last_point, radius, self.HISTORY_COLOR, 2, cv2.LINE_AA)
                label = f"L{self._action_number}"
                text_origin = (last_point[0] + radius + 3, last_point[1] + 18)
                cv2.putText(overlay, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (20, 20, 20), 5, cv2.LINE_AA)
                cv2.putText(overlay, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, self.HISTORY_COLOR, 2, cv2.LINE_AA)

        ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return encoded.tobytes() if ok else snapshot["jpeg_bytes"]

    def _save_goal_reached_map(self, snapshot, step_number):
        """Add L<n> only to the local terminal debug map, never to an AI input."""
        try:
            terminal_map = self._map_jpeg_with_history(
                snapshot, step_number, show_last_label=True
            )
            self._atomic_write(self.debug_dir / "latest_map.jpg", terminal_map)
            self._result_data["terminal_map_label"] = f"L{self._action_number}"
        except OSError as error:
            self._result_data["debug_save_error"] = str(error)

    def _revalidate(self, original, frame, destination):
        pass # skip revalidating for now

    def _same_pose(self, first, second):
        distance = math.hypot(first['x'] - second['x'], first['y'] - second['y'])
        delta = first['yaw'] - second['yaw']
        return (distance <= self.POSITION_TOLERANCE
                and abs(math.atan2(math.sin(delta), math.cos(delta))) <= self.YAW_TOLERANCE)

    def handle_navigation_event(self, payload):
        if (not self.active or self._stop_requested
                or self.stage not in {'dispatching', 'navigating'}
                or payload.get('event') != 'navigation'
                or payload.get('action_id') != self.action_id):
            return False
        if self._navigation_future is not None and not self._navigation_future.done():
            self._navigation_future.set_result(dict(payload))
        return True

    async def wait_until_finished(self):
        return await asyncio.shield(self.completion_future)

    async def _stop_motion(self):
        try:
            async with self._command_lock:
                feedback = await self.send_robot_command({
                    'command': 'stop_moving', 'action_id': self.action_id,
                })
            if not isinstance(feedback, dict) or feedback.get('status') != 'accepted':
                raise RuntimeError(str(feedback))
            return True
        except Exception as error:
            self._result_data['stop_error'] = str(error)
            return False

    async def stop(self, reason_code='USER_REQUESTED'):
        async with self._stop_lock:
            if not self.active:
                return None
            self._stop_requested = True
            # Finish any in-flight REQ/REP transaction before sending stop.
            # Cancelling its receive could break the shared command socket.
            stopped = await self._stop_motion() if self._command_sent else True
            if self._worker is not None and not self._worker.done():
                self._worker.cancel()
                await asyncio.gather(self._worker, return_exceptions=True)
            return self._finish('cancelled' if stopped else 'failed',
                                'stopped' if stopped else 'stop_failed',
                                reason_code if stopped else 'NAVIGATION_STOP_FAILED')

    def _finish(self, status, outcome, reason_code=None):
        result = ActionResult(self.action_id, 'navigate_action', status, target=self.target,
                              outcome=outcome, reason_code=reason_code,
                              retryable=status == 'failed', data=dict(self._result_data))
        self.active = False
        self.stage = 'idle'
        self._request_id = None
        for future in (self._selection_future, self._navigation_future):
            if future is not None and not future.done():
                future.cancel()
        if not self.completion_future.done():
            self.completion_future.set_result(result)
        return result
