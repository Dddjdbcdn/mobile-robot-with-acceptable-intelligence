"""Use repeated OOB vision assessments and Nav2 waypoints to reach a goal."""
import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import cv2
import zmq

from actions.action_result import ActionResult


FIND_OBJECT_QUERY = (
    "Find {target}. Select the next {mode} search waypoint using only visible "
    "contextual evidence, camera coverage, and unvisited map space."
)

MAP_GUIDANCE = """GENERAL MAP GUIDANCE
Image 1 is the occupancy and camera-coverage map. Map +X points right and +Y
points up. Black cells are occupied, light cells are known traversable space,
and gray cells are unknown or unavailable. Light-blue tint means one camera
observation; pink tint means repeated observations; untinted traversable space
has not been viewed.

The red circle and arrow are the robot pose and heading. Yellow S is the start
of this search and the magenta line is its traveled trace. Blue labeled circles
are selectable local poses; green F diamonds are selectable frontiers between
known free and unknown space. A marker's tick shows the camera heading after
arrival. For a clue observation, the orange cone outlines the reliable
close-range map region associated with Image 2; its direction continues beyond
the drawn range.

Choose the listed pose whose position and final heading best satisfy the query.
Use visual evidence, the observation cone, coverage, frontiers, distance, and
the existing trace together. Prefer meaningful progress over revisiting covered
space or moving only to reassess. Choose the best position without worrying about obstacles.
Nav2 handles path planning and obstacle avoidance.
Only IDs in valid_candidate_ids may be selected. For move, return
exactly one listed ID; for a terminal decision, return pose_id=null."""

MODE_GUIDANCE = {
    "goal": """MODE: goal
Image 2 is the current camera view. Choose between the local poses and
frontiers to best complete the user's navigation query. R+90, R-90, and R+180
rotate in place; use them only when looking in another direction is itself the
best next step. Return goal_reached when the complete semantic goal is already
satisfied, or blocked when no listed pose can help.""",
    "context": """MODE: context
Image 2 is the clue frame selected by search. The orange outline shows the
reliable close-range portion of that observation; its angular direction
continues beyond the outline. Numbered C poses investigate the clue nearby.
CF is the furthest reachable pose on the observation's center ray before a
blockage. Context mode never offers frontier poses.

Use remaining_waypoints, Image 2, and contextual_clue to choose one nearby
investigative pose. The listed poses already enforce the movement scope. Do not
choose a farther pose merely because it is farther away.

Never return goal_reached because target verification happens after arrival.
Return blocked only when no listed pose can investigate the clue.""",
    "destination": """MODE: destination
Image 2 is the grounded destination clue selected by search. Choose the safe
context pose that best follows that clue. Destination poses use the same cone
logic as local investigation but may reach up to 4 metres to encourage useful
wide movement. No frontier poses are offered.

Never return goal_reached because target verification happens after arrival.
Return blocked only when no listed pose can follow the grounded clue.""",
    "exploration": """MODE: exploration
Image 2 is a fresh center-facing camera view. The map offers the best nearby
inspection pose and EF, the furthest reachable pose continuing forward through
known traversable space. Space between the two poses may not have been visually
inspected even when it is traversable.

Choose EF only when the centered camera view and map show that advancing farther
is likely more useful than inspecting from the nearby pose. Prefer the nearby
pose when objects, openings, side areas, or other useful search detail could be
missed between the choices. Never return goal_reached because target
verification happens after arrival. Return blocked only when neither pose can
support exploration.""",
}


class NavigationError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class NavigateAction:
    VISION_TIMEOUT = 15.0
    NAVIGATION_TIMEOUT = 90.0
    CAMERA_MAX_AGE = 2.0
    JPEG_QUALITY = 75
    MAX_STEPS = 20
    MAX_DURATION = 300.0

    def __init__(
        self, ws, camera, send_robot_command, debug_dir=None,
        request_map_snapshot=None, map_crop_size_m=None,
        see_action=None, send_map_overlay=None,
    ):
        self.ws = ws
        self.camera = camera
        self.send_robot_command = send_robot_command
        self.request_map_snapshot = request_map_snapshot
        self.send_map_overlay = send_map_overlay
        self.see_action = see_action
        path = Path(__file__).resolve().parents[1] / 'tools' / 'vision_oob_tools.json'
        tool_definitions = json.loads(path.read_text())
        self.selection_tool_template = next(
            tool for tool in tool_definitions
            if tool['name'] == 'select_navigation_pose'
        )
        configured_debug_dir = debug_dir or os.environ.get('NAVIGATION_DEBUG_DIR')
        self.debug_dir = Path(configured_debug_dir) if configured_debug_dir else (
            Path(__file__).resolve().parents[1] / 'results' / 'navigation'
        )
        self.active = False
        self.map_crop_size_m = float(
            map_crop_size_m
            if map_crop_size_m is not None
            else os.environ.get('NAVIGATION_MAP_CROP_SIZE_M', '6.0')
        )
        if not 1.0 <= self.map_crop_size_m <= 20.0:
            raise ValueError('navigation map crop size must be between 1 and 20 metres')
        self.action_id = self.target = None
        self.stage = 'idle'
        self.completion_future = None
        self._worker = self._selection_future = self._navigation_future = None
        self._request_id = self._snapshot_id = None
        self._overlay_action_id = None
        self._overlay_revision = 0
        self._owns_goal_overlay = False
        self._goal_search_poses = []
        self._candidates = {}
        self._stop_requested = self._command_sent = False
        self._command_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._result_data = {}
        self._mode = 'goal'
        self._observation = None
        self._max_steps = self.MAX_STEPS
        self._stop_after_first_move = False

    async def start(
        self,
        query,
        action_id,
        *,
        mode='goal',
        observation=None,
        max_steps=None,
        stop_after_first_move=False,
        overlay_action_id=None,
        overlay_revision=0,
    ):
        if self.active:
            return ActionResult(action_id, 'navigate_action', 'already_running', target=query,
                                reason_code='NAVIGATION_BUSY', retryable=True)
        if not isinstance(query, str) or not query.strip():
            return ActionResult(action_id, 'navigate_action', 'failed',
                                reason_code='NAVIGATION_QUERY_REQUIRED')
        if mode not in {'goal', 'context', 'destination', 'exploration'}:
            return ActionResult(action_id, 'navigate_action', 'failed',
                                reason_code='UNKNOWN_NAVIGATION_MODE')
        self.active = True
        self.action_id, self.target = action_id, query.strip()
        self._result_data = {}
        self._mode = mode
        self._observation = (
            dict(observation)
            if isinstance(observation, dict)
            and isinstance(observation.get('jpeg_bytes'), (bytes, bytearray))
            else None
        )
        self._owns_goal_overlay = (
            mode == 'goal' and self.send_map_overlay is not None
        )
        self._overlay_action_id = (
            action_id if self._owns_goal_overlay else overlay_action_id
        )
        self._overlay_revision = (
            0 if self._owns_goal_overlay else int(overlay_revision or 0)
        )
        self._goal_search_poses = []
        self._max_steps = max(
            1, min(self.MAX_STEPS, int(max_steps or self.MAX_STEPS))
        )
        self._stop_after_first_move = stop_after_first_move is True
        self.stage = 'selecting'
        self._stop_requested = self._command_sent = False
        self.completion_future = asyncio.get_running_loop().create_future()
        self._worker = asyncio.create_task(self._run(), name=f'navigate-{action_id}')
        return ActionResult(action_id, 'navigate_action', 'running', target=self.target,
                            outcome='selecting_pose')

    async def start_find_object_step(
        self,
        *,
        target,
        action_id,
        mode,
        observation,
        overlay_action_id,
        overlay_revision,
    ):
        """Select and execute exactly one waypoint for the find-object loop."""
        return await self.start(
            query=FIND_OBJECT_QUERY.format(target=target, mode=mode),
            action_id=action_id,
            mode=mode,
            observation=observation,
            max_steps=1,
            stop_after_first_move=True,
            overlay_action_id=overlay_action_id,
            overlay_revision=overlay_revision,
        )

    async def _run(self):
        try:
            started_at = time.monotonic()
            completed_steps = []
            self._result_data['steps'] = completed_steps

            if self._owns_goal_overlay:
                await self._publish_goal_overlay()

            for step_number in range(1, self._max_steps + 1):
                if time.monotonic() - started_at > self.MAX_DURATION:
                    raise NavigationError(
                        'NAVIGATION_DURATION_LIMIT',
                        'The overall navigation time limit was reached before the goal was confirmed',
                    )

                self.stage = 'selecting'
                self._command_sent = False
                snapshot, camera_jpeg = await self._capture()
                selection = (
                    self._deterministic_selection(snapshot)
                    if self._mode == 'exploration' and len(self._candidates) == 1
                    else await self._select_pose(
                        snapshot, camera_jpeg, step_number
                    )
                )
                decision = selection['decision']
                self._result_data.update(
                    last_selection=selection,
                    last_snapshot_id=self._snapshot_id,
                    step_count=len(completed_steps),
                )

                if decision == 'goal_reached':
                    self._finish('succeeded', 'goal_reached')
                    return
                if decision == 'blocked':
                    raise NavigationError('NAVIGATION_BLOCKED', selection['reason'])
                # Resolve the ID against the exact snapshot shown to vision.
                destination = dict(self._candidates[selection['pose_id']])
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
                self._result_data['destination'] = dict(destination)
                if self._stop_after_first_move:
                    self._finish('succeeded', 'waypoint_reached')
                    return

            raise NavigationError(
                'NAVIGATION_STEP_LIMIT',
                'The waypoint limit was reached before the overall goal was confirmed',
            )
        except asyncio.CancelledError:
            return  # stop() owns cleanup and the terminal result.
        except Exception as error:
            if self._stop_requested:
                return
            error_type = type(error).__name__
            error_message = str(error) or repr(error)
            print(
                "\n[NavigateAction error] "
                f"action_id={self.action_id} stage={self.stage} "
                f"{error_type}: {error_message}"
            )
            traceback.print_exc()
            if self._command_sent:
                await self._stop_motion()
            if not self._stop_requested:
                self._result_data['error_type'] = error_type
                self._result_data['error'] = error_message
                self._result_data['error_stage'] = self.stage
                self._finish('failed', 'navigation_failed',
                             getattr(error, 'code', 'NAVIGATION_ERROR'))

    async def _capture(self, refresh_goal_overlay=True):
        request_map_snapshot = getattr(self, 'request_map_snapshot', None)
        if request_map_snapshot is None:
            raise NavigationError(
                'MAP_STREAM_UNAVAILABLE', 'Map request client is not configured'
            )
        map_request_id = uuid.uuid4().hex
        request = {
            'schema_version': 1,
            'operation': 'snapshot',
            'action_id': self.action_id,
            'request_id': map_request_id,
            'overlay_action_id': self._overlay_action_id,
            'overlay_revision': self._overlay_revision,
        }
        if self._mode != 'exploration':
            request['crop_size_m'] = getattr(self, 'map_crop_size_m', 4.0)
        try:
            snapshot = await request_map_snapshot(request)
        except (asyncio.TimeoutError, TimeoutError, RuntimeError, zmq.ZMQError) as error:
            raise NavigationError('MAP_STREAM_UNAVAILABLE', str(error)) from error

        metadata = snapshot['metadata']
        if str(metadata.get('snapshot_request_id')) != map_request_id:
            raise NavigationError(
                'MAP_REQUEST_MISMATCH', 'Map service returned the wrong request'
            )
        overlay = metadata.get('search_overlay')
        if self._overlay_action_id is not None and not (
            isinstance(overlay, dict)
            and str(overlay.get('action_id')) == str(self._overlay_action_id)
            and int(overlay.get('revision', 0)) >= self._overlay_revision
        ):
            raise NavigationError(
                'MAP_OVERLAY_TIMEOUT',
                'Map service did not render the requested find overlay revision',
            )
        self._snapshot_id = metadata['snapshot_id']
        if (
            refresh_goal_overlay
            and self._remember_goal_pose(metadata.get('robot_pose'))
        ):
            await self._publish_goal_overlay()
            return await self._capture(refresh_goal_overlay=False)

        self._save_map_snapshot(snapshot)
        self._candidates = {
            candidate['id']: dict(candidate)
            for candidate in metadata['candidates']
        }
        if not self._candidates:
            raise NavigationError('NO_NAVIGATION_CANDIDATES', 'No reachable map candidates')

        # Contextual modes already carry the exact clue frame. A single
        # exploration candidate is map-ranked and needs no vision request.
        if self._mode in {'context', 'destination'}:
            return snapshot, None
        if self._mode == 'exploration' and len(self._candidates) == 1:
            return snapshot, None

        if self._mode == 'exploration':
            camera_mover = getattr(self, 'see_action', None)
            if camera_mover is None:
                raise NavigationError(
                    'CAMERA_CENTER_UNAVAILABLE',
                    'Exploration pose comparison requires camera centering',
                )
            centered = await camera_mover.move_to_region(
                region='center',
                action_id=f'{self.action_id}:center-camera',
            )
            if centered.status != 'succeeded':
                raise NavigationError(
                    'CAMERA_CENTER_FAILED',
                    centered.reason_code or 'Could not center the camera',
                )

        frame = self.camera.snapshot()
        if time.monotonic() - frame.captured_at > self.CAMERA_MAX_AGE:
            raise NavigationError('STALE_CAMERA', 'Current camera frame is stale')
        ok, jpeg = await asyncio.to_thread(
            cv2.imencode, '.jpg', frame.tracking_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY]
        )
        if not ok:
            raise NavigationError('CAMERA_ENCODING_FAILED', 'Could not encode camera image')
        return snapshot, jpeg.tobytes()

    def _remember_goal_pose(self, pose):
        if not getattr(self, '_owns_goal_overlay', False) or not isinstance(pose, dict):
            return False
        try:
            observed = {
                'x': float(pose['x']),
                'y': float(pose['y']),
                'yaw': float(pose['yaw']),
                'frame_id': str(pose.get('frame_id') or 'map'),
                'kind': 'navigation_pose',
                'reason': (
                    'navigation_start' if not self._goal_search_poses
                    else 'waypoint_reached'
                ),
                'step': len(self._goal_search_poses),
                'views': [],
            }
        except (KeyError, TypeError, ValueError):
            return False
        self._goal_search_poses.append(observed)
        self._overlay_revision += 1
        return True

    async def _publish_goal_overlay(self):
        await self.send_map_overlay({
            'schema_version': 1,
            'operation': 'set',
            'action_id': self.action_id,
            'revision': self._overlay_revision,
            'frame_id': (
                self._goal_search_poses[-1]['frame_id']
                if self._goal_search_poses else 'map'
            ),
            'mode': 'goal',
            'observation': None,
            'search_poses': list(self._goal_search_poses),
        })

    def _discard_goal_overlay(self):
        if (
            not getattr(self, '_owns_goal_overlay', False)
            or getattr(self, 'send_map_overlay', None) is None
        ):
            return
        asyncio.create_task(self.send_map_overlay({
            'schema_version': 1,
            'operation': 'clear',
            'action_id': self.action_id,
            'revision': self._overlay_revision,
        }))

    def _deterministic_selection(self, snapshot):
        metadata = snapshot.get('metadata') or {}
        pose_id = metadata.get('selected_pose_id')
        if not isinstance(pose_id, str) or pose_id not in self._candidates:
            raise NavigationError(
                'INVALID_DETERMINISTIC_SELECTION',
                'Map planner did not provide one valid selected pose',
            )
        candidate = self._candidates[pose_id]
        return {
            'decision': 'move',
            'pose_id': pose_id,
            'reason': str(
                candidate.get('selection_reason')
                or 'map planner deterministic ranking'
            ),
            'ranking_score': candidate.get('ranking_score'),
            'source': 'map_planner',
        }

    async def _select_pose(self, snapshot, camera_jpeg, step_number):
        self._request_id = uuid.uuid4().hex
        self._selection_future = asyncio.get_running_loop().create_future()
        tool = self._build_selection_tool()
        map_jpeg = snapshot['jpeg_bytes']

        context = {
            'query': self.target,
            'mode': self._mode,
            'step': step_number,
            'candidates': {
                pose_id: {
                    key: candidate.get(key)
                    for key in (
                        'kind', 'distance_m', 'selection_reason',
                        'ray_extension_m', 'uncovered_cell_count',
                        'candidate_type',
                        'remaining_waypoints',
                        'reassessment_limit', 'reassessments_used',
                    )
                    if candidate.get(key) is not None
                }
                for pose_id, candidate in self._candidates.items()
            },
            'observation': (
                {
                    'contextual_clue': self._observation.get('contextual_clue'),
                    'candidate_type': self._observation.get('candidate_type'),
                    'movement_limit': self._observation.get('movement_limit'),
                    'movements_used': self._observation.get('movements_used'),
                    'remaining_waypoints': self._observation.get(
                        'remaining_waypoints'
                    ),
                    'reassessment_limit': self._observation.get(
                        'reassessment_limit'
                    ),
                    'reassessments_used': self._observation.get(
                        'reassessments_used'
                    ),
                }
                if self._observation is not None else None
            ),
        }
        prompt = (
            MAP_GUIDANCE
            + '\n\n'
            + MODE_GUIDANCE[self._mode]
            + '\n\nLIVE CONTEXT\n'
            + json.dumps(context, allow_nan=False)
        )
        vision_jpeg = (
            camera_jpeg
            if self._mode in {'goal', 'exploration'}
            else bytes(self._observation['jpeg_bytes'])
        )
        self._save_debug_inputs(
            vision_jpeg, snapshot['metadata'], context, prompt, tool,
            step_number, self._request_id,
        )
        content = [{'type': 'input_text', 'text': prompt}]
        content.append({
            'type': 'input_image',
            'image_url': 'data:image/jpeg;base64,'
            + base64.b64encode(map_jpeg).decode('ascii'),
        })
        content.append({
            'type': 'input_image',
            'image_url': 'data:image/jpeg;base64,'
            + base64.b64encode(vision_jpeg).decode('ascii'),
        })
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

    def _build_selection_tool(self):
        """Limit the model's pose choices to candidates in the current snapshot."""
        tool = copy.deepcopy(self.selection_tool_template)
        valid_pose_ids = list(self._candidates)
        valid_pose_ids.append(None)

        pose_id_parameter = tool['parameters']['properties']['pose_id']
        pose_id_parameter['enum'] = valid_pose_ids
        return tool

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
        if self._mode != 'goal' and decision == 'goal_reached':
            decision = 'blocked'
            pose_id = None
            args = {
                **args,
                'reason': (
                    'Search target claims are verified by the detector/search action; '
                    + str(args.get('reason') or 'no navigation waypoint selected')
                ),
            }
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


    @staticmethod
    def _atomic_write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)

    def _save_debug_inputs(self, vision_jpeg, map_metadata, context,
                           prompt, tool, step_number, request_id):
        """Persist the exact text, tool, and image order sent to vision."""
        manifest = {
            'action_id': self.action_id,
            'request_id': request_id,
            'snapshot_id': self._snapshot_id,
            'step': step_number,
            'input_images': ['latest_map_crop.jpg', 'latest_vision.jpg'],
            'prompt': prompt,
            'tool': tool,
            'map_metadata': map_metadata,
            'selection_context': context,
        }
        try:
            self._atomic_write(self.debug_dir / 'latest_vision.jpg', vision_jpeg)
            self._atomic_write(
                self.debug_dir / 'latest_request.json',
                json.dumps(manifest, indent=2, allow_nan=False).encode('utf-8'),
            )
            self._result_data['debug_dir'] = str(self.debug_dir)
            self._result_data['last_debug_step'] = step_number
        except (OSError, TypeError, ValueError) as error:
            # Debug output must never prevent the robot from navigating.
            self._result_data['debug_save_error'] = str(error)

    def _save_map_snapshot(self, snapshot):
        """Save both the exact LLM crop and the uncropped map for debugging."""
        try:
            self._atomic_write(
                self.debug_dir / 'latest_map_crop.jpg', snapshot['jpeg_bytes']
            )
            self._atomic_write(
                self.debug_dir / 'latest_map_full.jpg',
                snapshot.get('full_jpeg_bytes', snapshot['jpeg_bytes']),
            )
        except OSError as error:
            self._result_data['debug_save_error'] = str(error)

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
        self._discard_goal_overlay()
        self._owns_goal_overlay = False
        for future in (self._selection_future, self._navigation_future):
            if future is not None and not future.done():
                future.cancel()
        if not self.completion_future.done():
            self.completion_future.set_result(result)
        return result
