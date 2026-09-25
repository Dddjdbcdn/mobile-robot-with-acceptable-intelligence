"""Use repeated OOB vision assessments and Nav2 waypoints to reach a goal."""
import asyncio
import base64
import copy
import json
import math
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

SEMANTIC_HEADINGS = {
    "forward": 0.0,
    "forward_left": math.pi / 4.0,
    "left": math.pi / 2.0,
    "back_left": 3.0 * math.pi / 4.0,
    "backward": math.pi,
    "back_right": -3.0 * math.pi / 4.0,
    "right": -math.pi / 2.0,
    "forward_right": -math.pi / 4.0,
}


MAP_GUIDANCE = """GENERAL MAP GUIDANCE
The map is robot-relative: the red robot always points UP. UP is forward in
the current camera view, LEFT/RIGHT match the image, and DOWN is behind the
robot. Black cells are occupied, light cells are known traversable space,
and gray cells are unknown or unavailable. For object finding mode, light-blue tint
means one camera observation; pink tint means repeated observations;
untinted traversable space has not been viewed.

For normal navigation mode, The cyan CURRENT VIEW cone is the obstacle-clipped center field of view that
corresponds to the current camera image. Yellow S is the start and the magenta
line is the traveled trace. Blue labeled circles are selectable local poses;
green F diamonds are frontiers; H means hold position and only rotate. A ray
that meets an obstacle ends at its furthest safe rendered candidate. For a clue
observation, the orange outline is the reliable map region of that clue frame.

Choose both a listed pose and a semantic final heading. Heading names use the
displayed map: forward=UP, left=LEFT, right=RIGHT, backward=DOWN, with diagonal
combinations between them. The heading controls where the base and centered
camera face after arrival; Nav2 still chooses the path.

Use visual evidence, the observation cone, distance, and the existing trace to
make direct goal progress. This is navigation, not active mapping: uncertainty
alone is not a reason to move, rotate, inspect frontiers, or gather extra views.
Do not trade task completion for greater confidence when the current map already
supports a reasonable choice. Explicitly compare the current evidence with the
original evidence and prior intent. Follow the prior intent unless concrete new
evidence proves the goal is reached, blocked, or a different move is necessary.
Only IDs in the tool enum may be selected. A move requires pose_id and heading;
terminal decisions require both to be null. Use move_and_finish when reaching
the selected pose directly completes a bounded command such as nudge, move
slightly, back up, or rotate. Use ordinary move only when the goal genuinely
requires another decision after arrival, not merely to gain confidence."""

MODE_GUIDANCE = {
    "goal": """MODE: goal
Choose between local poses and frontiers to best complete the user's navigation
query. For geometric goals such as the middle, side, or corner of a room, trust
the known free-space layout and choose the most direct plausible local pose;
the camera need not prove geometric symmetry. A frontier is a last resort only
when the requested destination clearly lies beyond known space or no local pose
can make direct goal progress. Never visit a frontier merely to estimate the
room more precisely.

H holds the current position. Use it only when the requested final heading is
itself the goal or one new view can resolve concrete contradictory evidence.
Do not perform a multi-heading verification scan. If the user explicitly asks
for visual confirmation, use the arrival view and at most one purposeful H
rotation. After reaching the intended goal pose, return goal_reached unless the
new evidence shows a specific reason that it is wrong; residual uncertainty is
not such a reason. For a bounded displacement that one listed pose fulfills,
return move_and_finish so the robot stops after that Nav2 goal. Return blocked
when no listed pose can make useful progress.""",
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
    # One planning assessment, one Nav2 movement, then one terminal assessment.
    MAX_STEPS = 1
    MAX_ASSESSMENTS = 2
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
        self._initial_map_jpeg = None
        self._initial_vision_jpeg = None
        self._navigation_history = []

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
        self._initial_map_jpeg = None
        self._initial_vision_jpeg = None
        self._navigation_history = []
        self._max_steps = max(
            1, min(self.MAX_STEPS, int(max_steps or self.MAX_STEPS))
        )
        self._stop_after_first_move = stop_after_first_move is True
        if self._stop_after_first_move:
            self._max_steps = 1
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

            # A normal semantic action gets exactly one planning call and one
            # terminal arrival call. One-shot bounded/search moves finish after
            # the first call and never enter the verification phase.
            assessment_limit = (
                1 if self._stop_after_first_move else self.MAX_ASSESSMENTS
            )
            for assessment_number in range(1, assessment_limit + 1):
                if time.monotonic() - started_at > self.MAX_DURATION:
                    raise NavigationError(
                        'NAVIGATION_DURATION_LIMIT',
                        'The overall navigation time limit was reached before the goal was confirmed',
                    )

                self.stage = 'selecting'
                self._command_sent = False
                snapshot, camera_jpeg = await self._capture()
                selection = await self._select_pose(
                    snapshot, camera_jpeg, assessment_number
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
                if len(completed_steps) >= self._max_steps:
                    raise NavigationError(
                        'NAVIGATION_STEP_LIMIT',
                        'The terminal arrival assessment requested another move',
                    )
                # Resolve the ID against the exact snapshot shown to vision.
                destination = self._resolve_destination(selection, snapshot)
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
                    'step': len(completed_steps) + 1,
                    'assessment': assessment_number,
                    'snapshot_id': self._snapshot_id,
                    'selection': selection,
                    'destination': destination,
                    'robot_status': event.get('status'),
                })
                self._navigation_history.append({
                    'step': len(completed_steps),
                    'assessment': assessment_number,
                    'pose_id': selection['pose_id'],
                    'heading': selection['heading'],
                    'reasoning': selection['reasoning'],
                    'next_intent': selection['next_intent'],
                    'destination': {
                        key: destination[key]
                        for key in ('x', 'y', 'yaw')
                    },
                    'result': 'waypoint_reached',
                })
                self._result_data['step_count'] = len(completed_steps)
                self._result_data['destination'] = dict(destination)
                if decision == 'move_and_finish':
                    self._finish('succeeded', 'goal_reached')
                    return
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

        # Contextual modes already carry the exact clue frame.
        if self._mode in {'context', 'destination'}:
            return snapshot, None

        if self._mode in {'goal', 'exploration'}:
            camera_mover = getattr(self, 'see_action', None)
            if camera_mover is None:
                raise NavigationError(
                    'CAMERA_CENTER_UNAVAILABLE',
                    'Navigation map/camera comparison requires camera centering',
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

    @staticmethod
    def _normalize_yaw(yaw):
        return math.atan2(math.sin(yaw), math.cos(yaw))

    def _resolve_destination(self, selection, snapshot):
        """Combine a sampled position with the model's robot-relative heading."""
        destination = dict(self._candidates[selection['pose_id']])
        robot_pose = (snapshot.get('metadata') or {}).get('robot_pose') or {}
        try:
            robot_yaw = float(robot_pose['yaw'])
            offset = SEMANTIC_HEADINGS[selection['heading']]
        except (KeyError, TypeError, ValueError) as error:
            raise NavigationError(
                'INVALID_NAVIGATION_HEADING',
                'Map snapshot cannot resolve the selected semantic heading',
            ) from error
        destination['sampled_yaw'] = destination.get('yaw')
        destination['heading'] = selection['heading']
        destination['yaw'] = self._normalize_yaw(robot_yaw + offset)
        return destination

    def _build_selection_context(self, step_number):
        observation = None
        if self._observation is not None:
            observation = {
                key: self._observation.get(key)
                for key in (
                    'contextual_clue', 'candidate_type', 'movement_limit',
                    'movements_used', 'remaining_waypoints',
                    'reassessment_limit', 'reassessments_used',
                )
            }
        return {
            'query': self.target,
            'mode': self._mode,
            'completion_mode': (
                'single_move' if self._stop_after_first_move else 'verify_goal'
            ),
            'assessment': step_number,
            'maximum_assessments': (
                1 if self._stop_after_first_move else self.MAX_ASSESSMENTS
            ),
            'maximum_moves': self._max_steps,
            'moves_completed': len(self._navigation_history),
            'moves_remaining': self._max_steps - len(self._navigation_history),
            'history': list(self._navigation_history),
            'previous_intent': (
                self._navigation_history[-1]['next_intent']
                if self._navigation_history else None
            ),
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
            'observation': observation,
        }

    def _build_selection_prompt(self, context, step_number):
        if context['completion_mode'] == 'single_move':
            sequence_guidance = (
                'This is a one-shot bounded movement. Select the one pose and '
                'heading that fulfills it, use move_and_finish, and do not plan '
                'a later reassessment.'
            )
        elif context['moves_remaining'] == 0:
            sequence_guidance = (
                'This is call 2 of 2: the final arrival assessment. Images 1 '
                'and 2 are the original state; Images 3 and 4 are the state '
                'after the one permitted movement. Compare them against the '
                'recorded arrival condition. Return goal_reached when the goal '
                'is reasonably satisfied; otherwise return blocked with the '
                'specific contradiction. This call is terminal: do not request '
                'another move, correction, rotation, or exploratory view.'
            )
        elif step_number == 1:
            sequence_guidance = (
                'This is call 1 of 2 and the only planning call. Choose the one '
                'pose and heading most likely to fulfill the complete goal. '
                'The next call can only judge the arrival; it cannot make a '
                'correction. State a concrete arrival condition, not an '
                'information-gathering or multi-step plan.'
            )
        else:
            sequence_guidance = (
                'Images 1 and 2 are the original state. Images 3 and 4 are the '
                'current state after the recorded moves. Check whether the prior '
                'intent succeeded before choosing another move. If the previous '
                'move targeted the intended goal pose and there is no concrete '
                'contradiction, declare goal_reached now. Do not move or rotate '
                'again merely to become more certain.'
            )
        return (
            MAP_GUIDANCE
            + '\n\n'
            + MODE_GUIDANCE[self._mode]
            + '\n\nSEQUENCE GUIDANCE\n'
            + sequence_guidance
            + '\n\nLIVE CONTEXT\n'
            + json.dumps(context, allow_nan=False)
        )

    def _build_selection_content(
        self, prompt, map_jpeg, vision_jpeg, step_number
    ):
        if self._initial_map_jpeg is None:
            self._initial_map_jpeg = bytes(map_jpeg)
            self._initial_vision_jpeg = bytes(vision_jpeg)
        if step_number == 1:
            image_pairs = (
                ('CURRENT MAP (Image 1)', map_jpeg),
                ('CURRENT VISION (Image 2)', vision_jpeg),
            )
        else:
            image_pairs = (
                ('ORIGINAL MAP (Image 1)', self._initial_map_jpeg),
                ('ORIGINAL VISION (Image 2)', self._initial_vision_jpeg),
                ('CURRENT MAP (Image 3)', map_jpeg),
                ('CURRENT VISION (Image 4)', vision_jpeg),
            )
        content = [{'type': 'input_text', 'text': prompt}]
        for label, jpeg_bytes in image_pairs:
            content.extend((
                {'type': 'input_text', 'text': label},
                {
                    'type': 'input_image',
                    'image_url': 'data:image/jpeg;base64,'
                    + base64.b64encode(jpeg_bytes).decode('ascii'),
                },
            ))
        return content

    async def _select_pose(self, snapshot, camera_jpeg, step_number):
        self._request_id = uuid.uuid4().hex
        self._selection_future = asyncio.get_running_loop().create_future()
        tool = self._build_selection_tool()
        map_jpeg = snapshot['jpeg_bytes']
        context = self._build_selection_context(step_number)
        prompt = self._build_selection_prompt(context, step_number)
        vision_jpeg = (
            camera_jpeg
            if self._mode in {'goal', 'exploration'}
            else bytes(self._observation['jpeg_bytes'])
        )
        self._save_debug_inputs(
            vision_jpeg, snapshot['metadata'], context, prompt, tool,
            step_number, self._request_id,
        )
        content = self._build_selection_content(
            prompt, map_jpeg, vision_jpeg, step_number
        )
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
        """Expose only decisions valid for this phase of the two-call contract."""
        tool = copy.deepcopy(self.selection_tool_template)
        properties = tool['parameters']['properties']
        terminal_assessment = (
            not self._stop_after_first_move
            and len(self._navigation_history) >= self._max_steps
        )
        if terminal_assessment:
            properties['decision']['enum'] = ['goal_reached', 'blocked']
            properties['pose_id']['enum'] = [None]
            properties['heading']['enum'] = [None]
        else:
            properties['pose_id']['enum'] = [*self._candidates, None]
            if self._stop_after_first_move:
                properties['decision']['enum'] = ['move_and_finish', 'blocked']
            else:
                properties['decision']['enum'] = [
                    'move', 'goal_reached', 'blocked'
                ]
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
        heading = args.get('heading')
        if self._mode != 'goal' and decision == 'goal_reached':
            decision = 'blocked'
            pose_id = None
            heading = None
            args = {
                **args,
                'reasoning': (
                    'Search target claims are verified by the detector/search action; '
                    + str(args.get('reasoning') or 'no navigation waypoint selected')
                ),
            }
        move_decisions = {'move', 'move_and_finish'}
        if decision not in move_decisions | {'goal_reached', 'blocked'}:
            future.set_exception(NavigationError(
                'INVALID_NAVIGATION_DECISION', 'Vision returned an unknown decision'
            ))
            return
        if decision == 'move_and_finish' and not self._stop_after_first_move:
            future.set_exception(NavigationError(
                'INVALID_NAVIGATION_DECISION',
                'move_and_finish is only valid for a one-shot bounded movement',
            ))
            return
        if (
            decision in move_decisions
            and not self._stop_after_first_move
            and len(self._navigation_history) >= self._max_steps
        ):
            future.set_exception(NavigationError(
                'INVALID_NAVIGATION_DECISION',
                'The second navigation assessment must be terminal',
            ))
            return
        if 'pose_id' not in args or (pose_id is not None and (
                not isinstance(pose_id, str) or pose_id not in self._candidates)):
            future.set_exception(NavigationError('INVALID_POSE_ID', 'Vision returned an unknown pose ID'))
            return
        if (decision in move_decisions) != (pose_id is not None):
            future.set_exception(NavigationError(
                'INVALID_NAVIGATION_DECISION',
                'move decisions require a pose ID; terminal decisions require pose_id=null',
            ))
            return
        if decision in move_decisions:
            if heading not in SEMANTIC_HEADINGS:
                future.set_exception(NavigationError(
                    'INVALID_NAVIGATION_HEADING',
                    'move requires one valid semantic heading',
                ))
                return
        elif heading is not None:
            future.set_exception(NavigationError(
                'INVALID_NAVIGATION_HEADING',
                'terminal decisions require heading=null',
            ))
            return
        reasoning = str(args.get('reasoning') or '').strip()
        next_intent = str(args.get('next_intent') or '').strip()
        if not reasoning or not next_intent:
            future.set_exception(NavigationError(
                'INVALID_NAVIGATION_RATIONALE',
                'Every decision requires explicit reasoning and a next-intent summary',
            ))
            return
        future.set_result({
            'decision': decision,
            'pose_id': pose_id,
            'heading': heading,
            'reasoning': reasoning,
            'next_intent': next_intent,
            'reason': reasoning,
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
