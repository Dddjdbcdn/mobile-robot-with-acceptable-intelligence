# Vision-selected local navigation

The `navigate_action` tool takes a natural-language `query`, for example:

```json
{"query": "Move toward the opening on the left, keeping clear of the chair."}
```

`SMALL_BRAIN/actions/navigate_action.py` keeps one action active across local steps:

1. Read one fresh, atomic map JPEG and candidate table from `CameraStream`.
   MapLogic has already generated, filtered, and ranked candidates for the
   snapshot's selection mode.
2. For normal navigation and contextual clue investigation, send the map plus
   the selected USB/Astra camera frame to OOB vision. For exploration snapshots
   marked `selection_policy=deterministic`, immediately use
   `selected_pose_id` without encoding a camera frame or calling the model.
3. When vision is used, it returns `decision`, `pose_id`, and `reason`.
   `move` requires an ID
   from this snapshot: a numeric local pose, an `F` frontier, or an `R` in-place
   rotation. `goal_reached` and `blocked` require `pose_id: null`.
4. Resolve the ID against the original frozen table. Check a fresh map before
   moving: same frame, robot pose within 5 cm / 0.15 radians, and the selected
   coordinates/yaw still represented by a candidate within those tolerances.
   A new snapshot's ID cannot redirect the goal to another coordinate.
5. Send `navigate_to_pose` through the existing ZeroMQ command callback, using
   `frame_id`, `x`, `y`, and `angle` (the chosen yaw). Nav2 plans and executes it.
6. Wait for the bridge's navigation event with the matching action ID. After `Goal Reached`, capture fresh images and reassess the complete original query inside the same action. No intermediate lifecycle result is sent to the realtime model.
7. Finish successfully only when OOB vision returns `goal_reached`. `blocked`, selection errors, rejection, navigation failure, timeout, or a safety limit produce terminal results with a structured reason.

The existing realtime WebSocket handles OOB replies; `CognitionManager` routes
`select_navigation_pose` to the action. No separate model client is needed.
The images go to the OOB response, not the normal voice conversation.

Before every selection, the exact JPEGs sent to vision and a JSON manifest are
atomically refreshed under `SMALL_BRAIN/results/navigation/` as
`latest_map.jpg`, `latest_vision.jpg`, and `latest_request.json`. Only the newest
loop is retained. Set `NAVIGATION_DEBUG_DIR` to override this directory. Save
failures are reported in the action result but do not stop navigation.

Each navigation call shows one yellow `S` at that call's starting pose and a
magenta trail for motion within that call. Both are cleared and recreated on the
next call; numbered session starts and terminal markers are not retained. During
`find_object`, a light-blue camera-coverage overlay persists for the goal and is
ray-clipped at map obstacles. Context candidates are sampled directly in the clue
cone. Exploration samples camera-covered reachable cells (plus the robot's
current cell) and immediately uses the pose and heading that expose the most
uncovered known-free cells; frontiers are the fallback after coverage is exhausted.
`stop_navigation` cancels this action during either OOB selection or Nav2 movement.
Cancellation during vision prevents later replies from starting movement.
Cancellation during command dispatch waits for the request/reply transaction to
finish before sending stop. Camera switching, idle mode, and shutdown also stop
this action. Other robot and camera actions cannot start while it owns the robot;
it refuses to start while another robot action is active.

Tune `VISION_TIMEOUT` (15 s), `NAVIGATION_TIMEOUT` (90 s), `CAMERA_MAX_AGE` (2 s),
`POSITION_TOLERANCE` (0.05 m), `YAW_TOLERANCE` (0.15 rad), `MAX_STEPS`
(20), and `MAX_DURATION` (300 s) at the top of `NavigateAction`. Map freshness uses `CameraStream.map_snapshot()` (3 s).
Candidate clearance settings remain in BIG BRAIN's `MapRenderSettings`.

Numeric candidates translate to reachable 0.5, 1, 2, and 3 meter endpoints at
45-degree bearings. Three metadata-only `R` candidates provide +90-degree,
-90-degree, and 180-degree in-place turns. During object search, previously used
final rotation headings are filtered out. The selector is instructed to spin when
that exposes an uninspected view and to translate only for a concrete reason.

One call may execute multiple local candidate poses. Reaching an individual waypoint is progress, not completion. The action retains the original query and completed-step history, and only reports success after a fresh OOB assessment confirms the complete goal. `MAX_STEPS` (20) and `MAX_DURATION` (300 seconds) bound the loop.

Restart SMALL BRAIN to load the new tools and action. BIG BRAIN needs its existing
map stream and Nav2 bridge running. No new ROS command is required.

Run simulated checks without moving hardware or calling a model:

```bash
python3 -m unittest discover -s tests -p 'test_navigate_action.py' -v
```
