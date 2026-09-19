# Find object action

`find_object(target=...)` is a single long-running goal that owns the camera and
robot motion until the object is approached, the searchable space is exhausted,
a safety limit is reached, or `stop_goal` is called.

## Search policy

The executable sequence is highlighted above `FindLoopConfig` in
`SMALL_BRAIN/cognition/goal_executor.py`. Editable waypoint-selection LLM prompts
are grouped at the top of `SMALL_BRAIN/actions/navigate_action.py` under
`IMPORTANT LLM PROMPTS`; prompt text is intentionally kept separate from the
websocket and image-encoding code.

1. Center the camera and check local YOLO detections first for supported targets.
2. Assess the center image with the VLM. A definitive `found` result stops the
   scan immediately. For `candidate` or `not_found`, capture only the eligible
   side views and reassess center plus those sides together. Left and right are
   60-degree pans from center, so all three views span a conservative 180 degrees.
   YOLO is checked after every settled camera move and still stops the scan early.
3. Assessment returns `found` for a definitive target, `candidate` for one
   useful contextual clue, or `not_found` when neither target nor clue exists.
4. For `candidate`, search aims at the selected frame and navigation receives
   only that JPEG, its clue, and the map. For `not_found`, discovery receives the
   coverage-rendered map without camera JPEGs.
5. At an investigative waypoint, center the camera and check YOLO. A left or
   right pan is captured only when at least 50% of that direction's independent,
   obstacle-clipped 60-degree fan remains uncovered within 2 meters.
   Contextual investigation can continue while grounded clues remain.
6. When the first side has no clues, turn the body 180 degrees and apply the same
   forward-coverage test before deciding between a centered check and a sweep.
7. When both local searches are exhausted, MapLogic deterministically ranks
   unshaded local viewpoints and useful rotations, then light-blue viewpoints
   scanned once. It sends exactly one selected pose to NavigateAction, so no
   waypoint-selection model call is made. Only after known local space is
   exhausted does it rank and select a frontier; all frontiers remain rendered
   as debug markers.
   Every arrival gets a centered-camera check. Any arrival, including a frontier,
   translation, or rotation, may add a sweep when the forward-coverage test passes.
8. Once found, tracking tolerates brief glitches and attempts same-view detector
   reacquisition without recentering. When tracking becomes stable, it saves the
   target's map point, range, camera pose, observer pose, and confidence.
9. Tracking loss while the robot approaches is expected and does not fail the
   approach. After navigation finishes, stop any remaining motion-era tracker,
   aim from the new robot pose toward the saved target point, and run a one-frame
   search. If that misses, run the general three-row sweep. Only then send the
   resulting camera view for approach verification.

## Search memory and candidates

GoalExecutor keeps one ordered `search_poses` list. A robot rotation creates a
new pose, while camera pans and tilts at the same `(x, y, yaw)` are stored in that
pose's `views` list. The same structure supplies route rendering and coverage.
The map passed to each search decision overlays the accumulated, obstacle-clipped
horizontal field of view in light blue, capped at the reliable 2-meter vision
range. Areas covered by two or more captures are pale pink. Every capture
contributes to coverage even though only one clue frame may be sent to navigation.
Candidate generation does not reject previously visited coordinates.
Context mode samples positions directly inside the selected clue cone and adds
one reachable backward viewpoint. Exploration mode samples camera-covered,
reachable map cells (plus the robot's current cell), points each candidate
toward its best uncovered camera view, and chooses the candidate that exposes
the most uncovered grid cells. It uses a frontier only when the covered known
space has no remaining view.

GoalExecutor sends the complete coverage and route state to MapImageStream over
the dedicated overlay socket. MapImageStream raycasts against the raw occupancy
grid, renders coverage before candidate labels, and publishes both the finished
map and structured coverage metrics. NavigateAction waits for that overlay
revision and forwards the finished map to the LLM; it has no JPEG-threshold
raycaster or second overlay renderer. See [Map images for
exploration](map_stream.md#find-object-overlay-ownership) for the protocol.

Memory lasts for the complete `find_object` call. The terminal action result
also includes the memory records and waypoint count for logging.

## Bounds and cancellation

The defaults in `GoalExecutor` are:

- 4 consecutive contextual waypoints per clue chain
- 12 frontier iterations
- 20 total search waypoints
- 600 seconds total search time
- Pan left/right independently when 50% of that 60-degree, 2-meter fan is unshaded

All find-loop controls can be tuned without editing code:

| Environment variable | Default |
| --- | ---: |
| `FIND_MAX_CONTEXT_WAYPOINTS` | `4` |
| `FIND_MAX_EXPLORATION_WAYPOINTS` | `12` |
| `FIND_MAX_TOTAL_WAYPOINTS` | `20` |
| `FIND_MAX_DURATION_SECONDS` | `600` |
| `FIND_CAMERA_CENTER_PAN_DEG` | `95` |
| `FIND_CAMERA_HORIZONTAL_FOV_DEG` | `60` |
| `FIND_SWEEP_TRIGGER_FOV_DEG` | `180` |
| `FIND_SWEEP_SIDE_OFFSET_DEG` | `60` |
| `FIND_CAMERA_RELIABLE_RANGE_M` | `2` |
| `FIND_SWEEP_UNCOVERED_FRACTION` | `0.50` |

The sweep fraction counts uncovered known-free occupancy cells. Occupied cells
terminate visibility rays; the renderer only projects the resulting coverage grid.

`stop_goal`, idle mode, camera-source changes, and process shutdown cancel any
active nested navigation/search/tracking/approach action and stop robot motion.
