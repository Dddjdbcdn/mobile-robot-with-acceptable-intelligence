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
3. Assessment returns `found`, `not_found`, or one typed candidate: `visual`,
   `local`, `destination`, or `speculative`. The type is required for a candidate;
   numeric candidate confidence is not used.
4. GoalExecutor owns actionable candidate hypotheses and movement budgets. The
   first candidate type fixes the chain allowance: visual 0, local 2, and
   destination 10. Speculative candidates are ignored and create no actionable
   hypothesis, candidate navigation request, or retry. Their camera views still
   contribute to ordinary coverage. A visual candidate first receives one
   single-frame reassessment from the already-aimed camera. If it remains a
   candidate, GoalExecutor attempts tracking with GroundingDINO enabled and enters
   the normal approach pipeline only when acquisition succeeds. Failed acquisition
   rejects the clue as not found; visual candidates never navigate to a context
   pose. Later frames receive the hypothesis ID, original clue, type, and remaining
   allowance, but no historical images.
5. MapStream copies the hypothesis fields onto every sampled pose. Visual and
   local candidates cannot expose a frontier; destination candidates may expose
   an aligned long-ray or frontier pose. NavigateAction receives the map, the
   latest clue JPEG, and concise structured hypothesis metadata.
6. Exploration first ranks uncovered known-space viewpoints and useful rotations
   with frontiers disabled. When those local views are exhausted, exploration may
   select a frontier. A speculative result does not change this sequence.
7. Coverage exploration sends exactly one deterministic pose to NavigateAction,
   so no waypoint-selection model call is made. Context navigation uses the LLM
   only to choose among poses already filtered by the candidate policy.
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
one reachable backward viewpoint. The sampled poses inherit candidate type,
hypothesis ID, remaining budget, and frontier permission. Exploration mode
samples camera-covered, reachable map cells (plus the robot's current cell),
points each candidate toward its best uncovered camera view, and chooses the
candidate that exposes the most uncovered grid cells. It uses a frontier only
after local coverage is exhausted.

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

- Candidate movement allowances: visual 0, local 2, destination 10; speculative ignored
- 10-waypoint global safety ceiling for any candidate chain
- 20 exploration iterations
- 50 total search waypoints
- 600 seconds total search time
- 60-degree camera FOV and 2-meter reliable coverage range

The sweep fraction counts uncovered known-free occupancy cells. Occupied cells
terminate visibility rays; the renderer only projects the resulting coverage grid.

`stop_goal`, idle mode, camera-source changes, and process shutdown cancel any
active nested navigation/search/tracking/approach action and stop robot motion.
