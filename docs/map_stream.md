# Map images for exploration

`LLMRosBridge` starts `robot/map_stream.py` automatically. No extra ROS
executable is needed. It refreshes expensive semantic map analysis in a
background worker at 1 Hz, then draws the current TF pose and local candidates
onto that cached background at a target 5 Hz. Publisher timing is logged every
five seconds. SMALL BRAIN's `CameraStream` receives this independently of the selected
USB/Astra camera and replaces these inspection files on incoming updates:

- `SMALL_BRAIN/results/map/latest.jpg`
- `SMALL_BRAIN/results/map/latest.json`

The files remain on disk when the stream stops. Check their modification time;
`map_snapshot()` and the LLM tool reject cached images older than 3 seconds.
Writes replace each file atomically, but the JPEG/JSON pair is not a filesystem
transaction. The JSON and LLM input text carry the snapshot ID; the image has no title.

## Inputs and coordinates

Static map-stream settings:

| Setting | Value |
| --- | --- |
| map topic | `/map` |
| global costmap topic | `/global_costmap/costmap` |
| base frame | `base_footprint` |
| image size | `1024` pixels maximum |
| publish rate | `5 Hz` |
| analysis rate | `1 Hz` |
| live camera-FOV debug layer | disabled |
| camera horizontal FOV | `60 degrees` |
| camera reliable range | `2 m` |
| image endpoint | `tcp://127.0.0.1:5559` |
| overlay endpoint | `tcp://127.0.0.1:5560` |

The renderer prefers fresh TF from the map frame to `base_footprint`, supporting
SLAM and localization. A fresh pose topic is the fallback. Map, global costmap,
and pose must share a frame; differing grid origins, origin rotations, and
resolutions are supported. Pose freshness is limited to 2 seconds, costmap to
3 seconds. ROS timestamps use the node clock, including `use_sim_time`.

The global costmap configuration now sends full grids every publication. Restart
Nav2 with the updated configuration; the renderer does not consume incremental
`costmap_updates`. The static occupancy map may be old while the robot is moving.
A missing/stale costmap or pose suppresses publication with a throttled log.

The live camera-FOV debug layer is disabled. The `live_camera_fov` metadata
still records pan, horizontal FOV, and center yaw.

## Find-object overlay ownership

During `find_object`, GoalExecutor owns one ordered `search_poses` list. Each
unique `(x, y, robot yaw)` stores a `views` list of camera pan/tilt captures.
The overlay also carries the selected clue
observation and current mode (`context` or `exploration`). It sends versioned,
full-state `set` messages over a dedicated ZeroMQ PUSH socket.
`MapImageStream` binds the matching PULL socket, plans and renders the matching
candidate snapshot, and removes it after GoalExecutor sends `clear`. SMALL
BRAIN can override its connection with `MAP_OVERLAY_ENDPOINT`; it must match
the static overlay endpoint.

MapLogic computes coverage as an occupancy-grid-sized count array. It raycasts
each captured camera FOV against occupied cells and increments every visible map
cell. The renderer does not calculate visibility: it projects that grid onto the
image, draws once-covered cells light blue and repeated cells pale pink, then
draws the route, robot, and candidate labels.

Consequently, dark text, pose circles, trails, and other annotations can never
terminate coverage rays. Navigation receives the resulting JPEG as-is and reads
structured coverage-count and per-side uncovered-fraction fields; it does not
reconstruct coverage from pixels.

Every streamed frame identifies the applied state under
`metadata.search_overlay` with `action_id`, `revision`, `mode`, observation
count, and pose count. NavigateAction waits for the requested action/revision
before using the snapshot, preventing a decision against older planning state.
In context mode, MapLogic creates candidates directly inside the selected clue
cone at three ranges and three bearings, keeps up to six reachable choices, and
adds one reachable backward viewpoint (`CB`) that still looks toward the clue.
An orange outline and ray show the source observation. In exploration mode,
MapLogic samples reachable cells every 0.5 meters, shortlists cells near dense
uncovered coverage, tests eight camera headings, and publishes only the pose and
yaw that reveal the most uncovered known-free cells. Frontiers remain visible
for debugging and become selectable when no sampled view has gain. The selected
candidate reports `uncovered_cell_count` and `uncovered_ahead_fraction`.
GoalExecutor always captures the center, left, and right views at each
inspection pose. Coverage is used to rank exploration viewpoints, not to gate
camera sweep directions.

The entire JPEG is the map, with a compact color legend in the top-left corner
and no title, sidebar, or coordinate table. Set `MapRenderSettings.show_legend`
to `False` to hide the legend. The image preserves the map aspect ratio and
shows world +X right and +Y upward. Candidate labels show their numbers. The map includes 1 meter grid lines, a red
robot position/heading marker, and a blue 1 meter sampling circle. The legend
and exact candidate coordinates are supplied in the LLM input text and JSON.

Candidates use **0.5, 1, 2, and 3 meter rings** with **45-degree angular
steps**. Angles start at body forward and increase counterclockwise: 0, 45, 90,
135, 180, 225, 270, 315 degrees. This gives 32 translation proposals before
reachability filtering. Three `R` poses rotate in place by +90, -90, or 180
degrees. The robot's own
position is excluded. Each translation candidate yaw faces radially away from the robot. Candidate
metadata includes `distance_m` and signed `relative_angle_degrees`. Rotation
poses stay at the robot position and only change yaw; their labels are omitted
from the image to avoid overlap. IDs count unfiltered positions from the inner ring outward, starting forward on
each ring, so blocked samples leave gaps. Wall, corner, center, and exit
estimation are not used in this renderer.

Blocked, unknown, and disconnected proposals are omitted. Clearance includes a
conservative circular footprint and cell discretization margin. The default
circle covers the square footprint plus padding in `nav2.yaml`; change it when
the robot footprint changes. The cost threshold is 65 on the OccupancyGrid
0–100 scale.

## Code ownership

The implementation is split into two modules:

- `BIG_BRAIN/src/robot/robot/map_logic.py`: occupancy/costmap processing,
  reachability, frontiers, doors, rooms, local samples, FOV/coverage sampling,
  context filtering, and deterministic exploration ranking.
- `BIG_BRAIN/src/robot/robot/map_renderer.py`: standalone OpenCV background and
  overlay rendering from already-computed candidates and grid masks.
- `BIG_BRAIN/src/robot/robot/map_stream.py`: MapLogic/renderer orchestration,
  overlay command consumption, ROS/TF integration, metadata, JPEG encoding, and
  ZeroMQ publication.

The renderer calls `MapLogic.prepare()` for the slow map analysis and
`MapLogic.plan_candidates()` for every current-pose snapshot. The rendered
candidate IDs and `metadata.candidates` therefore come from the same plan.
`selection_policy=deterministic` includes one `selected_pose_id`; otherwise
NavigateAction may ask vision to choose among the supplied IDs.

IDs are only valid for their accompanying snapshot. Four-connected grid
reachability is **not a Nav2 path check**. Nav2 remains responsible for path
planning and obstacle avoidance.

## LLM access

The realtime agent can call `get_map_snapshot` with `include_camera: true`
(default). It adds the latest map image, candidate metadata, and current selected
camera image to the conversation. Camera and map are asynchronous; the tool
labels the source and freshness and notes that camera pan can differ from body
heading. Use `include_camera: false` for map-only inspection.

Python access: `camera.map_snapshot()` returns `jpeg_bytes`, decoded BGR `image`,
`metadata`, `received_at` (local monotonic clock), and `age_seconds`.

## Transport and deployment

The default endpoint is bound to loopback on the NUC. To display the map on a PC
without moving either robot process off the NUC, create a local SSH forward from
the PC (replace the SSH destination):

```bash
ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes \
  -L 5559:127.0.0.1:5559 user@nuc-address
```

Clone this repository on the PC, then run the standalone viewer in another PC
terminal:

```bash
python3 SMALL_BRAIN/map_stream_viewer.py
```

Press `q` or Escape to close it. The standalone viewer does not write images
or metadata to disk. Use `--no-window` for stream measurements without an
OpenCV window. The PC needs `python3-zmq`, `python3-opencv`, and
`python3-numpy`, or SMALL BRAIN's existing Python environment.

The viewer keeps only the newest available map, so a slow display skips old
frames instead of accumulating delay. Its status line reports receive Hz,
publisher overlay and analysis times, cache age, queue skips, sequence gaps,
and approximate end-to-end age. Age is meaningful only when the PC and robot
clocks are synchronized. The
SSH tunnel is recommended because the
ZeroMQ stream has no authentication or encryption of its own.

For a direct LAN connection instead, change `IMAGE_ENDPOINT` to BIG BRAIN's
specific private interface address and pass that address to
`map_stream_viewer.py --endpoint`. The full SMALL BRAIN runtime also remains
supported: set `MAP_STREAM_ENDPOINT` to that address and
`MAP_SNAPSHOT_PATH` to override its local inspection JPEG path.

Protocol: ZeroMQ PUB/SUB, two frames: `map/image` and a payload containing a
4-byte big-endian JSON length, UTF-8 JSON, then JPEG bytes. The publisher timer targets 5 Hz while semantic analysis refreshes at 1 Hz.
The live robot pose and local candidates are redrawn for every publication;
frontiers, doors, rooms, reachability, and the base raster use the newest
completed analysis cache. Images are sent to the LLM only when its tool is
called. Static stream settings can be changed at the top of `map_stream.py`.

Rebuild the ROS `robot` package (or use an existing symlink build) and restart the
bridge and SMALL BRAIN. `python3-opencv` is now declared as a ROS runtime dependency.

Offline checks, without ROS hardware or motion:

```bash
python3 -m unittest discover -s tests -p 'test_map_stream.py' -v
```
