# Map images for exploration

`LLMRosBridge` starts `robot/map_image_stream.py` automatically. No extra ROS
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

ROS parameters on the existing `llm_bridge` node:

| Parameter | Default |
| --- | --- |
| `map_image_map_topic` | `/map` (`nav_msgs/OccupancyGrid`) |
| `map_image_costmap_topic` | `/global_costmap/costmap` (`nav_msgs/OccupancyGrid`, **not** `costmap_raw`) |
| `map_image_pose_topic` | `/amcl_pose` (`geometry_msgs/PoseWithCovarianceStamped`) |
| `map_image_base_frame` | `base_footprint` |
| `map_image_clearance_m` | `0.25` |
| `map_image_sample_radius_m` | `1.0` |
| `map_image_sample_spacing_m` | `0.2` (radial spacing) |
| `map_image_sample_angle_degrees` | `45.0` |
| `map_image_include_rotation_poses` | `true` |
| `map_image_max_size` | `512` pixels on the longest edge |
| `map_image_publish_hz` | `5.0` target pose-overlay/stream rate |
| `map_image_analysis_hz` | `1.0` cached semantic-background refresh rate |
| `map_image_bind` | `tcp://127.0.0.1:5559` |

The renderer prefers fresh TF from the map frame to `base_footprint`, supporting
SLAM and localization. A fresh pose topic is the fallback. Map, global costmap,
and pose must share a frame; differing grid origins, origin rotations, and
resolutions are supported. Pose freshness is limited to 2 seconds, costmap to
3 seconds. ROS timestamps use the node clock, including `use_sim_time`.

The global costmap configuration now sends full grids every publication. Restart
Nav2 with the updated configuration; the renderer does not consume incremental
`costmap_updates`. The static occupancy map may be old while the robot is moving.
A missing/stale costmap or pose suppresses publication with a throttled log.

The entire JPEG is the map, with a compact color legend in the top-left corner
and no title, sidebar, or coordinate table. Set `MapRenderSettings.show_legend`
to `False` to hide the legend. The image preserves the map aspect ratio and
shows world +X right and +Y upward. Candidate labels show their numbers. The map includes 1 meter grid lines, a red
robot position/heading marker, and a blue 1 meter sampling circle. The legend
and exact candidate coordinates are supplied in the LLM input text and JSON.

Candidates use **radial sampling**: rings at **0.2, 0.4, 0.6, 0.8, and 1.0 m**,
with **45-degree angular steps** on each ring. Angles start at body forward and
increase counterclockwise: 0, 45, 90, 135, 180, 225, 270, 315 degrees. This gives
8 poses per ring and 40 translation proposals before obstacle filtering. Seven
additional `R` poses rotate in place by 45-degree increments. The robot's own
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

## Tuning the code

`BIG_BRAIN/src/robot/robot/map_image_stream.py` is organized in processing order:

- `MapRenderSettings`: sampling distances, clearance, thresholds, and image settings.
- `COLORS`: OpenCV BGR palette.
- `Grid`: occupancy grid/world coordinate conversions.
- `MapRenderer._reachable_cells()`: clearance and connectivity filtering.
- `MapRenderer._sample_candidates()`: sample coordinates, numbers, and headings.
- `MapRenderer._make_view()` and `_draw_*()`: map extent and individual overlays.
- `MapRenderer._metadata()`: the structured data accompanying the image.
- `MapImageStream`: ROS subscriptions, freshness checks, timer, and network I/O.

Radius, radial spacing, angular spacing, clearance, and image size also have the ROS parameters listed
above. Defaults for those parameters come from `MapRenderSettings`, so edits to
that settings class affect startup unless overridden. Angular spacing must divide
360 degrees evenly; radius limits omit any partial outer ring. `PUBLISH_PERIOD_SECONDS`
and `JPEG_QUALITY` are at the top of `MapImageStream`. Legend wording lives in
`SeeAction.get_map_snapshot()` in SMALL BRAIN.

IDs are only valid for their accompanying snapshot. Four-connected grid
reachability is **not a Nav2 path check**. Map streaming supplies perception and
proposals. The [navigate_action](navigation.md) can select and execute one local
waypoint using OOB vision; it does not run an autonomous exploration loop.

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

For a direct LAN connection instead, configure `map_image_bind` with BIG
BRAIN's specific private interface address and pass that address to
`map_stream_viewer.py --endpoint`. The full SMALL BRAIN runtime also remains
supported: set `MAP_STREAM_ENDPOINT` to that address and
`MAP_SNAPSHOT_PATH` to override its local inspection JPEG path.

Protocol: ZeroMQ PUB/SUB, two frames: `map/image` and a payload containing a
4-byte big-endian JSON length, UTF-8 JSON, then JPEG bytes. The publisher timer targets 5 Hz while semantic analysis refreshes at 1 Hz.
The live robot pose and local candidates are redrawn for every publication;
frontiers, doors, rooms, reachability, and the base raster use the newest
completed analysis cache. Images are sent to the LLM only when its tool is
called. If overlay time exceeds 200 ms, use a smaller
`map_image_max_size` or lower `map_image_publish_hz`.

Rebuild the ROS `robot` package (or use an existing symlink build) and restart the
bridge and SMALL BRAIN. `python3-opencv` is now declared as a ROS runtime dependency.

Offline checks, without ROS hardware or motion:

```bash
python3 -m unittest discover -s tests -p 'test_map_stream.py' -v
```
