# System watchdog

`system_watchdog` starts automatically with `robot/bringup.launch.py` and
publishes one `std_msgs/msg/String` JSON report per second on
`/system/watchdog`.

After sourcing `BIG_BRAIN/setup.sh`, inspect one complete report with:

```bash
health
```

Without the alias:

```bash
ros2 topic echo --once --full-length /system/watchdog
```

Example payload:

```json
{"ok":0,"seq":12,"up":"26/28","g":{"micro":{"imu_raw":"ok"},"nav":{"scan":"stale:1.4s"},"core":{"micro_agent":"no_node"}}}
```

The `g` object contains every configured check, grouped into `micro`,
`bridge`, `nav`, `layers`, and `core`. `ok` is `1` only when every check is healthy and
`up` is the healthy/total count.

Status values:

- `ok`: expected ROS endpoints exist and, for a stream, data is fresh.
- `no_pub` / `no_sub`: a required publisher or real consumer is missing.
- `no_msg`: endpoints exist but the watchdog has never received the stream.
- `stale:<age>s`: the stream stopped updating.
- `no_node`: a required ROS node is absent.
- `no_service`: a required plugin service is absent.
- `wait_*`: the same condition during the startup grace period.

Edit `BIG_BRAIN/src/robot/config/watchdog.yaml` to add topics or change stale
timeouts. A `stream` entry verifies publisher presence, downstream subscriber
presence, and message freshness. An `endpoint` entry only verifies graph
connectivity, which is appropriate for idle command topics and latched maps.

The `layers` group treats the Nav2 costmaps as pipelines rather than checking
only the final combined grid. It checks the local and global debug grid for
each range, obstacle, and STVL layer, both combined costmap streams, both STVL
`voxel_grid` streams, and the four STVL mark/clear toggle services. The
sensor checks also require three consumers of each range stream (local range
layer, global range layer, and collision monitor), two consumers of the ToF
cloud (the two obstacle layers), and four consumers of the depth cloud (STVL
mark and clear buffers in both costmaps). Inflation has no independent ROS
endpoint, so fresh combined costmaps are its available external health signal.
