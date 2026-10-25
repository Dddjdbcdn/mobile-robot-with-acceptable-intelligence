# DJ Robot FACE

Fullscreen, NUC-local control for starting and stopping the robot without a
client computer.

## Install

```bash
cd FACE
./install.sh
```

The installer adds user services, an application-menu entry, and a GNOME
autostart entry. The FACE window itself runs as a restartable user service so
a UI failure cannot leave the Ubuntu desktop exposed. The installer also
disables notification banners, screen locking,
blanking, desktop icons, and persistent dock display for the dedicated robot
screen. The fullscreen window stays above other desktop windows, and Ubuntu's
update/crash notification popups are suppressed for this dedicated session.

The interface starts automatically after graphical login. To test it in a
normal window:

```bash
python3 FACE/face.py --windowed
```

Press `F11` to toggle fullscreen. `Ctrl+Shift+Q` is the maintenance shortcut
to close the interface.

The interface uses GTK's native XInput2 support for the WaveShare touchscreen.
Start and Stop use a large touch target and respond on initial finger contact,
without requiring a synthetic mouse-button event from the panel.

While the robot is running, FACE becomes a compact Stop panel in the lower
corner so the fullscreen `Robot Vision` camera/debug overlay remains visible.
Stopping or detecting a failure restores the full control screen.

The compact panel also has a `MAP` button. It opens a touchscreen overview from
BIG_BRAIN's local map service, refreshing every two seconds only while the page
is visible; FACE does not need ROS Python libraries. The overview uses the full
screen while preserving its aspect ratio, with status and controls overlaid so
they do not shrink the map. A compact `LOCAL` control switches to the stream's
robot-centred local render and becomes `FULL` to return to the overview. The
adjacent minus and plus controls adjust the local area from 3 m × 3 m through
10 m × 10 m in one-metre steps and are hidden in full-overview mode. These map
controls are stacked on the right.
The status overlay reports compute time, render time, and total end-to-end map
time in milliseconds, plus whole-system CPU usage. The compact `CAMERA`,
`RESET`, and `STOP ROBOT` controls form a vertical lower-right panel, matching
the camera view's vertical `MAP`, `RESET`, and `STOP ROBOT` panel. `RESET` sends
a local ZeroMQ request to BIG_BRAIN's lift/reset coordinator. It stops movement,
clears accumulated behavior and tracking state, resets SLAM and both costmaps,
then reports success only after the spatial reset completes. Lift recovery uses
the same operations but waits for stable upright placement before resetting the
map.

Lift and reset progress is presented as a centered high-priority overlay in
both camera and map modes. It instructs the operator to place a lifted robot
down, reports map/manual reset progress, and confirms `READY` briefly once
movement is safe again; lift state is not duplicated in the debug board.

At the same low display rate, BIG_BRAIN overlays the current robot footprint,
up to 180 fresh LiDAR returns, the active Nav2 goal, and up to 200 points from
the current global plan. LiDAR and footprint data disappear when stale, while
the goal and plan are cleared when navigation completes or is cancelled.

While stopped, the lobby shows the active Wi-Fi connection and provides a
`CHANGE WI-FI` button that opens Ubuntu's network panel. During a run, Wi-Fi
and STM32 firmware health appear in the right-side diagnostics panel. If the
Wi-Fi or Realtime connection is lost, local vision, gestures, ROS control, and
the Stop panel remain active so the operator can stop safely and return to the
lobby to select another network.

## Runtime configuration

Optional service variables can be placed in:

```text
~/.config/dj-robot/environment
```

For example, to start saved-map navigation with the robot:

```text
DJ_ROS_LAUNCH_ARGS=amcl:=true nav2:=true
```

By default the Start button uses `ros2 launch robot bringup.launch.py`, then
starts `SMALL_BRAIN/main.py` after the local ZeroMQ bridge is ready.

Service logs are available with:

```bash
journalctl --user -u dj-ros.service -u dj-small-brain.service
```

The Stop button sends a zero velocity command before shutting down both user
services. It is a controlled software stop, not a safety-rated emergency stop.
