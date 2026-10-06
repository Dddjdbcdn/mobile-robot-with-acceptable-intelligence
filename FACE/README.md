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
