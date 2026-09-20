# USB camera image tuning

The Small Brain applies optional native V4L2 controls when it opens the USB
camera. With no variables set, the camera keeps its driver defaults/current
values. These settings do not affect the Astra camera stream.

## Connected camera controls

The current `USB Camera: USB Camera` on `/dev/video0` reports:

| Environment variable | V4L2 control | Supported values | Driver default |
|---|---|---:|---:|
| `USB_CAMERA_BRIGHTNESS` | `brightness` | -64–64 | 0 |
| `USB_CAMERA_CONTRAST` | `contrast` | 0–100 | 38 |
| `USB_CAMERA_SATURATION` | `saturation` | 0–100 | 64 |
| `USB_CAMERA_HUE` | `hue` | -180–180 | 0 |
| `USB_CAMERA_GAMMA` | `gamma` | 100–500 | 400 |
| `USB_CAMERA_SHARPNESS` | `sharpness` | 0–100 | 93 |
| `USB_CAMERA_BACKLIGHT_COMPENSATION` | `backlight_compensation` | 0–121 | 1 |
| `USB_CAMERA_AUTO_EXPOSURE` | `auto_exposure` | `auto`, `manual`, 3, or 1 | auto (3) |
| `USB_CAMERA_EXPOSURE_TIME_ABSOLUTE` | `exposure_time_absolute` | 50–10000 | 166 |
| `USB_CAMERA_EXPOSURE_DYNAMIC_FRAMERATE` | `exposure_dynamic_framerate` | true/false | false |
| `USB_CAMERA_WHITE_BALANCE_AUTOMATIC` | `white_balance_automatic` | true/false | true |
| `USB_CAMERA_WHITE_BALANCE_TEMPERATURE` | `white_balance_temperature` | 2800–6500, step 10 | 4600 |
| `USB_CAMERA_POWER_LINE_FREQUENCY` | `power_line_frequency` | `disabled`, `50hz`, `60hz`, or 0–2 | 50hz (1) |

Setting an exposure time automatically selects manual exposure unless
`USB_CAMERA_AUTO_EXPOSURE` is explicitly set. Setting a white-balance
temperature similarly disables automatic white balance.

## Starting point for a backlit subject

Start with automatic exposure, reset the unusually dark brightness value, and
raise backlight compensation gradually:

```bash
export USB_CAMERA_AUTO_EXPOSURE=auto
export USB_CAMERA_BRIGHTNESS=0
export USB_CAMERA_BACKLIGHT_COMPENSATION=20
python3 SMALL_BRAIN/main.py
```

Try backlight compensation in small increments (for example 10, 20, 30). A
larger value may reveal the foreground while washing out the sunlit background.
If automatic exposure remains unstable, switch to a fixed exposure and tune it:

```bash
export USB_CAMERA_AUTO_EXPOSURE=manual
export USB_CAMERA_EXPOSURE_TIME_ABSOLUTE=300
export USB_CAMERA_BRIGHTNESS=0
export USB_CAMERA_BACKLIGHT_COMPENSATION=20
python3 SMALL_BRAIN/main.py
```

Do not set every control at once. First tune exposure/backlight compensation,
then brightness and gamma, and finally color controls. Restart Small Brain after
changing environment variables.

For live experiments without restarting, use the same native controls directly:

```bash
v4l2-ctl -d /dev/video0 --set-ctrl=brightness=0
v4l2-ctl -d /dev/video0 --set-ctrl=backlight_compensation=20
v4l2-ctl -d /dev/video0 --get-ctrl=brightness,backlight_compensation
```

Inspect the device again after replacing the camera because supported controls
and ranges are hardware-specific:

```bash
v4l2-ctl -d /dev/video0 --list-ctrls-menus
```
