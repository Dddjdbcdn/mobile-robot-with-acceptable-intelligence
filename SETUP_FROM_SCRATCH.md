# Robot clean-room setup: Ubuntu 24.04 and ROS 2 Jazzy

This is the canonical start-to-finish procedure for rebuilding the physical
robot on a new SSD. Run the sections in order. Commands assume this exact
checkout location because several current configuration files still contain
absolute paths:

```text
/home/nguyendang/ROS2/full-stack-mobile-robot-with-acceptable-intelligence
```

Official references:

- ROS 2 Jazzy deb installation: <https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html>
- ROS Python environments: <https://docs.ros.org/en/jazzy/How-To-Guides/Using-Python-Packages.html>
- micro-ROS setup: <https://github.com/micro-ROS/micro_ros_setup>
- STM32 micro-ROS utility: <https://github.com/micro-ROS/micro_ros_stm32cubemx_utils>
- Slamtec ROS 2 driver: <https://github.com/Slamtec/sllidar_ros2>
- Orbbec Astra ROS 2 driver: <https://github.com/orbbec/ros2_astra_camera>
- OpenVINO system requirements: <https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html>
- OpenVINO Intel GPU setup: <https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html>
- OpenAI Realtime model: <https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini>

## 1. Project map

```text
full-stack-mobile-robot-with-acceptable-intelligence/
├── BIG_BRAIN/                    Ubuntu/ROS 2 runtime
│   ├── src/robot/                launch, URDF, Nav2, bridges, control
│   ├── src/custom_nav2_plugins/  robot-specific Nav2 plugins
│   ├── src/yolo_vision/          YOLO source and model export tools
│   └── src/<third parties>/      restored from third_party.repos
├── MICRO_ROS/                    STM32G474 firmware and Makefile
├── SMALL_BRAIN/                  realtime voice/vision application
│   ├── actions/                  robot behavior implementations
│   ├── database/                 identity, memory, in-process state
│   ├── services/                 camera, audio, tracking, AI inference
│   ├── tools/                    OpenAI function schemas
│   ├── utilities/                shared application helpers
│   └── vision_models/            export tools plus external model assets
├── patches/                      local changes to pinned upstream repos
├── third_party.repos             exact upstream URLs and revisions
├── MODEL_ASSETS.sha256           required external model inventory
└── .gitignore                    Git/source boundary
```

Runtime flow:

```text
STM32 sensors/motors
  └─ UART 921600 ─> micro-ROS agent ─> ROS topics
                                          ├─ ros2_control / EKF / Nav2
RPLIDAR C1 ────────────────────────────────┤
Orbbec Astra ──────────────────────────────┤
                                          └─ BIG_BRAIN bridges
                                              ├─ TCP 5555 request/reply
                                              ├─ TCP 5556 robot events
                                              ├─ TCP 5557 commands
RGB camera ─> SMALL_BRAIN ── TCP 5558 ───────> YOLO
     └─ audio + OpenVINO + OpenAI Realtime
```

Important ROS interfaces:

- STM32 publishes `/stm32/imu_msg`, `/stm32/ultrasonic_msg`,
  `/stm32/tof_raw_data`, `/stm32/debug_msg`, `/stm32/pwm_msg`, and
  `/stm32/wheel_states`.
- STM32 subscribes to `/stm32/wheel_commands`, `/stm32/servo_pan`, and
  `/stm32/servo_tilt`.
- The bridge publishes `/imu`, `/ultrasonic/left`, `/camera_tof`,
  `/ultrasonic/right`, and `/tof_pointcloud`.
- The lidar publishes `/scan`; Astra publishes `/camera/depth/points`.
- Motion commands use `/diff_drive_controller/cmd_vel`.

## 2. Non-negotiable migration backup

Do this before removing or formatting the old SSD. A GitHub clone of the current
repository is not yet a complete backup: the working tree contains untracked
first-party files, external models, nested Git repositories, and local upstream
patches.

Connect an external drive with at least 50 GB free, then copy the entire tree,
including hidden nested `.git` directories:

```bash
rsync -aHAX --info=progress2 \
  /home/nguyendang/ROS2/full-stack-mobile-robot-with-acceptable-intelligence/ \
  /path/to/external-backup/full-stack-mobile-robot/
```

Save audit data separately:

```bash
cd /home/nguyendang/ROS2/full-stack-mobile-robot-with-acceptable-intelligence

git status --short > /path/to/external-backup/root-git-status.txt
git diff --binary > /path/to/external-backup/root-worktree.patch
apt-mark showmanual > /path/to/external-backup/apt-manual.txt
```

Verify the backup before proceeding:

```bash
du -sh \
  /home/nguyendang/ROS2/full-stack-mobile-robot-with-acceptable-intelligence \
  /path/to/external-backup/full-stack-mobile-robot

rsync -aHAXn --checksum \
  /home/nguyendang/ROS2/full-stack-mobile-robot-with-acceptable-intelligence/ \
  /path/to/external-backup/full-stack-mobile-robot/
```

The second `rsync` must not report files that still need copying. Never use
`git clean` or `git reset --hard` as a substitute for organizing this checkout.

## 3. AI hardware gate

The unchanged code requires a supported Intel GPU:

- `BIG_BRAIN/src/yolo_vision/yolo_vision/yolo_node.py` uses
  `device="intel:gpu"`.
- `SMALL_BRAIN/main.py` and its inference services use `device="GPU"`.

The previously inspected i5-14400F/NVIDIA host exposed only `CPU` to OpenVINO.
An SSD or Ubuntu reinstall does not change that. Before expecting the complete
AI stack to run, choose one of these paths:

1. Install a supported Intel Arc GPU and the Intel compute runtime.
2. Change the inference devices to `CPU` and accept slower inference.
3. Port the inference stack to an NVIDIA-capable backend.

The rest of the ROS and STM32 stack can still be installed and tested without
the AI accelerator.

## 4. Install Ubuntu 24.04

1. Install Ubuntu Desktop 24.04 LTS, 64-bit, using UEFI/GPT.
2. Prefer the username `nguyendang`; it preserves the current absolute paths.
3. Allocate at least 100 GB free. A 512 GB SSD is a comfortable target for
   models, Docker images, builds, bags, and logs.
4. Enable networking and third-party drivers during installation.
5. Complete updates and reboot:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

Use Ubuntu Desktop because the SMALL_BRAIN GUI path expects `DISPLAY=:0` and
Qt XCB. If windows fail under Wayland, select an "Ubuntu on Xorg" session at
login.

## 5. Install ROS 2 Jazzy

Configure locale and the Ubuntu Universe repository:

```bash
sudo apt update
sudo apt install -y locales software-properties-common curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo add-apt-repository universe
```

Install the current ROS apt-source package:

```bash
ROS_APT_SOURCE_VERSION="$(
  curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest |
  grep -F '"tag_name"' |
  awk -F'"' '{print $4}'
)"

curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")_all.deb"

sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt update
sudo apt full-upgrade -y
```

Install ROS and system dependencies:

```bash
sudo apt install -y \
  ros-jazzy-desktop ros-dev-tools \
  python3-pip python3-venv python3-dev \
  python3-colcon-common-extensions python3-rosdep python3-vcstool \
  build-essential cmake ninja-build pkg-config git git-lfs curl wget rsync zstd \
  ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
  ros-jazzy-slam-toolbox ros-jazzy-robot-localization \
  ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
  ros-jazzy-controller-manager ros-jazzy-joint-state-broadcaster \
  ros-jazzy-diff-drive-controller ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-spatio-temporal-voxel-layer \
  ros-jazzy-image-tools ros-jazzy-rqt-image-view \
  ros-jazzy-sensor-msgs-py ros-jazzy-tf2-geometry-msgs \
  ros-jazzy-tf2-sensor-msgs ros-jazzy-rmw-cyclonedds-cpp \
  ros-jazzy-teleop-twist-keyboard \
  libgflags-dev libgoogle-glog-dev libusb-1.0-0-dev \
  libuvc-dev libeigen3-dev libudev-dev libasio-dev libtinyxml2-dev \
  portaudio19-dev ffmpeg v4l-utils libgl1 libglib2.0-0t64 \
  gcc-arm-none-eabi libnewlib-arm-none-eabi \
  stlink-tools openocd gdb-multiarch \
  flex bison libncurses-dev clang-tidy clang-format usbutils \
  docker.io clinfo
```

Initialize rosdep. If `rosdep init` reports that it was already initialized,
continue:

```bash
sudo rosdep init
rosdep update
```

Source Jazzy automatically:

```bash
printf '\nsource /opt/ros/jazzy/setup.bash\n' >> /home/nguyendang/.bashrc
source /opt/ros/jazzy/setup.bash
```

Use `/usr/bin/python3` for the project environments. Do not use Conda for ROS
nodes; the Python interpreter must remain compatible with the apt-installed ROS
packages.

## 6. Configure users, Docker, and Intel GPU support

```bash
sudo usermod -aG dialout,video,audio,render,docker nguyendang
sudo systemctl enable --now docker
sudo reboot
```

The `docker` group is root-equivalent; only trusted users should be members.
After reboot:

```bash
groups
docker run --rm hello-world
```

If using a supported Intel GPU, install the runtime and reboot again:

```bash
sudo apt install -y \
  ocl-icd-libopencl1 intel-opencl-icd \
  intel-level-zero-gpu level-zero clinfo
sudo usermod -aG render nguyendang
sudo reboot
```

Verify that a GPU device exists:

```bash
clinfo -l
```

## 7. Clone first-party source

```bash
mkdir -p /home/nguyendang/ROS2
cd /home/nguyendang/ROS2

git clone \
  https://github.com/Dddjdbcdn/full-stack-mobile-robot-with-acceptable-intelligence.git

cd /home/nguyendang/ROS2/full-stack-mobile-robot-with-acceptable-intelligence
export ROBOT_ROOT="$PWD"
```

The repository must first be cleaned and pushed using the policy in section 18.
Until that has happened, restore the old-SSD backup rather than trusting a fresh
GitHub clone.

## 8. Restore pinned third-party source

All external source revisions are recorded in `third_party.repos`:

```bash
cd "$ROBOT_ROOT"
vcs import . < third_party.repos
```

Keep only the three intentionally overridden Nav2 packages:

```bash
git -C BIG_BRAIN/src/nav2_src sparse-checkout init --cone
git -C BIG_BRAIN/src/nav2_src sparse-checkout set \
  nav2_behavior_tree nav2_costmap_2d nav2_planner
```

Apply the versioned robot-specific patches:

```bash
git -C BIG_BRAIN/src/nav2_src apply \
  ../../../patches/nav2-jazzy.patch

git -C BIG_BRAIN/src/ros2_astra_camera apply \
  ../../../patches/astra-camera.patch

git -C BIG_BRAIN/src/sllidar_ros2 apply \
  ../../../patches/sllidar-c1.patch

git -C BIG_BRAIN/src/topic_based_ros2_control apply \
  ../../../patches/topic-based-ros2-control.patch

git -C MICRO_ROS/micro_ros_stm32cubemx_utils apply \
  ../../patches/micro-ros-stm32.patch
```

Every command should complete without rejected hunks. Verify revisions and
changes:

```bash
vcs status .
```

Do not replace these pinned checkouts with arbitrary latest branches.

## 9. Restore the external model bundle

The models are deliberately excluded from Git. Several runtime files exceed
GitHub's normal 100 MiB file limit. GitHub recommends Releases, Git LFS, or an
external file service for large binaries:

<https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github>

The preferred project policy is:

1. Package the exact required files listed in `MODEL_ASSETS.sha256`.
2. Upload the archive as a versioned GitHub Release asset or to stable object
   storage.
3. Record the download URL and release version in this section.
4. Extract it at the repository root.
5. Verify every file with SHA-256.

Create the bundle on the known-working computer:

```bash
cd "$ROBOT_ROOT"
tar --zstd -cf robot-models-2026-08-20.tar.zst \
  --files-from <(awk '{print $2}' MODEL_ASSETS.sha256)
sha256sum robot-models-2026-08-20.tar.zst
```

On the new SSD, download or copy that archive, then:

```bash
cd "$ROBOT_ROOT"
tar --zstd -xf /path/to/robot-models-2026-08-20.tar.zst
sha256sum -c MODEL_ASSETS.sha256
```

Every line must report `OK`. A permanent model-bundle URL still needs to be
filled in after the first release is uploaded.

## 10. Create Python environments

Create a dedicated environment for SMALL_BRAIN:

```bash
cd "$ROBOT_ROOT"
/usr/bin/python3 -m venv SMALL_BRAIN/venv
source SMALL_BRAIN/venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r SMALL_BRAIN/requirements-runtime.txt

SAM2_BUILD_CUDA=0 python -m pip install -e \
  SMALL_BRAIN/vision_models/sam2_tools/sam2

# Preload the tokenizer used by the pinned GroundingDINO configuration.
python -c \
  'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("bert-base-uncased")'

deactivate
```

Create a separate YOLO environment. Source ROS first so the venv-launched node
can import `rclpy` from Jazzy:

```bash
cd "$ROBOT_ROOT"
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 -m venv BIG_BRAIN/src/yolo_vision/.venv
source BIG_BRAIN/src/yolo_vision/.venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r BIG_BRAIN/src/yolo_vision/requirements-runtime.txt
deactivate
```

Initialize mutable memory from its committed template:

```bash
cp -n SMALL_BRAIN/database/memory.example.json \
  SMALL_BRAIN/database/memory.json
```

Validate imports and accelerator visibility:

```bash
source /opt/ros/jazzy/setup.bash

BIG_BRAIN/src/yolo_vision/.venv/bin/python -c \
  'import rclpy, ultralytics, openvino as ov; print(ov.Core().available_devices)'

SMALL_BRAIN/venv/bin/python -c \
  'import cv2, torch, openvino as ov; print(cv2.__version__, torch.__version__, ov.Core().available_devices)'
```

The unchanged full AI runtime requires a GPU to appear in both OpenVINO device
lists.

## 11. Configure USB/udev rules

### STM32 ST-Link serial

The current launch file expects:

```text
/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066BFF485270535067113035-if02
```

Create `/etc/udev/rules.d/99-stm32-low-latency.rules` with:

```udev
ACTION=="add", SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="374b", MODE:="0660", GROUP:="dialout", RUN+="/bin/stty -F /dev/%k 921600 raw -echo -crtscts -ixon -ixoff"
ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="374b", ATTR{power/control}="on"
```

If `/dev/serial/by-id/` shows a different ST-Link serial, update both:

- `BIG_BRAIN/src/robot/launch/hardware.launch.py`
- `BIG_BRAIN/setup.sh`

The ST-Link probe serial used by `st-flash` is currently
`066BFF485270535067113035`.

### RPLIDAR C1

Create `/etc/udev/rules.d/rplidar.rules` with:

```udev
KERNEL=="ttyUSB*", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE:="0660", GROUP:="dialout", SYMLINK+="rplidar"
```

### Orbbec Astra

Install the rules supplied by the pinned driver:

```bash
cd "$ROBOT_ROOT/BIG_BRAIN/src/ros2_astra_camera/astra_camera/scripts"
sudo bash install.sh
```

Reload all rules, then unplug and reconnect the devices:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Inventory the hardware:

```bash
lsusb
ls -l /dev/serial/by-id/
ls -l /dev/rplidar
st-info --probe
v4l2-ctl --list-devices
arecord -l
aplay -l
```

## 12. Build BIG_BRAIN

Do not copy old `build`, `install`, or `log` directories to the new OS.

```bash
cd "$ROBOT_ROOT/BIG_BRAIN"
source /opt/ros/jazzy/setup.bash

rosdep update
rosdep install --from-paths src --ignore-src -r -y

colcon build \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
```

Verify the critical packages:

```bash
ros2 pkg prefix robot
ros2 pkg prefix micro_ros_agent
ros2 pkg prefix sllidar_ros2
ros2 pkg prefix astra_camera
ros2 pkg prefix custom_nav2_plugins
ros2 pkg prefix topic_based_ros2_control
```

## 13. Build and flash MICRO_ROS

The firmware targets an STM32G474CEU6, uses FreeRTOS, and transports micro-ROS
through USART1 DMA at 921600 baud.

Build the static micro-ROS library from the patched utility:

```bash
cd "$ROBOT_ROOT/MICRO_ROS"

docker run --rm -it \
  -v "$PWD:/project" \
  --env MICROROS_LIBRARY_FOLDER=micro_ros_stm32cubemx_utils/microros_static_library \
  microros/micro_ros_static_library_builder:jazzy
```

Build and inspect the firmware:

```bash
make clean
make -j"$(nproc)"
arm-none-eabi-size build/DJ_AMR_CUBEMX.elf
st-info --probe
```

Flash it:

```bash
st-flash --reset write build/DJ_AMR_CUBEMX.bin 0x08000000
```

## 14. Test hardware independently

Lift the drive wheels off the floor and keep a physical emergency stop within
reach.

In every ROS terminal:

```bash
cd "$ROBOT_ROOT/BIG_BRAIN"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

### STM32 and micro-ROS

```bash
ros2 run micro_ros_agent micro_ros_agent serial \
  -b 921600 \
  --dev /dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066BFF485270535067113035-if02
```

Reset the board after the agent starts. In another terminal:

```bash
ros2 node list
ros2 topic hz /stm32/imu_msg
ros2 topic hz /stm32/wheel_states
ros2 topic hz /stm32/ultrasonic_msg
ros2 topic hz /stm32/tof_raw_data
```

Expected rates are approximately 50, 50, 20, and 10 Hz.

### Lidar

```bash
ros2 launch sllidar_ros2 sllidar_c1_launch.py serial_port:=/dev/rplidar
ros2 topic hz /scan
```

### Astra

```bash
ros2 launch astra_camera astra.launch.xml
ros2 topic hz /camera/depth/points
```

### RGB camera and audio

```bash
v4l2-ctl --list-devices
ffplay /dev/video0
arecord -l
aplay -l
```

The SMALL_BRAIN camera index is currently hard-coded to `0`.

## 15. Run the full robot

### Terminal A: ROS runtime

Run from `BIG_BRAIN`; the YOLO launch command currently uses relative paths:

```bash
cd "$ROBOT_ROOT/BIG_BRAIN"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot bringup.launch.py
```

For live SLAM plus navigation:

```bash
ros2 launch robot bringup.launch.py slam:=true nav2:=true
```

For a saved map plus AMCL:

```bash
ros2 launch robot bringup.launch.py amcl:=true nav2:=true
```

Do not use `sim_run` or `sim_auto_run` yet. They reference the currently absent
`sim_bringup.launch.py`.

### Terminal B: SMALL_BRAIN

Use a graphical login and provide the OpenAI key only through the environment:

```bash
cd "$ROBOT_ROOT/SMALL_BRAIN"
source venv/bin/activate

export OPENAI_API_KEY='replace-with-your-key'
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb

python main.py
```

Never commit the populated key or save it in shell history on a shared machine.
The application needs internet access and paid access to the configured OpenAI
Realtime model. The first setup also downloads the pinned GroundingDINO
`bert-base-uncased` tokenizer unless that Hugging Face cache is restored.

## 16. Mapping workflow

Start mapping:

```bash
cd "$ROBOT_ROOT/BIG_BRAIN"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot bringup.launch.py slam:=true nav2:=true
```

Drive slowly and cover all accessible space. Save the completed map:

```bash
ros2 run nav2_map_server map_saver_cli \
  -f "$ROBOT_ROOT/BIG_BRAIN/src/robot/map/my_map"
```

Before pushing a map to a public repository, decide whether it reveals a private
home, lab, or facility layout. Store sensitive maps outside public Git.

## 17. Acceptance checklist

- [ ] `clinfo -l` shows the intended accelerator, or CPU fallback was selected.
- [ ] Both Python validation commands complete.
- [ ] `sha256sum -c MODEL_ASSETS.sha256` reports all files `OK`.
- [ ] `/dev/rplidar` exists.
- [ ] The expected STM32 `/dev/serial/by-id/` entry exists.
- [ ] `st-info --probe` detects the correct ST-Link.
- [ ] The intended RGB camera is `/dev/video0`.
- [ ] Microphone and speaker appear in `arecord -l` and `aplay -l`.
- [ ] `joint_state_broadcaster` is active.
- [ ] `diff_drive_controller` is active.
- [ ] STM32 topics publish at their expected rates.
- [ ] `/scan`, `/camera/depth/points`, `/imu`, and `/tof_pointcloud` publish.
- [ ] Wheel signs, encoder signs, and positive angular direction are correct.
- [ ] Lidar, Astra, IMU, and base TF frames align.
- [ ] Servo pan/tilt direction and limits are safe.
- [ ] `/yolo/detections` publishes after SMALL_BRAIN sends frames.
- [ ] TCP ports 5555 through 5558 are listening locally.
- [ ] Nav2 exposes `/navigate_to_pose` when `nav2:=true`.
- [ ] A low-speed navigation test succeeds with wheels first lifted, then on a
      clear floor.
- [ ] Zero-velocity and physical emergency-stop behavior are verified.

Useful commands:

```bash
ros2 node list
ros2 topic list
ros2 control list_controllers
ros2 topic hz /stm32/wheel_states
ros2 topic hz /imu
ros2 topic hz /scan
ros2 topic hz /camera/depth/points
ros2 topic hz /tof_pointcloud
ros2 topic echo --once /stm32/debug_msg
ros2 action list
ss -lntp | grep -E '5555|5556|5557|5558'
```

To stop safely, publish zero velocity while ROS is still alive, stop
SMALL_BRAIN, then terminate the BIG_BRAIN launch. `BIG_BRAIN/setup.sh` provides
a `stop` alias after it is sourced.

## 18. What belongs on GitHub

### Commit and push

- `README.md`, this guide, `.gitignore`, `.env.example`.
- `third_party.repos`, all files in `patches/`, and `MODEL_ASSETS.sha256`.
- `BIG_BRAIN/src/robot/` source, launch files, URDF, configuration, behavior
  trees, package metadata, maps that are safe to publish, and tests.
- `BIG_BRAIN/src/custom_nav2_plugins/` source and package metadata.
- `BIG_BRAIN/src/yolo_vision/` Python/package/export source and
  `requirements-runtime.txt`; exclude models and its `.venv`.
- `BIG_BRAIN/setup.sh`.
- `MICRO_ROS/` STM32 source, headers, linker scripts, CubeMX `.ioc`, Makefile,
  startup code, HAL/FreeRTOS source required by the firmware, and setup script.
- `SMALL_BRAIN/` first-party Python source, tool schemas, static identity after
  privacy review, `memory.example.json`, setup script, and
  `requirements-runtime.txt`.
- First-party model conversion/export scripts and documentation located outside
  ignored third-party checkouts.
- Small calibration/configuration artifacts that are required at runtime and
  contain no credentials or private data.

### Do not push

- `OPENAI_API_KEY`, `.env`, credentials, private keys, certificates, or tokens.
- `build/`, `install/`, `log/`, compiler objects, firmware binaries, or linker
  `.map` files.
- Python `venv/`, `.venv/`, `__pycache__/`, test caches, or IDE state.
- OpenVINO caches, kernel error logs, captured images, result directories, ROS
  bags, or mutable `SMALL_BRAIN/database/memory.json`.
- Model weights, checkpoints, exported OpenVINO `.bin` files, or the model
  bundle archive.
- Cloned third-party repositories represented by `third_party.repos`.
- Private maps of homes, labs, or facilities.

### Store outside normal Git

- The versioned model archive described in section 9.
- Raw datasets and training runs.
- Large ROS bags and camera recordings.
- Backups of mutable robot memory.
- Sensitive environment maps.

GitHub warns above 50 MiB and blocks normal Git objects larger than 100 MiB.
Keep the source repository comfortably below 1 GB. Git LFS is an alternative,
but it has storage/bandwidth implications and every contributor must install it:

<https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage>

## 19. One-time cleanup before the first clean push

First make the full external backup from section 2. The new `.gitignore` does
not automatically untrack generated files or old weights that were committed in
the past.

### Remove the unpushed oversized commit from the branch

Audit note from 2026-08-20: local `main` is two commits ahead of `origin/main`:
`703faaa` followed by `42b5d50`. The first commit contains a 353 MB CLIP weight
and two YOLO files larger than 100 MiB. They are not present in `origin/main`.
The second commit deletes them, but that is insufficient because a push still
includes the oversized parent commit.

Confirm that the branch is still exactly one commit ahead and zero behind:

```bash
git fetch origin
git rev-list --left-right --count origin/main...HEAD
```

Proceed only if it prints `0 2` and the external backup exists. Preserve a local
recovery branch, then squash the two unpushed commits back into the index while
retaining the current working tree:

```bash
git branch backup/pre-clean-42b5d50
git reset --soft origin/main
```

This intentionally rewrites only the unpushed local branch. Do not use `--hard`.
Do not delete the recovery branch until the clean replacement commit has been
pushed, cloned into a separate directory, built, and checked.

### Remove generated files from the index

Preview tracked files that now match `.gitignore`:

```bash
cd "$ROBOT_ROOT"
git ls-files -ci --exclude-standard
```

After reviewing that list, remove those paths from Git's index while preserving
the local copies. The audit found 184 such paths in `42b5d50`, mostly firmware
build output and Python bytecode:

```bash
git ls-files -ci --exclude-standard -z |
  xargs -0 -r git rm -r --cached --
```

The following command must then print nothing:

```bash
git ls-files -ci --exclude-standard
```

Review all existing deletions individually. Some are intentional source
replacements; others may be accidental. In particular, do not blindly accept
the current deleted SMALL_BRAIN source or old FACE tree without deciding whether
each feature is still required.

Check for unexpectedly large tracked files:

```bash
git ls-files -z |
  xargs -0 -r du -b |
  sort -nr |
  head -n 30
```

Check for likely secrets before staging:

```bash
git grep -nEI \
  'OPENAI_API_KEY[[:space:]]*=|api[_-]?key[[:space:]]*[:=]|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY'
```

Stage source deliberately rather than using a blind `git add -A` initially:

```bash
git add \
  README.md SETUP_FROM_SCRATCH.md .gitignore .env.example \
  third_party.repos MODEL_ASSETS.sha256 patches \
  BIG_BRAIN/src/robot BIG_BRAIN/src/custom_nav2_plugins \
  BIG_BRAIN/src/yolo_vision BIG_BRAIN/setup.sh \
  MICRO_ROS \
  SMALL_BRAIN/actions SMALL_BRAIN/database SMALL_BRAIN/services \
  SMALL_BRAIN/test_code SMALL_BRAIN/tools SMALL_BRAIN/utilities \
  SMALL_BRAIN/vision_models SMALL_BRAIN/main.py \
  SMALL_BRAIN/setup.sh SMALL_BRAIN/requirements-runtime.txt
```

Then inspect exactly what will be pushed:

```bash
git status --short
git diff --cached --stat
git diff --cached
```

Only after that review:

```bash
git commit -m "make robot setup reproducible"

# This must print no files at or above 100 MiB.
git rev-list --objects origin/main..HEAD |
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
  awk '$1 == "blob" && $3 >= 104857600 {print $3, $4}'

git push origin main
```

## 20. Reproducibility contract

A clean machine is considered reproducible only when all of these are true:

1. The root Git commit is recorded.
2. `third_party.repos` imports without floating branches.
3. Every file in `patches/` applies cleanly.
4. Python dependencies install from the two committed requirements files.
5. The external model bundle has a stable versioned URL.
6. `sha256sum -c MODEL_ASSETS.sha256` succeeds.
7. The ROS workspace and STM32 firmware build from clean output directories.
8. The hardware and runtime acceptance checklist passes.

Whenever code, firmware, dependencies, patches, or models change, update their
manifest, checksum, version, and this guide in the same pull request.
