# Mobile robot setup from scratch

Follow this document from top to bottom on Ubuntu 24.04. Do not copy an old
`build`, `install`, `log`, virtual environment, model cache, or STM32 generated
vendor directory onto the new computer.

The repository contains three parts:

- `BIG_BRAIN`: ROS 2 hardware, control, localization, SLAM, and navigation.
- `SMALL_BRAIN`: camera, audio, OpenVINO vision, YOLO, behavior, and OpenAI.
- `MICRO_ROS`: first-party STM32G474 application code and its CubeMX `.ioc`.

The expected machine is Ubuntu 24.04 with ROS 2 Jazzy. The AI services currently
request an Intel OpenVINO GPU. ROS and the STM32 can still be tested without one,
but AI inference must be changed to CPU if OpenVINO reports only `CPU`.

---

## 1. Project repository

### 1.1 Install the tools needed to clone source

```bash
sudo apt update
sudo apt install -y git python3-vcstool software-properties-common
```

### 1.2 Clone this repository

```bash
mkdir -p "$HOME/ROS2"
cd "$HOME/ROS2"

git clone \
  https://github.com/Dddjdbcdn/mobile-robot-with-acceptable-intelligence.git

cd mobile-robot-with-acceptable-intelligence
export ROBOT_ROOT="$PWD"
```

Whenever this guide says `$ROBOT_ROOT`, it means the cloned repository. In a
new terminal, restore it with:

```bash
cd "$HOME/ROS2/mobile-robot-with-acceptable-intelligence"
export ROBOT_ROOT="$PWD"
```

### 1.3 Clone the latest third-party repositories

`third_party.repos` uses readable branch names. Each setup clones the latest
source from that branch. ROS-specific micro-ROS repositories use `jazzy`; the
others use `main`, a compatibility branch, or the GroundingDINO OpenVINO branch.

| Repository | Installed path | Purpose |
|---|---|---|
| micro_ros_setup (`jazzy`) | `BIG_BRAIN/src/micro_ros_setup` | ROS-side micro-ROS tooling |
| ros2_astra_camera (Jazzy PR branch) | `BIG_BRAIN/src/ros2_astra_camera` | Orbbec Astra camera |
| sllidar_ros2 (`main`) | `BIG_BRAIN/src/sllidar_ros2` | RPLIDAR C1 |
| topic_based_ros2_control (`main`) | `BIG_BRAIN/src/topic_based_ros2_control` | ROS control transport |
| micro-ROS-Agent (`jazzy`) | `BIG_BRAIN/src/uros/micro-ROS-Agent` | STM32-to-ROS serial agent |
| STM32 micro-ROS utils | `MICRO_ROS/micro_ros_stm32cubemx_utils` | firmware static library |
| Depth Anything V2 | `SMALL_BRAIN/.../Depth-Anything-V2` | depth model source |
| GroundingDINO | `SMALL_BRAIN/.../GroundingDINO` | open-vocabulary detection |
| SAM2 | `SMALL_BRAIN/.../sam2` | segmentation model source |

Import all of them from the repository root:

```bash
cd "$ROBOT_ROOT"
vcs import . < third_party.repos
```

`drive_base` is not cloned because this robot uses the standard
`sensor_msgs/JointState` type for wheel commands and states. The Agent requires
`micro_ros_msgs`, but Jazzy provides that dependency as an apt package in
section 2.3, so its source repository is not cloned either.

The official Astra `master` branch does not yet contain its Jazzy/Kilted build
fix. This manifest follows Robert Gruberski's `fix/astra-kilted-build` branch
from upstream pull request #20:

<https://github.com/orbbec/ros2_astra_camera/pull/20>

The separate `patches/astra-camera.patch` only disables the driver's TF
publication so the robot's own TF tree remains authoritative.

### 1.4 Clone only the required Nav2 packages

Navigation2 is one Git repository containing many ROS packages. The three
required packages are directories inside that monorepo, not three separate Git
repositories. Use a shallow, blob-filtered sparse clone so Git downloads the
latest Jazzy source and populates only the selected package directories:

```bash
cd "$ROBOT_ROOT"

git clone \
  --depth 1 \
  --filter=blob:none \
  --sparse \
  --single-branch \
  --branch jazzy \
  https://github.com/ros-navigation/navigation2.git \
  BIG_BRAIN/src/nav2_src

git -C BIG_BRAIN/src/nav2_src sparse-checkout set \
  nav2_behavior_tree nav2_costmap_2d nav2_planner
```

The working tree now contains only:

```text
BIG_BRAIN/src/nav2_src/nav2_behavior_tree/
BIG_BRAIN/src/nav2_src/nav2_costmap_2d/
BIG_BRAIN/src/nav2_src/nav2_planner/
```

Git cone mode may also retain a few small files from the root of the Navigation2
repository. It does not populate the other Nav2 package directories.

### 1.5 Apply the project changes to third-party source

The cloned upstream repositories are changed using the small patch files
committed under `patches/`:

```bash
cd "$ROBOT_ROOT"

git -C BIG_BRAIN/src/nav2_src apply \
  ../../../patches/nav2-jazzy.patch

git -C BIG_BRAIN/src/ros2_astra_camera apply \
  ../../../patches/astra-camera.patch

git -C BIG_BRAIN/src/topic_based_ros2_control apply \
  ../../../patches/topic-based-ros2-control.patch

git -C MICRO_ROS/micro_ros_stm32cubemx_utils apply \
  ../../patches/micro-ros-stm32.patch
```

Confirm that every checkout is on its expected branch and that only the expected
patch changes are present:

```bash
vcs status .
```

This setup deliberately tracks the latest branch versions. An upstream update
can eventually make a patch fail or introduce a build error. If that happens,
update the affected patch for the new upstream source.

---

## 2. Ubuntu and ROS 2 Jazzy requirements

### 2.1 Configure Ubuntu 24.04

ROS 2 Jazzy deb packages officially target Ubuntu 24.04 Noble. Configure a UTF-8
locale and enable Ubuntu Universe:

```bash
sudo apt update
sudo apt install -y locales software-properties-common curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo add-apt-repository universe
```

### 2.2 Add the official ROS 2 apt repository

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
sudo apt upgrade -y
```

Official reference:
<https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html>

### 2.3 Install ROS core and build tools

This group provides ROS Desktop, colcon, rosdep, vcstool, compilers, and Python
environment support:

```bash
sudo apt install -y \
  ros-jazzy-desktop ros-dev-tools \
  ros-jazzy-micro-ros-msgs \
  python3-pip python3-venv python3-dev \
  python3-colcon-common-extensions python3-rosdep python3-vcstool \
  build-essential cmake ninja-build pkg-config git curl wget rsync zstd
```

### 2.4 Install navigation and localization

- Navigation2: path planning, controllers, behavior trees, and recovery.
- SLAM Toolbox: create and use maps.
- Robot Localization: EKF sensor fusion for odometry and IMU data.

```bash
sudo apt install -y \
  ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
  ros-jazzy-slam-toolbox \
  ros-jazzy-robot-localization \
  ros-jazzy-spatio-temporal-voxel-layer
```

### 2.5 Install robot control and descriptions

- ros2_control: controller framework.
- Diff Drive Controller: converts velocity commands into wheel commands.
- Xacro and Robot State Publisher: robot model and TF tree.

```bash
sudo apt install -y \
  ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
  ros-jazzy-controller-manager \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-diff-drive-controller \
  ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-joint-state-publisher-gui
```

### 2.6 Install sensor, transform, and debugging packages

```bash
sudo apt install -y \
  ros-jazzy-image-tools ros-jazzy-rqt-image-view \
  ros-jazzy-sensor-msgs-py \
  ros-jazzy-tf2-geometry-msgs ros-jazzy-tf2-sensor-msgs \
  ros-jazzy-rmw-cyclonedds-cpp \
  ros-jazzy-teleop-twist-keyboard \
  libgflags-dev libgoogle-glog-dev libusb-1.0-0-dev \
  libuvc-dev libeigen3-dev libudev-dev libasio-dev libtinyxml2-dev \
  portaudio19-dev ffmpeg v4l-utils libgl1 libglib2.0-0t64 usbutils
```

`rosdep` will install additional dependencies declared by cloned ROS packages.

### 2.7 Install firmware and container tools

```bash
sudo apt install -y \
  gcc-arm-none-eabi libnewlib-arm-none-eabi \
  stlink-tools openocd gdb-multiarch \
  docker.io \
  flex bison libncurses-dev clang-tidy clang-format
```

### 2.8 Initialize ROS and configure the shell

Run `rosdep init` once. If it says it is already initialized, continue.

```bash
sudo rosdep init
rosdep update

grep -qxF 'source /opt/ros/jazzy/setup.bash' "$HOME/.bashrc" || \
  echo 'source /opt/ros/jazzy/setup.bash' >> "$HOME/.bashrc"

grep -qxF 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' "$HOME/.bashrc" || \
  echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> "$HOME/.bashrc"

source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 --help >/dev/null
```

Use `/usr/bin/python3` for project virtual environments. Do not use Conda with
apt-installed ROS Python packages.

### 2.9 Configure device access and optional Intel GPU support

```bash
sudo usermod -aG dialout,video,audio,render,docker "$USER"
sudo systemctl enable --now docker

sudo apt install -y \
  ocl-icd-libopencl1 intel-opencl-icd clinfo intel-gpu-tools
```

Log out and back in after changing groups. Then verify:

```bash
groups
docker run --rm hello-world
clinfo -l
```

If `clinfo -l` does not show an Intel GPU, ROS can still run, but services that
request `GPU` or `intel:gpu` need a CPU fallback before running.

---

## 3. Build BIG_BRAIN (ROS 2 workspace)

Install package dependencies and build from the `BIG_BRAIN` directory:

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

Verify the packages required by the robot:

```bash
ros2 pkg prefix robot
ros2 pkg prefix custom_nav2_plugins
ros2 pkg prefix micro_ros_agent
ros2 pkg prefix sllidar_ros2
ros2 pkg prefix astra_camera
ros2 pkg prefix topic_based_ros2_control
```

After changing ROS source, rebuild with the same `colcon build` command. Do not
commit the generated `build`, `install`, or `log` directories.

---

## 4. Create the SMALL_BRAIN Python environment

YOLO is part of `SMALL_BRAIN`; it is not a ROS package and does not need a
separate ROS Python environment.

### 4.1 Create and populate the environment

```bash
cd "$ROBOT_ROOT"

/usr/bin/python3 -m venv SMALL_BRAIN/venv
source SMALL_BRAIN/venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r SMALL_BRAIN/requirements-runtime.txt

SAM2_BUILD_CUDA=0 python -m pip install -e \
  SMALL_BRAIN/vision_models/sam2_tools/sam2

python -c \
  'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("bert-base-uncased")'

deactivate
```

### 4.2 Create mutable robot memory

The real memory file is local runtime data and is intentionally not on GitHub:

```bash
cd "$ROBOT_ROOT"
cp -n SMALL_BRAIN/database/memory.example.json \
  SMALL_BRAIN/database/memory.json
```

### 4.3 Restore or export AI models

Model checkpoints and OpenVINO exports are intentionally excluded from Git.
Restore them from your model backup, or use the scripts under
`SMALL_BRAIN/vision_models/` to recreate them.

The application currently expects these paths:

```text
SMALL_BRAIN/vision_models/groundingdino_tools/GroundingDINO/
SMALL_BRAIN/vision_models/groundingdino_tools/models/groundingdino_swint_512x768_onnx.xml
SMALL_BRAIN/vision_models/depthanything_tools/openvino_models/dav2_metric_indoor_vitb_896x504_fp16.xml
SMALL_BRAIN/vision_models/sam2_tools/models/sam2.1_hiera_b+_openvino/
SMALL_BRAIN/vision_models/yolo_tools/yolo11m-pose_openvino_model/
SMALL_BRAIN/vision_models/yolo_tools/yoloe-11m_openvino_model/
```

OpenVINO `.xml` models require their matching `.bin` files in the same
directory. SAM2 and the export tools also require their original checkpoints.

Relevant tools:

```text
SMALL_BRAIN/vision_models/groundingdino_tools/README.md
SMALL_BRAIN/vision_models/depthanything_tools/export_depthanything.py
SMALL_BRAIN/vision_models/sam2_tools/export_sam2_openvino.py
SMALL_BRAIN/vision_models/yolo_tools/export_yolo.py
SMALL_BRAIN/vision_models/yolo_tools/export_yolo_pose.py
```

The model export workflow is still under development. Confirm that an export's
output filename matches the path above before starting the full application.

### 4.4 Validate the environment

```bash
cd "$ROBOT_ROOT"
source SMALL_BRAIN/venv/bin/activate

python -c \
  'import cv2, openvino as ov, torch, ultralytics; print(ov.Core().available_devices)'

deactivate
```

---

## 5. Regenerate and build MICRO_ROS with STM32CubeMX

The repository intentionally stores the `.ioc` and first-party STM32 code, but
not the generated STM32 HAL, CMSIS, FreeRTOS, startup, or linker trees.

### 5.1 Install STM32CubeMX

Download the Linux STM32CubeMX installer from STMicroelectronics and install it.
Ubuntu 24.04 is supported by current STM32CubeMX releases:

<https://www.st.com/en/development-tools/stm32cubemx.html>

Also keep the ARM compiler, Docker, and ST-Link tools installed in section 2.7.

### 5.2 Generate the vendor firmware source

1. Open `MICRO_ROS/DJ_AMR_CUBEMX.ioc` in STM32CubeMX.
2. Do not create a new project.
3. In Project Manager, select `Makefile` as the toolchain.
4. Generate code into the existing `MICRO_ROS` directory.
5. Close STM32CubeMX.

CubeMX can overwrite the Makefile. Restore the repository's micro-ROS-aware
version immediately after generation:

```bash
cd "$ROBOT_ROOT"
git restore MICRO_ROS/Makefile
git status --short
```

Do not restore or discard changes to first-party `Core/` files without reviewing
them. Confirm that CubeMX created:

```bash
test -d MICRO_ROS/Drivers
test -d MICRO_ROS/Middlewares
test -f MICRO_ROS/startup_stm32g474xx.s
test -f MICRO_ROS/STM32G474xx_FLASH.ld
```

These generated files remain local and are ignored by Git.

Delete the TIM7 callback in main.c after building the project from CubeMX (it is ticked in motor.c)

```bash
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  /* USER CODE BEGIN Callback 0 */

  /* USER CODE END Callback 0 */
  if (htim->Instance == TIM7)
  {
    HAL_IncTick();
  }
  /* USER CODE BEGIN Callback 1 */

  /* USER CODE END Callback 1 */
}
```

### 5.3 Build the micro-ROS static library

```bash
cd "$ROBOT_ROOT/MICRO_ROS"

docker run --rm -it \
  -v "$PWD:/project" \
  --env MICROROS_LIBRARY_FOLDER=micro_ros_stm32cubemx_utils/microros_static_library \
  microros/micro_ros_static_library_builder:jazzy
```

### 5.4 Build and flash the STM32

```bash
cd "$ROBOT_ROOT/MICRO_ROS"
make clean
make -j"$(nproc)"
arm-none-eabi-size build/DJ_AMR_CUBEMX.elf
st-info --probe
```

Lift the wheels off the ground before flashing or testing motors:

```bash
st-flash --reset write build/DJ_AMR_CUBEMX.bin 0x08000000
```

---

## 6. Configure robot hardware permissions

### 6.1 STM32 ST-Link serial

The current project expects this device:

```text
/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066BFF485270535067113035-if02
```

Create `/etc/udev/rules.d/99-stm32-low-latency.rules`:


```bash
sudo nano /etc/udev/rules.d/99-stm32-low-latency.rules
```

```udev
ACTION=="add", SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="374b", MODE:="0660", GROUP:="dialout", RUN+="/bin/stty -F /dev/%k 921600 raw -echo -crtscts -ixon -ixoff"
ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="374b", ATTR{power/control}="on"
```

If your ST-Link serial differs, update it in:

```text
BIG_BRAIN/src/robot/launch/hardware.launch.py
BIG_BRAIN/setup.sh
```

### 6.2 RPLIDAR C1

Create `/etc/udev/rules.d/rplidar.rules`:

```bash
sudo nano /etc/udev/rules.d/rplidar.rules
```

```udev
KERNEL=="ttyUSB*", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE:="0660", GROUP:="dialout", SYMLINK+="rplidar"
```

### 6.3 Orbbec Astra

```bash
cd "$ROBOT_ROOT/BIG_BRAIN/src/ros2_astra_camera/astra_camera/scripts"
sudo bash install.sh
```

Reload rules, unplug the devices, and reconnect them:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger

lsusb
ls -l /dev/serial/by-id/
ls -l /dev/rplidar
st-info --probe
v4l2-ctl --list-devices
arecord -l
aplay -l
```

---

## 7. Test each subsystem

Keep the drive wheels lifted and keep an emergency stop within reach.

In every ROS terminal:

```bash
cd "$ROBOT_ROOT/BIG_BRAIN"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

### 7.1 STM32 and micro-ROS

```bash
ros2 run micro_ros_agent micro_ros_agent serial \
  -b 921600 \
  --dev /dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066BFF485270535067113035-if02
```

Reset the STM32, then check topics from another ROS terminal:

```bash
ros2 node list
ros2 topic hz /stm32/imu_msg
ros2 topic hz /stm32/wheel_states
ros2 topic hz /stm32/ultrasonic_msg
ros2 topic hz /stm32/tof_raw_data
```

### 7.2 RPLIDAR

```bash
ros2 launch sllidar_ros2 sllidar_c1_launch.py serial_port:=/dev/rplidar
ros2 topic hz /scan
```

### 7.3 Astra depth camera

```bash
ros2 launch astra_camera astra.launch.xml
ros2 topic hz /camera/depth/points
```

### 7.4 RGB camera and audio

```bash
v4l2-ctl --list-devices
ffplay /dev/video0
arecord -l
aplay -l
```

---

## 8. Run the complete robot

### Terminal A: ROS 2 robot

```bash
cd "$ROBOT_ROOT/BIG_BRAIN"
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch robot bringup.launch.py
```

For live mapping and navigation:

```bash
ros2 launch robot bringup.launch.py slam:=true nav2:=true
```

For localization using the saved map:

```bash
ros2 launch robot bringup.launch.py amcl:=true nav2:=true
```

Do not use `sim_run` or `sim_auto_run`; `sim_bringup.launch.py` is not currently
part of the repository.

### Terminal B: SMALL_BRAIN

Use a graphical Ubuntu login:

```bash
cd "$ROBOT_ROOT/SMALL_BRAIN"
source venv/bin/activate

export OPENAI_API_KEY='replace-with-your-key'
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb

python main.py
```

Never commit the OpenAI key. The application requires internet access, camera
index `0`, microphone/speaker access, and access to the configured OpenAI
Realtime model.

---

## 9. Save a map

Start the robot with SLAM:

```bash
cd "$ROBOT_ROOT/BIG_BRAIN"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot bringup.launch.py slam:=true nav2:=true
```

After driving through the mapped area:

```bash
ros2 run nav2_map_server map_saver_cli \
  -f "$ROBOT_ROOT/BIG_BRAIN/src/robot/map/my_map"
```

Do not publish a map if it reveals a private home, laboratory, or facility.

---

## 10. Final acceptance checklist

- [ ] The GitHub repository clones without Git LFS or missing first-party code.
- [ ] `vcs import . < third_party.repos` completes.
- [ ] The sparse Nav2 checkout contains only the three required packages.
- [ ] All four patch commands complete successfully.
- [ ] `source /opt/ros/jazzy/setup.bash` works.
- [ ] `colcon build` completes in `BIG_BRAIN`.
- [ ] All six critical ROS packages report a prefix.
- [ ] `clinfo -l` shows the intended device, or CPU fallback is configured.
- [ ] The SMALL_BRAIN import validation succeeds.
- [ ] Required models exist beside their matching OpenVINO `.bin` files.
- [ ] CubeMX regenerates the STM32 vendor source.
- [ ] The micro-ROS static library and STM32 firmware build successfully.
- [ ] ST-Link, RPLIDAR, Astra, RGB camera, microphone, and speakers are detected.
- [ ] STM32 topics publish at stable rates.
- [ ] Lidar publishes `/scan` and Astra publishes `/camera/depth/points`.
- [ ] The ROS robot starts before `SMALL_BRAIN/main.py`.
- [ ] `git status --short` shows no models, environments, builds, or runtime data.

When every item passes, the installation is reproducible from the repository.
