# Full-stack mobile robot with acceptable intelligence

This repository contains a physical two-wheel robot split across three layers:

- `MICRO_ROS`: STM32G474 firmware for motors, encoders, IMU, ultrasonic sensors,
  VL53L7CX ToF, servos, and micro-ROS transport.
- `BIG_BRAIN`: ROS 2 Jazzy hardware integration, ros2_control, localization,
  mapping/navigation, lidar, depth camera, and robot/LLM bridges.
- `SMALL_BRAIN`: RGB/audio capture, YOLO and OpenVINO vision services,
  behavior logic, and OpenAI Realtime integration.

For a new Ubuntu installation, follow [SETUP_FROM_SCRATCH.md](SETUP_FROM_SCRATCH.md)
from section 1 through the final acceptance checklist. Do not assume a plain
clone contains model binaries: third-party source is pinned in
`third_party.repos`, local upstream changes live in `patches/`, and runtime
models are restored or exported separately.

## Repository policy

- Commit first-party source, configuration, manifests, patches, requirements,
  tests, and non-sensitive maps.
- Keep builds, environments, caches, secrets, mutable robot memory, third-party
  checkouts, datasets, and model binaries outside normal Git.
- Store versioned model bundles separately from normal Git history.

The complete installation sequence is in [SETUP_FROM_SCRATCH.md](SETUP_FROM_SCRATCH.md).
