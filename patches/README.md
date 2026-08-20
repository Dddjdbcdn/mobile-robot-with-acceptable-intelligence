# Third-party patches

These patches contain the robot-specific modifications made on top of the exact
upstream commits recorded in `third_party.repos`. Apply them only after importing
that manifest. `SETUP_FROM_SCRATCH.md` contains the commands and order.

| Patch | Purpose |
| --- | --- |
| `nav2-jazzy.patch` | Fix range-layer bounds and return invalid planner indices. |
| `astra-camera.patch` | Disable the Astra driver's duplicate TF publisher. |
| `sllidar-c1.patch` | Reduce irrelevant CycloneDDS log output. |
| `topic-based-ros2-control.patch` | Add configurable command/state QoS. |
| `micro-ros-stm32.patch` | Increase micro-ROS entity limits for this firmware. |
