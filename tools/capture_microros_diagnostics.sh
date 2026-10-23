#!/usr/bin/env bash

set -u

phase="${1:-snapshot}"
case "$phase" in
  pre-reboot|post-reboot|snapshot) ;;
  *)
    echo "Usage: $0 [pre-reboot|post-reboot|snapshot]" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d-%H%M%S)"
output_dir="$repo_root/diagnostics/microros-reboot"
report="$output_dir/${timestamp}-${phase}.txt"
serial_link="/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066BFF485270535067113035-if02"

mkdir -p "$output_dir"

section() {
  printf '\n===== %s =====\n' "$1"
}

run() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
  "$@" 2>&1 || printf '[exit=%s]\n' "$?"
}

capture() {
  section "CAPTURE METADATA"
  run date --iso-8601=seconds
  run uptime
  run cat /proc/sys/kernel/random/boot_id
  run uname -a
  run bash -c 'test -r /etc/os-release && cat /etc/os-release'
  run id

  section "RELEVANT PROCESSES"
  run bash -c "ps -eo user,pid,ppid,lstart,stat,cmd --width 300 | rg -i 'micro_ros_agent|st-util|gdb-multiarch|ros2 launch|ModemManager|brltty' | rg -v 'rg -i'"
  run bash -c "pgrep -af 'micro_ros_agent|st-util|gdb-multiarch' || true"

  section "SERIAL DEVICE AND OWNERSHIP"
  run ls -l /dev/ttyACM0 "$serial_link"
  run readlink -f "$serial_link"
  run stat /dev/ttyACM0
  run udevadm info --query=property --name=/dev/ttyACM0
  run udevadm info --query=path --name=/dev/ttyACM0
  run stty -F /dev/ttyACM0 -a
  run lsof /dev/ttyACM0
  run fuser -v /dev/ttyACM0

  section "USB TOPOLOGY AND DESCRIPTORS"
  run lsusb
  run lsusb -t
  run usb-devices
  run lsusb -v -d 0483:374b

  section "USB AND TTY SYSFS STATE"
  local tty_device usb_interface usb_device
  tty_device="$(readlink -f /sys/class/tty/ttyACM0/device 2>/dev/null || true)"
  usb_interface="$tty_device"
  usb_device="$(readlink -f "$tty_device/.." 2>/dev/null || true)"
  printf 'tty_device=%s\nusb_interface=%s\nusb_device=%s\n' \
    "$tty_device" "$usb_interface" "$usb_device"
  if [[ -n "$tty_device" ]]; then
    run ls -la "$tty_device"
    run bash -c 'for path in "$@"; do if [[ -e "$path" ]]; then printf "%s=" "$path"; cat "$path"; fi; done' _ \
      "$tty_device/bInterfaceNumber" \
      "$tty_device/bAlternateSetting" \
      "$tty_device/power/control" \
      "$tty_device/power/runtime_status" \
      "$usb_device/authorized" \
      "$usb_device/bConfigurationValue" \
      "$usb_device/power/control" \
      "$usb_device/power/runtime_status" \
      "$usb_device/power/autosuspend" \
      "$usb_device/power/autosuspend_delay_ms"
  fi
  run bash -c "readlink -f /sys/class/tty/ttyACM0/device/driver; readlink -f /sys/class/tty/ttyACM0/device/../driver"

  section "KERNEL MODULES"
  run bash -c "lsmod | rg 'cdc_acm|usbcore|usb_storage|xhci|ehci|ohci'"
  run modinfo cdc_acm

  section "CURRENT-BOOT USB KERNEL LOG"
  run bash -c "journalctl --no-pager -b -k | rg -i 'usb|cdc_acm|ttyACM|st-?link|disconnect|reset|stall|timeout|error'"

  section "CURRENT-BOOT RELATED SERVICE LOG"
  run bash -c "journalctl --no-pager -b | rg -i 'ttyACM|micro.?ros|modemmanager|brltty|st-?link'"

  if [[ "$phase" == "post-reboot" ]]; then
    section "PREVIOUS-BOOT USB KERNEL LOG"
    run bash -c "journalctl --no-pager -b -1 -k | rg -i 'usb|cdc_acm|ttyACM|st-?link|disconnect|reset|stall|timeout|error'"

    section "PREVIOUS-BOOT RELATED SERVICE LOG"
    run bash -c "journalctl --no-pager -b -1 | rg -i 'ttyACM|micro.?ros|modemmanager|brltty|st-?link'"
  fi

  section "POTENTIAL SERIAL-PORT SERVICES"
  run systemctl is-active ModemManager.service
  run systemctl status --no-pager ModemManager.service
  run systemctl is-active brltty.service
  run systemctl status --no-pager brltty.service

  section "MICRO-ROS AGENT IDENTITY"
  local agent_pid
  agent_pid="$(pgrep -n -f '/micro_ros_agent.*serial' 2>/dev/null || true)"
  printf 'agent_pid=%s\n' "$agent_pid"
  if [[ -n "$agent_pid" && -r "/proc/$agent_pid/cmdline" ]]; then
    run bash -c 'tr "\0" " " < "$1/cmdline"; echo' _ "/proc/$agent_pid"
    run readlink -f "/proc/$agent_pid/exe"
    run sha256sum "/proc/$agent_pid/exe"
    run bash -c 'tr "\0" "\n" < "$1/environ" | rg "^(ROS_|RMW_|CYCLONEDDS|FASTRTPS)" || true' _ "/proc/$agent_pid"
  fi
  run bash -lc "source /opt/ros/jazzy/setup.bash; source '$repo_root/BIG_BRAIN/install/setup.bash'; ros2 pkg prefix micro_ros_agent"
  run bash -lc "source /opt/ros/jazzy/setup.bash; source '$repo_root/BIG_BRAIN/install/setup.bash'; timeout 5 ros2 node list | sort"
  run bash -lc "source /opt/ros/jazzy/setup.bash; source '$repo_root/BIG_BRAIN/install/setup.bash'; timeout 5 ros2 topic list | sort | rg 'stm32|micro_ros|wheel|servo'"

  section "FIRMWARE AND SOURCE IDENTITY"
  run stat "$repo_root/MICRO_ROS/build/DJ_AMR_CUBEMX.elf" "$repo_root/MICRO_ROS/build/DJ_AMR_CUBEMX.bin"
  run sha256sum "$repo_root/MICRO_ROS/build/DJ_AMR_CUBEMX.elf" "$repo_root/MICRO_ROS/build/DJ_AMR_CUBEMX.bin"
  run arm-none-eabi-size "$repo_root/MICRO_ROS/build/DJ_AMR_CUBEMX.elf"
  run git -C "$repo_root" rev-parse HEAD
  run git -C "$repo_root" status --short -- MICRO_ROS BIG_BRAIN/src/robot/launch/hardware.launch.py
  run git -C "$repo_root" diff --stat -- MICRO_ROS
  run sha256sum \
    "$repo_root/MICRO_ROS/Core/Src/Controller/microros_app.c" \
    "$repo_root/MICRO_ROS/Core/Src/Controller/microros_dma_transport.c" \
    "$repo_root/MICRO_ROS/Core/Src/usart.c" \
    "$repo_root/MICRO_ROS/Core/Src/dma.c" \
    "$repo_root/BIG_BRAIN/src/robot/launch/hardware.launch.py"

  section "ST-LINK PROBE VISIBILITY"
  run st-info --probe
}

capture >"$report" 2>&1
printf '%s\n' "$report"
