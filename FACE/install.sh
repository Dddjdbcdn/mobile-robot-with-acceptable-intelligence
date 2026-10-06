#!/usr/bin/env bash
set -euo pipefail

face_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
robot_root="$(cd "$face_dir/.." && pwd)"
unit_dir="$HOME/.config/systemd/user"
app_dir="$HOME/.local/share/applications"
autostart_dir="$HOME/.config/autostart"
config_dir="$HOME/.config/dj-robot"

mkdir -p "$unit_dir" "$app_dir" "$autostart_dir" "$config_dir"
chmod +x "$face_dir/face.py" "$face_dir/scripts/"*.sh

sed -e "s|@FACE_DIR@|$face_dir|g" -e "s|@ROBOT_ROOT@|$robot_root|g" \
    "$face_dir/systemd/dj-ros.service.in" > "$unit_dir/dj-ros.service"
sed -e "s|@FACE_DIR@|$face_dir|g" -e "s|@ROBOT_ROOT@|$robot_root|g" \
    "$face_dir/systemd/dj-small-brain.service.in" > "$unit_dir/dj-small-brain.service"
sed -e "s|@FACE_DIR@|$face_dir|g" \
    "$face_dir/systemd/dj-face-ui.service.in" > "$unit_dir/dj-face-ui.service"
install -m 0644 "$face_dir/systemd/dj-robot.target" "$unit_dir/dj-robot.target"

sed "s|@FACE_DIR@|$face_dir|g" \
    "$face_dir/desktop/dj-robot-face.desktop.in" > "$app_dir/dj-robot-face.desktop"
install -m 0644 "$app_dir/dj-robot-face.desktop" "$autostart_dir/dj-robot-face.desktop"

systemctl --user daemon-reload
systemctl --user import-environment OPENAI_API_KEY 2>/dev/null || true

# Dedicated-display defaults for the current GNOME user session.
gsettings set org.gnome.desktop.notifications show-banners false 2>/dev/null || true
gsettings set org.gnome.desktop.notifications show-in-lock-screen false 2>/dev/null || true
gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null || true
gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null || true
gsettings set org.gnome.desktop.screensaver idle-activation-enabled false 2>/dev/null || true
gsettings set org.gnome.desktop.interface enable-animations false 2>/dev/null || true
gsettings set org.gnome.desktop.background show-desktop-icons false 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power idle-dim false 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' 2>/dev/null || true
gsettings set org.gnome.shell.extensions.dash-to-dock autohide true 2>/dev/null || true
gsettings set org.gnome.shell.extensions.dash-to-dock dock-fixed false 2>/dev/null || true
gsettings set org.gnome.shell.extensions.dash-to-dock intellihide true 2>/dev/null || true
gsettings set com.ubuntu.update-notifier no-show-notifications true 2>/dev/null || true
gsettings set com.ubuntu.update-notifier hide-reboot-notification true 2>/dev/null || true
gsettings set com.ubuntu.update-notifier show-apport-crashes false 2>/dev/null || true
gsettings set com.ubuntu.update-notifier show-livepatch-status-icon false 2>/dev/null || true

echo "DJ Robot FACE installed. It will open automatically at the next graphical login."
echo "Launch it now from the app menu as 'DJ Robot'."
