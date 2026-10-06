#!/usr/bin/env bash
set -euo pipefail

face_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
robot_root="$(cd "$face_dir/.." && pwd)"

# Desktop sessions normally do not source .bashrc. Recover only the exported
# key from the user's established interactive-shell setup without printing it.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    OPENAI_API_KEY="$(bash -ic 'printf %s "${OPENAI_API_KEY:-}"' 2>/dev/null)"
    export OPENAI_API_KEY
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not available to the robot service." >&2
    exit 1
fi

# ROS owns the local ZeroMQ endpoint used by Small Brain. Wait for it rather
# than relying only on process startup order.
/usr/bin/python3 - <<'PY'
import socket
import sys
import time

deadline = time.monotonic() + 45
while time.monotonic() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", 5555), timeout=0.5):
            sys.exit(0)
    except OSError:
        time.sleep(0.5)
print("Timed out waiting for the ROS ZeroMQ bridge on port 5555", file=sys.stderr)
sys.exit(1)
PY

cd "$robot_root/SMALL_BRAIN"
export DISPLAY="${DISPLAY:-:0}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
exec "$robot_root/SMALL_BRAIN/venv/bin/python" main.py
