small_brain_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$small_brain_root/venv/bin/activate"

alias chat='clear && python main.py'

export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
