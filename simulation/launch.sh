#!/usr/bin/env bash
# Desktop launcher for the EMG BiDexHand simulation. Log: simulation/launcher.log
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
unset PYTHONPATH
export MUJOCO_GL=egl
PY="$HERE/../.venv/bin/python"
[ -x "$PY" ] || PY="$HOME/EMG_Prosthetic_Hand_Presentation/demo/.venv/bin/python"
"$PY" "$HERE/app.py" "$@" >>"$HERE/launcher.log" 2>&1 || \
  zenity --error --title="EMG BiDexHand Simulation" --text="Simulation failed. See $HERE/launcher.log" 2>/dev/null
