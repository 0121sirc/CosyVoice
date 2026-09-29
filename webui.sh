#!/usr/bin/env bash
# Control the CosyVoice Gradio WebUI (webui.py).
#
# Usage: webui.sh [start|stop|status] [--host HOST] [--port PORT] [--model_dir DIR] [--force]
#   start   Launch the WebUI in the background (default when no command is given)
#   stop    Gracefully stop it (TERM -> wait -> KILL)
#   status  Report whether it is running and whether the endpoint responds
#
# NOTE webui.py always binds 0.0.0.0; --host only selects the address used to
# probe health (default: tailscale IP, else 127.0.0.1).
#
# The 6GB GPU cannot hold this and the OpenAI API server at the same time, so
# start refuses while ./openai_api_server.sh has an instance up (override: --force).
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"

# PATH (ffprobe for gradio) / LD_LIBRARY_PATH (onnxruntime CUDA EP) / malloc tuning
source "$HERE/env.sh"

# Default probe host: the tailscale IP, else fall back to 127.0.0.1 with a warning.
default_host() {
  local ip=""
  ip="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  if [[ -z "$ip" ]]; then
    ip="$(ip -4 -o addr show tailscale0 2>/dev/null | awk '{split($4,a,"/"); print a[1]}' || true)"
  fi
  if [[ -z "$ip" ]]; then
    echo "WARNING: no tailscale IP found (tailscale not running?); falling back to 127.0.0.1" >&2
    echo "127.0.0.1"
  else
    echo "$ip"
  fi
}

HOST="${HOST:-$(default_host)}"
PORT="${PORT:-8000}"
# -RL = symlink view of Fun-CosyVoice3-0.5B whose llm.pt points at llm.rl.pt (RL
# post-trained LLM). Revert with --model_dir pretrained_models/Fun-CosyVoice3-0.5B.
MODEL_DIR="${MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B-RL}"
FORCE="${FORCE:-0}"
PYTHON="$HERE/.conda_env/bin/python"
RUN_DIR="$HERE/.conda_env/.run"
PID_FILE="$RUN_DIR/webui.pid"
LOG_FILE="$RUN_DIR/webui.log"
API_PID_FILE="$RUN_DIR/openai_api_server.pid"
API_PROBE="http://127.0.0.1:8091/v1/health"
API_PORT="${API_PORT:-8091}"

# 0.0.0.0 is a bind address, not a dialable one.
HEALTH_HOST="$HOST"
[[ "$HEALTH_HOST" == "0.0.0.0" ]] && HEALTH_HOST="127.0.0.1"

usage() {
  cat <<EOF
Usage: $(basename "$0") [start|stop|status] [--host HOST] [--port PORT] [--model_dir DIR] [--force]

  start    Start the CosyVoice WebUI in the background (default).
           Waits until the HTTP endpoint is ready (model loading takes ~1-2 min).
  stop     Gracefully stop the WebUI (TERM -> wait -> KILL fallback).
  status   Show whether the WebUI is running and its endpoint responds.

Options:
  --host HOST      Probe host (default: $HOST); the server itself binds 0.0.0.0
  --port PORT      Port (default: $PORT)
  --model_dir DIR  Model directory (default: $MODEL_DIR)
  --force          Start even if the OpenAI API server appears to be running
  -h, --help       Show this help
EOF
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

health_ok() {
  curl -fsS --max-time 3 --noproxy '*' "http://$HEALTH_HOST:$PORT/" >/dev/null 2>&1
}

# True when something is listening on a TCP port -- catches a server that was
# started by hand, since these scripts may bind a tailscale IP rather than
# 127.0.0.1 (in which case the HTTP probe below cannot see it).
port_listening() {
  ss -H -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":$1$"
}

api_server_alive() {
  if [[ -f "$API_PID_FILE" ]]; then
    local pid
    pid="$(cat "$API_PID_FILE" 2>/dev/null || true)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && return 0
  fi
  curl -fsS --max-time 2 --noproxy '*' "$API_PROBE" 2>/dev/null | grep -q '"status"' && return 0
  port_listening "$API_PORT"
}

check_gpu_exclusive() {
  [[ "$FORCE" == "1" ]] && return 0
  if api_server_alive; then
    echo "ERROR: the CosyVoice OpenAI API server is running (pid file: $API_PID_FILE, probe: $API_PROBE)." >&2
    echo "       This 6GB GPU cannot hold both servers at once (~4.6GB + ~4.6GB)." >&2
    echo "       Stop it first, then retry:" >&2
    echo "           ./openai_api_server.sh stop" >&2
    echo "       or pass --force if you know there is room." >&2
    return 1
  fi
}

cmd_start() {
  if is_running; then
    if health_ok; then
      echo "Already running (pid $(cat "$PID_FILE")) on http://$HOST:$PORT"
      return 0
    fi
    echo "Stale process $(cat "$PID_FILE") without a healthy endpoint; restarting."
    cmd_stop
  fi

  # .conda_env/ is gitignored (it *is* the conda env), so a fresh clone ships no
  # interpreter at this path. Fail with the recipe instead of letting nohup die
  # quietly in the log.
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python interpreter not found at: $PYTHON" >&2
    echo "  The conda env (.conda_env/) is not part of this checkout. Create it, then retry:" >&2
    echo "      conda create -p .conda_env python=3.10 pip" >&2
    echo "      .conda_env/bin/pip install -r requirements.txt" >&2
    echo "  re-run: ./webui.sh start" >&2
    return 1
  fi

  check_gpu_exclusive

  # Refuse to start when the port is already taken, otherwise health_ok() would
  # answer for a process we did not start.
  if port_listening "$PORT"; then
    echo "ERROR: port $PORT is already in use; refusing to start a second WebUI." >&2
    ss -tlnp "sport = :$PORT" 2>/dev/null | sed 's/^/    /' >&2 || true
    echo "  stop the owner first: ./webui.sh stop" >&2
    return 1
  fi

  mkdir -p "$RUN_DIR"
  : > "$LOG_FILE"
  echo "Starting CosyVoice WebUI on http://$HOST:$PORT ..."
  nohup "$PYTHON" "$HERE/webui.py" --port "$PORT" --model_dir "$MODEL_DIR" \
    >>"$LOG_FILE" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"

  # Liveness before health: a pid that never bound the socket must lose even if
  # another process answers on this port.
  for _ in $(seq 1 300); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: WebUI exited during startup. Last log lines:" >&2
      tail -20 "$LOG_FILE" >&2 || true
      rm -f "$PID_FILE"
      return 1
    fi
    if health_ok; then
      echo "Up (pid $pid). Log: $LOG_FILE"
      return 0
    fi
    sleep 1
  done

  echo "WARNING: started (pid $pid) but the endpoint is not ready yet; see $LOG_FILE"
  return 0
}

cmd_stop() {
  if ! is_running; then
    rm -f "$PID_FILE"
    echo "Not running."
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  echo "Stopping pid $pid ..."
  kill -TERM "$pid" 2>/dev/null || true

  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "  force killing $pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "Stopped."
}

cmd_status() {
  if is_running; then
    echo "Running (pid $(cat "$PID_FILE")) on http://$HOST:$PORT"
    if health_ok; then
      echo "Health: ok"
    else
      echo "Health: FAILED (endpoint not responding; model still loading? see $LOG_FILE)"
      return 1
    fi
  else
    echo "Not running."
    return 1
  fi
}

CMD="start"
while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|status) CMD="$1"; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model_dir) MODEL_DIR="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# HEALTH_HOST/PORT may have changed via options.
HEALTH_HOST="$HOST"
[[ "$HEALTH_HOST" == "0.0.0.0" ]] && HEALTH_HOST="127.0.0.1"

case "$CMD" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
esac
