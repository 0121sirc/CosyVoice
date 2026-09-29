#!/usr/bin/env bash
# Control the CosyVoice OpenAI-compatible TTS server (openai_tts_server.py).
#
# Usage: openai_api_server.sh [start|stop|status] [--host HOST] [--port PORT] [--model_dir DIR] [--force]
#   start   Launch the server in the background (default when no command is given)
#   stop    Gracefully stop it (TERM -> wait -> KILL)
#   status  Report whether it is running and whether /v1/health responds
#
# Endpoints once up:
#   GET  /v1/health  GET /v1/models  GET /v1/audio/voices  (+ alias /v1/voices)
#   POST /v1/audio/speech   (OpenAI TTS shape, voices from ./voices/ + ./voices-ext/)
#
# The 6GB GPU cannot hold this server and the gradio webui at the same time, so
# start refuses while ./webui.sh has an instance up (override: --force).
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"

# PATH / LD_LIBRARY_PATH (onnxruntime CUDA EP) / malloc tuning (方案A + 方案B)
source "$HERE/.conda_env/env.sh"

# Speed tuning defaults (override by exporting the vars before calling this
# script; see .conda_env/speed_notes.md for the measurements behind each knob).
#   COSY_FAST      comma list of monkeypatched knobs (empty string = stock speed)
#   COSY_PREWARM   warm the ONNX speech tokenizer at boot instead of inside the
#                  first client request (saves ~22s of first-request latency)
#   COSY_PROFILE   per-request stage timings in the log (off by default)
export COSY_FAST="${COSY_FAST:-hop,hopmax,cache,cudnn,nocache,nfe5,f0f32}"
export COSY_PREWARM="${COSY_PREWARM:-1}"
export COSY_PROFILE="${COSY_PROFILE:-}"

# Default bind host: the tailscale IP, else fall back to 127.0.0.1 with a warning.
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
PORT="${PORT:-8091}"
# -RL is a symlink view of Fun-CosyVoice3-0.5B whose llm.pt points at llm.rl.pt
# (the GRPO-post-trained LLM). Fall back with --model_dir pretrained_models/Fun-CosyVoice3-0.5B.
MODEL_DIR="${MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B-RL}"
FORCE="${FORCE:-0}"
PYTHON="$HERE/.conda_env/bin/python"
# pid/log live inside the (already untracked) conda env dir so `git status` stays clean;
# ChatTTS_colab keeps them in .run/ instead.
RUN_DIR="$HERE/.conda_env/.run"
PID_FILE="$RUN_DIR/openai_api_server.pid"
LOG_FILE="$RUN_DIR/openai_api_server.log"
WEBUI_PID_FILE="$RUN_DIR/webui.pid"
WEBUI_PROBE="http://127.0.0.1:8000/"
WEBUI_PORT="${WEBUI_PORT:-8000}"

# 0.0.0.0 is a bind address, not a dialable one.
HEALTH_HOST="$HOST"
[[ "$HEALTH_HOST" == "0.0.0.0" ]] && HEALTH_HOST="127.0.0.1"

usage() {
  cat <<EOF
Usage: $(basename "$0") [start|stop|status] [--host HOST] [--port PORT] [--model_dir DIR] [--force]

  start    Start the CosyVoice OpenAI TTS server in the background (default).
           Waits until /v1/health is ready (model loading takes ~1-2 min).
  stop     Gracefully stop the server (TERM -> wait -> KILL fallback).
  status   Show whether the server is running and healthy.

Options:
  --host HOST      Bind host (default: $HOST)
  --port PORT      Bind port (default: $PORT)
  --model_dir DIR  Model directory (default: $MODEL_DIR)
  --force          Start even if the webui appears to be running
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
  curl -fsS --max-time 3 --noproxy '*' "http://$HEALTH_HOST:$PORT/v1/health" >/dev/null 2>&1
}

# True when something is listening on a TCP port -- catches a server that was
# started by hand, since these scripts may bind a tailscale IP rather than
# 127.0.0.1 (in which case the HTTP probe below cannot see it).
port_listening() {
  ss -H -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":$1$"
}

# True when something that looks like our gradio webui answers on its port.
webui_alive() {
  if [[ -f "$WEBUI_PID_FILE" ]]; then
    local pid
    pid="$(cat "$WEBUI_PID_FILE" 2>/dev/null || true)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && return 0
  fi
  curl -fsS --max-time 2 --noproxy '*' "$WEBUI_PROBE" 2>/dev/null | grep -qi gradio && return 0
  port_listening "$WEBUI_PORT"
}

check_gpu_exclusive() {
  [[ "$FORCE" == "1" ]] && return 0
  if webui_alive; then
    echo "ERROR: the CosyVoice webui is running (pid file: $WEBUI_PID_FILE, probe: $WEBUI_PROBE)." >&2
    echo "       This 6GB GPU cannot hold both servers at once (~4.6GB + ~4.6GB)." >&2
    echo "       Stop it first, then retry:" >&2
    echo "           ./webui.sh stop" >&2
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
  check_gpu_exclusive

  mkdir -p "$RUN_DIR"
  : > "$LOG_FILE"
  echo "Starting CosyVoice OpenAI TTS server on http://$HOST:$PORT (model: $MODEL_DIR) ..."
  nohup "$PYTHON" "$HERE/openai_tts_server.py" --host "$HOST" --port "$PORT" \
    --model_dir "$MODEL_DIR" \
    >>"$LOG_FILE" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"

  # Model loading takes ~1-2 minutes (llm.pt/flow.pt/hift.pt + wetext frontend).
  for _ in $(seq 1 300); do
    if health_ok; then
      echo "Up (pid $pid). Log: $LOG_FILE"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: server exited during startup. Last log lines:" >&2
      tail -20 "$LOG_FILE" >&2 || true
      rm -f "$PID_FILE"
      return 1
    fi
    sleep 1
  done

  echo "WARNING: started (pid $pid) but /v1/health is not ready yet; see $LOG_FILE"
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
      curl -fsS --max-time 3 --noproxy '*' "http://$HEALTH_HOST:$PORT/v1/health" 2>/dev/null && echo
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
