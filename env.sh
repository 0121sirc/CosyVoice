#!/usr/bin/env bash
# Shared runtime environment for the CosyVoice launchers:
#     webui.sh, openai_api_server.sh, run_webui.sh
#
# Source it, do not execute it (from the repo root):
#     source ./env.sh
#
# This file lives at the repo ROOT (not inside .conda_env/) so that it survives
# a fresh clone: .conda_env/ is gitignored because it holds the conda env itself,
# but this file is hand-written config the launchers cannot start without.
#
# Three concerns live here (each toggleable from the environment before sourcing):
#   1. PATH      -> the env's bin dir, so gradio/ffmpeg can find ffprobe
#   2. 方案A      -> LD_LIBRARY_PATH for the onnxruntime CUDA EP (cuDNN 9 lives in
#                   .conda_env/cudnn9/, torch ships cuDNN 8 under site-packages/nvidia)
#   3. 方案B      -> glibc malloc tuning + the malloc_trim thread in sitecustomize.py,
#                   which keeps the webui's RSS and load-time peak from filling 7.8GB
#
# Safe to source more than once.
[[ -n "${COSYVOICE_ENV_SOURCED:-}" ]] && return 0
COSYVOICE_ENV_SOURCED=1

# This file sits at the repo root, but everything it configures lives inside the
# (gitignored) .conda_env/, so anchor to ./env.sh -> ./.conda_env explicitly.
_COSYVOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_COSYVOICE_ENV_DIR="$_COSYVOICE_ROOT/.conda_env"

# --- 1. PATH -----------------------------------------------------------------
export PATH="$_COSYVOICE_ENV_DIR/bin:$PATH"

# --- 2. 方案A: onnxruntime CUDA EP -------------------------------------------
# onnxruntime-gpu 1.19.2 (CUDA 12 build) needs cuDNN 9, i.e. libcudnn.so.9, while
# torch 2.3.1 ships libcudnn.so.8. The SONAMEs differ so both can be visible at
# once; the standalone copy is unpacked into .conda_env/cudnn9/ by install.sh.
if [[ -d "$_COSYVOICE_ENV_DIR/cudnn9/nvidia/cudnn/lib" ]]; then
    export LD_LIBRARY_PATH="$_COSYVOICE_ENV_DIR/cudnn9/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# cublas/cudart/cufft/curand/cusolver/cusparse for the same EP, provided by the
# nvidia-* pip packages that come with the torch cu121 wheel.
for _d in "$_COSYVOICE_ENV_DIR"/lib/python3.10/site-packages/nvidia/*/lib; do
    [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:$LD_LIBRARY_PATH"
done
unset _d

# --- 3. 方案B: glibc malloc --------------------------------------------------
# The webui has 24 threads; glibc opens one arena per ~8 threads and each one
# fragments memory on its own.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
# glibc raises the mmap threshold up to 32MB when it sees many large alloc/free
# cycles, which pins model-load buffers (llm.pt 1.9GB, flow.pt 1.3GB) inside the
# sbrk heap instead of releasing them. Pin it low so big allocations go through
# mmap/munmap and the load-time peak (was VmHWM 6.5GB) stays manageable.
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-131072}"
export MALLOC_MMAP_MAX_="${MALLOC_MMAP_MAX_:-65536}"
# Keep auto-trim active for large frees (the dynamic threshold can grow huge
# after big allocations, which disables trimming entirely).
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-67108864}"
# sitecustomize.py (in this env's site-packages) runs a background thread that
# calls malloc_trim(0), returning memory freed after model loading to the OS.
export COSYVOICE_MALLOC_TRIM="${COSYVOICE_MALLOC_TRIM:-1}"
export COSYVOICE_MALLOC_TRIM_INTERVAL="${COSYVOICE_MALLOC_TRIM_INTERVAL:-5}"

unset _COSYVOICE_ENV_DIR
unset _COSYVOICE_ROOT
