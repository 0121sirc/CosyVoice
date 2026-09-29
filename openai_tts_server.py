#!/usr/bin/env python3
"""OpenAI-compatible TTS server for CosyVoice.

Exposes ``POST /v1/audio/speech`` so any OpenAI TTS client (the ``openai`` SDK,
an s2s pipeline, ...) can use this repo's ``Fun-CosyVoice3-0.5B`` weights and the
preset voices stored under ``voices/`` + ``voices-ext/`` (merged as a union).
Mirrors ChatTTS_colab/openai_tts_server.py.

Endpoints:
  GET  /v1/health        -> readiness probe
  GET  /v1/models        -> {"data": [{"id": ...}]}
  GET  /v1/audio/voices  -> {"voices": ["default", ...], "default": ...,
                             "details": {name: {"desc": ...}}}   # display only;
                             # the prompt text is server-side input, not exposed
  GET  /v1/voices        -> identical body (alias; see the route below)
  POST /v1/audio/speech  -> pcm (streamed) | wav | mp3 | flac | opus

Request body (OpenAI shape + ``seed`` / ``params`` extensions)::

    {
      "model": "Fun-CosyVoice3-0.5B",   # optional, ignored; /v1/health reports the loaded dir
      "input": "要合成的文本",
      "voice": "default" | "bfy" | ...,
      "response_format": "pcm" | "wav" | "mp3" | "flac" | "opus",
      "speed": 1.0,
      "seed": 42,
      "instructions": "带点笑意",           # OpenAI-style; params.instruction is the alias
      "params": {"instruction": "带点笑意", "text_frontend": true}
    }

Seeding: ``seed`` defaults to 42, so a request that omits it is reproducible
(fixed seed also keeps request latency variance down). Send any int to
override; send ``"seed": null`` to opt out and get fresh randomness per call.

A voice is one directory under any voice root. Roots are scanned in order and
merged as a union (``voices/`` wins over ``voices-ext/`` on a name collision)::

    voices/bfy/prompt.wav        # reference audio, 3-10s, >=16kHz, any channel count
    voices/bfy/meta.json         # {"name": ..., "prompt_text": "...", "desc": "..."}
    voices-ext/mine/prompt.wav   # same layout, second root
    voices-ext/mine/meta.json

Backend selection per request:
  * ``instructions`` / ``params.instruction`` present -> inference_instruct2
    (clone the reference voice *and* apply a natural-language instruction)
  * meta.prompt_text present                          -> inference_zero_shot
  * meta.prompt_text missing/empty                    -> inference_cross_lingual

``prompt_text`` should match what is spoken in ``prompt.wav``; without it the
clone quality degrades but synthesis still works.

The 6GB GPU cannot hold this server and the gradio webui at the same time --
openai_api_server.sh / webui.sh refuse to start while the other one is alive.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread
from typing import Iterable, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from cosyvoice.utils.common import set_all_random_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cosyvoice.openai")

# Optional runtime instrumentation / speed knobs (see .conda_env/fast_patch.py).
# Everything is monkeypatched onto the loaded model, so no tracked source moves.
_FAST_PATCH = None
if os.environ.get("COSY_PROFILE") or os.environ.get("COSY_FAST"):
    sys.path.insert(0, str(HERE / ".conda_env"))
    try:
        import fast_patch as _FAST_PATCH
    except Exception as exc:                      # pragma: no cover
        logger.warning("fast_patch unavailable: %r", exc)

DEFAULT_MODEL_NAME = "Fun-CosyVoice3-0.5B"
DEFAULT_VOICE = "default"
# CosyVoice3's LLM asserts that this token id (151646) appears in the concatenated
# prompt_text + text (cosyvoice/llm/llm.py:479); without it synthesis dies inside a
# worker thread and surfaces as an unrelated hift crash. Voices that do not spell
# the marker out get it inserted here.
ENDOFPROMPT = "<|endofprompt|>"
# webui.py limits speed to [0.5, 2.0]; the OpenAI API nominally allows 0.25..4.0
SPEED_MIN, SPEED_MAX = 0.5, 2.0
MIN_PROMPT_SR = 16000          # cosyvoice.utils.file_utils.load_wav asserts this
MAX_PROMPT_S = 30              # frontend._extract_speech_token() asserts this
# Voice roots: scanned in order and merged as a union; an earlier root wins when
# two roots define the same voice name (so ./voices overrides ./voices-ext).
DEFAULT_VOICE_DIRS = (HERE / "voices", HERE / "voices-ext")
NATIVE_FORMATS = ("pcm", "wav")
FFMPEG_FORMATS = ("mp3", "flac", "opus")
MEDIA_TYPES = {
    "pcm": "audio/pcm",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/ogg",
}

app = FastAPI(title="CosyVoice OpenAI-compatible TTS")

_model = None
_sample_rate = 24000
_load_lock = Lock()
# One synthesis at a time: the model plus the ORT session already fill the card.
_synth_lock = Lock()
_SYNTH_WAIT_S = 120
_voices: dict[str, dict] = {}
_voice_dirs: list[Path] = list(DEFAULT_VOICE_DIRS)
_model_dir = HERE / "pretrained_models" / DEFAULT_MODEL_NAME


# --------------------------------------------------------------------------- model
def _ensure_loaded():
    global _model, _sample_rate
    if _model is None:
        with _load_lock:
            if _model is None:
                logger.info("loading CosyVoice model from %s", _model_dir)
                _model = AutoModel(model_dir=str(_model_dir))
                _sample_rate = int(_model.sample_rate)
                logger.info("CosyVoice model loaded, sample_rate=%d", _sample_rate)
                if _FAST_PATCH is not None:
                    _FAST_PATCH.install(_model, logger)
    return _model


# COSY_PREWARM=1: load the model and run the ONNX speech tokenizer once, in the
# background, right after boot.  Its CUDA EP init costs ~27s and would otherwise
# land inside the first client request (measured: first request 41.6s -> ~9s).
@app.on_event("startup")
def _prewarm_on_boot() -> None:
    if not os.environ.get("COSY_PREWARM") or _FAST_PATCH is None:
        return

    def _job() -> None:
        try:
            cosy = _ensure_loaded()
            _FAST_PATCH.prewarm(cosy, logger, HERE / "voices" / "default" / "prompt.wav")
        except Exception as exc:                      # pragma: no cover
            logger.warning("prewarm failed: %r", exc)

    Thread(target=_job, daemon=True).start()


# --------------------------------------------------------------------------- voices
def _scan_voices() -> dict[str, dict]:
    """Merge every voice root: <root>/<name>/{prompt.*, meta.json} -> {name: info}.

    The roots form a union; when two roots define the same name the earlier root
    wins and the shadowed one is logged.
    """
    voices: dict[str, dict] = {}
    roots = [d for d in _voice_dirs if d.is_dir()]
    for missing in (d for d in _voice_dirs if not d.is_dir()):
        logger.info("voices root %s does not exist (skipping)", missing)
    if not roots:
        logger.warning("no voice root exists: %s", ", ".join(map(str, _voice_dirs)))
        return voices
    for root in roots:
        for voice_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if voice_dir.name in voices:
                logger.warning("duplicate voice %s: keeping %s, ignoring %s",
                               voice_dir.name, voices[voice_dir.name]["path"].parent, voice_dir)
                continue
            prompts = sorted(voice_dir.glob("prompt.*"))
            if not prompts:
                logger.warning("skipping voice %s: no prompt.* file", voice_dir)
                continue
            meta = {}
            meta_file = voice_dir / "meta.json"
            if meta_file.is_file():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    logger.warning("voice %s: ignoring broken meta.json (%s)", voice_dir, exc)
            prompt_text = str(meta.get("prompt_text") or "").strip()
            instruction = str(meta.get("instruction") or "").strip()
            if prompt_text and ENDOFPROMPT not in prompt_text:
                # Official demo shape is "You are a helpful assistant.<|endofprompt|>希望...";
                # an empty meta.instruction yields "<|endofprompt|>希望...".
                prompt_text = f"{instruction}{ENDOFPROMPT}{prompt_text}" if instruction \
                    else f"{ENDOFPROMPT}{prompt_text}"
                logger.info("voice %s: inserted %s into prompt_text", voice_dir.name, ENDOFPROMPT)
            voices[voice_dir.name] = {
                "path": prompts[0],
                "prompt_text": prompt_text,
                "desc": str(meta.get("desc") or "").strip(),
            }
    if voices:
        logger.info("loaded %d voices from %d root(s): %s",
                    len(voices), len(roots), ", ".join(voices))
    else:
        logger.warning("no voices found under %s", ", ".join(map(str, roots)))
    return voices


def _get_voice(name: str) -> dict:
    """Look up a voice, rescanning once so any voice root can be edited without a restart."""
    key = (name or "").strip()
    if not key or key.lower() == "default":
        key = DEFAULT_VOICE
    if key not in _voices:
        _voices.update(_scan_voices())
    voice = _voices.get(key)
    if voice is None:
        available = ", ".join(sorted(_voices)) or "<none>"
        raise HTTPException(status_code=400,
                            detail=f"unknown voice {key!r}; available: {available}")
    return voice


def _validate_prompt(voice_name: str, voice: dict) -> None:
    import torchaudio
    try:
        info = torchaudio.info(str(voice["path"]))
    except Exception as exc:  # unreadable/corrupt audio
        raise HTTPException(status_code=400,
                            detail=f"voice {voice_name!r}: cannot read {voice['path'].name}: {exc}")
    if info.sample_rate < MIN_PROMPT_SR:
        raise HTTPException(status_code=400,
                            detail=f"voice {voice_name!r}: prompt sample rate "
                                   f"{info.sample_rate} < {MIN_PROMPT_SR}")
    seconds = info.num_frames / info.sample_rate
    if seconds > MAX_PROMPT_S:
        # The model asserts this inside frontend._extract_speech_token() during
        # synthesis (and the failure surfaces as an opaque 500), so reject it
        # here instead. 3-10s of clean single-speaker speech works best anyway.
        raise HTTPException(status_code=400,
                            detail=f"voice {voice_name!r}: prompt audio is {seconds:.1f}s, "
                                   f"longer than the {MAX_PROMPT_S}s limit "
                                   f"(cosyvoice/cli/frontend.py:97); trim it to 3-10s")


# ---------------------------------------------------------------------- synthesis
def _iter_audio(text: str, voice: dict, req: "SpeechRequest", stream: bool) -> Iterator[np.ndarray]:
    """Yield float32 mono chunks in [-1, 1]. The caller owns _synth_lock."""
    cosy = _ensure_loaded()

    extra = dict(req.params or {})
    instruction = (req.instructions or extra.get("instruction") or "").strip()
    text_frontend = bool(extra.get("text_frontend", True))

    if _FAST_PATCH is not None:
        _FAST_PATCH.begin(
            "instruct2" if instruction else
            ("zero_shot" if voice["prompt_text"] else "cross_lingual"),
            stream, getattr(getattr(cosy, "model", None), "token_hop_len", None))

    speed = 1.0 if req.speed is None else float(req.speed)
    if not (SPEED_MIN <= speed <= SPEED_MAX):
        clamped = min(max(speed, SPEED_MIN), SPEED_MAX)
        logger.warning("speed %.3f outside [%s, %s]; clamping to %.1f",
                       speed, SPEED_MIN, SPEED_MAX, clamped)
        speed = clamped

    if req.seed is not None:
        set_all_random_seed(int(req.seed))

    prompt_wav = str(voice["path"])
    if instruction:
        # instruct2 feeds instruct_text in as prompt_text with the speech prompt
        # dropped, so the marker belongs at the end of the instruction: the LLM
        # then sees "instruction<|endofprompt|>tts_text", same layout as zero_shot.
        if ENDOFPROMPT not in instruction:
            instruction = f"{instruction}{ENDOFPROMPT}"
        logger.info("instruct2 synthesis (%d chars, instruction=%r)", len(text), instruction)
        outputs = cosy.inference_instruct2(text, instruction, prompt_wav,
                                           stream=stream, speed=speed, text_frontend=text_frontend)
    elif voice["prompt_text"]:
        logger.info("zero_shot synthesis (%d chars)", len(text))
        outputs = cosy.inference_zero_shot(text, voice["prompt_text"], prompt_wav,
                                           stream=stream, speed=speed, text_frontend=text_frontend)
    else:
        # No prompt_text: CosyVoice3 still needs the marker, so it has to ride in
        # the synthesis text. Note frontend.text_normalize skips normalization and
        # splitting whenever <|...|> is present in the text.
        if ENDOFPROMPT not in text:
            text = f"{ENDOFPROMPT}{text}"
            logger.warning("voice has no prompt_text; prepending %s to the input text "
                           "(text frontend will be skipped for this request)", ENDOFPROMPT)
        logger.info("cross_lingual synthesis (%d chars, no prompt_text)", len(text))
        outputs = cosy.inference_cross_lingual(text, prompt_wav,
                                               stream=stream, speed=speed, text_frontend=text_frontend)

    try:
        first_yield = True
        for model_output in outputs:
            if first_yield:
                first_yield = False
                if _FAST_PATCH is not None:
                    _FAST_PATCH.event("first_yield")
            yield model_output["tts_speech"].numpy().reshape(-1)
    except RuntimeError as exc:
        # Symptom of llm_job dying inside its thread (its exception never reaches
        # us), most often an input that could not satisfy the CosyVoice3 marker
        # assert, leaving the flow/hift to chew on zero speech tokens.
        if "Kernel size can't be greater than actual input size" in str(exc):
            raise HTTPException(
                status_code=500,
                detail="the LLM produced no speech tokens; for CosyVoice3 the "
                       "prompt_text (or the input text) must contain "
                       f"{ENDOFPROMPT} -- see the server log for the original error")
        raise
    finally:
        if _FAST_PATCH is not None:
            for line in _FAST_PATCH.report(
                    hop_end=getattr(getattr(cosy, "model", None), "token_hop_len", None)):
                logger.info(line)


def _pcm_stream(text: str, voice: dict, req: "SpeechRequest") -> Iterator[bytes]:
    """Stream s16le chunks and release the synthesis lock when the stream ends,
    including on client disconnect (GeneratorExit runs the finally clause)."""
    try:
        for chunk in _to_pcm16(_iter_audio(text, voice, req, stream=True)):
            yield chunk
    finally:
        _synth_lock.release()


def _to_pcm16(chunks: Iterable[np.ndarray]) -> Iterator[bytes]:
    for chunk in chunks:
        samples = np.clip(np.asarray(chunk, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if samples.size:
            yield (samples * 32767.0).astype("<i2").tobytes()


def _wav_bytes(pcm: bytes) -> bytes:
    import wave
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def _ffmpeg_convert(pcm: bytes, fmt: str) -> bytes:
    """s16le mono -> mp3/flac/opus via the env's ffmpeg (conda install)."""
    cmd = ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(_sample_rate),
           "-ac", "1", "-i", "-", "-f", fmt]
    if fmt == "opus":
        cmd += ["-b:a", "64k"]
    cmd += ["-"]
    try:
        proc = subprocess.run(cmd, input=pcm, capture_output=True, check=False)
    except FileNotFoundError:
        raise HTTPException(status_code=501,
                            detail="ffmpeg not found on PATH; use response_format 'pcm' or 'wav'")
    if proc.returncode != 0:
        raise HTTPException(status_code=500,
                            detail=f"ffmpeg failed for {fmt}: {proc.stderr.decode(errors='ignore')[:300]}")
    return proc.stdout


def _collect(chunks: Iterable[np.ndarray]) -> bytes:
    pcm = b"".join(_to_pcm16(chunks))
    if not pcm:
        raise HTTPException(status_code=500, detail="synthesis produced no audio")
    return pcm


# ------------------------------------------------------------------------- api
class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    response_format: str = "pcm"
    speed: float | None = None
    seed: int | None = 42
    instructions: str | None = None
    params: dict | None = None


@app.get("/v1/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "model_loaded": _model is not None,
        # The directory name, not a constant: pretrained_models/Fun-CosyVoice3-0.5B-RL
        # reports itself as such, so /v1/health tells you which LLM checkpoint is live.
        "model": _model_dir.name,
        "sample_rate": _sample_rate,
        "voices": sorted(_voices) or sorted(_scan_voices()),
    })


@app.get("/v1/models")
def models() -> JSONResponse:
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": _model_dir.name,
            "object": "model",
            "owned_by": "FunAudioLLM",
        }],
    })


# OpenAI ships no voice-list endpoint at all; LocalAI documents
# /v1/audio/voices, while ElevenLabs-style and most community
# "OpenAI-compatible" clients guess /v1/voices. Serve one handler from both
# paths (identical body) so their probe does not 404.
@app.get("/v1/voices")
@app.get("/v1/audio/voices")
def voices() -> JSONResponse:
    current = _scan_voices()
    _voices.clear()
    _voices.update(current)
    default = DEFAULT_VOICE if DEFAULT_VOICE in current else (next(iter(current), ""))
    return JSONResponse({
        "voices": sorted(current),
        "default": default,
        "details": {
            # display metadata only: the prompt text stays server-side (it is
            # synthesis input, never something a caller needs)
            name: {"desc": info["desc"]}
            for name, info in current.items()
        },
    })


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty input")

    fmt = (req.response_format or "pcm").lower()
    if fmt not in NATIVE_FORMATS and fmt not in FFMPEG_FORMATS:
        raise HTTPException(status_code=400,
                            detail=f"response_format must be one of "
                                   f"{list(NATIVE_FORMATS + FFMPEG_FORMATS)}, got {req.response_format!r}")

    voice_name = req.voice or DEFAULT_VOICE
    voice = _get_voice(voice_name)
    _validate_prompt(voice_name, voice)

    stream = fmt == "pcm"
    logger.info("tts request: %d chars, voice=%s, format=%s, speed=%s, seed=%s",
                len(text), voice_name, fmt, req.speed, req.seed)

    # Claim the synthesis slot before anything is sent; _pcm_stream releases it
    # when the stream ends, the buffered branch releases it in its finally.
    if not _synth_lock.acquire(timeout=_SYNTH_WAIT_S):
        raise HTTPException(status_code=503,
                            detail=f"another synthesis is still running (waited {_SYNTH_WAIT_S}s)")

    if stream:
        return StreamingResponse(_pcm_stream(text, voice, req), media_type=MEDIA_TYPES[fmt])

    try:
        pcm = _collect(_iter_audio(text, voice, req, stream=False))
        payload = _wav_bytes(pcm) if fmt == "wav" else _ffmpeg_convert(pcm, fmt)
    finally:
        _synth_lock.release()
    return Response(content=payload, media_type=MEDIA_TYPES[fmt])


def main() -> None:
    global _voice_dirs, _model_dir
    parser = argparse.ArgumentParser(description="CosyVoice OpenAI-compatible TTS server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model_dir", default=str(HERE / "pretrained_models" / DEFAULT_MODEL_NAME))
    parser.add_argument("--voices_dirs", nargs="+", default=None,
                        help="voice roots merged as a union, earlier roots win on a name "
                             "collision (default: %s)"
                             % " ".join(str(p) for p in DEFAULT_VOICE_DIRS))
    args = parser.parse_args()

    _model_dir = Path(args.model_dir)
    if args.voices_dirs:
        _voice_dirs = [Path(p) for p in args.voices_dirs]
    _voices.update(_scan_voices())

    # Load before serving so /v1/health only answers once the model is usable.
    _ensure_loaded()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
