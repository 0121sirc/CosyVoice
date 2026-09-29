#!/usr/bin/env python3
"""Runtime instrumentation (and later: speed knobs) for the CosyVoice server.

Everything here monkeypatches the *loaded model object*; no tracked source file
is ever modified.  Imported by openai_tts_server.py when an env var is set:

    COSY_PROFILE=1   per-request stage timings, logged after every synthesis
    COSY_FAST=1      speed knobs (Phase 2/3; keep off for A/B baselines)

Typical output:

    [profile] mode=pcm hop=25->100 wall=11.07s first_yield=6.41s
    [profile]   frontend=0.84s x1 (normalize=0.10 text_tok=0.01 spk_tok=0.30 feat=0.29 emb=0.15)
    [profile]   llm=5.12s x1 n=224 (43.7 tok/s) marks @25=0.61 @50=1.18 @100=2.35 @200=4.70
    [profile]   flow=3.44s x5 calls=[tok100/p100, ...] solver=3.20s x5 dit=50 x5
    [profile]   hift=1.31s x5 f0=0.42s
    [profile]   events frontend_end=0.84 first_token=1.02 first_flow=2.60 first_hift=5.90
"""
from __future__ import annotations

import os
import time

PROFILE = os.environ.get("COSY_PROFILE", "") == "1"

# Speed knobs: COSY_FAST=1 / "all" enables the safe set, otherwise a comma list.
#   safe:  hop      reset token_hop_len per request (fixes the leaked 25->100)
#          scale4   stream_scale_factor 2 -> 4 (first chunk 25, then straight to 100)
#          hopmax   jump to a single big chunk after the first one (fewest flow calls)
#          cache    cache prompt feature extraction per voice (mtime keyed)
#          cudnn    cudnn.benchmark=True for fixed-shape convs
#          nocache  skip torch.cuda.empty_cache() at the end of every request
#          prewarm  run the ORT speech tokenizer once at startup (22s one-off)
#   risky: nfe5     flow Euler steps 10 -> 5
#          cfg      classifier-free guidance off (batch-1 solver)
#          f0f32    hift f0 predictor in fp32 instead of fp64
#          fastsample  nucleus sampling with 1 CUDA sync instead of ~50
_SAFE_KNOBS = ("hop", "cache", "cudnn", "nocache", "prewarm")
_ALL_KNOBS = set(_SAFE_KNOBS) | {"scale4", "hopmax", "nfe5", "cfg", "f0f32", "fastsample"}
_RAW_FAST = os.environ.get("COSY_FAST", "").strip()
if _RAW_FAST.lower() in ("1", "all", "yes", "true"):
    _RAW_FAST = ",".join(_SAFE_KNOBS)
FAST_SET = {s.strip() for s in _RAW_FAST.split(",") if s.strip()} - {""}
FAST = bool(FAST_SET)

# token counts at which we timestamp LLM progress (25 tok == 1 s of speech)
MARKS = (25, 50, 75, 100, 150, 200, 250, 300, 400, 500)

_LOGGER = None
_INSTALLED = False
_B: dict | None = None


# ------------------------------------------------------------------ accumulation
def _bucket() -> dict | None:
    return _B


def _add(name: str, dt: float) -> None:
    b = _B
    if b is None:
        return
    b["t"][name] = b["t"].get(name, 0.0) + dt
    b["c"][name] = b["c"].get(name, 0) + 1


def event(name: str) -> None:
    b = _B
    if b is None:
        return
    b["ev"].setdefault(name, round(time.perf_counter() - b["t0"], 3))


def begin(mode: str, stream: bool, hop: int | None = None) -> None:
    global _B
    if not PROFILE:
        return
    _B = {"t0": time.perf_counter(), "t": {}, "c": {}, "ev": {},
          "flow": [], "t2w": [], "marks": {}, "n_tok": 0,
          "mode": mode, "stream": stream, "hop": hop}


def _fmt_calls(calls: list, limit: int = 6) -> str:
    if not calls:
        return "-"
    shown = ", ".join(str(c) for c in calls[:limit])
    if len(calls) > limit:
        shown += f", ...(+{len(calls) - limit})"
    return shown


def report(hop_end: int | None = None) -> list[str]:
    """Consume the current bucket and return log lines."""
    global _B
    if _B is None:
        return []
    b, _B = _B, None
    t, c, ev = b["t"], b["c"], b["ev"]
    out = [f"[profile] mode={b['mode']} hop={b['hop']}->{hop_end} "
           f"wall={time.perf_counter() - b['t0']:.3f}s "
           f"first_yield={ev.get('first_yield', '-')}"]

    fe = t.get("frontend_zero_shot") or t.get("frontend_cross_lingual") or \
        t.get("frontend_instruct2")
    if fe is not None:
        parts = [f"{k}={t.get(k, 0):.2f}" for k in
                 ("text_normalize", "_extract_text_token", "_extract_speech_token",
                  "_extract_speech_feat", "_extract_spk_embedding")
                 if k in t]
        out.append(f"[profile]   frontend={fe:.2f}s x{c.get('frontend_zero_shot', 1)}"
                   f" ({' '.join(parts)}) ev={ev.get('frontend_end', '-')}")

    if "llm_job" in t:
        n = b["n_tok"]
        gen = t.get("llm_gen", t["llm_job"])
        rate = n / gen if gen else 0.0
        marks = " ".join(f"@{k}={v:.2f}" for k, v in sorted(b["marks"].items()))
        line = (f"[profile]   llm={t['llm_job']:.2f}s x{c['llm_job']} gen={gen:.2f}s "
                f"n={n} ({rate:.1f} tok/s) {marks} ev_first_token={ev.get('first_token', '-')}")
        if "llm_sample" in t:
            line += (f" | sampling={t['llm_sample']:.2f}s x{c['llm_sample']}"
                     f" ({t['llm_sample'] / gen * 100:.0f}% of gen, "
                     f"{t['llm_sample'] / max(n, 1) * 1000:.1f}ms/tok)")
        out.append(line)

    if "flow_inference" in t:
        out.append(f"[profile]   flow={t['flow_inference']:.2f}s x{c['flow_inference']} "
                   f"calls=[{_fmt_calls(b['flow'])}] solver={t.get('flow_solver', 0):.2f}s "
                   f"x{c.get('flow_solver', 0)} dit={c.get('dit', 0)} "
                   f"ev_first={ev.get('first_flow', '-')}")

    if "hift_inference" in t:
        out.append(f"[profile]   hift={t['hift_inference']:.2f}s "
                   f"x{c['hift_inference']} f0={t.get('f0_predictor', 0):.2f}s "
                   f"x{c.get('f0_predictor', 0)} ev_first={ev.get('first_hift', '-')}")

    if b["t2w"]:
        out.append(f"[profile]   token2wav={t.get('token2wav', 0):.2f}s "
                   f"x{c.get('token2wav', 0)} calls=[{_fmt_calls(b['t2w'])}]")
    if ev:
        out.append(f"[profile]   events {ev}")
    return out


# -------------------------------------------------------------------- patching
def _wrap(obj, name: str, label: str | None = None, tail_hook=None,
          ev_start: str | None = None, ev_end: str | None = None) -> None:
    """Patch obj.<name> with a timing wrapper (instance attribute wins)."""
    orig = getattr(obj, name, None)
    if orig is None or getattr(orig, "_fast_patched", False):
        return
    key = label or name

    def wrapper(*args, **kwargs):
        if ev_start:
            event(ev_start)          # setdefault -> first occurrence only
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            _add(key, dt)
            if ev_end:
                event(ev_end)
            if tail_hook is not None:
                tail_hook(dt, args, kwargs)

    wrapper._fast_patched = True
    wrapper._fp_wrapped = name              # lets knobs tell who owns this slot
    try:
        setattr(obj, name, wrapper)
    except Exception:
        if _LOGGER:
            _LOGGER.warning("fast_patch: cannot patch %r", name)


def _wrap_llm_generator(llm) -> None:
    orig = getattr(llm, "inference", None)
    if orig is None or getattr(orig, "_fast_patched", False):
        return

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        gen = orig(*args, **kwargs)
        n = 0
        while True:
            try:
                tok = next(gen)
            except StopIteration:
                break
            n += 1
            el = time.perf_counter() - t0
            if n == 1:
                event("first_token")
            b = _B
            if b is not None:
                if n in MARKS and n not in b["marks"]:
                    b["marks"][n] = round(el, 3)
                b["n_tok"] = n
            yield tok
        _add("llm_gen", time.perf_counter() - t0)

    wrapper._fast_patched = True
    setattr(llm, "inference", wrapper)


def _patch_flow(flow) -> None:
    def flow_hook(dt, args, kwargs):
        b = _B
        if b is None:
            return
        tok = kwargs.get("token")
        ptok = kwargs.get("prompt_token")
        b["flow"].append(
            f"{tok.shape[1] if tok is not None else '?'}/p{ptok.shape[1] if ptok is not None else '?'}"
            f"{'S' if kwargs.get('streaming') else ''}{'F' if kwargs.get('finalize') else ''}"
            f"@{dt:.2f}")

    _wrap(flow, "inference", "flow_inference", tail_hook=flow_hook,
          ev_start="first_flow")

    decoder = getattr(flow, "decoder", None)
    if decoder is not None:
        _wrap(decoder, "forward", "flow_solver")
        est = getattr(decoder, "estimator", None)
        if est is not None and hasattr(est, "forward"):
            _wrap(est, "forward", "dit")


def _patch_hift(hift) -> None:
    _wrap(hift, "inference", "hift_inference", ev_start="first_hift")
    f0 = getattr(hift, "f0_predictor", None)
    if f0 is not None:
        _wrap(f0, "forward", "f0_predictor")


def _patch_frontend(fe) -> None:
    for name in ("frontend_zero_shot", "frontend_cross_lingual", "frontend_instruct2"):
        if hasattr(fe, name):
            _wrap(fe, name, ev_end="frontend_end")
    for name in ("text_normalize", "_extract_text_token", "_extract_speech_token",
                 "_extract_speech_feat", "_extract_spk_embedding"):
        if hasattr(fe, name):
            _wrap(fe, name)


def _patch_model(model) -> None:
    _wrap(model, "llm_job")

    def t2w_hook(dt, args, kwargs):
        b = _B
        if b is not None:
            b["t2w"].append(
                f"off{kwargs.get('token_offset', '?')}"
                f"{'F' if kwargs.get('finalize') else ''}@{dt:.2f}")

    _wrap(model, "token2wav", "token2wav", tail_hook=t2w_hook)
    # sampling = ras_sampling/nucleus_sampling: a Python loop over the sorted
    # vocab whose `cum_prob < top_p` test syncs the GPU ~25x per token.
    # Patch the fast path first so the profiler measures it and the rare
    # fallback can still reach the original functools.partial.
    if "fastsample" in FAST_SET:
        _patch_sampling(model.llm)
    if getattr(model.llm, "sampling", None) is not None:
        _wrap(model.llm, "sampling", "llm_sample")
    _wrap_llm_generator(model.llm)
    _patch_flow(model.flow)
    _patch_hift(model.hift)


def install(cosy, logger) -> bool:
    """Patch a loaded CosyVoice* instance.  Idempotent."""
    global _INSTALLED, _LOGGER
    _LOGGER = logger
    if _INSTALLED:
        return PROFILE
    try:
        _patch_frontend(cosy.frontend)
        _patch_model(cosy.model)
        _INSTALLED = True
        if logger:
            logger.info("fast_patch installed (PROFILE=%s FAST=%s)",
                        PROFILE, ",".join(sorted(FAST_SET)) or "-")
        apply_knobs(cosy, logger)
    except Exception as exc:  # never break the server because of instrumentation
        if logger:
            logger.warning("fast_patch install failed: %r", exc)
    return PROFILE


# ------------------------------------------------------------------ speed knobs
_MISS = object()


def _memo(obj, name: str, key_fn, limit: int = 8) -> None:
    orig = getattr(obj, name, None)
    if orig is None or getattr(orig, "_memo_patched", False):
        return
    cache: dict = {}

    def wrapper(*args, **kwargs):
        try:
            key = key_fn(*args, **kwargs)
        except Exception:
            return orig(*args, **kwargs)
        hit = cache.get(key, _MISS)
        if hit is not _MISS:
            return hit
        val = orig(*args, **kwargs)
        cache[key] = val
        while len(cache) > limit:
            cache.pop(next(iter(cache)))
        return val

    wrapper._memo_patched = True
    wrapper._fast_patched = True
    try:
        setattr(obj, name, wrapper)
    except Exception:
        if _LOGGER:
            _LOGGER.warning("fast_patch: cannot memoize %r", name)


def _wav_key(*args, **kwargs):
    """(path, mtime_ns) for the prompt-audio extractors (args exclude self)."""
    import os as _os
    path = args[0] if args else kwargs.get("prompt_wav")
    st = _os.stat(str(path))
    return (str(path), st.st_mtime_ns)


def _patch_prompt_cache(fe) -> None:
    for name in ("_extract_speech_token", "_extract_speech_feat", "_extract_spk_embedding"):
        _memo(fe, name, _wav_key)
    _memo(fe, "_extract_text_token", lambda text=None, *a, **kw: text, limit=16)


def _patch_hop_reset(model) -> None:
    """cli/model.py:360 grows token_hop_len 25->100 inside a request and never
    restores it, so after the first stream call every later request waits for a
    100-token (4 s) first chunk.  Reset it at the start of every tts()."""
    init = int(getattr(model, "token_hop_len", 25))
    orig = model.tts

    def tts(*args, **kwargs):
        model.token_hop_len = init
        return orig(*args, **kwargs)

    tts._fast_patched = True
    model.tts = tts
    if _LOGGER:
        _LOGGER.info("fast_patch: token_hop_len reset to %d on every request", init)


def _patch_chunking(model, scale: int | None, jump: bool = False) -> None:
    """cli/model.py:360 grows token_hop_len by stream_scale_factor per chunk
    (25 -> 50 -> 100, cap token_max_hop_len).  Each extra chunk is a full flow
    pass over the whole prefix (2.2x redundant DiT work measured), so a bigger
    step after the first small chunk cuts total streaming time without touching
    the first-chunk latency."""
    if scale is not None:
        model.stream_scale_factor = int(scale)
    if jump:
        model.token_max_hop_len = 10 ** 6
    if _LOGGER:
        _LOGGER.info("fast_patch: chunking scale=%s max_hop=%s",
                     model.stream_scale_factor, model.token_max_hop_len)


def _patch_sampling(llm) -> None:
    """common.py:138 ras_sampling -> nucleus_sampling loops over the sorted vocab
    on the GPU: `cum_prob += sorted_value[i]` turns cum_prob into a tensor, so
    every `cum_prob < top_p` test calls .item() -> ~25 CUDA syncs per token
    (measured 3.5-5.7 ms/token = 16-22% of LLM time).  Same algorithm, same
    multinomial draw on the same device; only the boundary test moves to the
    CPU with one .tolist() sync.

    The yaml wires sampling through `!name:...ras_sampling {top_p, top_k, ...}`,
    i.e. HyperPyYAML hands back a functools.partial with keyword-only args, so
    the parameters are read from `keywords` and the rare fallback is invoked
    with just the three positional arguments.
    """
    import torch
    cur = getattr(llm, "sampling", None)
    if getattr(cur, "_fp_fast", False) or getattr(cur, "_fp_wrapped", "") == "sampling":
        return                              # already patched, or profiling owns it
    orig = llm.sampling                      # ras_sampling, usually a partial
    kw = dict(getattr(orig, "keywords", {}) or {})
    top_p = float(kw.get("top_p", 0.8))
    top_k = int(kw.get("top_k", 25))
    win_size = int(kw.get("win_size", 10))
    tau_r = float(kw.get("tau_r", 0.1))

    def sampling(weighted_scores, decoded_tokens, sampling_):
        sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(
            descending=True, stable=True)
        # Same loop as common.py:151, but over a CPU slice: `cum_prob` still
        # becomes a float32 tensor after the first add and the `< top_p` test
        # still compares float32, so the cut-off is bit-identical to the CUDA
        # version -- only the ~25 .item() syncs move off the critical path.
        sv = sorted_value[:top_k].cpu()
        si = sorted_idx[:top_k].cpu()
        prob, indices = [], []
        cum_prob = 0.0
        for i in range(len(si)):
            if cum_prob < top_p and len(prob) < top_k:
                cum_prob += sv[i]
                prob.append(sv[i])
                indices.append(si[i])
            else:
                break
        if not prob:
            prob, indices = sv[:1], si[:1]
        prob = torch.tensor(prob).to(weighted_scores)
        indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
        top_ids = indices[prob.multinomial(1, replacement=True)].item()
        rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device)
                   == top_ids).sum().item()
        if rep_num >= win_size * tau_r:       # rare: fall back to the original
            weighted_scores[top_ids] = -float('inf')
            return orig(weighted_scores, decoded_tokens, sampling_)
        return top_ids

    sampling._fp_fast = True
    llm.sampling = sampling
    if _LOGGER:
        _LOGGER.info("fast_patch: vectorized nucleus sampling "
                     "(top_p=%s top_k=%s win=%s tau=%s)", top_p, top_k, win_size, tau_r)


def _patch_cudnn() -> None:
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def _patch_nocache() -> None:
    import torch
    torch.cuda.empty_cache = lambda: None


def _patch_nfe(decoder, n: int) -> None:
    """flow.py hardcodes n_timesteps=10; override the kwarg (chained on top of
    whatever the profiler already installed)."""
    orig = decoder.forward

    def forward(*args, **kwargs):
        kwargs["n_timesteps"] = n
        return orig(*args, **kwargs)

    forward._fast_patched = True
    decoder.forward = forward
    if _LOGGER:
        _LOGGER.info("fast_patch: flow Euler steps -> %d", n)


def _patch_cfg(flow) -> None:
    """Classifier-free guidance off: inference_cfg_rate=0 plus a batch-1 solver,
    so the DiT runs once per step instead of twice (flow_matching.py:95-117)."""
    import torch
    dec = flow.decoder
    dec.inference_cfg_rate = 0.0

    def solve_euler(x, t_span, mu, mask, spks, cond, streaming=False):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        sol = []
        x_in = torch.zeros([1, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([1, 1, x.size(2)], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([1, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([1], device=x.device, dtype=spks.dtype)
        spks_in = torch.zeros([1, 80], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([1, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = dec.forward_estimator(x_in, mask_in, mu_in, t_in,
                                            spks_in, cond_in, streaming)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1].float()

    solve_euler._fast_patched = True
    dec.solve_euler = solve_euler
    if _LOGGER:
        _LOGGER.info("fast_patch: CFG off (batch-1 solver, inference_cfg_rate=0)")


def _patch_f0_f32(hift) -> None:
    """generator.py:716 forces the f0 predictor to float64 on every call; Pascal
    runs fp64 at 1/32 of fp32.  Keep it in fp32 (quality gated by A/B)."""
    import torch
    f0 = hift.f0_predictor
    orig_to, orig_fwd = f0.to, f0.forward

    def to(*args, **kwargs):
        dt = args[0] if args else kwargs.get("dtype")
        if dt == torch.float64:            # ignore the fp64 cast
            return f0
        return orig_to(*args, **kwargs)

    def forward(x, *args, **kwargs):
        return orig_fwd(x.float(), *args, **kwargs)

    to._fast_patched = True
    forward._fast_patched = True
    f0.to = to
    f0.forward = forward
    if _LOGGER:
        _LOGGER.info("fast_patch: f0 predictor kept in fp32")


def prewarm(cosy, logger, wav_path) -> bool:
    """Run the ORT speech tokenizer once so its CUDA EP init (~27 s) happens at
    startup instead of inside the first client request."""
    import time as _time
    t0 = _time.perf_counter()
    try:
        cosy.frontend._extract_speech_token(str(wav_path))
    except Exception as exc:
        if logger:
            logger.warning("fast_patch prewarm failed: %r", exc)
        return False
    dt = _time.perf_counter() - t0
    if logger:
        logger.info("fast_patch prewarm: speech tokenizer first run %.2fs (%s)",
                    dt, wav_path)
    return True


def apply_knobs(cosy, logger) -> None:
    if not FAST_SET:
        return
    m, fe = cosy.model, cosy.frontend
    flow = m.flow
    if "hop" in FAST_SET:
        _patch_hop_reset(m)
    if "scale4" in FAST_SET:
        _patch_chunking(m, scale=4)
    if "hopmax" in FAST_SET:
        _patch_chunking(m, scale=10 ** 6, jump=True)
    if "cache" in FAST_SET:
        _patch_prompt_cache(fe)
    if "cudnn" in FAST_SET:
        _patch_cudnn()
    if "nocache" in FAST_SET:
        _patch_nocache()
    if "nfe5" in FAST_SET:
        _patch_nfe(flow.decoder, 5)
    if "cfg" in FAST_SET:
        _patch_cfg(flow)
    if "f0f32" in FAST_SET:
        _patch_f0_f32(m.hift)
    if "fastsample" in FAST_SET:
        _patch_sampling(m.llm)
    unknown = FAST_SET - _ALL_KNOBS
    if unknown and logger:
        logger.warning("fast_patch: unknown knob(s) ignored: %s",
                       ",".join(sorted(unknown)))
