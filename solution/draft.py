from __future__ import annotations

import os
import threading
import time

from solution.common import clean_asr_artifact, common_word_prefix, pick_best, should_escalate

_SR = 16000
_BYTES_PER_SEC = _SR * 2  # int16 mono

_MIN_AUDIO_BYTES = int(_SR * 0.75) * 2          # ~0.75s before the first draft, like the skeleton
_FAST_REDECODE_MIN_NEW_BYTES = int(_BYTES_PER_SEC * 0.3)   # redecode fast model every ~0.3s of new audio
_ESCALATE_TRIGGER_BYTES = int(_BYTES_PER_SEC * 1.0)        # don't bother escalating on <1s of audio
_ESCALATE_RETRIGGER_BYTES = int(_BYTES_PER_SEC * 1.5)      # re-run escalate after this much NEW audio
ESCALATE_FINAL_WAIT_S = float(os.environ.get("ESCALATE_FINAL_WAIT_S", "1.2"))

FASTPATH_MODEL = os.environ.get("FASTPATH_MODEL", "small")
ESCALATE_MODEL = os.environ.get("ESCALATE_MODEL", "Oriserve/Whisper-Hindi2Hinglish-Swift")
HI_PROB_THRESHOLD = float(os.environ.get("HI_PROB_THRESHOLD", "0.15"))

_fast_model = None
_fast_model_failed = False
_escalate_pipe = None
_escalate_pipe_failed = False


# --- per-clip state (draft_reset() is called by the harness between clips) -

class _ClipState:
    def __init__(self) -> None:
        self.generation = 0
        self.prev_shown_text = ""
        self.committed = ""
        self.last_fast_decode_len = 0
        self.last_fast_text = ""
        self.escalate_lock = threading.Lock()
        self.escalate_thread: threading.Thread | None = None
        self.escalate_result_text: str | None = None
        self.escalate_result_gen: int = -1
        self.escalate_started_len = 0


_state = _ClipState()


def draft_reset() -> None:
    """Called by the sealed harness at the start of each clip."""
    global _state
    # Bump the generation on the OLD state object so a still-running
    # background thread from the previous clip writes a result that the
    # generation check below will simply ignore -- we never block here.
    _state.generation += 1
    _state = _ClipState()
    _state.generation = 0


# --- fast path ---------------------------------------------------------

def _get_fast_model():
    global _fast_model, _fast_model_failed
    if _fast_model_failed:
        raise RuntimeError("fast model previously failed to load this run; not retrying")
    if _fast_model is not None:
        return _fast_model
    try:
        import platform
        if platform.system() == "Darwin":
            try:
                import mlx_whisper  # noqa: F401
                _fast_model = ("mlx", FASTPATH_MODEL)
                return _fast_model
            except ImportError:
                pass
        from faster_whisper import WhisperModel
        _fast_model = ("ct2", WhisperModel(FASTPATH_MODEL, device="cpu", compute_type="int8"))
        return _fast_model
    except Exception:
        _fast_model_failed = True
        raise


def _pcm_to_float32(audio_buffer: bytes):
    import numpy as np
    audio = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def _fast_decode(audio_buffer: bytes) -> tuple[str, list[tuple[str, float]]]:
    """Decode the rolling PCM prefix with the fast model. Returns
    (text, all_language_probs) -- probs are best-effort (mlx path can't give
    a full distribution; see solution.transcribe for the same caveat)."""
    kind, model = _get_fast_model()
    audio = _pcm_to_float32(audio_buffer)
    if audio.size == 0:
        return "", []
    if kind == "mlx":
        import mlx_whisper
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=f"mlx-community/whisper-{model}")
        text = (result.get("text") or "").strip()
        return text, [(result.get("language", "en"), 1.0)]
    segments, info = model.transcribe(
        audio, language=None, task="transcribe", language_detection_segments=1,
    )
    text = " ".join(s.text for s in segments).strip()
    probs = info.all_language_probs or [(info.language, info.language_probability)]
    return text, probs


# --- escalate path (faithful Hinglish finalizer) -----------------------

def _get_escalate_pipe():
    global _escalate_pipe, _escalate_pipe_failed
    if _escalate_pipe_failed:
        raise RuntimeError("escalate model previously failed to load this run; not retrying")
    if _escalate_pipe is not None:
        return _escalate_pipe
    try:
        import torch
        from transformers import pipeline
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _escalate_pipe = pipeline(
            "automatic-speech-recognition", model=ESCALATE_MODEL,
            device=device, torch_dtype=torch.float32,
        )
        return _escalate_pipe
    except Exception:
        _escalate_pipe_failed = True
        raise


def _escalate_decode_sync(audio_buffer: bytes) -> str:
    audio = _pcm_to_float32(audio_buffer)
    if audio.size == 0:
        return ""
    pipe = _get_escalate_pipe()
    out = pipe({"array": audio, "sampling_rate": _SR},
               generate_kwargs={"task": "transcribe"}, chunk_length_s=30)
    return clean_asr_artifact((out.get("text") or "").strip())


def _escalate_worker(audio_snapshot: bytes, generation: int, state: _ClipState) -> None:
    try:
        text = _escalate_decode_sync(audio_snapshot)
    except Exception:  # noqa: BLE001 - escalate must never crash the stream
        text = ""
    with state.escalate_lock:
        if generation == state.generation:  # discard stale results from a prior clip
            state.escalate_result_text = text
            state.escalate_result_gen = generation


def _maybe_start_escalate(audio_buffer: bytes, all_probs, fast_text: str, state: _ClipState) -> None:
    if len(audio_buffer) < _ESCALATE_TRIGGER_BYTES:
        return
    with state.escalate_lock:
        busy = state.escalate_thread is not None and state.escalate_thread.is_alive()
    if busy:
        return
    grown_enough = (len(audio_buffer) - state.escalate_started_len) >= _ESCALATE_RETRIGGER_BYTES
    if state.escalate_started_len > 0 and not grown_enough:
        return  # already covered this much audio recently; don't re-run for a sliver of new audio
    escalate, _reason = should_escalate(all_probs, fast_text, HI_PROB_THRESHOLD)
    if not escalate:
        return
    snapshot = bytes(audio_buffer)  # copy: the harness's buffer keeps growing under us
    gen = state.generation
    t = threading.Thread(target=_escalate_worker, args=(snapshot, gen, state), daemon=True)
    state.escalate_thread = t
    state.escalate_started_len = len(audio_buffer)
    t.start()


def _current_escalate_text(state: _ClipState) -> str | None:
    with state.escalate_lock:
        if state.escalate_result_gen == state.generation and state.escalate_result_text:
            return state.escalate_result_text
    return None


# --- the contract function ----------------------------------------------

def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    state = _state

    if not is_final and len(audio_buffer) < _MIN_AUDIO_BYTES:
        return (state.committed, len(state.committed))

    # Throttle fast redecodes: whisper-family decode is far more expensive
    # than the 20ms frame cadence we're called at, so only redecode once
    # enough new audio has arrived (or unconditionally on the final call, so
    # the fast fallback is as fresh as possible if escalate doesn't land).
    new_bytes = len(audio_buffer) - state.last_fast_decode_len
    fast_text, all_probs = state.last_fast_text, []
    if is_final or new_bytes >= _FAST_REDECODE_MIN_NEW_BYTES or not state.last_fast_text:
        try:
            fast_text, all_probs = _fast_decode(audio_buffer)
            fast_text = clean_asr_artifact(fast_text) or state.last_fast_text
            state.last_fast_decode_len = len(audio_buffer)
            state.last_fast_text = fast_text
        except Exception:  # noqa: BLE001 - never crash the stream; hold the previous text
            fast_text = state.last_fast_text

    _maybe_start_escalate(audio_buffer, all_probs, fast_text, state)

    if is_final:
        # Give an in-flight faithful pass a bounded extra chance to land --
        # do NOT wait unboundedly, the end-to-final cap punishes that hard.
        with state.escalate_lock:
            thread = state.escalate_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=ESCALATE_FINAL_WAIT_S)
        escalate_text = _current_escalate_text(state)
        final_text = pick_best(escalate_text or "", fast_text, state.committed)
        state.committed = final_text
        state.prev_shown_text = final_text
        return (final_text, len(final_text))

    escalate_text = _current_escalate_text(state)
    text_to_show = pick_best(escalate_text or "", fast_text, state.committed)

    stable = common_word_prefix(state.prev_shown_text, text_to_show)
    if len(stable) >= len(state.committed):
        state.committed = stable
    state.prev_shown_text = text_to_show

    return (text_to_show, len(state.committed))
