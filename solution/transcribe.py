from __future__ import annotations

import argparse
import json
import os
import time

from solution.common import clean_asr_artifact, pick_best, should_escalate

FASTPATH_MODEL = os.environ.get("FASTPATH_MODEL", "small")
ESCALATE_MODEL = os.environ.get("ESCALATE_MODEL", "Oriserve/Whisper-Hindi2Hinglish-Swift")
HI_PROB_THRESHOLD = float(os.environ.get("HI_PROB_THRESHOLD", "0.15"))

_fast_model = None
_fast_model_failed = False
_escalate_pipe = None
_escalate_pipe_failed = False


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


def _fast_transcribe(wav_path: str) -> tuple[str, list[tuple[str, float]], float]:
    """Returns (text, all_language_probs, elapsed_ms)."""
    kind, model = _get_fast_model()
    t0 = time.time()
    if kind == "mlx":
        import mlx_whisper
        result = mlx_whisper.transcribe(wav_path, path_or_hf_repo=f"mlx-community/whisper-{model}")
        text = (result.get("text") or "").strip()
        # mlx-whisper doesn't expose a full language-probability distribution
        # the way faster-whisper does; fall back to its single detected
        # language at prob 1.0 -- the Devanagari/lexical backstop in
        # solution.common still catches code-switch clips this misses.
        lang = result.get("language", "en")
        probs = [(lang, 1.0)]
    else:
        segments, info = model.transcribe(
            wav_path, language=None, task="transcribe",
            language_detection_segments=1,
        )
        text = " ".join(s.text for s in segments).strip()
        probs = info.all_language_probs or [(info.language, info.language_probability)]
    return text, probs, (time.time() - t0) * 1000


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


def _escalate_transcribe(wav_path: str) -> tuple[str, float]:
    t0 = time.time()
    pipe = _get_escalate_pipe()
    out = pipe(wav_path, generate_kwargs={"task": "transcribe"}, chunk_length_s=30)
    text = (out.get("text") or "").strip()
    return text, (time.time() - t0) * 1000


def transcribe(wav_path: str, mode: str = "auto") -> dict:
    t0 = time.time()
    candidates: list[dict] = []
    model_ids: list[str] = []
    fast_text, fast_ms, escalate_text, escalate_ms = "", 0.0, "", 0.0
    route, route_reason, lang_guess = "no_model_available", "", "unknown"

    try:
        fast_text, all_probs, fast_ms = _fast_transcribe(wav_path)
        fast_text = clean_asr_artifact(fast_text)
        candidates.append({"engine": f"faster-whisper-{FASTPATH_MODEL}", "text": fast_text})
        model_ids.append(f"faster-whisper-{FASTPATH_MODEL}-int8")
        lang_guess = all_probs[0][0] if all_probs else "unknown"

        escalate, route_reason = should_escalate(all_probs, fast_text, HI_PROB_THRESHOLD)
        if mode == "fast":
            escalate = False
        elif mode in ("hinglish", "verbatim"):
            escalate = True

        if escalate:
            lang_guess = "hinglish"
            route = "fast"  # overwritten below if escalate succeeds
            try:
                escalate_text, escalate_ms = _escalate_transcribe(wav_path)
                escalate_text = clean_asr_artifact(escalate_text)
                candidates.append({"engine": ESCALATE_MODEL, "text": escalate_text})
                model_ids.append(ESCALATE_MODEL)
                route = "hinglish_escalate"
            except Exception as e:  # noqa: BLE001 - escalate must never sink the run
                route = "fast_fallback_escalate_error"
                candidates.append({"engine": ESCALATE_MODEL, "text": "",
                                    "note": f"escalate failed: {type(e).__name__}"})
        else:
            route = "fast"

    except Exception as e:  # noqa: BLE001 - no model installed: stay contract-valid
        candidates.append({"engine": "none", "text": "",
                            "note": f"plug your engine here ({type(e).__name__}: {e})"})

    # Final pick: prefer the escalate (faithful) result when we took that
    # path, but never let it return something worse than the fast result --
    # pick_best skips blank/looping candidates in priority order.
    if route == "hinglish_escalate":
        final_text = pick_best(escalate_text, fast_text)
    else:
        final_text = pick_best(fast_text)

    total_ms = (time.time() - t0) * 1000
    return {
        "text": final_text,
        "mode_used": mode,
        "language_guess": lang_guess,
        "timings_ms": {
            "total": round(total_ms),
            "asr": round(fast_ms + escalate_ms),
            "postprocess": round(max(0.0, total_ms - fast_ms - escalate_ms)),
        },
        "raw_candidates": candidates,
        "model_ids": model_ids,
        "local_only": True,
        "route": route,
        "route_reason": route_reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "fast", "hinglish", "verbatim"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = transcribe(args.input, args.mode)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.output}  ({result['timings_ms']['total']}ms, route={result['route']}, "
          f"local_only={result['local_only']})")


if __name__ == "__main__":
    main()
