from __future__ import annotations

import re
from collections import Counter

# --- text normalization / matching ----------------------------------------

_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

HINDI_MARKERS = frozenset({
    "hai", "hain", "tha", "thi", "nahi", "nahin", "kya", "kyun", "kyon",
    "kaise", "kaun", "kahan", "kab", "matlab", "abhi", "pehle", "baad",
    "lekin", "magar", "aur", "karo", "karna", "karenge", "sikhenge", "dekho",
    "dekhna", "chahiye", "bhi", "yeh", "woh", "iska", "uska", "humko", "humne",
    "aapko", "aapne", "tumko", "mein", "unka", "uska", "raha", "rahi", "rahe",
    "gaya", "gayi", "gaye", "mat", "abhi", "kyunki", "isliye", "phir", "wala",
    "wali", "wale", "bata", "batao", "sunna", "sunno", "kijiye", "kijiyega",
})


def tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text or ""))


def has_hindi_lexical_marker(text: str) -> bool:
    toks = {t.lower() for t in tokens(text)}
    return bool(toks & HINDI_MARKERS)


def should_escalate(all_language_probs: list[tuple[str, float]] | None,
                     fast_text: str,
                     hi_prob_threshold: float = 0.15) -> tuple[bool, str]:
    
    probs = dict(all_language_probs or [])
    hi_prob = probs.get("hi", 0.0)
    if hi_prob >= hi_prob_threshold:
        return True, f"audio_lid_hi_prob={hi_prob:.2f}"
    if has_devanagari(fast_text):
        return True, "devanagari_in_fast_text"
    if has_hindi_lexical_marker(fast_text):
        return True, "hindi_lexical_marker_in_fast_text"
    return False, f"english_confident(hi_prob={hi_prob:.2f})"


# --- reliability guards -----------------------------------------------------

def has_repetition_loop(text: str, n: int = 3, k: int = 4) -> bool:
    """Same definition as scorecard.has_repetition_loop: an n-gram repeated
    >= k times is treated as a degenerate decode loop."""
    toks = [t.lower() for t in tokens(text)]
    if len(toks) < n * k:
        return False
    grams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return max(grams.values(), default=0) >= k


def is_blank(text: str) -> bool:
    return not tokens(text)


def clean_asr_artifact(text: str) -> str:
    """Strip whisper-family bracketed non-speech tags so they never leak into
    a final transcript (e.g. '[BLANK_AUDIO]', '[MUSIC]')."""
    s = (text or "").strip()
    if re.fullmatch(r"\[(?:BLANK_AUDIO|MUSIC|NOISE|SILENCE|INAUDIBLE)\]", s, re.I):
        return ""
    return s


# --- streaming commit logic (shared with the reference draft.py shape) -----

def common_word_prefix(left: str, right: str) -> str:
    """Longest common leading-word prefix between two decodes of overlapping
    audio -- the part that's stopped changing across re-decodes, so it's safe
    to commit (never rewrite) per the streaming contract."""
    lw, rw = tokens(left), tokens(right)
    out: list[str] = []
    for a, b in zip(lw, rw):
        if a.lower() != b.lower():
            break
        out.append(b)
    return " ".join(out)


def pick_best(*candidates: str) -> str:
    """First non-blank, non-looping candidate, in priority order. Guarantees
    we never return a blank or a degenerate loop while a usable alternative
    exists -- matches the 'never embarrass itself' requirement."""
    for c in candidates:
        c = clean_asr_artifact(c)
        if c and not has_repetition_loop(c):
            return c
    # everything was blank/looping: return the least-bad non-empty option,
    # or empty string as an absolute last resort (never crash).
    for c in candidates:
        c = clean_asr_artifact(c)
        if c:
            return c
    return ""
