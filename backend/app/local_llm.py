"""Secure bridge to a locally-hosted LLM (Ollama) such as DeepSeek-R1.

Privacy & cybersecurity posture
------------------------------
This module deliberately:

* Only talks to a loopback address by default (``http://127.0.0.1:11434``).
  If ``LLM_OLLAMA_URL`` is overridden, we **refuse** any non-loopback host
  unless ``LLM_ALLOW_REMOTE=1`` is also set — this protects the user from
  accidentally piping prompts to a public Ollama server.
* Strips any reasoning traces emitted by DeepSeek-R1 (``<think>...</think>``)
  before forwarding tokens to the browser, so internal chain-of-thought
  never leaves the machine.
* Caps every request size (``LLM_MAX_INPUT_CHARS``) and response size
  (``LLM_MAX_OUTPUT_TOKENS``) to avoid prompt-injection or runaway costs.
* Implements an LRU cache so repeated identical questions return instantly
  without hitting the model again.
* Uses ``keep_alive`` so Ollama keeps the model warm in RAM/VRAM between
  requests, eliminating the cold-start penalty.

The module exposes:

* ``llm_health()``                       — returns ``{enabled, ok, model, ...}``
* ``stream_summary(prompt, fea_result)`` — generator that yields text chunks

No prompt content is ever logged at INFO level. Failures fall back to a
deterministic summary so the website never breaks if Ollama is offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Config (all env-driven so deploy never needs code changes)
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "deepseek-r1:latest"

LLM_ENABLED = os.getenv("LLM_ENABLED", "1").lower() in ("1", "true", "yes", "on")
LLM_OLLAMA_URL = os.getenv("LLM_OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
LLM_ALLOW_REMOTE = os.getenv("LLM_ALLOW_REMOTE", "0").lower() in ("1", "true", "yes")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.15"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
# We force ``think: False`` on every Ollama call (see below), so the model
# emits ONLY the user-visible answer. 512 tokens is plenty for a structural
# executive summary and keeps perceived latency under a few seconds.
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "512"))
LLM_KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "60m")
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "2048"))
LLM_CACHE_SIZE = int(os.getenv("LLM_CACHE_SIZE", "128"))
LLM_MAX_INPUT_CHARS = int(os.getenv("LLM_MAX_INPUT_CHARS", "8000"))
# Hard wall-clock budget for any single LLM call. 60 s is generous for a
# warm 8B Q4 model on CPU; we surface a fallback summary if we blow past it
# so the user never stares at a blank chat bubble.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
# Optional CPU thread hint for Ollama. 0 = let Ollama choose; positive = pin.
LLM_NUM_THREAD = int(os.getenv("LLM_NUM_THREAD", "0"))

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", flags=re.IGNORECASE)


def _is_loopback(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).hostname or ""
        return host.lower() in _LOOPBACK_HOSTS
    except Exception:
        return False


def _guard_target() -> Tuple[bool, str]:
    """Enforce the loopback-only privacy contract."""
    if not LLM_ENABLED:
        return False, "LLM bridge disabled (LLM_ENABLED=0)"
    if not _is_loopback(LLM_OLLAMA_URL) and not LLM_ALLOW_REMOTE:
        return (
            False,
            "LLM_OLLAMA_URL is non-loopback; set LLM_ALLOW_REMOTE=1 to opt out of "
            "the loopback-only safety rule (not recommended).",
        )
    return True, ""


# ---------------------------------------------------------------------------
# Tiny thread-safe LRU cache (sha256(prompt) -> text)
# ---------------------------------------------------------------------------


class _LRU:
    def __init__(self, capacity: int) -> None:
        self._cap = max(1, capacity)
        self._data: "OrderedDict[str, str]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            v = self._data.get(key)
            if v is None:
                return None
            self._data.move_to_end(key)
            return v

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._cap:
                self._data.popitem(last=False)


_CACHE = _LRU(LLM_CACHE_SIZE)


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _compact_fea_for_llm(fea: Dict[str, Any]) -> Dict[str, Any]:
    """Strip megabyte-scale arrays before showing the result to the LLM."""
    if not isinstance(fea, dict):
        return {}
    keep = {
        "analysis_type",
        "engine",
        "load_combination",
        "elapsed_ms",
        "parse_notes",
        "assumptions",
        "totals",
        "p_delta_note",
        "design_criteria",
    }
    safe: Dict[str, Any] = {k: fea[k] for k in keep if k in fea}

    cards = fea.get("result_cards") or []
    safe["result_cards"] = [
        {
            "label": c.get("label"),
            "value": c.get("value"),
            "unit": c.get("unit"),
            "tone": c.get("tone"),
        }
        for c in cards[:24]
        if isinstance(c, dict)
    ]
    beams = fea.get("beams") or []
    safe["beams_top"] = beams[:5]
    safe["beam_count"] = len(beams)
    columns = fea.get("columns") or []
    safe["columns_top"] = columns[:5]
    safe["column_count"] = len(columns)
    reactions = fea.get("base_reactions") or []
    safe["reactions_top"] = reactions[:6]
    safe["reaction_count"] = len(reactions)
    drifts = fea.get("storey_drifts") or []
    safe["storey_drifts"] = drifts[:12]
    return safe


_SYSTEM_PROMPT = (
    "You are Balmores AI, a senior licensed structural engineer. The PyNite "
    "FEM kernel has already produced numeric results — your job is a client-ready "
    "Markdown report with explicit engineering judgement.\n\n"
    "Required structure (use these exact headings):\n"
    "## Executive summary\n"
    "3–5 bullets: load combination, controlling member/action, governing reaction, "
    "drift or deflection check, and overall adequacy.\n"
    "## Recommendations\n"
    "3–6 actionable bullets (verification steps, detailing, ETABS cross-check, "
    "load-case review, section optimisation) grounded ONLY in the JSON.\n"
    "## Conclusion\n"
    "One short paragraph: PASS / MARGINAL / FAIL style verdict with the "
    "critical number that drives it.\n\n"
    "Hard rules:\n"
    "- Use ONLY numbers from the supplied PyNite JSON. Never invent values.\n"
    "- Flag DCR > 1.0, drift > h/400, or deflection exceedance explicitly.\n"
    "- Reference NSCP 2015 / ASCE 7 only when design_criteria already cites it.\n"
    "- No chain-of-thought. Output the final answer directly."
)


# ---------------------------------------------------------------------------
# Prompt canonicalisation — DeepSeek-R1 rewrites loose / typo-ridden briefs
# (e.g. "design 2m beam simply supptd 2kn.m") into the strict canonical form
# the regex parser understands. This is the rescue path when the deterministic
# parser raises ValueError.
# ---------------------------------------------------------------------------

_CANONICALIZE_SYSTEM = (
    "You normalise loose structural-engineering prompts into one canonical "
    "line that a strict regex parser can read. Output ONE line only. No "
    "reasoning, no preamble, no markdown — just the canonical sentence.\n\n"
    "Unit rules (NON-NEGOTIABLE):\n"
    "  - Beam loads are per metre: use kN/m (distributed) or kN (point).\n"
    "  - Building floor loads are area pressures: use kPa.\n"
    "  - Lengths in m, slab thickness in mm.\n\n"
    "Abbreviation hints: supptd→supported, simp→simply, cant→cantilever, "
    "rc→RC, kn.m/knm/kn-m→kNm (or kN/m for distributed), "
    "pt→point, sty/storey/floor→storey, bldg→building, hgt→storey height.\n\n"
    "Few-shot examples (mimic the OUT line exactly):\n\n"
    "USER: design 2m beam simply supptd 2kn.m\n"
    "OUT: Simply supported concrete beam, span 2 m, UDL 2 kN/m DL.\n\n"
    "USER: rc beam 6m, dl 12, ll 8\n"
    "OUT: Simply supported RC beam, span 6 m, DL 12 kN/m, LL 8 kN/m.\n\n"
    "USER: cant beam 4m fixed left 25kn pt tip\n"
    "OUT: Cantilever concrete beam, span 4 m, 25 kN point load at 4 m from the left.\n\n"
    "USER: continuous beam 4 spans 6m dl12 ll8\n"
    "OUT: Continuous concrete beam, 4 spans of 6 m, DL 12 kN/m, LL 8 kN/m.\n\n"
    "USER: 3 bay 6m 4 storey 3.5 dl20 ll8 lat25\n"
    "OUT: 2D RC moment frame, 3 bays of 6 m, 4 storeys at 3.5 m, DL 20 kN/m, LL 8 kN/m, 25 kN lateral per floor.\n\n"
    "USER: 5sty rc bldg manila spans 6 8 6 / 5 5 hgt 3.5 dl4.5 ll3\n"
    "OUT: 5-storey RC building in Manila, X-spans (6, 8, 6 m), Y-spans (5, 5 m), 3.5 m storey heights, 4.5 kPa DL, 3 kPa LL.\n\n"
    "Hard rules:\n"
    "  1. Preserve every number the user actually wrote.\n"
    "  2. If supports are missing for a beam, default to 'Simply supported'.\n"
    "  3. If material is missing, assume 'concrete'.\n"
    "  4. NEVER invent values that aren't stated or implied.\n"
)


def canonicalize_prompt(message: str, timeout: float = 90.0) -> Optional[str]:
    """Use the local LLM to rewrite ``message`` into the canonical regex form.

    Returns the canonical line, or ``None`` if the LLM is unavailable / fails.
    The output is sanitised: <think> blocks stripped, code-fences removed,
    only the FIRST non-empty line kept (LLM sometimes adds extra commentary
    despite the instructions).
    """
    enabled, _ = _guard_target()
    if not enabled:
        return None
    msg = (message or "").strip()
    if not msg:
        return None
    cache_key = _digest(LLM_MODEL, "canonicalize_v2", msg)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached or None

    canon_options: Dict[str, Any] = {
        "temperature": 0.05,
        "top_p": 0.9,
        "num_ctx": LLM_NUM_CTX,
        "num_predict": 160,
    }
    if LLM_NUM_THREAD > 0:
        canon_options["num_thread"] = LLM_NUM_THREAD
    body = {
        "model": LLM_MODEL,
        "stream": False,
        "keep_alive": LLM_KEEP_ALIVE,
        # Tell DeepSeek-R1 to skip its internal <think> trace — for this task
        # the few-shot examples are enough and silent reasoning would burn the
        # entire token budget. Drops latency from ~170 s to ~3 s.
        "think": False,
        "options": canon_options,
        "messages": [
            {"role": "system", "content": _CANONICALIZE_SYSTEM},
            {"role": "user", "content": msg[:LLM_MAX_INPUT_CHARS]},
        ],
    }
    try:
        req = Request(
            f"{LLM_OLLAMA_URL}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
    except Exception:  # noqa: BLE001
        return None

    raw = (payload.get("message") or {}).get("content", "") or ""
    cleaned = _post_clean(raw)
    cleaned = re.sub(r"^```[a-zA-Z]*\n?|```$", "", cleaned, flags=re.MULTILINE).strip()
    # Keep only the first non-empty line
    first = ""
    for line in cleaned.splitlines():
        line = line.strip().lstrip("-•").strip()
        if line:
            first = line
            break
    if not first:
        _CACHE.put(cache_key, "")
        return None
    # Reject obviously empty/echo responses
    if first.lower() == msg.lower() or len(first) < 12:
        _CACHE.put(cache_key, "")
        return None
    _CACHE.put(cache_key, first)
    return first


def _build_user_prompt(user_message: str, fea_compact: Dict[str, Any]) -> str:
    msg = user_message[:LLM_MAX_INPUT_CHARS]
    body = json.dumps(fea_compact, ensure_ascii=False, default=str)
    if len(body) > LLM_MAX_INPUT_CHARS:
        body = body[:LLM_MAX_INPUT_CHARS] + "... [truncated]"
    return (
        f"User question:\n{msg}\n\n"
        f"PyNite FEM result JSON (authoritative):\n```json\n{body}\n```\n\n"
        "Write the executive summary now."
    )


# ---------------------------------------------------------------------------
# Token streaming with <think>...</think> scrubbing
# ---------------------------------------------------------------------------


class _ThinkStripper:
    """Removes DeepSeek-R1 ``<think>...</think>`` blocks from a token stream.

    The block can span many tokens so we buffer until we see the close tag.
    """

    def __init__(self) -> None:
        self._inside = False
        self._buf = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []
        while True:
            if self._inside:
                close = self._buf.lower().find("</think>")
                if close == -1:
                    self._buf = self._buf[-16:]  # keep a tail just in case
                    return "".join(out)
                self._buf = self._buf[close + len("</think>") :]
                self._inside = False
            else:
                open_at = self._buf.lower().find("<think>")
                if open_at == -1:
                    if len(self._buf) > 8:
                        out.append(self._buf[:-8])
                        self._buf = self._buf[-8:]
                    return "".join(out)
                out.append(self._buf[:open_at])
                self._buf = self._buf[open_at + len("<think>") :]
                self._inside = True

    def flush(self) -> str:
        if self._inside:
            return ""
        tail = self._buf
        self._buf = ""
        return tail


def _post_clean(text: str) -> str:
    """Final safety net for any residual reasoning blocks."""
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# HTTP layer (stdlib only — no extra deps in requirements.txt)
# ---------------------------------------------------------------------------


def _ollama_post_stream(path: str, body: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    url = f"{LLM_OLLAMA_URL}{path}"
    data = json.dumps(body).encode("utf-8")
    req = Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
    )
    with urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def warm_model(timeout: float = 5.0) -> bool:
    """Trigger Ollama to load the model into RAM/VRAM without generating output.

    Returns True if the warm-up was dispatched successfully. Failures are
    swallowed — warm-up is best-effort.
    """
    enabled, _ = _guard_target()
    if not enabled:
        return False
    try:
        body = {
            "model": LLM_MODEL,
            "keep_alive": LLM_KEEP_ALIVE,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "options": {"num_predict": 1, "temperature": 0.0},
        }
        req = Request(
            f"{LLM_OLLAMA_URL}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout) as resp:
            resp.read(1)
        return True
    except Exception:
        return False


def llm_health() -> Dict[str, Any]:
    """Cheap probe — used by frontend to show the privacy badge."""
    enabled, reason = _guard_target()
    out: Dict[str, Any] = {
        "enabled": LLM_ENABLED,
        "model": LLM_MODEL,
        "endpoint": LLM_OLLAMA_URL,
        "loopback_only": _is_loopback(LLM_OLLAMA_URL),
        "ok": False,
    }
    if not enabled:
        out["reason"] = reason
        return out
    try:
        req = Request(f"{LLM_OLLAMA_URL}/api/tags", method="GET")
        with urlopen(req, timeout=3.0) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
        names = {m.get("name") for m in payload.get("models", []) if isinstance(m, dict)}
        out["installed_models"] = sorted([n for n in names if n])
        out["ok"] = LLM_MODEL in names
        if not out["ok"]:
            out["reason"] = (
                f"Model '{LLM_MODEL}' not pulled. "
                f"Run: ollama pull {LLM_MODEL}"
            )
    except URLError as e:
        out["reason"] = f"Ollama not reachable: {e.reason}"
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"Probe error: {e}"
    return out


# ---------------------------------------------------------------------------
# Public streaming summariser
# ---------------------------------------------------------------------------


def stream_summary(
    user_message: str,
    fea_result: Dict[str, Any],
    on_token: Optional[Iterable] = None,
) -> Iterator[str]:
    """Yield cleaned summary chunks. Falls back to deterministic text if the
    LLM is offline or guarded out, so the UX never breaks.
    """
    enabled, reason = _guard_target()
    if not enabled:
        yield _fallback_summary(fea_result, note=reason)
        return

    compact = _compact_fea_for_llm(fea_result)
    user_prompt = _build_user_prompt(user_message, compact)
    cache_key = _digest(LLM_MODEL, _SYSTEM_PROMPT, user_prompt)
    cached = _CACHE.get(cache_key)
    if cached:
        yield cached
        return

    options: Dict[str, Any] = {
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "num_ctx": LLM_NUM_CTX,
        "num_predict": LLM_MAX_OUTPUT_TOKENS,
    }
    if LLM_NUM_THREAD > 0:
        options["num_thread"] = LLM_NUM_THREAD
    body = {
        "model": LLM_MODEL,
        "stream": True,
        "keep_alive": LLM_KEEP_ALIVE,
        # Skip DeepSeek-R1's internal <think> block entirely. The summary is a
        # straightforward template; reasoning would just burn tokens and add
        # 30–90 s of perceived latency on CPU. This is the single biggest
        # speed win we have.
        "think": False,
        "options": options,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    stripper = _ThinkStripper()
    pieces: list[str] = []
    started_at = time.perf_counter()
    try:
        for event in _ollama_post_stream("/api/chat", body):
            msg = event.get("message") or {}
            chunk = msg.get("content") or ""
            if not chunk:
                if event.get("done"):
                    break
                continue
            clean = stripper.feed(chunk)
            if clean:
                pieces.append(clean)
                yield clean
            if event.get("done"):
                break
            if time.perf_counter() - started_at > LLM_TIMEOUT_SECONDS:
                yield "\n\n_(LLM timed out — see PyNite tables below for full results.)_"
                break
    except URLError as e:
        yield _fallback_summary(fea_result, note=f"Local LLM offline: {e.reason}")
        return
    except Exception as e:  # noqa: BLE001
        yield _fallback_summary(fea_result, note=f"Local LLM error: {e}")
        return

    tail = stripper.flush()
    if tail:
        pieces.append(tail)
        yield tail

    full = _post_clean("".join(pieces))
    if full:
        _CACHE.put(cache_key, full)


# ---------------------------------------------------------------------------
# General-purpose chat (no PyNite involved)
# ---------------------------------------------------------------------------
#
# When the user's message isn't a solvable structural brief (e.g. "hello",
# "what is buckling?", "how do I install ollama?") we still need to answer
# them so the chat never feels broken. This function streams a conversational
# DeepSeek-R1 reply with a system prompt that keeps it on-topic for structural
# engineering but allows general engineering / tool questions too.

_CHAT_SYSTEM = (
    "You are Balmores AI, a privacy-first structural-engineering assistant "
    "that runs **entirely on the user's local PC**. The user is talking to "
    "you through a chat box. Reply directly and concisely in Markdown.\n\n"
    "Behaviour rules:\n"
    "- If the user asks a structural-engineering question (beams, frames, "
    "buildings, loads, drift, code checks), answer it with practical "
    "engineering insight and, when appropriate, suggest the canonical brief "
    "they can paste back to trigger a PyNite solve (e.g. 'Simply supported "
    "concrete beam, span 6 m, DL 12 kN/m, LL 8 kN/m.').\n"
    "- If the user greets you or asks who you are, introduce yourself in one "
    "or two sentences and tell them what you can do (PyNite FEA + AI "
    "commentary, supports beams / 2D frames / 3D buildings up to "
    "60 storeys with P-Δ, drift and base reactions). Never mention the "
    "underlying model name — refer to yourself only as 'Balmores AI'.\n"
    "- If the user asks an off-topic question, answer briefly and gently "
    "steer them back to structural engineering.\n"
    "- Never reveal internal reasoning or <think> blocks. Be the final "
    "answer only.\n"
    "- Never claim to call PyNite — only the backend orchestrator runs the "
    "solver. If the user wants numbers, ask them to give spans, supports, "
    "and loads, then they can press Enter to run a real FEM solve.\n"
)


def stream_general_chat(
    user_message: str,
    history: Optional[list] = None,
) -> Iterator[str]:
    """Stream a DeepSeek-R1 reply to ``user_message`` without going through PyNite.

    Used as the catch-all so that *any* user input gets a response. ``history``,
    if provided, is a list of ``{"role": "user"|"assistant", "content": str}``
    dicts representing prior turns (oldest first, capped to the last ~6 turns
    by the caller).
    """
    enabled, reason = _guard_target()
    msg = (user_message or "").strip()
    if not msg:
        yield "Type a question or a structural brief and press Enter."
        return
    if not enabled:
        yield (
            "_Balmores AI is not reachable right now "
            f"({reason}). I can still solve structural prompts through the "
            "deterministic PyNite parser — try: "
            "'Simply supported concrete beam, span 6 m, DL 12 kN/m, LL 8 kN/m.'_"
        )
        return

    cache_key = _digest(LLM_MODEL, "chat_v1", _CHAT_SYSTEM, msg)
    cached = _CACHE.get(cache_key)
    if cached:
        yield cached
        return

    # Cap context: max 6 prior turns, strip to LLM_MAX_INPUT_CHARS per turn.
    messages: list[Dict[str, str]] = [{"role": "system", "content": _CHAT_SYSTEM}]
    if history:
        for turn in list(history)[-6:]:
            role = (turn.get("role") or "").lower()
            if role not in ("user", "assistant"):
                continue
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            messages.append({"role": role, "content": content[:LLM_MAX_INPUT_CHARS]})
    messages.append({"role": "user", "content": msg[:LLM_MAX_INPUT_CHARS]})

    chat_options: Dict[str, Any] = {
        "temperature": 0.3,
        "top_p": 0.9,
        "num_ctx": LLM_NUM_CTX,
        "num_predict": min(LLM_MAX_OUTPUT_TOKENS, 384),
    }
    if LLM_NUM_THREAD > 0:
        chat_options["num_thread"] = LLM_NUM_THREAD
    body = {
        "model": LLM_MODEL,
        "stream": True,
        "keep_alive": LLM_KEEP_ALIVE,
        # Disable visible chain-of-thought; chat answers should be the direct
        # reply, not a reasoning trace. Drops first-token latency from minutes
        # to a couple of seconds.
        "think": False,
        "options": chat_options,
        "messages": messages,
    }

    stripper = _ThinkStripper()
    pieces: list[str] = []
    started_at = time.perf_counter()
    try:
        for event in _ollama_post_stream("/api/chat", body):
            ev_msg = event.get("message") or {}
            chunk = ev_msg.get("content") or ""
            if not chunk:
                if event.get("done"):
                    break
                continue
            clean = stripper.feed(chunk)
            if clean:
                pieces.append(clean)
                yield clean
            if event.get("done"):
                break
            if time.perf_counter() - started_at > LLM_TIMEOUT_SECONDS:
                yield "\n\n_(Local LLM took too long — please try again or rephrase.)_"
                break
    except URLError as e:
        yield f"_Balmores AI is temporarily unreachable ({e.reason})._"
        return
    except Exception as e:  # noqa: BLE001
        yield f"_Balmores AI error: {e}_"
        return

    tail = stripper.flush()
    if tail:
        pieces.append(tail)
        yield tail

    full = _post_clean("".join(pieces))
    if full:
        _CACHE.put(cache_key, full)


def summarize_fea_result(user_message: str, fea_result: Dict[str, Any]) -> str:
    """Non-streaming summary for POST /llm/summarize (reliable after FEA completes)."""
    parts: list[str] = []
    for chunk in stream_summary(user_message, fea_result):
        if chunk:
            parts.append(chunk)
    text = _post_clean("".join(parts))
    if text:
        return text
    return _fallback_summary(fea_result, note="LLM commentary unavailable.")


def _fallback_summary(fea_result: Dict[str, Any], note: str = "") -> str:
    """Deterministic, useful text when the LLM is unavailable."""
    if not isinstance(fea_result, dict):
        return "Analysis complete. (LLM commentary unavailable.)"
    cards = {c.get("label"): c for c in fea_result.get("result_cards", []) if isinstance(c, dict)}

    def card(label: str) -> str:
        c = cards.get(label) or {}
        val = c.get("value", "—")
        unit = c.get("unit") or ""
        return f"**{val}{(' ' + unit) if unit else ''}**"

    lines = [
        "## Executive summary",
        f"- **Solver:** {fea_result.get('engine', 'PyNite FEM')} · "
        f"**Combo:** {fea_result.get('load_combination', '—')}",
    ]
    drifts = fea_result.get("storey_drifts") or []
    if drifts:
        worst = max(drifts, key=lambda d: d.get("max_drift_mm", 0) or 0)
        lines.append(
            f"- **Worst storey drift:** {worst.get('max_drift_mm', 0):.1f} mm at storey {worst.get('storey_index', '—')}"
        )
    if cards:
        for label in ("Max drift", "Beam max moment", "Beam max shear", "Column axial", "DCR proxy"):
            if label in cards:
                lines.append(f"- {label}: {card(label)}")
    lines.extend(
        [
            "",
            "## Recommendations",
            "- Cross-check member envelopes and support reactions in ETABS (import the exported .e2k).",
            "- Confirm load combinations and pattern directions match the governing ULS case above.",
            "- Review detailing and connection capacity outside this linear elastic envelope check.",
            "",
            "## Conclusion",
            "Deterministic PyNite results are shown in the tables; enable the local Balmores AI "
            "bridge (Ollama + deepseek-r1) for a model-generated executive summary.",
        ]
    )
    if note:
        lines.append(f"\n_{note}_")
    return "\n".join(lines)
