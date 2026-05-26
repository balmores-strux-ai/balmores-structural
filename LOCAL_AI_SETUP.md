# Local AI Mode — DeepSeek-R1 + PyNite (private, on-device)

This guide turns the Balmores Structural website into a **fully local** structural-engineering copilot:

- **PyNite** — open-source finite-element kernel (already vendored).
- **DeepSeek-R1** — 8B reasoning LLM running in your Ollama on `127.0.0.1:11434`.
- **FastAPI** backend bound to loopback, behind a token-bucket rate limiter.

Nothing about your project leaves your PC. The backend physically refuses to talk to a non-loopback LLM host unless you opt out of the safety rule.

## 1. Install Ollama and the model (one-time)

```powershell
winget install Ollama.Ollama
ollama pull deepseek-r1
```

Check it's serving:

```powershell
Invoke-WebRequest http://127.0.0.1:11434/api/tags | Select-Object -ExpandProperty Content
```

You should see `deepseek-r1:latest` in the JSON.

## 2. Start the website in PRIVATE LOCAL AI MODE

In one terminal:

```powershell
.\run-local-ai.bat
```

In another:

```powershell
.\run-frontend.bat
```

Open <http://127.0.0.1:3000>. You should see the **🔒 Private · deepseek-r1:latest** badge in the top-right and a "DeepSeek-R1 summary" checkbox already on.

The default sample is **6-storey RC building in Manila** so first impression is fast and consistent.

## 3. What's protected (cybersecurity posture)

| Risk | Mitigation |
|---|---|
| LLM endpoint accidentally pointed at a public Ollama | `LLM_ALLOW_REMOTE=0` (default). Backend refuses non-loopback URLs. |
| API exposed on LAN/Wi-Fi | Uvicorn bound to `127.0.0.1` in `run-local-ai.bat`. |
| Prompt-injection bloating prompts | `LLM_MAX_INPUT_CHARS=16000` truncates both the user prompt and the FEM JSON sent to the LLM. |
| Runaway DeepSeek-R1 reasoning loops | `LLM_MAX_OUTPUT_TOKENS=2048` ceiling + `LLM_TIMEOUT_SECONDS=300` hard cutoff. |
| Cross-site request forgery | CORS locked to `http://127.0.0.1:3000`, `http://localhost:3000`. |
| Reasoning traces leaking through to the client | `<think>...</think>` blocks are stripped server-side by `ThinkStripper`, with a final regex post-clean. |
| Repeated identical questions wasting GPU time | Thread-safe LRU cache keyed by SHA-256 of (model + system prompt + user prompt). |
| Brute-force flooding | Per-IP token bucket on `/llm/*` (30 capacity, 0.5/s refill by default). |
| Oversized request bodies | `MAX_BODY_BYTES=2097152` middleware rejects >2 MiB chat bodies. |
| Browser-level XSS / clickjacking | `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy` headers via `SecurityHeadersMiddleware`. |
| Server logs leaking prompt content | Access log is structured JSON of method/path/status/duration only — prompt bodies are never logged. |

If you ever want to expose the LLM to your LAN (e.g. multiple PCs in a studio), set:

```powershell
$env:LLM_OLLAMA_URL = "http://192.168.1.50:11434"
$env:LLM_ALLOW_REMOTE = "1"          # explicit opt-in
$env:ALLOWED_ORIGINS = "http://192.168.1.50:3000"
```

…but only do that behind a private VPN/firewall. Never expose Ollama directly to the public internet — it has no authentication.

## 4. Performance tuning

DeepSeek-R1 8B on CPU is the bottleneck (typically 5–15 tokens/s). The website already mitigates this by:

- **Pre-warming** the model on FastAPI startup (`warm_model()` keeps it in RAM).
- **Streaming** tokens to the browser so the user sees the summary materialise live.
- **Keep-alive 30 m** so subsequent prompts skip the cold-load.
- **LRU cache** so the same question returns instantly.

If you have a GPU, install the Ollama GPU build — token throughput typically jumps 3–10×.

If you want maximum speed and don't need R1's reasoning quality:

```powershell
$env:LLM_MODEL = "llama3.2:3b"
ollama pull llama3.2:3b
```

The site auto-detects this and re-labels the badge.

## 5. How the request flows

```
Browser ── POST /llm/ask/stream ──► FastAPI ──► run_in_executor(_run_prompt_pipeline)
                                       │              ↓
                                       │           PyNite FEM solve
                                       │              ↓
                                       └────► Ollama (127.0.0.1:11434)
                                                      │
                                                      ▼
                                                DeepSeek-R1 streams tokens
                                                      │
                                              <think> blocks stripped
                                                      │
                                          NDJSON `{"type":"llm_token"}` lines
                                                      │
                                                      ▼
                                       Browser renders Markdown live
```

Each NDJSON line is one of: `stage`, `tick`, `llm_token`, `complete`, `error`. The frontend mutates the assistant bubble token-by-token so the user sees the reply form in real time, exactly like ChatGPT.

## 6. Quick sanity check

```powershell
# Backend health (privacy badge data)
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 4
# LLM bridge only
Invoke-RestMethod http://127.0.0.1:8000/llm/health
```

`/llm/health` should return `ok: true` once the model is pulled and Ollama is running.
