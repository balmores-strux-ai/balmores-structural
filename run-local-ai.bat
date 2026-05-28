@echo off
REM ==========================================================================
REM  Balmores Structural — PRIVATE LOCAL AI MODE
REM
REM  Starts the FastAPI backend bound to 127.0.0.1 (no public network) and
REM  enables the local DeepSeek-R1 bridge via Ollama. Use this when you want
REM  to run the website entirely on your own PC. No prompt ever leaves the
REM  machine.
REM
REM  Prerequisites (one-time):
REM    1) winget install Ollama.Ollama        (or download from ollama.com)
REM    2) ollama pull deepseek-r1
REM ==========================================================================

setlocal

cd /d "%~dp0backend"

if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt -q

REM ----- Privacy hardening (override only if you know what you are doing) -----
set ALLOWED_ORIGINS=http://127.0.0.1:3000,http://localhost:3000
set SECURITY_HEADERS=1
set ACCESS_LOG_JSON=1
set MAX_BODY_BYTES=2097152

REM ----- Local LLM bridge (speed-tuned for snappy chat) -------------------
set LLM_ENABLED=1
set LLM_OLLAMA_URL=http://127.0.0.1:11434
set LLM_ALLOW_REMOTE=0
if "%LLM_MODEL%"=="" set LLM_MODEL=deepseek-r1:latest
REM Keep the model loaded in RAM/VRAM so consecutive prompts are instant.
set LLM_KEEP_ALIVE=60m
REM Hard caps + small context for "first token in 1–3 s" UX. Internal
REM <think> reasoning is disabled in code; the model emits the answer only.
set LLM_MAX_OUTPUT_TOKENS=700
set LLM_NUM_CTX=2048
REM DeepSeek-R1 is a reasoning model and needs headroom to finish the full
REM Executive summary / Recommendations / Conclusion. The web UI already shows
REM PyNite results instantly, so this longer budget only affects the AI prose
REM that streams in afterwards — the page never blocks on it.
set LLM_TIMEOUT_SECONDS=180
set LLM_PHASE_BUDGET_SECONDS=180
set LLM_RATE_CAPACITY=30
set LLM_RATE_REFILL_PER_SEC=0.5
set LLM_LOCAL_ONLY=1

REM ----- Free port 8000 if a previous backend is still hogging it ---------
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue; if ($c) { foreach ($x in $c) { try { Stop-Process -Id $x.OwningProcess -Force -ErrorAction SilentlyContinue } catch {} }; Start-Sleep -Seconds 1; Write-Host 'Freed port 8000.' -ForegroundColor Yellow }"

REM ----- Verify Ollama is up before starting (best-effort) -----------------
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri http://127.0.0.1:11434/api/tags -UseBasicParsing -TimeoutSec 3 | Out-Null; Write-Host 'Ollama OK' -ForegroundColor Green } catch { Write-Host 'WARNING: Ollama not responding on 127.0.0.1:11434. Start it with `ollama serve` in another terminal, or install via winget install Ollama.Ollama.' -ForegroundColor Yellow }"

echo.
echo ==========================================================================
echo  PRIVATE LOCAL AI MODE  -  Balmores Structural
echo  -------------------------------------------------------------------------
echo  API     : http://127.0.0.1:8000   (loopback only - not reachable on LAN)
echo  LLM     : %LLM_MODEL%
echo  Ollama  : %LLM_OLLAMA_URL%
echo  Routes  : /llm/health  /llm/summarize  /fea/analyze-prompt  (+ /llm/ask/stream legacy)
echo  Front   : run-frontend.bat  in another terminal, then open
echo            http://127.0.0.1:3000
echo  -------------------------------------------------------------------------
echo  After the server boots, open the chat page. The header should show
echo  "Private . deepseek-r1:latest" with a green dot. Then you can type
echo  anything (incl. "hello") and DeepSeek-R1 will reply.
echo ==========================================================================
echo.

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

endlocal
