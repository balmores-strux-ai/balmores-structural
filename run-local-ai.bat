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

REM ----- Local LLM bridge -----
set LLM_ENABLED=1
set LLM_OLLAMA_URL=http://127.0.0.1:11434
set LLM_ALLOW_REMOTE=0
if "%LLM_MODEL%"=="" set LLM_MODEL=deepseek-r1:latest
set LLM_KEEP_ALIVE=30m
set LLM_MAX_OUTPUT_TOKENS=2048
set LLM_TIMEOUT_SECONDS=300
set LLM_RATE_CAPACITY=30
set LLM_RATE_REFILL_PER_SEC=0.5

REM ----- Verify Ollama is up before starting (best-effort) -----
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri http://127.0.0.1:11434/api/tags -UseBasicParsing -TimeoutSec 3 | Out-Null; Write-Host 'Ollama OK' -ForegroundColor Green } catch { Write-Host 'WARNING: Ollama not responding on 127.0.0.1:11434. Start it with `ollama serve` in another terminal, or install via winget install Ollama.Ollama.' -ForegroundColor Yellow }"

echo.
echo ==========================================================================
echo  PRIVATE LOCAL AI MODE
echo    API   : http://127.0.0.1:8000  (loopback only — not reachable from LAN)
echo    LLM   : %LLM_MODEL% via %LLM_OLLAMA_URL%
echo    Front : run-frontend.bat  in another terminal
echo ==========================================================================
echo.

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

endlocal
