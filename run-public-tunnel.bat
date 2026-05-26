@echo off
REM ==========================================================================
REM  Balmores Structural — PUBLIC TUNNEL MODE
REM
REM  Exposes the local AI engine (Ollama + PyNite) to the public website via
REM  a Cloudflare Tunnel. Your PC stays the brain — the public URL is just a
REM  thin pipe protected by an API key + rate limit.
REM
REM  Prerequisites (one-time):
REM    1) winget install Cloudflare.cloudflared
REM    2) Ollama already installed + `ollama pull deepseek-r1`
REM    3) (optional) cloudflared login + create a named tunnel for stable URLs
REM
REM  Security model:
REM    * Backend refuses /llm/* unless API_KEY matches X-API-Key header
REM    * Per-IP token-bucket rate limit (LLM_RATE_CAPACITY tokens / refill)
REM    * Output and input caps prevent runaway generation / prompt injection
REM    * No prompt content is logged
REM    * If the API key isn't set the backend will boot but REFUSE every
REM      /llm/* request — fail-closed by design
REM ==========================================================================

setlocal EnableDelayedExpansion

cd /d "%~dp0backend"

if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt -q

REM ----- Generate a fresh API key if none was supplied --------------------
if "%API_KEY%"=="" (
  for /f "delims=" %%i in ('powershell -NoProfile -Command "[guid]::NewGuid().ToString('N')"') do set API_KEY=%%i
)

REM ----- Public-tunnel security posture -----------------------------------
set LLM_LOCAL_ONLY=0
set LLM_PUBLIC_TUNNEL=1
set SECURITY_HEADERS=1
set ACCESS_LOG_JSON=1
set MAX_BODY_BYTES=2097152
REM Tighter rate limit in public mode so a single client can't pin the GPU.
set LLM_RATE_CAPACITY=12
set LLM_RATE_REFILL_PER_SEC=0.25

REM ----- Local LLM bridge (speed-tuned for public chat) -------------------
set LLM_ENABLED=1
set LLM_OLLAMA_URL=http://127.0.0.1:11434
set LLM_ALLOW_REMOTE=0
if "%LLM_MODEL%"=="" set LLM_MODEL=deepseek-r1:latest
REM Keep the model in RAM/VRAM for an hour so consecutive public requests
REM avoid the cold-load penalty (~5–15 s on an 8B Q4 model).
set LLM_KEEP_ALIVE=60m
REM Hard caps tuned for "fast first answer" UX. The summary template needs
REM only ~300–500 tokens; thinking is disabled in code.
set LLM_MAX_OUTPUT_TOKENS=512
set LLM_NUM_CTX=2048
set LLM_TIMEOUT_SECONDS=60
set LLM_PHASE_BUDGET_SECONDS=45

REM ----- CORS — restrict to your real public website + localhost ----------
REM Public-tunnel mode should not inherit a local-only ALLOWED_ORIGINS value
REM from a prior run-local-ai.bat shell.
set ALLOWED_ORIGINS=https://www.balmoreslab.com,https://balmoreslab.com,http://127.0.0.1:3000,http://localhost:3000

REM ----- Free port 8000 so a stale backend can't hijack the boot ----------
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue; if ($c) { foreach ($x in $c) { try { Stop-Process -Id $x.OwningProcess -Force -ErrorAction SilentlyContinue } catch {} }; Start-Sleep -Seconds 1 }"

REM ----- Verify Ollama is up ----------------------------------------------
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri http://127.0.0.1:11434/api/tags -UseBasicParsing -TimeoutSec 3 | Out-Null; Write-Host 'Ollama OK' -ForegroundColor Green } catch { Write-Host 'WARNING: Ollama not responding on 127.0.0.1:11434. Start it before continuing.' -ForegroundColor Yellow }"

REM ----- Verify cloudflared is installed ----------------------------------
REM WinGet's MSI sometimes installs successfully before the current shell's
REM PATH refreshes. Prefer PATH, then fall back to the standard MSI location.
set CLOUDFLARED_EXE=cloudflared
where cloudflared >NUL 2>&1
if errorlevel 1 (
  for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$c=@((Join-Path ${env:ProgramFiles(x86)} 'cloudflared\cloudflared.exe'), (Join-Path $env:ProgramFiles 'cloudflared\cloudflared.exe')); $p=$c | Where-Object { Test-Path $_ } | Select-Object -First 1; if ($p) { $p }"`) do set "CLOUDFLARED_EXE=%%i"
  if "!CLOUDFLARED_EXE!"=="cloudflared" (
    echo.
    echo cloudflared is not installed.
    echo Install with:  winget install Cloudflare.cloudflared
    echo Or download:   https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
    pause
    exit /b 1
  )
)

echo.
echo ==========================================================================
echo  PUBLIC TUNNEL MODE  -  Balmores Structural
echo  -------------------------------------------------------------------------
echo  API_KEY ........ %API_KEY%
echo  Local backend .. http://127.0.0.1:8000
echo  CORS ........... %ALLOWED_ORIGINS%
echo  Tunnel mode .... LLM_PUBLIC_TUNNEL=%LLM_PUBLIC_TUNNEL%
echo  Rate limit ..... %LLM_RATE_CAPACITY% reqs burst, refill %LLM_RATE_REFILL_PER_SEC%/s
echo  -------------------------------------------------------------------------
echo  Save the API_KEY above. Configure your public site to send it as:
echo      X-API-Key: %API_KEY%
echo  In Render (or wherever the public frontend is hosted) set:
echo      BACKEND_PROXY_URL=https://YOUR-CLOUDFLARE-URL
echo      BACKEND_API_KEY=%API_KEY%
echo  Leave NEXT_PUBLIC_API_URL and NEXT_PUBLIC_API_KEY empty for best security.
echo  -------------------------------------------------------------------------
echo  A new window will open running cloudflared. Watch its output for the
echo  trycloudflare.com URL (or your named-tunnel hostname) and paste that
echo  into BACKEND_PROXY_URL.
echo ==========================================================================
echo.

REM Start the tunnel in a separate window so the backend logs stay readable.
start "Balmores - Cloudflare Tunnel" cmd /k ""%CLOUDFLARED_EXE%" tunnel --url http://127.0.0.1:8000"

REM Boot the backend in the foreground.
uvicorn app.main:app --host 127.0.0.1 --port 8000

endlocal
