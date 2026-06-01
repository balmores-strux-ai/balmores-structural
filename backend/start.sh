#!/usr/bin/env bash
# Render web service: bind 0.0.0.0:$PORT immediately (see app lifespan — no blocking prewarm).
set -o errexit
export MPLBACKEND="${MPLBACKEND:-Agg}"
PORT="${PORT:-8000}"
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1
