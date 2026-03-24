# Structural Brain Prototype

Prototype full-stack workspace for a chat-first structural analysis product.

## What this prototype includes
- Chat-first UI inspired by ChatGPT/Grok
- Right-side live results panel
- Center 3D structural viewer in-browser
- FastAPI backend with:
  - natural-language project parsing
  - assumption/default engine
  - prototype structural inference service
  - ETABS verification job stub
  - report payload generation stub

## Important
This prototype is built to show the **overall architecture and look**.
It uses a **prototype inference engine** right now.
When your 5000-model brain is ready, replace:
- `backend/app/services/inference.py`
with your trained-model inference service and load your real bundle there.

## Run locally

### Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:3000  
Backend docs: http://localhost:8000/docs

## Repo structure
```text
frontend/   Next.js App Router UI
backend/    FastAPI API and services
```
