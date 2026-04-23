# BALMORES STRUCTURAL

Prototype full-stack website that turns a **natural-language structural brief** into a **finite-element analysis** and produces structured output for engineers.

The analysis kernel is the open-source **PyNite** FEM library (MIT-licensed), vendored directly into this repo so no external service is needed.

## What it solves

The assistant recognises three classes of problem from a plain-English prompt and runs the matching PyNite model:

- **2D beam** – simply supported / fixed / cantilever, with UDLs, DL + LL, and point loads. Returns **shear, moment and deflection diagrams** plus reactions and envelopes.
- **2D moment frame** – multi-bay, multi-storey planar frame with gravity and per-floor lateral loads. Returns beam / column envelopes, storey drift and reactions.
- **3D building frame** – irregular bay grid, storey heights, DL/LL (kPa), wind (kPa), seismic zone, SBC. Returns coloured 3D geometry, drift, reactions and member envelopes with an optional P-Δ analysis.

All outputs are reflected to the right side of the chat form in real time, including a coloured 3D viewer powered by React Three Fiber.

## Stack

- **Frontend:** Next.js 14 (App Router) + React + React Three Fiber
- **Backend:** FastAPI
- **FEM kernel:** [PyNite](./Pynite-main/Pynite-main) — vendored, MIT-licensed, open-source

## Quick start

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

Open:

- Frontend: http://localhost:3000
- Backend docs: http://localhost:8000/docs

### Environment

The frontend reaches the backend at `http://localhost:8000` by default. Override with `NEXT_PUBLIC_API_URL` if needed.

## Example prompts

Try these in the chat box on the homepage:

- `Simply supported steel beam, span 8 m, UDL 12 kN/m DL and 8 kN/m LL, with a 40 kN point load at midspan.`
- `Concrete cantilever beam, fixed at the left, span 4 m, 25 kN point load at 4 m from the left, DL 8 kN/m, LL 4 kN/m.`
- `2D RC moment frame, 3 bays of 6 m, 4 storeys at 3.5 m, DL 20 kN/m LL 8 kN/m on each beam, 25 kN lateral per floor.`
- `6-storey RC building, X-spans (6, 8, 6m), Y-spans (5, 5m), 3.5m storey heights, 4.5 kPa DL, 3 kPa LL, 200mm slab, 1 kPa wind, Seismic Zone 3, 200 kPa SBC.`

## License

PyNite is MIT-licensed (see [Pynite-main/Pynite-main/LICENSE](./Pynite-main/Pynite-main/LICENSE)).
All results produced by this prototype must be verified with a licensed engineer before they are used for real design.
