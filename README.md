# BALMORES STRUCTURAL

Prototype full-stack website that turns a **natural-language structural brief** into a **finite-element analysis** and produces structured output for engineers.

The analysis kernel is the open-source **PyNite** FEM library (MIT-licensed), vendored directly into this repo so no external service is needed.

## What it solves

The assistant recognises three classes of problem from a plain-English prompt and runs the matching PyNite model:

- **2D beam** â€” simply supported / fixed / cantilever, **plus continuous beams with 2, 3, 4, or 5 supports** (multi-span). Returns shear, moment and deflection diagrams plus reactions and envelopes.
- **2D moment frame** â€” multi-bay, multi-storey planar frame with gravity and per-floor lateral loads. Returns beam / column envelopes, storey drift and reactions.
- **3D building frame** â€” irregular bay grid in X and Y, up to **60 storeys** (the practical limit for the shared-instance solver), DL/LL (kPa), wind (kPa), seismic zone, SBC. Returns drift, reactions and member envelopes with optional P-Î” analysis.

## Location-aware design criteria

Mention a city in your prompt â€” *"30-storey RC tower in Cebu"*, *"office in Tokyo"*, *"warehouse in Singapore"* â€” and the backend automatically resolves:

| Parameter | Source |
|---|---|
| Design wind speed V (m/s, 3-s gust 50-yr) | NSCP 2015 (PH) / ASCE 7-22 (US) / Eurocode (EU) / AS-NZS 1170 / KBC / GB-50009 etc. |
| Velocity pressure q = 0.613Â·VÂ² | Derived |
| Seismic zone, PGA, base-shear coeff. V/W | Same code per region |
| Allowable soil bearing (SBC) | Curated city table |
| Load combinations (ULS / SLS) | Generic limit-state envelope |

Every assumption is shown to the user in a **Design criteria** card on the right-hand side of the chat. If no location is provided, generic moderate values are used and clearly tagged `ASSUMED FALLBACK`.

Built-in cities include: Manila, Quezon City, Makati, Taguig, Cebu, Davao, Iloilo, Tacloban, Baguio, Cagayan de Oro, Bacolod, Tokyo, Osaka, Singapore, Hong Kong, Bangkok, Jakarta, Kuala Lumpur, Ho Chi Minh, Hanoi, Seoul, Taipei, Shanghai, Beijing, New York, San Francisco, Los Angeles, Chicago, Miami, Houston, London, Paris, Berlin, Rome, Madrid, Sydney, Melbourne, Auckland, Wellington.

## Live progress overlay

Long 3D solves (40 + storeys with P-Î” can run for 30â€“60 s on free hosting) used to feel frozen. The solver now streams **STAAD-style progress events** over an NDJSON endpoint (`POST /fea/analyze-prompt/stream`), and the frontend shows a live percent counter, gradient progress bar, and named stage transitions:

```
Connecting to PyNite kernel
Building nodes, members, sections, supports
Assembling stiffness blocks
Applying gravity, wind and seismic load cases
Sparse Cholesky factorisation of K
Solving KÂ·u = F (load combinations)
P-Î” second-order iteration
Extracting member envelopes and storey drift
Formatting tables and design criteria
Solve complete
```

## Performance notes

- PyNite is configured with `sparse=True` everywhere â€” assembly + `scipy.sparse.linalg.spsolve` keeps the linear-algebra path fast even at 21 000 + DOF.
- The FastAPI app pre-warms PyNite + matplotlib on startup so the **first** request after a deploy doesn't pay the cold-import cost.
- A **storey cap of 60** raises a friendly `ValueError` instead of letting a large 100-storey job time out the worker. (Local benchmark on the Render-equivalent dev profile: 60 storeys with full P-Î” â‰ˆ 43 s; 100 storeys linear â‰ˆ 45 s.)
- Sample prompts in the UI cover the full range â€” single-bay portal up to a 60-storey stress test in Taipei.

## Stack

- **Frontend:** Next.js 14 (App Router) + React + a small live-progress component.
- **Backend:** FastAPI with NDJSON streaming.
- **FEM kernel:** [PyNite](./Pynite-main/Pynite-main) â€” vendored, MIT-licensed, open-source.

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

Try these in the chat box:

```
Simply supported steel beam, span 8 m, UDL 12 kN/m DL and 8 kN/m LL, with a 40 kN point load at midspan.
Concrete cantilever beam, fixed at the left, span 4 m, 25 kN point load at 4 m from the left, DL 8 kN/m, LL 4 kN/m.
Continuous concrete beam, 2 spans of 6 m, DL 15 kN/m, LL 10 kN/m.
Continuous steel beam with 3 spans of 5, 6, 5 m. DL 18 kN/m, LL 12 kN/m, 50 kN at 8.5 m.
Continuous concrete beam, 4 spans of 7 m, DL 20 kN/m, LL 15 kN/m.
Continuous concrete beam, spans (6, 8, 10, 8, 6 m), 6 supports. DL 22 kN/m, LL 18 kN/m. Left and right ends fixed.
2D RC moment frame, 3 bays of 6 m, 4 storeys at 3.5 m, DL 20 kN/m LL 8 kN/m, 25 kN lateral per floor.
2D structural steel moment frame, 5 bays of 7 m, 6 storeys at 3.6 m, DL 18 kN/m, LL 10 kN/m, 35 kN wind per floor.
6-storey RC building in Manila, X-spans (6, 8, 6m), Y-spans (5, 5m), 3.5m storey heights, 4.5 kPa DL, 3 kPa LL, 200mm slab.
30-storey RC tower in Cebu, X-spans (6, 8, 12, 8, 6m), Y-spans (5, 9, 9, 5m), 3.5m storey heights, 4.5 kPa DL, 3 kPa LL, 200mm slab.
60-storey RC tower in Taipei, X-spans (6, 8, 12, 8, 6m), Y-spans (5, 9, 9, 5m), 3.5m storey heights, 4.5 kPa DL, 3 kPa LL, 200mm slab.
25-storey structural steel tower in Tokyo, X-spans (8, 10, 8m), Y-spans (6, 6m), 3.8m storey heights, 3 kPa DL, 4 kPa LL.
12-storey RC building in Singapore, X-spans (7, 7, 7m), Y-spans (6, 6m), 3.6m storey heights, 4 kPa DL, 3 kPa LL, 180 mm slab.
```

## License

PyNite is MIT-licensed (see [Pynite-main/Pynite-main/LICENSE](./Pynite-main/Pynite-main/LICENSE)).
All results produced by this prototype must be verified with a licensed engineer before they are used for real design.

