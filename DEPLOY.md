# Deployment notes (Render, GitHub, env)

## GitHub clone failures on Render (`Couldn't connect to server` to github.com:443)

This is a **network path issue between Render’s build network and GitHub**, not your application code.

1. Retry the deploy when [GitHub Status](https://www.githubstatus.com) and [Render Status](https://status.render.com) are green.
2. Confirm the **repository URL and organization** in the Render dashboard match the repo that exists (e.g. `owner/repo` spelling).
3. For **private** repos, reconnect the **GitHub integration** and ensure Render’s GitHub App has access to that repository.
4. If failures continue for hours, open a ticket with **Render support** and attach the build log.

## Render cold start ("SERVICE WAKING UP" screen)

On the **free** plan, Render sleeps web services after ~15 minutes without traffic. The first visit then shows Render's loading terminal for 30–60 seconds.

**Mitigations (this repo):**

1. **GitHub Actions** — workflow `.github/workflows/keep-render-awake.yml` pings `www.balmoreslab.com`, `balmoreslab.com`, and the backend `/health` every 10 minutes. Enable **Actions** on the GitHub repo and ensure the default branch runs workflows.
2. **Always-on** — upgrade `balmores-structural-frontend` (and optionally the backend) to **Starter** in the Render dashboard for instant loads with no wake screen.

## Backend environment

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Optional. **SQLite** (`sqlite:///./data/balmores.db`) or **Postgres** (use Render’s internal URL in production). If unset, the API uses **in-memory** storage (resets on restart). |
| `ALLOWED_ORIGINS` | Comma-separated allowed browser origins for CORS in production. Example: `https://www.balmoreslab.com`. Default `*` if unset. |
| Rate limiting | Prefer **edge** limits (Render/nginx/Cloudflare). In-app rate limiting was removed to avoid FastAPI/Pydantic conflicts; add `slowapi` or similar in a follow-up if needed. |
| `SENTRY_DSN` | Optional error reporting. |
| `DEBUG` | Set to `1` to include exception text in JSON 500 responses (dev only). |

## New API behavior

- **`POST /chat/stream`**: NDJSON stream (`application/x-ndjson`) with `meta`, `delta`, and `complete` events. The frontend uses this for live typing; it falls back to `POST /chat` if the stream route is missing (404).
- **`X-Request-ID`**: Echoed on responses when not provided by the client.
- **JSON errors**: Many errors return `{ "error": { "message": "..." }, "request_id": "..." }`.

## OpenAPI types (frontend)

With the backend running locally:

```bash
cd frontend && npm install && npm run gen:api
```

This writes `lib/api-schema.d.ts` from `http://127.0.0.1:8000/openapi.json`.

## E2E tests

```bash
cd frontend && npm install && npx playwright install
# Terminal 1: backend on 8000, frontend on 3000
npm run e2e
```

Set `PLAYWRIGHT_BASE_URL` if the app is not on port 3000.
