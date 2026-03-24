# BALMORES STRUCTURAL

Chat-style structural assistant: plain English → 3D frame, linear FEM, drift check, graphs, ETABS-oriented text export.

## Local run

See `START_HERE.txt` or:

```bash
pip install -r requirements.txt
copy .env.example .env   # add OPENAI_API_KEY
uvicorn app:app --host 127.0.0.1 --port 8000
```

## Deploy on Render

1. Push this repo to **GitHub** (see below).
2. In [Render](https://render.com): **New +** → **Blueprint** → connect the repo → apply `render.yaml`.
3. In the Render dashboard, set **Environment**:
   - `OPENAI_API_KEY` — required for Build & Analyze / Ask.
   - Optional: `BALMORES_BRAIN_PT` — only if you host a small `.pt` somewhere reachable (Render disk is ephemeral; prefer bundling in image or external URL — see Render docs for persistent disk if needed).

**Note:** `.pt` brain files are gitignored by default. Train/deploy weights separately if they are large.

## Push to GitHub (automatic on your PC)

Git and GitHub CLI should already be installed (or run `winget install Git.Git` and `winget install GitHub.cli`).

**Two commands** (login once, then push + create repo):

```powershell
cd c:\Users\dell\Desktop\balmores-strux-ai
gh auth login
.\scripts\auto-push-to-github.ps1
```

This creates a **public** repo named `balmores-structural` on your GitHub account (if `origin` is not set yet) and pushes `main`.

- Different repo name: `.\scripts\auto-push-to-github.ps1 my-custom-name`
- Non-interactive: set a classic PAT with **repo** scope as `GH_TOKEN`, or put the PAT in one-line file `gh_token.txt` (gitignored), then run the script.

Manual `git push` still works if you prefer GitHub Desktop or your own remote.

## License

Use and modify for your projects; verify all structural work with a licensed PE and ETABS (or equivalent).
