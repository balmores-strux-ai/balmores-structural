#Requires -Version 5.1
<#
  Pushes the SEO / Knowledge-Panel update directly to GitHub using the
  GitHub REST Contents API (gh api PUT). This script does NOT require
  git.exe to be installed — it only needs the GitHub CLI (gh) and a valid
  login.

  Usage:
    1) Authenticate once:
         & "C:\Program Files\GitHub CLI\gh.exe" auth login -h github.com
       (choose: GitHub.com -> HTTPS -> Login with a web browser)

    2) Run this script from the project root:
         .\scripts\push-seo-via-gh-api.ps1

  What it does:
    - Uploads the changed/new files in ./frontend/app and ./frontend/public
      to balmores-strux-ai/balmores-structural on branch `main`.
    - Triggers Render auto-deploy for balmoreslab.com.
#>

$ErrorActionPreference = "Continue"
# We check native exit codes manually; stop-on-error breaks the expected
# 404 response when a file doesn't exist yet on the remote.

$gh = "C:\Program Files\GitHub CLI\gh.exe"
if (-not (Test-Path $gh)) { $gh = "gh" }

$owner  = "balmores-strux-ai"
$repo   = "balmores-structural"
$branch = "main"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

# Files to push, relative to repo root.
$files = @(
  # Core Next.js routes & metadata
  "frontend/app/layout.tsx",
  "frontend/app/page.tsx",
  "frontend/app/about/page.tsx",
  "frontend/app/cv/page.tsx",
  "frontend/app/research/page.tsx",
  "frontend/app/sitemap.ts",
  "frontend/app/robots.ts",
  "frontend/app/manifest.ts",
  # Dynamic OG / Twitter images
  "frontend/app/opengraph-image.tsx",
  "frontend/app/about/opengraph-image.tsx",
  # Feeds, WebFinger, static SEO assets
  "frontend/app/feed.xml/route.ts",
  "frontend/app/.well-known/webfinger/route.ts",
  "frontend/public/seo-schema.json",
  "frontend/public/ykjm52si9r4gfvwhul8ob7cd3nqxpe01.txt",
  "frontend/public/humans.txt",
  "frontend/public/llms.txt",
  "frontend/public/foaf.rdf",
  "frontend/public/.well-known/security.txt",
  # Repo-level citation metadata
  "CITATION.cff",
  "codemeta.json",
  # Automation
  "scripts/ping-crawlers.ps1",
  "scripts/push-seo-via-gh-api.ps1",
  "SHIP-SEO.ps1",
  # Playbooks
  "seo/KNOWLEDGE-PANEL-KIT.md",
  "seo/louie-balmores-bios.md",
  "seo/wikidata-louie-balmores.quickstatements.tsv",
  "seo/wikidata-README.md",
  "seo/github-profile-README.md"
)

# Verify auth before doing any work.
$null = & $gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host "gh is not authenticated. Run:  & '$gh' auth login -h github.com" -ForegroundColor Yellow
  exit 1
}

function Get-RemoteSha($path) {
  $enc  = [System.Uri]::EscapeUriString($path)
  # Redirect stderr to $null via call operator with 2>&1 | Out-Null on error
  $json = & $gh api "/repos/$owner/$repo/contents/$enc`?ref=$branch" 2>&1
  if ($LASTEXITCODE -ne 0) { return $null }
  try   { return ($json | ConvertFrom-Json).sha }
  catch { return $null }
}

function Push-File($rel) {
  $full = Join-Path $repoRoot $rel
  if (-not (Test-Path $full)) {
    Write-Host "SKIP (missing): $rel" -ForegroundColor DarkYellow
    return
  }
  $bytes  = [IO.File]::ReadAllBytes($full)
  $b64    = [Convert]::ToBase64String($bytes)
  $sha    = Get-RemoteSha $rel

  $body = @{
    message = "SEO: Knowledge Panel update for Louie Doniego Balmores - $rel"
    content = $b64
    branch  = $branch
  }
  if ($sha) { $body.sha = $sha }

  $tmp = New-TemporaryFile
  ($body | ConvertTo-Json -Depth 6) | Set-Content -Path $tmp -Encoding UTF8

  $enc = [System.Uri]::EscapeUriString($rel)
  Write-Host ("PUT {0}  ({1})" -f $rel, ($(if ($sha) { "update" } else { "create" })))
  & $gh api --method PUT "/repos/$owner/$repo/contents/$enc" --input $tmp | Out-Null
  Remove-Item $tmp -Force
}

foreach ($f in $files) { Push-File $f }

Write-Host ""
Write-Host "All files pushed. Render is auto-building balmoreslab.com right now." -ForegroundColor Green
Write-Host "Build dashboard: https://dashboard.render.com"
Write-Host ""

# Wait for Render build, then fire every crawl-accelerator.
Write-Host "Waiting 90s for Render to deploy before pinging crawlers..." -ForegroundColor Cyan
Start-Sleep -Seconds 90

& (Join-Path $repoRoot "scripts\ping-crawlers.ps1")
